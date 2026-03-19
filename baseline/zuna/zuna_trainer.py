"""
ZUNA Trainer using Abstract Base Class.

ZUNA is a 380M-parameter masked diffusion autoencoder for EEG.
The encoder is always used frozen. Input EEG is tokenised into
raw 32-sample windows per channel; all channels are packed into
a single sequence with 4-D RoPE positional encoding (x, y, z, t).

Compatible datasets (5 s @ 256 Hz = 1280 samples):
  - TUEV   (tuev/01_tcp_ar,   21 ch)
  - Mimul-11 (mimul_11/10_20, 60 ch)
  - Things-EEG-2 (things_eeg_2/10_20, 63 ch)

Shape pipeline per forward pass:
  raw EEG          [B, n_ch, 1280]
  tokenise         [B, n_ch, 40, 32]   (40 windows of 32 samples)
  pack             [1, B*n_ch*40, 32]
  EncoderTransformer → [1, B*n_ch*40, latent_dim=32]
  reshape+permute  [B, T=40, C=n_ch, D=32]   → MultiHeadClassifier

Embedding cache path
--------------------
When cfg.training.cache_features=True the trainer:
  1. Runs the frozen encoder once over train/val/test splits.
  2. Saves features [N, T, C, D] to disk as numpy .npy files.
  3. Re-trains only the lightweight classifier head on the cached features.
"""

import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import datasets as hf_datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.zuna.zuna_config import ZunaConfig, ZunaModelArgs
from common.distributed.env import get_is_master, clean_torch_distributed
from common.distributed.loader import DistributedGroupBatchSampler
from data.processor.wrapper import get_dataset_montage

logger = logging.getLogger('baseline')

# ── Constants matching DREAMER / Zuna training setup ──────────────────────────

_ZUNA_XYZ_EXTREMES = torch.tensor([[-0.12, -0.12, -0.12], [0.12, 0.12, 0.12]])
_ZUNA_NUM_BINS = 50

# Mastoid electrodes used in TUH TCP-AR montage → nearest standard positions
_CH_NAME_ALIASES: Dict[str, str] = {
    'A1': 'TP9',
    'A2': 'TP10',
    # TUEV TCP montage uses all-caps; MNE standard montages use mixed case
    'FP1': 'Fp1',
    'FP2': 'Fp2',
    'FZ':  'Fz',
    'CZ':  'Cz',
    'PZ':  'Pz',
}


# ── Channel position helpers ───────────────────────────────────────────────────

def _get_mne_positions(ch_names: List[str]) -> torch.Tensor:
    """
    Return (n_ch, 3) float32 xyz positions (metres) for a list of channel names.
    Looks up MNE standard_1020, then standard_1005, then falls back to (0,0,0)
    with a warning.
    """
    import mne
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos_1020 = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
        pos_1005 = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']

    positions = []
    for ch in ch_names:
        name = _CH_NAME_ALIASES.get(ch, ch)
        if name in pos_1020:
            positions.append(pos_1020[name])
        elif name in pos_1005:
            positions.append(pos_1005[name])
        else:
            logger.warning(f"ZUNA: channel '{ch}' (alias '{name}') not in MNE montage — using (0,0,0)")
            positions.append(np.zeros(3, dtype=np.float32))

    return torch.tensor(np.stack(positions), dtype=torch.float32)


def _build_chan_pos_dict(
    ds_info: dict,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    For each montage in ds_info build (chan_pos, chan_pos_disc) tensors.

    Returns
    -------
    {montage_key: (chan_pos [n_ch,3], chan_pos_disc [n_ch,3])}
    """
    from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import discretize_chan_pos

    result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for ds_name, info in ds_info.items():
        montages = get_dataset_montage(ds_name, info['config'])
        for montage_key, ch_names in montages.items():
            chan_pos = _get_mne_positions(ch_names)
            chan_pos_disc = discretize_chan_pos(chan_pos, _ZUNA_XYZ_EXTREMES, _ZUNA_NUM_BINS)
            result[montage_key] = (chan_pos, chan_pos_disc)
            logger.info(f"ZUNA: built positions for {montage_key} ({len(ch_names)} ch)")

    return result


# ── DataLoader factory ─────────────────────────────────────────────────────────

class ZunaDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(self, dataset: HFDataset, dataset_names, dataset_configs) -> HFDataset:
        return dataset


# ── Unified model ──────────────────────────────────────────────────────────────

class ZunaUnifiedModel(nn.Module):
    """
    Wraps the frozen ZUNA EncoderTransformer + MultiHeadClassifier.

    chan_pos_dict maps montage_key → (chan_pos [n_ch,3], chan_pos_disc [n_ch,3]).
    These tensors are moved to the correct device on the first forward call.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: MultiHeadClassifier,
        chan_pos_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        n_fine: int = 32,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.chan_pos_dict = chan_pos_dict
        self.n_fine = n_fine

    def encode(self, batch: dict) -> torch.Tensor:
        """Run the frozen encoder only; return [B, T, C, D] features."""
        from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import chop_and_reshape_signals

        x: torch.Tensor = batch['data'].float()   # [B, n_ch, n_t]
        montage: str = batch['montage'][0]
        B, n_ch, n_t = x.shape

        # Normalise to match Zuna training distribution.
        mean = x.mean(dim=(1, 2), keepdim=True)               # [B, 1, 1]
        std  = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        x    = ((x - mean) / std / 10.0).clamp(-1.0, 1.0)
        n_coarse = n_t // self.n_fine             # 40 for 1280-sample epochs

        chan_pos, chan_pos_disc = self.chan_pos_dict[montage]
        chan_pos      = chan_pos.to(x.device)
        chan_pos_disc = chan_pos_disc.to(x.device)

        # ── Tokenise each sample in the batch ─────────────────────────────────
        tokens_list:  List[torch.Tensor] = []
        tok_idx_list: List[torch.Tensor] = []
        seq_lens_list: List[int] = []

        for i in range(B):
            eeg_r, _, cpd_r, _, tc_r, seq_len = chop_and_reshape_signals(
                eeg_signal        = x[i],           # [n_ch, n_t]
                chan_pos          = chan_pos,
                chan_pos_discrete = chan_pos_disc,
                chan_dropout      = [],
                tf                = self.n_fine,    # 32
                use_coarse_time   = "B",
            )
            # chop_and_reshape_signals may return CPU tensors regardless of input device
            dev = x.device
            tokens_list.append(eeg_r.to(dev))
            tok_idx_list.append(torch.cat([cpd_r.to(dev), tc_r.to(dev)], dim=1))  # [seq_len, 4]
            seq_lens_list.append(int(seq_len))

        # ── Pack into single sequence ──────────────────────────────────────────
        packed_tokens  = torch.stack(tokens_list).reshape(1, -1, self.n_fine).to(x.device)
        packed_tok_idx = torch.stack(tok_idx_list).reshape(1, -1, 4).to(x.device)
        seq_lens       = torch.tensor(seq_lens_list, dtype=torch.long, device=x.device)

        # ── Encode ────────────────────────────────────────────────────────────
        try:
            enc_out, _ = self.encoder(
                token_values = packed_tokens,
                seq_lens     = seq_lens,
                tok_idx      = packed_tok_idx,
                attn_impl    = "flex_attention",
            )
        except Exception:
            enc_out, _ = self.encoder(
                token_values = packed_tokens,
                seq_lens     = seq_lens,
                tok_idx      = packed_tok_idx,
                attn_impl    = "sdpa",
            )

        # enc_out: [1, B * n_ch * n_coarse, latent_dim]
        latent_dim = enc_out.shape[-1]

        # ── Reshape to [B, T, C, D] ───────────────────────────────────────────
        features = (
            enc_out.squeeze(0)                              # [B*n_ch*n_coarse, D]
                   .reshape(B, n_ch, n_coarse, latent_dim)  # [B, n_ch, 40, D]
                   .permute(0, 2, 1, 3)                     # [B, T=40, C=n_ch, D]
                   .contiguous()
        )
        return features

    def forward(self, batch: dict) -> torch.Tensor:
        features = self.encode(batch)
        montage: str = batch['montage'][0]
        return self.classifier(features, montage)


# ── Cached-feature dataset & model ─────────────────────────────────────────────

class ZunaCachedDataset(Dataset):
    """
    Dataset backed by pre-computed ZUNA encoder embeddings stored as .npy files.

    Supports both:
    - Integer indexing  → {'features': Tensor[T,C,D], 'label': int, 'montage': str}
    - String column access (dataset['montage']) required by DistributedGroupBatchSampler
    """

    def __init__(self, features_path: str, labels: np.ndarray, montages: List[str]):
        self._features_path = features_path
        self.features = np.load(features_path, mmap_mode='r')   # [N, T, C, D]
        self.labels   = np.asarray(labels, dtype=np.int64)
        self.montages = list(montages)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            # HF-style column access used by DistributedGroupBatchSampler
            if idx == 'montage':
                return self.montages
            if idx == 'label':
                return self.labels.tolist()
            raise KeyError(f"ZunaCachedDataset: unknown column '{idx}'")
        # Copy out of the memmap to get a proper numpy array before converting
        feat = torch.tensor(np.array(self.features[idx]), dtype=torch.float32)
        return {
            'features': feat,
            'label':    int(self.labels[idx]),
            'montage':  self.montages[idx],
        }


class ZunaCachedModel(nn.Module):
    """Classifier-only model that operates on pre-computed ZUNA embeddings."""

    def __init__(self, classifier: MultiHeadClassifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, batch: dict) -> torch.Tensor:
        features = batch['features'].float()   # [B, T, C, D]
        montage  = batch['montage'][0]
        return self.classifier(features, montage)


# ── Trainer ────────────────────────────────────────────────────────────────────

class ZunaTrainer(AbstractTrainer):
    """ZUNA trainer that inherits from AbstractTrainer."""

    def __init__(self, cfg: ZunaConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.dataloader_factory = ZunaDataLoaderFactory(
            batch_size  = self.cfg.data.batch_size,
            num_workers = self.cfg.data.num_workers,
            seed        = self.cfg.seed,
        )

        self.encoder  = None
        self.classifier = None

        self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self) -> nn.Module:
        """Download ZUNA from HuggingFace, extract encoder, build classifier."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as safe_load
        from zuna.inference.AY2l.lingua.apps.AY2latent_bci.transformer import (
            DecoderTransformerArgs, EncoderDecoder,
        )
        from zuna.inference.AY2l.lingua.lingua.args import dataclass_from_dict

        model_cfg: ZunaModelArgs = self.cfg.model
        logger.info("Setting up ZUNA encoder (downloading from HuggingFace if needed)…")

        # ── Load HF config → model args ───────────────────────────────────────
        cfg_path = hf_hub_download(
            repo_id  = model_cfg.pretrained_repo,
            filename = model_cfg.pretrained_config_file,
        )
        with open(cfg_path) as f:
            hf_cfg = json.load(f)
        model_args: DecoderTransformerArgs = dataclass_from_dict(
            DecoderTransformerArgs, hf_cfg["model"]
        )

        # ── Load weights → extract encoder ────────────────────────────────────
        weights_path = hf_hub_download(
            repo_id  = model_cfg.pretrained_repo,
            filename = model_cfg.pretrained_weights_file,
        )
        sd_raw = safe_load(weights_path, device="cpu")
        sd = {k.removeprefix("model."): v for k, v in sd_raw.items()}

        enc_dec = EncoderDecoder(model_args)
        enc_dec.load_state_dict(sd, strict=True)

        self.encoder = enc_dec.encoder
        latent_dim   = model_args.encoder_output_dim
        del enc_dec

        logger.info(f"ZUNA encoder loaded — latent_dim={latent_dim}, "
                    f"params={sum(p.numel() for p in self.encoder.parameters()):,}")

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        # ── Build per-montage channel position tensors ─────────────────────────
        chan_pos_dict = _build_chan_pos_dict(self.ds_info)

        # ── Build classifier ──────────────────────────────────────────────────
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg     = model_cfg.classifier_head
        n_coarse     = int(self.cfg.fs * 5) // model_cfg.n_fine  # 1280 // 32 = 40

        ds_shape_info: Dict[str, Tuple[int, int, int]] = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                ds_shape_info[montage_key] = (n_coarse, n_channels, latent_dim)

        self.classifier = MultiHeadClassifier(
            embed_dim    = latent_dim,
            head_configs = head_configs,
            head_cfg     = head_cfg,
            ds_shape_info = ds_shape_info,
            t_sne        = model_cfg.t_sne,
        )
        logger.info(f"ZUNA classifier built for: {list(head_configs.keys())}")

        # ── Assemble unified model ────────────────────────────────────────────
        model = ZunaUnifiedModel(
            encoder       = self.encoder,
            classifier    = self.classifier,
            chan_pos_dict = chan_pos_dict,
            n_fine        = model_cfg.n_fine,
        )

        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=False)
        self.model = model

        return model

    def load_checkpoint(self, checkpoint_path: str):
        """Not used — weights are always fetched from HuggingFace in setup_model."""
        logger.info(f"load_checkpoint called with {checkpoint_path} — "
                    "ZUNA weights are loaded from HuggingFace in setup_model, ignoring.")

    # ── Embedding-cache helpers ────────────────────────────────────────────────

    def _make_precompute_factory(self) -> ZunaDataLoaderFactory:
        """DataLoader factory using precompute_batch_size instead of training batch_size."""
        return ZunaDataLoaderFactory(
            batch_size  = self.cfg.training.precompute_batch_size,
            num_workers = self.cfg.data.num_workers,
            seed        = self.cfg.seed,
        )

    def _precompute_split(
        self,
        raw_model: ZunaUnifiedModel,
        split: hf_datasets.NamedSplit,
        split_name: str,
        cache_dir: str,
        precompute_factory: ZunaDataLoaderFactory,
    ):
        """Encode one data split and save features/labels/montages to disk."""
        feat_path    = os.path.join(cache_dir, f'{split_name}_features.npy')
        labels_path  = os.path.join(cache_dir, f'{split_name}_labels.npy')
        montage_path = os.path.join(cache_dir, f'{split_name}_montages.json')

        if os.path.exists(feat_path):
            logger.info(f"[cache] '{split_name}' already exists at {feat_path}, skipping")
            return

        logger.info(f"[cache] Encoding '{split_name}' split …")

        # Build a dataloader at the precompute batch size
        mixed = (split == hf_datasets.Split.TRAIN and self.cfg.multitask)
        loaders, _ = precompute_factory.create_dataloader(
            datasets_config = self.ds_conf,
            mixed           = mixed,
            fs              = self.cfg.fs,
            num_replicas    = self.world_size,
            rank            = self.local_rank,
            split           = split,
        )
        # create_dataloader returns a single loader for mixed=True,
        # a list of loaders for mixed=False
        if isinstance(loaders, list):
            loader = loaders[0]
        else:
            loader = loaders

        all_features: List[np.ndarray] = []
        all_labels:   List[int]        = []
        all_montages: List[str]        = []

        raw_model.encoder.eval()
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if step % 100 == 0:
                    logger.info(f"  [{split_name}] step {step} / {len(loader)}")
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                features = raw_model.encode(batch)   # [B, T, C, D]
                all_features.append(features.cpu().numpy().astype(np.float32))
                all_labels.extend(batch['label'].cpu().numpy().tolist())
                all_montages.extend(list(batch['montage']))

        features_arr = np.concatenate(all_features, axis=0)   # [N, T, C, D]
        labels_arr   = np.array(all_labels, dtype=np.int64)

        np.save(feat_path,    features_arr)
        np.save(labels_path,  labels_arr)
        with open(montage_path, 'w') as f:
            json.dump(all_montages, f)

        logger.info(f"[cache] '{split_name}' saved: {features_arr.shape} → {feat_path}")

    def _load_cached_dataset(self, cache_dir: str, split_name: str) -> ZunaCachedDataset:
        """Load a previously saved cached dataset."""
        feat_path    = os.path.join(cache_dir, f'{split_name}_features.npy')
        labels_path  = os.path.join(cache_dir, f'{split_name}_labels.npy')
        montage_path = os.path.join(cache_dir, f'{split_name}_montages.json')

        labels = np.load(labels_path)
        with open(montage_path) as f:
            montages = json.load(f)

        ds = ZunaCachedDataset(feat_path, labels, montages)
        logger.info(f"[cache] Loaded '{split_name}': {len(ds)} samples from {feat_path}")
        return ds

    # ── Cached training path ───────────────────────────────────────────────────

    def _cached_train_epoch(
        self,
        train_loader: DataLoader,
        train_sampler: DistributedGroupBatchSampler,
    ):
        """Train one epoch on cached embeddings (classifier only, no encoder)."""
        self.model.train()
        train_sampler.set_epoch(self.epoch)

        for step_in_epoch, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels  = batch['label']
            ds_name = batch['montage'][0].split('/')[0]

            logits, loss = self.train_step(batch, labels)

            if torch.isnan(loss):
                logger.warning(f"NaN loss at step {self.current_step}")

            self.scaler.scale(loss).backward()
            grad_norm = self._clip_grad_norm_()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.current_step % self.cfg.logging.log_step_interval == 0:
                preds    = torch.argmax(logits, dim=-1)
                step_acc = (preds == labels).float().mean()

                loss_t = loss.clone().detach()
                acc_t  = step_acc.clone().detach()

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(acc_t,  op=torch.distributed.ReduceOp.AVG)

                if get_is_master():
                    log_data = {
                        'train/epoch':    self.epoch,
                        'train/step':     self.current_step,
                        'train/loss_ce':  loss_t.cpu().item(),
                        'train/acc':      acc_t.cpu().item(),
                        'train/grad_norm': grad_norm,
                        'train/header_lr': self.get_current_lr()[0],
                    }
                    if not self.multitask:
                        log_data = {f"{ds_name}/{k}": v for k, v in log_data.items()}
                    if self.cfg.logging.use_cloud:
                        self._log_to_cloud(log_data)
                    from baseline.abstract.trainer import format_console_log_dict
                    logger.info(format_console_log_dict(log_data, prefix='train'))

            self.current_step += 1
            self.scheduler.step()

    def _run_cached_training(self):
        """Full training pipeline using pre-computed embedding cache."""
        torch.distributed.barrier()
        self.collect_dataset_info(mixed=True)

        # 1. Build full model (encoder + classifier) — needed for precomputation
        full_model = self.setup_model()
        raw_model  = full_model.module if hasattr(full_model, 'module') else full_model

        # 2. Determine / create cache directory
        cache_dir = self.cfg.training.features_cache_dir
        if cache_dir is None:
            ds_tag    = '_'.join(self.ds_conf.keys())
            cache_dir = os.path.join('assets', 'data', 'cache', f'zuna_{ds_tag}')
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"[cache] Cache directory: {cache_dir}")

        # 3. Pre-compute embeddings for all splits (skips if already on disk)
        precompute_factory = self._make_precompute_factory()
        for split_enum, split_name in [
            (hf_datasets.Split.TRAIN,      'train'),
            (hf_datasets.Split.VALIDATION, 'valid'),
            (hf_datasets.Split.TEST,       'test'),
        ]:
            self._precompute_split(
                raw_model, split_enum, split_name, cache_dir, precompute_factory
            )

        # 4. Build cached datasets
        train_ds = self._load_cached_dataset(cache_dir, 'train')
        valid_ds = self._load_cached_dataset(cache_dir, 'valid')
        test_ds  = self._load_cached_dataset(cache_dir, 'test')

        # 5. Replace self.model with classifier-only ZunaCachedModel
        cached_model = ZunaCachedModel(raw_model.classifier)
        cached_model = cached_model.to(self.device)
        cached_model = self.maybe_wrap_ddp(cached_model, find_unused_parameters=False)
        self.model   = cached_model

        # 6. Build training DataLoader with DistributedGroupBatchSampler
        train_sampler = DistributedGroupBatchSampler(
            dataset      = train_ds,
            batch_size   = self.cfg.data.batch_size,
            num_replicas = self.world_size,
            rank         = self.local_rank,
            seed         = self.cfg.seed,
            drop_last    = False,
        )
        num_workers = self.cfg.data.num_workers
        train_loader = DataLoader(
            train_ds,
            batch_sampler = train_sampler,
            num_workers   = num_workers,
            pin_memory    = True,
            persistent_workers = num_workers > 0,
        )

        # 7. Build eval DataLoaders (simple, no custom sampler needed)
        valid_loader = DataLoader(
            valid_ds,
            batch_size  = self.cfg.data.batch_size,
            shuffle     = False,
            num_workers = num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size  = self.cfg.data.batch_size,
            shuffle     = False,
            num_workers = num_workers,
        )

        # 8. Setup optimizer / scheduler using cached train loader
        self.setup_optimizer_and_scheduler(cached_model, train_loader)

        logger.info(
            f"[cache] Classifier training: {len(train_ds)} train / "
            f"{len(valid_ds)} val / {len(test_ds)} test samples"
        )
        logger.info(f"[cache] Starting {self.cfg.training.max_epochs} epochs …")

        # 9. Training loop
        for epoch in range(self.cfg.training.max_epochs):
            self.epoch = epoch
            torch.distributed.barrier()

            self._cached_train_epoch(train_loader, train_sampler)
            self.eval_epoch([valid_loader], 'eval')
            self.eval_epoch([test_loader],  'test')

            if (epoch + 1) % self.cfg.logging.ckpt_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint(is_milestone=True)
        self.finish_cloud_logging()
        clean_torch_distributed(self.local_rank)
        logger.info("[cache] Cached training completed!")

    # ── Entry point override ───────────────────────────────────────────────────

    def run_unified_training(self):
        """Route to cached or standard training depending on config."""
        if self.cfg.training.cache_features:
            self._run_cached_training()
        else:
            super().run_unified_training()


def main():
    import sys
    from omegaconf import OmegaConf

    if len(sys.argv) != 2:
        print("Usage: python zuna_trainer.py config.yaml")
        sys.exit(1)

    file_cfg   = OmegaConf.load(sys.argv[1])
    code_cfg   = OmegaConf.create(ZunaConfig().model_dump())
    merged     = OmegaConf.merge(code_cfg, file_cfg)
    cfg        = ZunaConfig.model_validate(OmegaConf.to_container(merged, resolve=True))

    ZunaTrainer(cfg).run()


if __name__ == "__main__":
    main()
