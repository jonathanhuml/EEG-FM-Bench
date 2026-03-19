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
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.zuna.zuna_config import ZunaConfig, ZunaModelArgs
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

    def forward(self, batch: dict) -> torch.Tensor:
        from zuna.inference.AY2l.lingua.apps.AY2latent_bci.eeg_data import chop_and_reshape_signals

        x: torch.Tensor = batch['data'].float()   # [B, n_ch, n_t]
        montage: str = batch['montage'][0]
        B, n_ch, n_t = x.shape

        # Normalise to match Zuna training distribution.
        # Step 1 — global z-score per epoch (approximates normalize_raw() which z-scores
        #   the continuous recording before epoching; std ≈ 1).
        # Step 2 — divide by 10 and clip to ±1 (eeg_eval.py make_batch_iterator:
        #   `eeg_signal = eeg_signal / data_norm` then `.clamp(-data_clip, data_clip)`;
        #   config_infer.yaml: data_norm=10.0, data_clip=1.0 → "ZUNA expects std=0.1").
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
            # eeg_r:  [n_ch * n_coarse, n_fine]
            # cpd_r:  [n_ch * n_coarse, 3]
            # tc_r:   [n_ch * n_coarse, 1]
            # chop_and_reshape_signals may return CPU tensors regardless of input device
            dev = x.device
            tokens_list.append(eeg_r.to(dev))
            tok_idx_list.append(torch.cat([cpd_r.to(dev), tc_r.to(dev)], dim=1))  # [seq_len, 4]
            seq_lens_list.append(int(seq_len))

        # ── Pack into single sequence ──────────────────────────────────────────
        # packed_tokens:  [1, B * n_ch * n_coarse, n_fine]
        # packed_tok_idx: [1, B * n_ch * n_coarse, 4]
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
