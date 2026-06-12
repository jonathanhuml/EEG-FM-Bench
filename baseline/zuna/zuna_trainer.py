"""Frozen ZUNA encoder integration using the original AY2latent modules.

The adapter follows neuroai's ZUNA port: average reference, per-channel
time-axis z-score, native-frame channel coordinates, coarse-time-major token
packing, and restoration to FM-Bench's ``[B, T, C, D]`` feature convention.
Segment length is derived from each input batch and can vary across datasets.

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
import sys
import warnings
from pathlib import Path
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


# Copied from AY2latent/lingua/apps/AY2latent_bci/eeg_data.py.
# We keep these two small helpers local because that source file currently has
# an unrelated syntax error near the bottom that prevents importing it at all.
def chop_and_reshape_signals(
    eeg_signal,
    chan_pos=None,
    chan_pos_discrete=None,
    tf=128,
    use_coarse_time="B",
):
    """Reshape [channels, time] EEG into ZUNA's token sequence layout."""
    num_chans, num_tpts = eeg_signal.shape

    if use_coarse_time == "C":
        tc = 1
    else:
        assert num_tpts % tf == 0, f"{num_tpts=} is not divisible by tf={tf}. {num_chans=}"
        tc = num_tpts // tf

    if use_coarse_time == "A":
        seqlen = num_chans * tc
        eeg_reshaped = eeg_signal.reshape(num_chans, tc, tf).transpose(0, 1).reshape(seqlen, tf)
        chan_pos_reshaped = chan_pos.repeat((tc, 1)) if chan_pos is not None else None
        chan_pos_discrete_reshaped = chan_pos_discrete.repeat((tc, 1)) if chan_pos_discrete is not None else None
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1).repeat((tc, 1))
        tc_reshaped = torch.arange(tc).repeat((num_chans, 1)).T.reshape(seqlen, 1)

    elif use_coarse_time == "B" or use_coarse_time == "D":
        seqlen = num_chans * tc
        eeg_reshaped = eeg_signal.reshape(num_chans, tc, tf).reshape(seqlen, tf)
        chan_pos_reshaped = chan_pos.repeat_interleave(repeats=tc, dim=0) if chan_pos is not None else None
        chan_pos_discrete_reshaped = chan_pos_discrete.repeat_interleave(repeats=tc, dim=0) if chan_pos_discrete is not None else None
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1).repeat_interleave(repeats=tc, dim=0)
        tc_reshaped = torch.arange(tc).repeat((num_chans, 1)).reshape(seqlen, 1)

    elif use_coarse_time == "C":
        seqlen = num_chans
        eeg_reshaped = eeg_signal[:, :tf]
        chan_pos_reshaped = chan_pos
        chan_pos_discrete_reshaped = chan_pos_discrete
        tc_reshaped = torch.zeros(num_chans, 1)
        chan_id_reshaped = torch.arange(num_chans).unsqueeze(-1)

    else:
        raise NotImplementedError(
            f"use_coarse_time={use_coarse_time!r} must be one of A, B, C, or D"
        )

    if use_coarse_time == "D":
        indices = list(range(0, tc * num_chans, tc))
        eeg_by_channel = []
        pos_by_channel = []
        pos_discrete_by_channel = []
        tc_by_channel = []
        chan_id_by_channel = []
        seq_lens = []
        for i in indices:
            st, nd = i, i + tc
            eeg_by_channel.append(eeg_reshaped[st:nd, :])
            pos_by_channel.append(chan_pos_reshaped[st:nd, :])
            pos_discrete_by_channel.append(chan_pos_discrete_reshaped[st:nd, :])
            tc_by_channel.append(tc_reshaped[st:nd, :])
            chan_id_by_channel.append(chan_id_reshaped[st:nd, :])
            seq_lens.append(tc)
        eeg_reshaped = eeg_by_channel
        chan_pos_reshaped = pos_by_channel
        chan_pos_discrete_reshaped = pos_discrete_by_channel
        tc_reshaped = tc_by_channel
        chan_id_reshaped = chan_id_by_channel
        seqlen = seq_lens

    return (
        eeg_reshaped,
        chan_pos_reshaped,
        chan_pos_discrete_reshaped,
        chan_id_reshaped,
        tc_reshaped,
        seqlen,
        num_chans,
    )


def discretize_chan_pos(chan_pos, xyz_extremes, num_bins):
    """Discretize continuous channel positions into integer xyz bins."""
    xyz_min = xyz_extremes[0]
    xyz_max = xyz_extremes[1]

    within_min = (chan_pos >= xyz_min).all()
    within_max = (chan_pos <= xyz_max).all()

    if not (within_min and within_max):
        out_of_bounds_min = chan_pos < xyz_min
        out_of_bounds_max = chan_pos > xyz_max
        warnings.warn(
            f"Channel positions out of bounds detected!\n"
            f"  Positions below min: {out_of_bounds_min.sum().item()} elements\n"
            f"  Positions above max: {out_of_bounds_max.sum().item()} elements\n"
            f"  xyz_min: {xyz_min.tolist()}\n"
            f"  xyz_max: {xyz_max.tolist()}\n"
            f"  chan_pos range: [{chan_pos.min(dim=0).values.tolist()}, "
            f"{chan_pos.max(dim=0).values.tolist()}]"
        )

    chan_pos_normalized = (chan_pos - xyz_min) / (xyz_max - xyz_min)
    chan_pos_discrete = (chan_pos_normalized * num_bins).long()
    return torch.clamp(chan_pos_discrete, 0, num_bins - 1)


# ── Channel position helpers ──────────────────────────────────────────────────

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
    # Additional midline channels uppercased by standardize_chs_names (e.g. THINGS-EEG2)
    'AFZ': 'AFz',
    'FCZ': 'FCz',
    'CPZ': 'CPz',
    'POZ': 'POz',
    'OZ':  'Oz',
    # Inion channel in Mimul-11 (60-ch montage)
    'IZ':  'Iz',
}


# ── Channel position helpers ───────────────────────────────────────────────────

def _get_mne_positions(
    ch_names: List[str],
    invalid_channel_position: float,
) -> torch.Tensor:
    """
    Return (n_ch, 3) float32 xyz positions (metres) for a list of channel names.
    Looks up MNE standard_1020, then standard_1005. Unknown channels use the
    same sentinel as neuroai and are excluded before ZUNA tokenization.
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
            logger.warning(
                "ZUNA: channel '%s' (alias '%s') not in MNE montage; excluding it",
                ch,
                name,
            )
            positions.append(
                np.full(3, invalid_channel_position, dtype=np.float32)
            )

    return torch.tensor(np.stack(positions), dtype=torch.float32)


def _build_chan_pos_dict(
    ds_info: dict,
    invalid_channel_position: float,
) -> Dict[str, torch.Tensor]:
    """Build MNE-head coordinate tensors for every FM-Bench montage."""
    result: Dict[str, torch.Tensor] = {}
    for ds_name, info in ds_info.items():
        montages = get_dataset_montage(ds_name, info['config'])
        for montage_key, ch_names in montages.items():
            result[montage_key] = _get_mne_positions(
                ch_names,
                invalid_channel_position,
            )
            logger.info(f"ZUNA: built positions for {montage_key} ({len(ch_names)} ch)")

    return result


# ── DataLoader factory ─────────────────────────────────────────────────────────

class ZunaDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(self, dataset: HFDataset, dataset_names, dataset_configs) -> HFDataset:
        return dataset


# ── Unified model ──────────────────────────────────────────────────────────────

class ZunaUnifiedModel(nn.Module):
    """Wrap the AY2latent encoder and FM-Bench classifier."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: MultiHeadClassifier,
        chan_pos_dict: Dict[str, torch.Tensor],
        n_fine: int = 32,
        data_norm: float = 10.0,
        data_clip: Optional[float] = 1.0,
        do_avg_ref: bool = True,
        num_bins: int = 100,
        channel_position_montage: str = "standard_1005",
        invalid_channel_position: float = -0.1,
        attn_impl: str = "flex_attention",
        skip_input_norm: bool = False,
        feature_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.chan_pos_dict = chan_pos_dict
        self.n_fine = n_fine
        self.data_norm = data_norm
        self.data_clip = data_clip
        self.do_avg_ref = do_avg_ref
        self.num_bins = num_bins
        self.invalid_channel_position = invalid_channel_position
        self.attn_impl = attn_impl
        self.skip_input_norm = skip_input_norm
        self.feature_norm = feature_norm
        native_to_head = self._native_to_head_transform(channel_position_montage)
        self.register_buffer(
            "native_to_head_rotation",
            native_to_head[:3, :3],
        )
        self.register_buffer(
            "native_to_head_translation",
            native_to_head[:3, 3],
        )
        self.register_buffer(
            "xyz_extremes",
            torch.tensor(
                [[-0.12, -0.12, -0.12], [0.12, 0.12, 0.12]],
                dtype=torch.float32,
            ),
        )

    @staticmethod
    def _native_to_head_transform(montage_name: str) -> torch.Tensor:
        if montage_name in {"standard_1005", "standard_1020"}:
            return torch.tensor(
                [
                    [0.999993681908, 0.003551873844, 0.000202048104, -0.001762724953],
                    [-0.003557568649, 0.998389124870, 0.056625857949, 0.031094350428],
                    [-0.000000594737, -0.056626219302, 0.998395442963, 0.039597249076],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )

        import mne

        montage = mne.channels.make_standard_montage(montage_name)
        transform = mne.channels.compute_native_head_t(montage)["trans"]
        return torch.as_tensor(transform, dtype=torch.float64)

    def _valid_channel_mask(self, channel_positions: torch.Tensor) -> torch.Tensor:
        sentinel = torch.isclose(
            channel_positions,
            torch.as_tensor(
                self.invalid_channel_position,
                dtype=channel_positions.dtype,
                device=channel_positions.device,
            ),
            rtol=0.0,
            atol=1e-6,
        ).all(dim=-1)
        valid = torch.isfinite(channel_positions).all(dim=-1) & ~sentinel
        if (~valid).all(dim=1).any():
            raise ValueError("ZUNA received a sample with no valid channel positions")
        return valid

    def _to_zuna_native_frame(
        self,
        channel_positions: torch.Tensor,
        valid_channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        native_positions = channel_positions.clone()
        points = channel_positions[valid_channel_mask].to(torch.float64)
        rotation = self.native_to_head_rotation.to(points.device)
        translation = self.native_to_head_translation.to(points.device)
        native_points = torch.linalg.solve(rotation, (points - translation).T).T
        native_positions[valid_channel_mask] = native_points.to(
            native_positions.dtype
        )
        return native_positions

    def _preprocess_eeg(
        self,
        x: torch.Tensor,
        valid_channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.skip_input_norm:
            return x.float()

        x = x.float()
        mask = valid_channel_mask.unsqueeze(-1).to(x.dtype)
        if self.do_avg_ref:
            valid_count = mask.sum(dim=1, keepdim=True)
            channel_mean = (x * mask).sum(dim=1, keepdim=True) / valid_count
            x = (x - channel_mean) * mask

        x = (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
        )
        x = x / self.data_norm
        if self.data_clip is not None:
            x = x.clamp(-self.data_clip, self.data_clip)
        return x * mask

    def _sequence_repacking(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
        valid_channel_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_inputs: List[torch.Tensor] = []
        seq_lens: List[torch.Tensor] = []
        tok_indices: List[torch.Tensor] = []

        for eeg, pos, valid in zip(
            x,
            channel_positions,
            valid_channel_mask,
            strict=True,
        ):
            eeg = eeg[valid]
            pos = pos[valid]
            pos_discrete = discretize_chan_pos(
                pos.float(),
                self.xyz_extremes.to(pos.device),
                self.num_bins,
            )
            (
                encoder_input,
                _chan_pos,
                pos_discrete,
                _chan_id,
                t_coarse,
                seq_len,
                _num_chans,
            ) = chop_and_reshape_signals(
                eeg_signal=eeg,
                chan_pos=pos,
                chan_pos_discrete=pos_discrete,
                tf=self.n_fine,
                use_coarse_time="A",
            )
            encoder_inputs.append(encoder_input.to(eeg.device))
            seq_lens.append(
                torch.tensor([seq_len], dtype=torch.long, device=eeg.device)
            )
            tok_indices.append(
                torch.cat(
                    (
                        pos_discrete.to(eeg.device).unsqueeze(0),
                        t_coarse.to(eeg.device).long().unsqueeze(0),
                    ),
                    dim=2,
                )
            )

        return (
            torch.cat(encoder_inputs, dim=0),
            torch.cat(seq_lens, dim=0),
            torch.cat(tok_indices, dim=1),
        )

    def _restore_dense_outputs(
        self,
        latent: torch.Tensor,
        seq_lens: torch.Tensor,
        valid_channel_mask: torch.Tensor,
        n_times: int,
    ) -> torch.Tensor:
        coarse_time = n_times // self.n_fine
        sparse_outputs = latent.squeeze(0).split(
            seq_lens.detach().cpu().tolist(),
            dim=0,
        )
        dense_outputs = []
        for sparse, valid in zip(
            sparse_outputs,
            valid_channel_mask,
            strict=True,
        ):
            dense = sparse.new_zeros(
                valid.shape[0],
                coarse_time,
                sparse.shape[-1],
            )
            dense[valid] = sparse.reshape(
                coarse_time,
                int(valid.sum()),
                sparse.shape[-1],
            ).transpose(0, 1)
            dense_outputs.append(dense)
        return torch.stack(dense_outputs)

    def encode(self, batch: dict) -> torch.Tensor:
        """Run the frozen encoder only; return [B, T, C, D] features."""
        x: torch.Tensor = batch['data'].float()
        montage: str = batch['montage'][0]
        if x.shape[-1] % self.n_fine != 0:
            raise ValueError(
                f"ZUNA requires T divisible by {self.n_fine}; got {x.shape[-1]}"
            )

        positions = self.chan_pos_dict[montage].to(x.device)
        positions = positions.unsqueeze(0).expand(x.shape[0], -1, -1)
        valid = self._valid_channel_mask(positions)
        native_positions = self._to_zuna_native_frame(positions, valid)
        x = self._preprocess_eeg(x, valid)
        encoder_input, seq_lens, tok_idx = self._sequence_repacking(
            x,
            native_positions,
            valid,
        )
        do_idx = encoder_input.sum(dim=-1) == 0
        encoder_result = self.encoder(
            token_values=encoder_input.unsqueeze(0),
            seq_lens=seq_lens,
            tok_idx=tok_idx,
            do_idx=do_idx,
            attn_impl=self.attn_impl,
        )
        latent = encoder_result[0]
        dense = self._restore_dense_outputs(
            latent,
            seq_lens,
            valid,
            x.shape[-1],
        )
        return dense.permute(0, 2, 1, 3).contiguous()

    def forward(self, batch: dict) -> torch.Tensor:
        features = self.encode(batch)   # [B, T, C, D]
        if self.feature_norm is not None:
            shape = features.shape
            features = self.feature_norm(features.reshape(shape[0], -1)).reshape(shape)
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

    def __init__(
        self,
        classifier: MultiHeadClassifier,
        feature_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.feature_norm = feature_norm

    def forward(self, batch: dict) -> torch.Tensor:
        features = batch['features'].float()   # [B, T, C, D]
        if self.feature_norm is not None:
            shape = features.shape
            features = self.feature_norm(features.reshape(shape[0], -1)).reshape(shape)
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
        """Load the original AY2latent encoder from a distributed checkpoint."""
        model_cfg: ZunaModelArgs = self.cfg.model
        ay2latent_root = Path(model_cfg.ay2latent_root).expanduser().resolve()
        checkpoint_path = Path(model_cfg.checkpoint_path).expanduser().resolve()
        if not (ay2latent_root / "apps" / "AY2latent_bci").is_dir():
            raise FileNotFoundError(
                f"AY2latent checkout not found at {ay2latent_root}"
            )
        if not (checkpoint_path / ".metadata").is_file():
            raise FileNotFoundError(
                f"ZUNA distributed checkpoint not found at {checkpoint_path}"
            )
        if str(ay2latent_root) not in sys.path:
            sys.path.insert(0, str(ay2latent_root))

        from apps.AY2latent_bci.transformer import (
            DecoderTransformerArgs,
            EncoderDecoder,
        )
        from lingua.args import dataclass_from_dict
        from lingua.checkpoint import load_from_checkpoint

        logger.info("Setting up ZUNA from AY2latent checkpoint %s", checkpoint_path)
        model_kwargs = dict(model_cfg.model_kwargs)
        model_kwargs["encoder_latent_downsample_factor"] = 1
        model_kwargs["ape_dim"] = 0
        model_args: DecoderTransformerArgs = dataclass_from_dict(
            DecoderTransformerArgs,
            model_kwargs,
        )
        enc_dec = EncoderDecoder(model_args)
        enc_dec.init_weights()
        load_from_checkpoint(
            str(checkpoint_path),
            enc_dec,
            model_key="model",
        )

        self.encoder = enc_dec.encoder
        latent_dim = model_args.encoder_output_dim
        del enc_dec

        logger.info(
            "ZUNA encoder loaded: latent_dim=%s, params=%s",
            latent_dim,
            f"{sum(p.numel() for p in self.encoder.parameters()):,}",
        )

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        encoder_trainable = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        logger.info("ZUNA encoder frozen: trainable encoder params=%s", encoder_trainable)

        # ── Build per-montage channel position tensors ─────────────────────────
        chan_pos_dict = _build_chan_pos_dict(
            self.ds_info,
            model_cfg.invalid_channel_position,
        )

        # ── Build classifier ──────────────────────────────────────────────────
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg     = model_cfg.classifier_head
        ds_shape_info: Dict[str, Tuple[int, int, int]] = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                if n_timepoints % model_cfg.n_fine != 0:
                    raise ValueError(
                        f"{montage_key}: {n_timepoints} samples is not divisible "
                        f"by ZUNA n_fine={model_cfg.n_fine}"
                    )
                n_coarse = n_timepoints // model_cfg.n_fine
                ds_shape_info[montage_key] = (n_coarse, n_channels, latent_dim)

        self.classifier = MultiHeadClassifier(
            embed_dim    = latent_dim,
            head_configs = head_configs,
            head_cfg     = head_cfg,
            ds_shape_info = ds_shape_info,
            t_sne        = model_cfg.t_sne,
        )
        logger.info(f"ZUNA classifier built for: {list(head_configs.keys())}")

        # ── Optional per-feature BatchNorm (mimics StandardScaler) ────────────
        feature_norm: Optional[nn.Module] = None
        if model_cfg.use_feature_norm:
            first_shape = next(iter(ds_shape_info.values()))  # (T=40, C, D)
            n_flat = first_shape[0] * first_shape[1] * first_shape[2]
            feature_norm = nn.BatchNorm1d(n_flat, track_running_stats=True, affine=False)
            logger.info(f"ZUNA: feature_norm enabled — BatchNorm1d({n_flat})")

        # ── Assemble unified model ────────────────────────────────────────────
        model = ZunaUnifiedModel(
            encoder          = self.encoder,
            classifier       = self.classifier,
            chan_pos_dict    = chan_pos_dict,
            n_fine           = model_cfg.n_fine,
            data_norm        = model_cfg.data_norm,
            data_clip        = model_cfg.data_clip,
            do_avg_ref       = model_cfg.do_avg_ref,
            num_bins         = model_cfg.num_bins_discretize_xyz_chan_pos,
            channel_position_montage = model_cfg.channel_position_montage,
            invalid_channel_position = model_cfg.invalid_channel_position,
            attn_impl        = model_cfg.attn_impl,
            skip_input_norm  = model_cfg.skip_input_norm,
            feature_norm     = feature_norm,
        )

        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=False)
        self.model = model

        return model

    def load_checkpoint(self, checkpoint_path: str):
        """ZUNA's AY2latent checkpoint is loaded during ``setup_model``."""
        logger.info("ZUNA checkpoint is loaded during setup_model: %s", checkpoint_path)

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
                    logger.info(
                        f"[cache] Encoding '{split_name}' split: step {step} / {len(loader)}"
                    )
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
        # Log param counts before swapping model (encoder still accessible via raw_model)
        self.log_model_param_counts()

        cached_model = ZunaCachedModel(
            classifier   = raw_model.classifier,
            feature_norm = raw_model.feature_norm,
        )
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
