#!/usr/bin/env python3
"""
PSD Trainer using Abstract Base Class

A fixed-feature-extractor baseline that computes per-channel Welch power
spectral density (via scipy.signal.welch) and feeds the log-PSD into the
framework's standard classification head (default: avg_pool).

Encoder output shape: [B, 1, C, n_freq_bins]
  T=1          — spectrum is treated as a single "frame"
  C            — EEG channels (preserved so the head pools over channels)
  n_freq_bins  — number of frequency bins retained after optional band filter

Because freeze_encoder=True and the encoder has no learnable parameters,
no gradients are needed through the scipy call.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from scipy.signal import welch
from torch import nn, Tensor
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.psd.psd_config import PSDConfig, PSDModelArgs


logger = logging.getLogger("baseline")


# ── Encoder ────────────────────────────────────────────────────────────────────

class WelchPSDEncoder(nn.Module):
    """
    Fixed (no-parameter) per-channel Welch PSD encoder.

    Steps per forward call:
      1. Move data to CPU numpy (scipy requirement)
      2. scipy.signal.welch → power in µV²/Hz per channel
      3. Optional band-pass: zero-out bins outside [fmin, fmax]
      4. Log-compress: log(1 + power)
      5. Reshape to [B, 1, C, n_freq_bins]

    Args:
        fs:       Sampling rate in Hz.
        nperseg:  Welch window length in samples (default 256 = 1 s at 256 Hz).
        noverlap: Welch overlap in samples (default nperseg // 2).
        fmin:     Low-frequency cutoff (Hz); None = no lower bound.
        fmax:     High-frequency cutoff (Hz); None = no upper bound.
    """

    def __init__(
        self,
        fs: int,
        nperseg: int,
        noverlap: Optional[int],
        fmin: Optional[float],
        fmax: Optional[float],
    ):
        super().__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.fmin = fmin
        self.fmax = fmax

        # Pre-compute the frequency axis and band mask from Welch params.
        # welch returns freqs of length nperseg // 2 + 1.
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        band_mask = np.ones(len(freqs), dtype=bool)
        if fmin is not None:
            band_mask &= freqs >= fmin
        if fmax is not None:
            band_mask &= freqs <= fmax

        self.n_freq_bins: int = int(band_mask.sum())
        # Store as buffer so it travels to the right device automatically.
        self.register_buffer(
            "band_mask",
            torch.from_numpy(band_mask),  # bool tensor
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, T]  (float tensor, any device)
        Returns:
            [B, 1, C, n_freq_bins]  (float tensor, same device as x)
        """
        device = x.device
        x_np = x.detach().cpu().numpy()  # [B, C, T]

        # scipy.signal.welch processes the last axis by default.
        _, psd_np = welch(
            x_np,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            axis=-1,
        )  # psd_np: [B, C, nperseg//2+1]

        psd = torch.from_numpy(psd_np).to(device=device, dtype=x.dtype)

        # Apply band mask and log-compress.
        psd = psd[..., self.band_mask]    # [B, C, n_freq_bins]
        psd = torch.log1p(psd)

        return psd.unsqueeze(1)            # [B, 1, C, n_freq_bins]


# ── Unified model ──────────────────────────────────────────────────────────────

class PSDUnifiedModel(nn.Module):
    """Welch PSD encoder + multi-head classification head."""

    def __init__(self, encoder: WelchPSDEncoder, classifier: MultiHeadClassifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        x = batch["data"]           # [B, C, T]
        montage = batch["montage"][0]
        features = self.encoder(x)  # [B, 1, C, n_freq_bins]
        return self.classifier(features, montage)


# ── DataLoader factory (identity — no special adapter needed) ─────────────────

class PSDDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> HFDataset:
        return dataset


# ── Trainer ────────────────────────────────────────────────────────────────────

class PSDTrainer(AbstractTrainer):
    """PSD trainer — fixed Welch encoder, trainable classification head only."""

    def __init__(self, cfg: PSDConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.dataloader_factory = PSDDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
        )

        self.encoder: Optional[WelchPSDEncoder] = None
        self.classifier: Optional[MultiHeadClassifier] = None
        self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Build Welch PSD encoder + multi-head classifier and wrap in DDP."""
        logger.info("Setting up PSD model architecture …")
        model_cfg: PSDModelArgs = self.cfg.model

        self.encoder = WelchPSDEncoder(
            fs=self.cfg.fs,
            nperseg=model_cfg.nperseg,
            noverlap=model_cfg.noverlap,
            fmin=model_cfg.fmin,
            fmax=model_cfg.fmax,
        )
        n_freq_bins = self.encoder.n_freq_bins
        embed_dim = n_freq_bins

        # ds_shape_info maps montage_key → (T_out, C, D)
        # T=1 (single spectrum frame), C=n_channels, D=n_freq_bins
        ds_shape_info: dict = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info["shape_info"].items():
                ds_shape_info[montage_key] = (1, n_channels, n_freq_bins)

        head_configs = {ds_name: info["n_class"] for ds_name, info in self.ds_info.items()}
        head_cfg = model_cfg.classifier_head

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_info,
            t_sne=model_cfg.t_sne,
        )
        logger.info(
            f"PSD: nperseg={model_cfg.nperseg}, band=[{model_cfg.fmin},{model_cfg.fmax}]Hz, "
            f"n_freq_bins={n_freq_bins}, heads={list(head_configs.keys())}"
        )

        model = PSDUnifiedModel(encoder=self.encoder, classifier=self.classifier)
        model = self.apply_lora(model)
        model = model.to(self.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        logger.info("PSD model setup complete.")
        self.model = model
        return model

    def load_checkpoint(self, checkpoint_path):
        """No-op — Welch PSD encoder has no pretrained weights."""
        logger.info("PSD encoder is parameter-free; no checkpoint to load.")
