"""
PSD (Power Spectral Density) Configuration that inherits from AbstractConfig.

WelchPSDEncoder is a fixed feature extractor — no learnable parameters.
It outputs log-power spectra shaped [B, 1, C, n_freq_bins], which feeds
directly into the framework's standard classification heads (avg_pool, etc.).
"""

from typing import Dict, List, Optional
from pydantic import Field

from baseline.abstract.config import (
    AbstractConfig,
    BaseDataArgs,
    BaseModelArgs,
    BaseTrainingArgs,
    BaseLoggingArgs,
)


class PSDDataArgs(BaseDataArgs):
    """PSD data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 64
    num_workers: int = 2


class PSDModelArgs(BaseModelArgs):
    """PSD model configuration."""
    # Welch PSD parameters
    nperseg: int = 256          # samples per segment (window length)
    noverlap: Optional[int] = None  # None → nperseg // 2
    # Frequency band to retain (Hz).  None = keep all bins.
    fmin: Optional[float] = None
    fmax: Optional[float] = None


class PSDTrainingArgs(BaseTrainingArgs):
    """PSD training configuration — only the classifier head is trained."""
    max_epochs: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lr_schedule: str = "cosine"
    max_lr: float = 4e-4
    encoder_lr_scale: float = 0.0   # encoder is fixed computation, no params
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2
    min_lr: float = 4e-6

    use_amp: bool = False
    freeze_encoder: bool = True     # no-op; kept for framework compatibility


class PSDLoggingArgs(BaseLoggingArgs):
    """PSD logging configuration."""
    experiment_name: str = "psd"
    run_dir: str = "assets/run"

    use_cloud: bool = False
    cloud_backend: str = "wandb"
    project: Optional[str] = "psd"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class PSDConfig(AbstractConfig):
    """PSD configuration that extends AbstractConfig."""

    model_type: str = "psd"
    fs: int = 256

    data: PSDDataArgs = Field(default_factory=PSDDataArgs)
    model: PSDModelArgs = Field(default_factory=PSDModelArgs)
    training: PSDTrainingArgs = Field(default_factory=PSDTrainingArgs)
    logging: PSDLoggingArgs = Field(default_factory=PSDLoggingArgs)

    def validate_config(self) -> bool:
        return True
