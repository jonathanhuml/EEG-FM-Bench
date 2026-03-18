"""
ZUNA Configuration that inherits from AbstractConfig.
"""

from typing import Dict, Optional, List
from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class ZunaDataArgs(BaseDataArgs):
    """ZUNA data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 4   # packed sequence is memory-heavy
    num_workers: int = 2


class ZunaModelArgs(BaseModelArgs):
    """ZUNA model configuration."""
    # HuggingFace repo — weights downloaded automatically on first run
    pretrained_repo: str = "Zyphra/ZUNA"
    pretrained_weights_file: str = "model-00001-of-00001.safetensors"
    pretrained_config_file: str = "config.json"

    # Tokenisation parameters (must match pretrained model)
    n_fine: int = 32          # raw time-samples per token
    encoder_output_dim: int = 32  # latent_dim; read from HF config at runtime


class ZunaTrainingArgs(BaseTrainingArgs):
    """ZUNA training configuration — encoder always frozen."""
    max_epochs: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lr_schedule: str = "cosine"
    max_lr: float = 4e-4
    encoder_lr_scale: float = 0.0   # unused; encoder is frozen
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2
    min_lr: float = 4e-6

    use_amp: bool = False
    freeze_encoder: bool = True     # Zuna encoder is always frozen


class ZunaLoggingArgs(BaseLoggingArgs):
    """ZUNA logging configuration."""
    experiment_name: str = "zuna"
    run_dir: str = "assets/run"

    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "zuna"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class ZunaConfig(AbstractConfig):
    """ZUNA configuration that extends AbstractConfig."""

    model_type: str = "zuna"
    fs: int = 256   # ZUNA was trained at 256 Hz

    data: ZunaDataArgs = Field(default_factory=ZunaDataArgs)
    model: ZunaModelArgs = Field(default_factory=ZunaModelArgs)
    training: ZunaTrainingArgs = Field(default_factory=ZunaTrainingArgs)
    logging: ZunaLoggingArgs = Field(default_factory=ZunaLoggingArgs)

    def validate_config(self) -> bool:
        if self.fs != 256:
            return False
        if self.model.n_fine <= 0:
            return False
        return True
