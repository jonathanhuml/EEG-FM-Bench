from typing import Dict, Optional, List

from pydantic import Field

from baseline.abstract.config import BaseModelArgs, BaseTrainingArgs, BaseDataArgs, BaseLoggingArgs, AbstractConfig


class ReveDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 64
    num_workers: int = 2


class ReveModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None
    pos_bank_pretrained_path: Optional[str] = None

    embed_dim: int = 512
    depth: int = 22
    heads: int = 8
    head_dim: int = 64
    mlp_dim_ratio: float = 2.66
    use_geglu: bool = True

    freqs: int = 4
    noise_ratio: float = 0.0025

    patch_size: int = 200
    patch_overlap: int = 20

    dropout: float = 0.1


class ReveTrainingArgs(BaseTrainingArgs):
    max_epochs: int = 50

    lr_schedule: str = "reduce_on_plateau"
    eps: float = 1e-9
    warmup_epochs: int = 5
    max_lr: float = 2.4e-4
    min_lr: float = 2.4e-5

    use_amp: bool = True
    warmup_freeze_encoder: bool = True
    warmup_freeze_encoder_epochs: int = 1

    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.95


class ReveLoggingArgs(BaseLoggingArgs):
    experiment_name: str = "reve"
    run_dir: str = "assets/run"

    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "reve"

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class ReveConfig(AbstractConfig):
    model_type: str = "reve"
    fs: int = 200

    data: ReveDataArgs = Field(default_factory=ReveDataArgs)
    model: ReveModelArgs = Field(default_factory=ReveModelArgs)
    training: ReveTrainingArgs = Field(default_factory=ReveTrainingArgs)
    logging: ReveLoggingArgs = Field(default_factory=ReveLoggingArgs)

    def validate_config(self) -> bool:
        return True

