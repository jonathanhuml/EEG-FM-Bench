"""
Mantis Configuration that inherits from AbstractConfig.

Mantis is a time series foundation model using Token Generation + ViT architecture.
This configuration supports:
1. Multi-variate time series (EEG with multiple channels)
2. Multi-dataset joint training with dynamic channel/time routing
3. Classification head for downstream tasks
4. LoRA fine-tuning support
"""

from typing import Dict, Optional, List

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class MantisDataArgs(BaseDataArgs):
    """Mantis data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2
    
    # Channel/time routing configuration
    target_seq_len: int = 512  # Target sequence length (must be multiple of 32)


class MantisModelArgs(BaseModelArgs):
    """Mantis model configuration."""
    # Pretrained model path
    pretrained_path: Optional[str] = None  # Will use HuggingFace if None
    
    # Model architecture parameters
    seq_len: int = 512  # Must be multiple of num_patches (32)
    hidden_dim: int = 256  # Embedding dimension per channel
    num_patches: int = 32  # Number of patches (tokens)
    
    # Scalar encoder parameters
    scalar_scales: Optional[List[float]] = None  # Default: [1e-4, 1e-3, ..., 1e4]
    hidden_dim_scalar_enc: int = 32
    epsilon_scalar_enc: float = 1.1
    
    # Transformer (ViT) parameters
    transf_depth: int = 6  # Number of transformer layers
    transf_num_heads: int = 8
    transf_mlp_dim: int = 512
    transf_dim_head: int = 128
    transf_dropout: float = 0.1


class MantisTrainingArgs(BaseTrainingArgs):
    """Mantis training configuration."""
    max_epochs: int = 30
    
    weight_decay: float = 0.05  # As per official implementation
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_schedule: str = "cosine"
    max_lr: float = 2e-4  # As per official implementation
    encoder_lr_scale: float = 1.0
    warmup_epochs: int = 3
    warmup_scale: float = 1e-1
    pct_start: float = 0.2
    min_lr: float = 2e-5
    
    use_amp: bool = True
    freeze_encoder: bool = False
    
    # Label smoothing
    label_smoothing: float = 0.1


class MantisLoggingArgs(BaseLoggingArgs):
    """Mantis logging configuration."""
    experiment_name: str = "mantis"
    run_dir: str = "assets/run"
    
    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "mantis"
    entity: Optional[str] = None
    
    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["mantis"])
    
    log_step_interval: int = 1
    ckpt_interval: int = 1


class MantisConfig(AbstractConfig):
    """Mantis configuration that extends AbstractConfig."""
    
    model_type: str = "mantis"
    fs: int = 256  # Default sampling frequency
    
    data: MantisDataArgs = Field(default_factory=MantisDataArgs)
    model: MantisModelArgs = Field(default_factory=MantisModelArgs)
    training: MantisTrainingArgs = Field(default_factory=MantisTrainingArgs)
    logging: MantisLoggingArgs = Field(default_factory=MantisLoggingArgs)

    def validate_config(self) -> bool:
        """Validate Mantis specific configuration."""
        # Check sequence length is multiple of num_patches
        if self.model.seq_len % self.model.num_patches != 0:
            return False
        
        # Check model dimensions
        if self.model.hidden_dim <= 0:
            return False
        
        # Check transformer parameters
        if self.model.transf_depth <= 0:
            return False
        
        return True
