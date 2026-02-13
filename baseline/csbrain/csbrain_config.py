"""
CSBrain Configuration that inherits from AbstractConfig.

CSBrain is a brain-region-aware EEG foundation model that uses:
- Temporal embedding layer for multiscale temporal patterns
- Brain region embedding layer for spatial relationships
- Region-aware attention mechanism
"""

from typing import Dict, Optional, List

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class CSBrainDataArgs(BaseDataArgs):
    """CSBrain data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 64
    num_workers: int = 2
    
    # CSBrain specific: patch configuration
    patch_size: int = 200  # Patch size in samples (at 200Hz: 1 second)
    # seq_len is now dynamic based on sample length, this is the maximum
    max_seq_len: Optional[int] = 2048


class CSBrainModelArgs(BaseModelArgs):
    """CSBrain model configuration."""
    # Pretrained model path
    pretrained_path: Optional[str] = None
    
    # Architecture parameters
    in_dim: int = 200  # Input dimension (patch size)
    out_dim: int = 200  # Output dimension
    d_model: int = 200  # Model dimension
    dim_feedforward: int = 800  # Feedforward dimension
    n_layer: int = 12  # Number of transformer layers
    nhead: int = 8  # Number of attention heads
    
    # Temporal embedding kernel sizes
    tem_embed_kernel_sizes: List[tuple] = Field(
        default_factory=lambda: [(1,), (3,), (5,)]
    )


class CSBrainTrainingArgs(BaseTrainingArgs):
    """CSBrain training configuration."""
    max_epochs: int = 12
    
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_schedule: str = "cosine"
    max_lr: float = 1e-4
    encoder_lr_scale: float = 1.0  # backbone_lr = max_lr * encoder_lr_scale
    warmup_epochs: int = 2
    warmup_scale: float = 1e-2
    min_lr: float = 1e-6
    
    use_amp: bool = True
    freeze_encoder: bool = False  # CSBrain typically fine-tunes the encoder
    
    label_smoothing: float = 0.1


class CSBrainLoggingArgs(BaseLoggingArgs):
    """CSBrain logging configuration."""
    experiment_name: str = "csbrain"
    run_dir: str = "assets/run"
    
    # Cloud logging options
    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "csbrain"
    entity: Optional[str] = None
    
    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["csbrain"])
    
    # Logging intervals
    log_step_interval: int = 1
    ckpt_interval: int = 1


class CSBrainConfig(AbstractConfig):
    """CSBrain configuration that extends AbstractConfig."""
    
    model_type: str = "csbrain"
    fs: int = 200
    
    data: CSBrainDataArgs = Field(default_factory=CSBrainDataArgs)
    model: CSBrainModelArgs = Field(default_factory=CSBrainModelArgs)
    training: CSBrainTrainingArgs = Field(default_factory=CSBrainTrainingArgs)
    logging: CSBrainLoggingArgs = Field(default_factory=CSBrainLoggingArgs)

    def validate_config(self) -> bool:
        """Validate CSBrain-specific configuration."""
        # Check model dimensions
        if self.model.d_model <= 0:
            return False
        
        # Check d_model is compatible with nhead
        if self.model.d_model % self.model.nhead != 0:
            return False
        
        # Check learning rate schedule
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
            
        return True
