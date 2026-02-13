"""
Abstract configuration base class for baseline models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class ClassifierHeadType(str, Enum):
    """Enumeration of supported classification head types."""
    AVG_POOL = "avg_pool"                      # Adaptive average pooling (current default)
    ATTENTION_POOL = "attention_pool"          # Attention pooling with learnable query
    DUAL_STREAM_FUSION = "dual_stream_fusion"  # Dual stream attention fusion
    FLATTEN_MLP = "flatten_mlp"                # Flatten + Large


class ClassifierHeadConfig(BaseModel):
    """Configuration for classification head."""
    head_type: ClassifierHeadType = ClassifierHeadType.AVG_POOL

    # Common parameters
    hidden_dims: list[int] = Field(default_factory=lambda: [128])
    dropout: float = 0.3

    # Attention Pool parameters (for ATTENTION_POOL type)
    attn_n_head: int = 4
    attn_head_dim: int = 64

    # Dual Stream Fusion parameters (for DUAL_STREAM_FUSION type)
    fusion_mode: str = "dual"  # "time_first", "channel_first", or "dual"
    fusion_n_head: int = 4
    fusion_head_dim: int = 64
    fusion_use_rope: bool = True
    fusion_rope_theta: float = 10000.0
    fusion_max_seq_len: int = 1024

    # Note: FLATTEN_MLP head uses fixed 3-layer structure based on dataset shape
    # Layer structure: (n_ch * n_patches * dim) -> (n_patches * dim) -> dim -> n_class


class BaseDataArgs(BaseModel):
    """Base data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class BaseModelArgs(BaseModel):
    """Base model configuration."""
    pretrained_path: Optional[str] = None

    grad_cam: bool = False
    t_sne: bool = False
    grad_cam_target: str = 'channel'

    # Classifier head configuration
    classifier_head: ClassifierHeadConfig = Field(default_factory=ClassifierHeadConfig)

class BaseLoRAArgs(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""
    use_lora: bool = False
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor (effective scaling = alpha/r)
    lora_dropout: float = 0.0  # Dropout for LoRA layers
    lora_target_modules: List[str] = Field(default_factory=lambda: ["default"])  # Target module patterns
    lora_exclude_modules: Optional[List[str]] = None  # Modules to exclude from LoRA
    lora_target_type: str = "default"  # Predefined target type: "default", "full", "attention", "ffn"
    lora_scope: str = "transformer"  # Scope: "transformer" (only in Transformer blocks) or "full" (all layers)
    lora_lr_scale: float = 1.0  # Learning rate scale for LoRA parameters relative to head_lr


class BaseTrainingArgs(BaseModel):
    """Base training configuration."""
    max_epochs: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    lr_schedule: str = "onecycle"  # 'onecycle' or 'cosine'
    max_lr: float = 1e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2  # For OneCycleLR
    min_lr: float = 1e-6  # For CosineAnnealingLR

    use_amp: bool = True
    freeze_encoder: bool = False

    # LoRA configuration
    lora: BaseLoRAArgs = Field(default_factory=BaseLoRAArgs)


class BaseLoggingArgs(BaseModel):
    """Base logging configuration."""
    experiment_name: str = "baseline"
    run_dir: str = "assets/run"

    use_cloud: bool = False
    cloud_backend: str = "wandb"
    project: Optional[str] = None
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: [])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class AbstractConfig(BaseModel, ABC):
    """Abstract base configuration class for all baseline models."""
    
    seed: int = 42
    master_port: int = 41216
    multitask: bool = False
    model_type: str = "base"  # To identify which model is being used
    conf_file: Optional[str] = None
    fs: int = 256
    
    data: BaseDataArgs = Field(default_factory=BaseDataArgs)
    model: BaseModelArgs = Field(default_factory=BaseModelArgs)
    training: BaseTrainingArgs = Field(default_factory=BaseTrainingArgs)
    logging: BaseLoggingArgs = Field(default_factory=BaseLoggingArgs)

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate model-specific configuration requirements."""
        pass
