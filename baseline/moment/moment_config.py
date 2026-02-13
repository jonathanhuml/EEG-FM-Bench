"""
MOMENT Configuration that inherits from AbstractConfig.

MOMENT is a family of open time-series foundation models using T5 encoder backbone.
This configuration supports:
1. Multi-variate time series (EEG with multiple channels)
2. Multi-dataset joint training with dynamic channel/time routing
3. Classification head for downstream tasks
4. LoRA fine-tuning support

Reference: AutonLab/MOMENT
"""

from typing import Dict, Optional, List

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs
from baseline.moment import TASKS


class MomentDataArgs(BaseDataArgs):
    """MOMENT data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2
    
    # Sequence length configuration
    target_seq_len: int = 512  # Target sequence length (must be compatible with patch size)


class MomentModelArgs(BaseModelArgs):
    """MOMENT model configuration."""
    # Pretrained model path
    pretrained_path: Optional[str] = None  # HuggingFace model ID or local path
    
    # Model architecture parameters
    seq_len: int = 512  # Input sequence length
    d_model: int = 512  # Transformer hidden dimension (T5-base: 768, T5-large: 1024)
    patch_len: int = 8  # Patch length for tokenization
    patch_stride_len: int = 8  # Patch stride (typically equal to patch_len)
    
    # T5 backbone configuration
    transformer_backbone: str = "google/flan-t5-small"  # Pretrained T5 model
    transformer_type: str = "encoder_only"  # encoder_only, decoder_only, encoder_decoder
    randomly_initialize_backbone: bool = True

    # emb head proc
    reduction: str = "none"
    
    # Freeze settings (for fine-tuning)
    freeze_embedder: bool = False
    freeze_encoder: bool = False
    freeze_head: bool = False

    # T5 config overrides (if any)
    t5_config: Optional[Dict] = None


class MomentTrainingArgs(BaseTrainingArgs):
    """MOMENT training configuration."""
    max_epochs: int = 30
    
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_schedule: str = "cosine"
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    encoder_lr_scale: float = 0.5
    warmup_epochs: int = 3
    warmup_scale: float = 1e-1
    pct_start: float = 0.2

    use_amp: bool = True
    freeze_encoder: bool = False
    
    # Label smoothing
    label_smoothing: float = 0.1


class MomentLoggingArgs(BaseLoggingArgs):
    """MOMENT logging configuration."""
    experiment_name: str = "moment"
    run_dir: str = "assets/run"
    
    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "moment"
    entity: Optional[str] = None
    
    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["moment"])
    
    log_step_interval: int = 1
    ckpt_interval: int = 1


class MomentConfig(AbstractConfig):
    """MOMENT configuration that extends AbstractConfig."""
    
    model_type: str = "moment"
    fs: int = 256  # Default sampling frequency
    
    data: MomentDataArgs = Field(default_factory=MomentDataArgs)
    model: MomentModelArgs = Field(default_factory=MomentModelArgs)
    training: MomentTrainingArgs = Field(default_factory=MomentTrainingArgs)
    logging: MomentLoggingArgs = Field(default_factory=MomentLoggingArgs)

    def validate_config(self) -> bool:
        """Validate MOMENT specific configuration."""
        # Check patch length compatibility with sequence length
        if self.model.seq_len % self.model.patch_len != 0:
            return False
        
        # Check model dimensions
        if self.model.d_model <= 0:
            return False
        
        # Check reduction type
        if self.model.reduction not in ["mean", "concat", "none"]:
            return False
        
        return True

    def build_moment_config(self) -> dict:
        """Build MOMENT configuration dictionary."""
        # Default T5 config for MOMENT
        t5_config = self.model.t5_config or {
            "architectures": [
                "T5ForConditionalGeneration"
            ],
            "vocab_size": 32128,
            "d_ff": 1024,
            "d_kv": 64,
            "d_model": 512,
            "decoder_start_token_id": 0,
            "dropout_rate": 0.1,
            "eos_token_id": 1,
            "feed_forward_proj": "gated-gelu",
            "initializer_factor": 1.0,
            "is_encoder_decoder": True,
            "layer_norm_epsilon": 1e-06,
            "model_type": "t5",
            "n_positions": 512,
            "num_decoder_layers": 8,
            "num_heads": 6,
            "num_layers": 8,
            "output_past": True,
            "pad_token_id": 0,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32,
            "tie_word_embeddings": False,
            "use_cache": True,
        }

        return {
            "task_name": TASKS.EMBED,
            "model_name": "MOMENT",
            "d_model": self.model.d_model,
            "seq_len": self.model.seq_len,
            "patch_len": self.model.patch_len,
            "patch_stride_len": self.model.patch_stride_len,
            "transformer_backbone": self.model.transformer_backbone,
            "transformer_type": self.model.transformer_type,
            "enable_gradient_checkpointing": False,
            "randomly_initialize_backbone": self.model.randomly_initialize_backbone,
            "reduction": self.model.reduction,
            "freeze_embedder": self.model.freeze_embedder,
            "freeze_encoder": self.model.freeze_encoder,
            "freeze_head": self.model.freeze_head,
            "t5_config": t5_config,
        }
