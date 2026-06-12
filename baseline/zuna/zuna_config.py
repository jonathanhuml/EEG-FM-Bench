"""ZUNA configuration backed by the local AY2latent checkout."""

from typing import Any, Dict, Optional, List
from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class ZunaDataArgs(BaseDataArgs):
    """ZUNA data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 4   # packed sequence is memory-heavy
    num_workers: int = 2


class ZunaModelArgs(BaseModelArgs):
    """ZUNA model configuration."""
    ay2latent_root: str = (
        "/data/groups/bci/jonhuml/workspace/AY2latent/lingua"
    )
    checkpoint_path: str = (
        "/data/groups/bci/checkpoints/bci/ZUNA2_5e-4/checkpoints/0000052500"
    )

    # Mirrors AY2latent's ZUNA2 evaluation model. The original implementation
    # remains the source of truth for the encoder and tokenization helpers.
    model_kwargs: Dict[str, Any] = Field(default_factory=lambda: {
        "dim": 1024,
        "n_layers": 16,
        "head_dim": 64,
        "seqlen_t": False,
        "huber_c": None,
        "input_dim": 32,
        "encoder_input_dim": 32,
        "encoder_output_dim": 32,
        "encoder_latent_downsample_factor": 1,
        "encoder_sliding_window": 65536,
        "sliding_window": 65536,
        "xattn_sliding_window": 65536,
        "max_seqlen": 256,
        "max_chans": 512,
        "model_dtype": "bf16",
        "stft_global_sigma": 0.1,
        "adaptive_loss_weighting": True,
        "num_fine_time_pts": 32,
        "rope_dim": 4,
        "rope_theta": 10000.0,
        "ape_dim": 0,
        "tok_idx_type": "{x,y,z,tc}",
        "dont_noise_chan_xyz": False,
        "zero_spatial": False,
        "dropout_vec_type": "zeros",
        "register_tok_idx": "mean_all",
    })

    n_fine: int = 32
    encoder_output_dim: int = 32
    data_norm: float = 10.0
    data_clip: Optional[float] = 1.0
    do_avg_ref: bool = True
    num_bins_discretize_xyz_chan_pos: int = 100
    channel_position_montage: str = "standard_1005"
    invalid_channel_position: float = -0.1
    attn_impl: str = "flex_attention"

    # If True, skip the ((x-mean)/std/10).clamp(-1,1) normalisation in encode().
    # Set True for DREAMER, where data is already z-scored and compare_models.py
    # feeds it directly to ZUNA (ZUNA_DATA_NORM=1.0, i.e. no-op).
    skip_input_norm: bool = False

    # If True, apply per-feature BatchNorm to encoder output before head.
    # Matches the StandardScaler(fit_transform) step in compare_models.py.
    use_feature_norm: bool = False


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

    # Embedding cache — pre-compute encoder output once, train classifier on disk cache
    cache_features: bool = False
    features_cache_dir: Optional[str] = None  # path to save/load cached embeddings
    precompute_batch_size: int = 8            # batch size used only during precomputation


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
