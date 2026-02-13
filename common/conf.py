import logging
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from common.path import create_parent_dir
from common.type import TrainStage, PretrainTaskListType, TemporalConvType, StdFactorType, SpectralType


class BasePreprocArgs(BaseModel):
    # Target sampling rate for preprocessing
    fs: int = 256
    clean_middle_cache: bool = False
    clean_shared_info: bool = False
    num_preproc_arrow_writers: int = 4
    num_preproc_mid_workers: int = 6
    pretrain_datasets: list[str] = Field(default_factory=lambda: [])
    finetune_datasets: dict[str, str] = Field(default_factory=lambda: {})


class BaseEnvVars(BaseModel):
    TORCH_DISTRIBUTED_DEBUG: str = "INFO"
    CUDA_LAUNCH_BLOCKING: str = "1"
    TORCH_USE_CUDA_DSA: str = "1"
    # Use GNU openMP (GOMP) instead of Intel OpenMP [Intel Math Kernel Library (MKL)]
    MKL_SERVICE_FORCE_INTEL: str = "GNU"
    OMP_NUM_THREADS: str = "1"
    MKL_NUM_THREADS: str = "1"
    # faster intra-node collectives, seems to be a cluster specific flag
    ENABLE_INTRA_NODE_COMM: str = "1"
    # avoids OOMs with long context
    # TORCH_NCCL_AVOID_RECORD_STREAMS: str = "1"
    # increasing NCCL timeout time before having some NCCL error 22 should give a 16s timeout
    NCCL_IB_TIMEOUT: str = "23"
    NCCL_DEBUG: str = "INFO"
    TORCH_NCCL_ASYNC_ERROR_HANDLING: str = "1"
    # wandb
    WANDB_CONSOLE: str = "off"
    # WANDB_API_KEY: str = WANDB_KEY


class BaseDataLoaderArgs(BaseModel):
    datasets: dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 1
    sample_ratio: float = 0.1


class BaseDistRunArgs(BaseModel):
    master_port: int = 41216
    is_port_random: bool = False

    seed: int = 42
    debug: bool = False
    use_cpu: bool = False
    use_amp: bool = True
    deterministic: bool = False


class BaseCloudLogArgs(BaseModel):
    project: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[list[str]] = Field(default_factory=lambda: [])
    job_type: Optional[str] = 'debug'
    mode: Optional[str] = 'online'


class BaseLogArgs(BaseModel):
    run_dir: str = ''
    log_train_step_interval: int = 10
    log_valid_step_interval: int = 5

    ckpt_epoch_interval: int = 1
    ckpt_step_ratio_interval_epoch: float = 0.1
    save_scaling_ckpt: bool = False

    use_cloud: bool = True
    cloud: BaseCloudLogArgs =  Field(default_factory=BaseCloudLogArgs)


class BaseOptimArgs(BaseModel):
    eps: float = 1e-8
    epochs: int = 10
    warmup: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-2
    clip: float = 1.0
    min_lr: float = 1e-5
    warmup_lr_factor: float = 1e-1
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])

    loss_scale_gpt: float = 0.34
    loss_scale_mae_tp: float = 0.33
    loss_scale_mae_ch: float = 0.33
    pretrain_task_list_type: PretrainTaskListType = PretrainTaskListType.ALL

    # DeepSeek V3 style MoE load balancing (auxiliary-loss-free via bias)
    # Note: bias is updated inside MoE layers based on load deviation.
    # These are trainer-level controls for the sequence-wise auxiliary loss.
    moe_seq_aux_global_coef: float = 1e-5  # Global scaling for seq aux loss (applied in trainer)


class BaseLoRAArgs(BaseModel):
    """LoRA configuration for Former model."""
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            'proj_q', 'proj_k', 'proj_v', 'proj_out',  # Attention layers
            'linear_in', 'linear_gate', 'linear_out'  # FFN layers (including MoE expert FFN)
        ]
    )


class BaseFinetuneArgs(BaseModel):
    multitask_mode: bool = False
    freeze_encoder: bool = False
    # Freeze encoder for the first N epochs, then unfreeze and continue finetuning.
    # - When freeze_encoder=True, the encoder stays frozen for all epochs (takes precedence).
    # - Ignored when LoRA is enabled (apply_lora will manage trainable params).
    freeze_encoder_epochs: int = 0

    batch_size: int = 32
    sample_ratio: float = 1.0

    epochs: int = 30
    warmup: int = 3
    lr: float = 1e-4
    head_lr_scale: float = 10.0
    fusion_lr_scale: float = 5.0
    min_lr: float = 1e-5
    clip: float = 5.0
    warmup_lr_factor: float = 1e-2

    with_reconstruct: bool = True
    lambda_recon: float = 0.1
    label_smoothing: float = 0.1

    enable_moe_balance: bool = True
    moe_seq_aux_global_coef: float = 1e-5

    # Contrastive learning loss weight (used when model.use_contrast=true)
    contrast_loss_weight: float = 0.05

    apply_loss_weight: bool = True
    loss_weight_type: str = 'sqrt'
    loss_scale_per_dataset: bool = False

    log_train_step_interval: int = 10
    log_valid_step_interval: int = 5

    ckpt_epoch_interval: int = 1
    ckpt_step_ratio_interval_epoch: float = 0.1

    checkpoint: Optional[str] = None
    datasets: dict[str, str] = Field(default_factory=lambda: {})

    lora: BaseLoRAArgs = Field(default_factory=BaseLoRAArgs)


class BaseClassifierHeaderArgs(BaseModel):
    # n_class will be automatically defined in runtime
    n_class: int = 2

    hidden_dims:  list[int] = Field(default_factory=lambda: [64])
    mlp_dropout: float = 0.3


class BaseStackConvArgs(BaseModel):
    stack_out_channels: list[int] = Field(
        default_factory=lambda: [16, 24, 48, 32, 64])
    stack_kernel_size: list[list[int]] = Field(
        default_factory=lambda: [[9, 5], [13, 9], [15, 11, 9], [33, 25, 17], [33, 25, 17, 9]])
    stack_stride: list[list[int]] = Field(
        default_factory=lambda: [[2, 1], [2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 2, 1]])


class BaseModelArgs(BaseModel):
    dim: int = 640
    dim_temporal: int = 512
    dim_fft: int = 128
    patch_size: int = 256
    max_rope_seq_len: int = 2304

    head_dim: int = 80
    n_head: int = 8
    n_kv_head: int = 4

    n_layer: int = 8

    mae_temporal_ratio: float = 0.5
    mae_channel_ratio: float = 0.5

    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = 4
    moe_ffn_dim_multiplier: Optional[float] = 1.0
    norm_eps: float = 1e-8
    rope_theta: float = float(1e5)

    attn_dropout_rate: float = 0.1
    ffn_dropout_rate: float = 0.1

    t_embed: TemporalConvType = TemporalConvType.MULTISCALE
    stack_args: BaseStackConvArgs = Field(default_factory=BaseStackConvArgs)
    patch_kernel_size: int = 7
    patch_stride: int = 5

    f_embed: SpectralType = SpectralType.STFT
    f_embed_fmax: float = 100.0
    stft_win_len: int = 160
    stft_hop_len: int = 64
    stft_f_hidden: int = 16

    is_finetune: bool = False
    classifier_args: BaseClassifierHeaderArgs = Field(default_factory=BaseClassifierHeaderArgs)

    std_base: Optional[float] = None
    std_factor: StdFactorType = StdFactorType.DISABLED

    grad_cam: bool = False
    t_sne: bool = False

    @model_validator(mode='after')
    def check_dim_match(self):
        if self.f_embed != SpectralType.NO:
            if self.dim != self.dim_temporal + self.dim_fft:
                raise ValueError('Feature dims do not match')

        if self.dim != self.head_dim * self.n_head:
            raise ValueError('Head dims do not match')
        return self


class BaseSetupArgs(BaseModel):
    model_type: str = 'default'
    conf_file: Optional[str] = None
    stage: TrainStage = TrainStage.PRETRAIN
    # Global sampling rate for data loading (must match preprocessed data)
    fs: int = 256

    data: BaseDataLoaderArgs = Field(default_factory=BaseDataLoaderArgs)
    model: BaseModelArgs = Field(default_factory=BaseModelArgs)
    optim: BaseOptimArgs = Field(default_factory=BaseOptimArgs)
    ft: BaseFinetuneArgs = Field(default_factory=BaseFinetuneArgs)

    env: BaseEnvVars = Field(default_factory=BaseEnvVars)
    dist: BaseDistRunArgs = Field(default_factory=BaseDistRunArgs)
    log: BaseLogArgs = Field(default_factory=BaseLogArgs)

    def dump_to_yaml(self, path: Optional[str]=None, sort_keys: bool = False):
        conf = self.model_dump()
        conf_yaml = yaml.dump(
            conf,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys
        )

        log = logging.getLogger()
        log.info('Config is as follows in this run:')
        log.info(conf_yaml)

        if path is not None:
            create_parent_dir(path)
            with open(path, 'w') as f:
                f.write(conf_yaml)
