"""Unified configuration for gradient and representation analysis.

This module centralizes all analysis-related configurations:
1. ExperimentParadigm: Three experimental paradigms for gradient analysis
2. AnalysisConfig: Unified configuration for all analysis components
3. OutputConfig: Settings for data persistence and visualization

The three experimental paradigms are:
1. SCRATCH_VS_PRETRAINED: Compare from-scratch vs pretrained finetuning
   - Goal: Evaluate how much pretraining helps downstream tasks
   
2. PRETRAIN_VS_FINETUNE: Compare pretraining vs finetuning gradient dynamics
   - Goal: Evaluate optimization direction alignment between stages
   
3. MULTI_DATASET_JOINT: Multi-dataset joint finetuning analysis
   - Goal: Evaluate knowledge sharing and gradient conflicts between datasets
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# =============================================================================
# Experimental Paradigms
# =============================================================================


class ExperimentParadigm(str, Enum):
    """Experimental paradigms for gradient analysis.
    
    Each paradigm answers different research questions:
    
    SCRATCH_VS_PRETRAINED:
        "Does pretraining help? Which components benefit most?"
        - Compares: from-scratch finetune vs pretrained finetune
        - Same data, same epochs, different initialization
        - Conditions: ["scratch", "pretrained"]
        
    PRETRAIN_VS_FINETUNE:
        "Are pretraining and finetuning optimizing in the same direction?"
        - Compares: reconstruction objective vs classification objective
        - Same data, same model, different objectives
        - Conditions: ["pretrain", "finetune"]
        
    MULTI_DATASET_JOINT:
        "Can datasets help each other? Where do conflicts occur?"
        - Analyzes: gradient consistency across datasets in joint training
        - Same model, different data sources
        - Conditions: dataset names (e.g., ["tuab", "tuev", "seed"])
    """
    SCRATCH_VS_PRETRAINED = "scratch_vs_pretrained"
    PRETRAIN_VS_FINETUNE = "pretrain_vs_finetune"
    MULTI_DATASET_JOINT = "multi_dataset_joint"


class ModelType(str, Enum):
    """Supported EEG foundation models."""
    CBRAMOD = "cbramod"
    LABRAM = "labram"
    REVE = "reve"
    CSBRAIN = "csbrain"
    MANTIS = "mantis"
    MOMENT = "moment"


class GroupingStrategy(str, Enum):
    """Parameter grouping strategies for analysis.
    
    BY_MODULE_TYPE:
        Groups by module type (embed, attention, ffn, norm, head)
        - General purpose, works for all transformer models
        
    BY_LAYER_INDEX:
        Groups by layer index (layer_0, layer_1, ..., layer_N)
        - Fine-grained, shows gradient flow through depth
        
    BY_MODEL_INNOVATION:
        Groups by model-specific architectural innovations
        - CBraMod: spatial_attn vs temporal_attn (criss-cross)
        - LaBraM: vq_tokenizer vs neural_transformer
        - REVE: fourier_4d_pe vs denoising_head
        - CSBrain: region_attn vs window_attn (cross-scale)
        - Mantis: tokgen_conv vs tokgen_scalar
        - Moment: t5_encoder vs patch_embed
    """
    BY_MODULE_TYPE = "by_module_type"
    BY_LAYER_INDEX = "by_layer_index"
    BY_MODEL_INNOVATION = "by_model_innovation"


# =============================================================================
# Gradient Collection Configuration
# =============================================================================


class GradientCollectionConfig(BaseModel):
    """Configuration for gradient collection during training."""
    
    # Projection settings (required for memory efficiency)
    projection_dim: int = Field(
        default=1024,
        description="Dimension for gradient projection (random projection for memory efficiency)"
    )
    projection_seed: int = Field(
        default=42,
        description="Random seed for projection matrix"
    )
    
    # Collection settings
    collect_interval: int = Field(
        default=10,
        description="Steps between gradient collections"
    )
    max_samples_per_condition: int = Field(
        default=512,
        description="Maximum gradient samples per condition (sliding window)"
    )
    track_raw_norms: bool = Field(
        default=True,
        description="Track raw gradient L2 norms before projection"
    )
    
    # Probe settings (for multi-dataset)
    probe_batches_per_condition: int = Field(
        default=2,
        description="Number of probe batches per condition for gradient snapshot"
    )


class FeatureCollectionConfig(BaseModel):
    """Configuration for intermediate feature collection."""
    
    enabled: bool = Field(
        default=False,
        description="Whether to collect intermediate features (CKA/RSA analysis)"
    )
    
    # Layer selection
    feature_layers: Optional[List[str]] = Field(
        default=None,
        description="Specific layers to hook for features (None = auto-detect)"
    )
    auto_detect_max_layers: int = Field(
        default=12,
        description="Maximum layers to auto-detect for feature collection"
    )
    
    # Projection settings
    projection_dim: int = Field(
        default=1024,
        description="Dimension for feature projection"
    )
    projection_seed: int = Field(
        default=42,
        description="Random seed for feature projection"
    )
    
    # Collection settings
    max_samples_per_condition: int = Field(
        default=512,
        description="Maximum feature samples per layer/condition"
    )
    probe_batches_per_condition: int = Field(
        default=2,
        description="Fixed probe batches for reference feature computation"
    )


# =============================================================================
# Analysis Metrics Configuration
# =============================================================================


class MetricsConfig(BaseModel):
    """Configuration for computed metrics."""
    
    # Gradient metrics
    compute_cosine_similarity: bool = True
    compute_subspace_affinity: bool = True
    compute_conflict_stats: bool = True
    compute_svcca: bool = True
    compute_energy_flow: bool = True
    
    # Subspace analysis settings
    subspace_ranks: List[int] = Field(
        default_factory=lambda: [2, 3, 4, 6, 8],
        description="Ranks for subspace affinity computation"
    )
    
    # SVCCA settings
    svcca_components: int = Field(default=10, description="Number of CCA components")
    svcca_variance_threshold: float = Field(default=0.99, description="Variance threshold for SVD")
    
    # Conflict detection
    conflict_sample_rate: float = Field(
        default=1.0,
        description="Sampling rate for conflict computation (1.0 = all pairs)"
    )
    
    # Feature metrics (only if feature collection enabled)
    cka_kernel: str = Field(default="linear", description="CKA kernel type")
    rsa_metric: str = Field(default="correlation", description="RSA distance metric")
    rsa_comparison: str = Field(default="spearman", description="RSA comparison method")


# =============================================================================
# Output Configuration
# =============================================================================


class OutputConfig(BaseModel):
    """Configuration for data persistence and output."""
    
    output_dir: str = Field(
        default="./analysis_results",
        description="Root output directory"
    )
    
    # Data saving
    save_format: str = Field(
        default="hdf5",
        description="Data format: 'hdf5' for compact storage, 'npz' for numpy"
    )
    save_metrics_jsonl: bool = Field(
        default=True,
        description="Save per-step metrics as JSONL for easy analysis"
    )
    save_summary_json: bool = Field(
        default=True,
        description="Save aggregated summary as JSON"
    )
    
    # Checkpointing
    checkpoint_interval: int = Field(
        default=10,
        description="Steps between saving intermediate results"
    )
    
    # Note: Visualization is DISABLED during training
    # Use post-processing script to generate figures from saved data


# =============================================================================
# Training Configuration (for analysis runs)
# =============================================================================


class TrainingBudgetConfig(BaseModel):
    """Training budget configuration for analysis runs."""
    
    num_steps: Optional[int] = Field(
        default=512,
        description="Training steps budget (takes priority over epochs)"
    )
    num_epochs: Optional[int] = Field(
        default=None,
        description="Training epochs budget (used if num_steps is None)"
    )
    
    # Pretrain-specific (for PRETRAIN_VS_FINETUNE paradigm)
    pretrain_steps: Optional[int] = Field(
        default=None,
        description="Pretrain steps (defaults to num_steps if None)"
    )
    
    # Warmup
    warmup_steps: int = Field(
        default=32,
        description="Warmup steps before gradient collection"
    )


class CrossDatasetConfig(BaseModel):
    """Configuration for cross-dataset consistency analysis.
    
    This enables running SCRATCH_VS_PRETRAINED or PRETRAIN_VS_FINETUNE
    on multiple datasets independently, then comparing whether conclusions
    are consistent across datasets.
    
    Research Questions:
    - Does pretrain help consistently across different datasets?
    - Are gradient conflict patterns similar across different EEG tasks?
    - Which layer groups benefit most from pretraining - is it task-dependent?
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable cross-dataset comparison analysis"
    )
    
    # Dataset selection for cross-dataset analysis
    comparison_datasets: Optional[List[str]] = Field(
        default=None,
        description="Datasets to include in cross-dataset comparison (None = all)"
    )
    
    # Metrics for cross-dataset consistency
    compute_conclusion_similarity: bool = Field(
        default=True,
        description="Compute similarity of conclusions across datasets"
    )
    
    compute_effect_correlation: bool = Field(
        default=True,
        description="Compute correlation of pretraining effect sizes across datasets"
    )
    
    compute_rank_agreement: bool = Field(
        default=True,
        description="Compute agreement in group-wise ranking across datasets"
    )
    
    # Significance testing
    effect_size_threshold: float = Field(
        default=0.05,
        description="Minimum cosine difference to consider 'pretraining helps'"
    )
    
    conflict_threshold: float = Field(
        default=0.3,
        description="Conflict frequency threshold to consider 'high conflict'"
    )
    
    # Aggregation
    aggregate_method: str = Field(
        default="mean",
        description="How to aggregate metrics across datasets: 'mean', 'median', 'vote'"
    )


class MaskingConfig(BaseModel):
    """Masking configuration for pretraining objective."""
    
    mask_ratio: float = Field(
        default=0.5,
        description="Ratio of patches to mask"
    )
    mask_strategy: str = Field(
        default="random_mixed",
        description="Masking strategy: 'random', 'temporal', 'channel', 'random_mixed'"
    )


# =============================================================================
# Main Analysis Configuration
# =============================================================================


class AnalysisConfig(BaseModel):
    """Unified configuration for gradient and representation analysis.
    
    This is the main configuration class that combines all analysis settings.
    """
    
    # Experiment settings
    paradigm: ExperimentParadigm = Field(
        default=ExperimentParadigm.SCRATCH_VS_PRETRAINED,
        description="Experimental paradigm to run"
    )
    model_type: ModelType = Field(
        default=ModelType.CBRAMOD,
        description="Model type to analyze"
    )
    
    # Trainer configuration
    trainer_config_path: Optional[str] = Field(
        default=None,
        description="Path to trainer configuration YAML"
    )
    trainer_overrides: List[str] = Field(
        default_factory=list,
        description="Dotlist overrides for trainer config"
    )
    
    # Checkpoint paths
    pretrained_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to pretrained checkpoint (for SCRATCH_VS_PRETRAINED)"
    )
    
    # Dataset selection
    datasets: Optional[Dict[str, str]] = Field(
        default=None,
        description="Datasets to analyze (None = use all from trainer config)"
    )
    
    # Grouping strategy
    grouping_strategy: GroupingStrategy = Field(
        default=GroupingStrategy.BY_MODULE_TYPE,
        description="How to group parameters for analysis"
    )
    
    # Sub-configurations
    gradient: GradientCollectionConfig = Field(
        default_factory=GradientCollectionConfig
    )
    feature: FeatureCollectionConfig = Field(
        default_factory=FeatureCollectionConfig
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig
    )
    training: TrainingBudgetConfig = Field(
        default_factory=TrainingBudgetConfig
    )
    masking: MaskingConfig = Field(
        default_factory=MaskingConfig
    )
    cross_dataset: CrossDatasetConfig = Field(
        default_factory=CrossDatasetConfig,
        description="Cross-dataset consistency analysis settings"
    )
    
    # Reproducibility
    seed: int = Field(default=42, description="Random seed")
    seeds: Optional[List[int]] = Field(
        default=None,
        description="Multiple seeds for repeated runs (overrides seed)"
    )
    
    # Runtime
    device: str = Field(default="cuda", description="Device to use")
    
    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        # Ensure output directory exists
        Path(self.output.output_dir).mkdir(parents=True, exist_ok=True)
        
        # For PRETRAIN_VS_FINETUNE, feature collection is not needed
        if self.paradigm == ExperimentParadigm.PRETRAIN_VS_FINETUNE:
            self.feature.enabled = False
    
    def get_conditions(self) -> List[str]:
        """Get condition names based on paradigm.
        
        Returns:
            List of condition names for gradient collection axes
        """
        if self.paradigm == ExperimentParadigm.SCRATCH_VS_PRETRAINED:
            return ["scratch", "pretrained"]
        elif self.paradigm == ExperimentParadigm.PRETRAIN_VS_FINETUNE:
            return ["pretrain", "finetune"]
        elif self.paradigm == ExperimentParadigm.MULTI_DATASET_JOINT:
            return list(self.datasets.keys()) if self.datasets else []
        else:
            return ["default"]
    
    def get_run_dir(self, timestamp: str, seed: Optional[int] = None) -> Path:
        """Get run directory path.
        
        Args:
            timestamp: Timestamp string (e.g., "20260125_143022")
            seed: Optional seed for multi-seed runs
            
        Returns:
            Path to run directory
        """
        base = Path(self.output.output_dir)
        run_name = f"{self.model_type.value}_{self.paradigm.value}_{timestamp}"
        if seed is not None:
            run_name = f"{run_name}_seed{seed}"
        return base / run_name


# =============================================================================
# Model Innovation Groups (for BY_MODEL_INNOVATION strategy)
# =============================================================================


# These define the core architectural innovations of each model
# Based on paper analysis

MODEL_INNOVATION_GROUPS: Dict[str, Dict[str, Tuple[str, List[str]]]] = {
    "cbramod": {
        # CBraMod: Criss-Cross Brain Transformer
        # Innovation: Dual-path spatial-temporal attention
        "spatial_attention": (
            "Spatial (channel-wise) self-attention",
            ["self_attn_s", "attn_s"]
        ),
        "temporal_attention": (
            "Temporal (time-wise) self-attention", 
            ["self_attn_t", "attn_t"]
        ),
        "spectral_embed": (
            "Spectral (frequency domain) patch embedding",
            ["spectral_proj", "fft"]
        ),
        "patch_embed": (
            "Spatial patch embedding convolutions",
            ["proj_in", "patch_embedding", "positional_encoding"]
        ),
    },
    "labram": {
        # LaBraM: Large Brain Model
        # Innovation: VQ-NSP tokenizer + large-scale pretraining
        "vq_tokenizer": (
            "Vector Quantization Neural Signal Processing tokenizer",
            ["vqnsp", "quantizer", "codebook"]
        ),
        "temporal_conv": (
            "Temporal convolution for patch embedding",
            ["temporal_conv", "conv1", "conv2", "conv3"]
        ),
        "neural_transformer": (
            "Neural Transformer encoder",
            ["blocks", "attn", "mlp"]
        ),
        "channel_embed": (
            "Per-channel embedding with brain topology",
            ["chan_embed", "pos_embed"]
        ),
    },
    "reve": {
        # REVE: Robust EEG Vision Encoder  
        # Innovation: 4D Fourier positional embedding + denoising
        "fourier_4d_pe": (
            "4D Fourier positional embedding (time, freq, channel, trial)",
            ["mlp4d", "fourier", "ln"]
        ),
        "patch_projection": (
            "Linear patch projection",
            ["to_patch_embedding"]
        ),
        "transformer_core": (
            "Standard transformer with RMSNorm",
            ["layers", "to_qkv", "to_out", "net"]
        ),
        "attention_pooling": (
            "Attention-based pooling for classification",
            ["cls_query_token", "final_layer"]
        ),
    },
    "csbrain": {
        # CSBrain: Cross-Scale Brain Transformer
        # Innovation: Multi-scale cross-region and cross-window attention
        "brain_region_embed": (
            "Brain region-specific embedding",
            ["brain_embed", "region_blocks"]
        ),
        "temporal_multiscale": (
            "Multi-scale temporal convolution embedding",
            ["tem_embed", "convs"]
        ),
        "region_attention": (
            "Inter-region cross-scale attention",
            ["inter_region_attn"]
        ),
        "window_attention": (
            "Inter-window cross-scale attention + global projection",
            ["inter_window_attn", "global_fc"]
        ),
    },
    "mantis": {
        # Mantis: Multi-scale Adaptive Neural Time-series Intelligence System
        # Innovation: Adaptive token generation + ViT
        "tokgen_conv": (
            "Token generator convolutions (multi-scale)",
            ["tokgen_unit", "convs"]
        ),
        "tokgen_scalar": (
            "Scalar statistics encoder for adaptive tokens",
            ["scalar_encoder", "linear_encoder"]
        ),
        "vit_attention": (
            "Vision Transformer attention layers",
            ["to_qkv", "to_out"]
        ),
        "vit_ffn": (
            "Vision Transformer feed-forward layers",
            ["vit_unit", "net"]
        ),
    },
    "moment": {
        # MOMENT: Modular Time-series Foundation Model
        # Innovation: T5 encoder backbone + universal patching
        "patch_embed": (
            "Universal patch embedding for any time series",
            ["patch_embedding", "value_embedding", "position_embedding"]
        ),
        "t5_attention": (
            "T5 self-attention (relative position)",
            ["selfattention", ".q.", ".k.", ".v.", ".o."]
        ),
        "t5_ffn": (
            "T5 feed-forward (gated)",
            ["wi", "wo", "densereludense"]
        ),
        "revin": (
            "Reversible Instance Normalization",
            ["normalizer", "revin"]
        ),
    },
}


def get_innovation_group_patterns(model_type: str) -> Dict[str, Tuple[str, List[str]]]:
    """Get innovation group patterns for a model type.
    
    Args:
        model_type: Model type string (e.g., "cbramod")
        
    Returns:
        Dict mapping group name to (description, patterns) tuple
    """
    return MODEL_INNOVATION_GROUPS.get(model_type.lower(), {})
