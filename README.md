# EEG-FM-Bench: A Comprehensive Benchmark for EEG Foundation Models

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2508.17742)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/xw1216/EEG-FM-Bench)
[![Datasets](https://img.shields.io/badge/Datasets-14_Curated-blue)](#-datasets)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

*A comprehensive benchmark for systematic and standardized evaluation of EEG foundation models*

[✨ Features](#-key-features) • [📈 Results](#-benchmark-results) • [📁 Project Structure](#-project-structure) • [🚀 Quick Start](#-quick-start) • [📊 Datasets](#-datasets) • [🏗️ Models](#-supported-models)

[🖥️ HPC](#-high-performance-computing) • [📖 Documentation](#-documentation) • [🤝 Limitations & Contributing](#-limitations--contributing) • [📚 Others](#-citations)

</div>

---

## 🌟 What is EEG-FM-Bench?

EEG-FM-Bench addresses a critical gap in neuroscience AI research: the lack of standardized evaluation frameworks for EEG foundation models. As these models rapidly proliferate, inconsistent evaluation methods have made fair comparisons nearly impossible, hindering scientific progress.



<div align="center">
  <img src="assets/img/pipeline.png" alt="EEG-FM-Bench Pipeline" width="600">
</div>

**Our contributions:**
- 🎯 **Unified Benchmark Platform**: An open-source suite integrating standardized protocols, diverse tasks, and diagnostic tools for end-to-end EEG-FM evaluation
- 📚 **Diverse Task Suite**: 14 datasets across 10 canonical EEG paradigms
- 🔬 **Systematic Baselines**: Empirical study of SOTA EEG-FMs under matched preprocessing and optimization across multiple fine-tuning strategies
- 🎨 **Diagnostic Toolkit**: Gradient- and representation-based analyses to probe transfer mechanisms and identify pre-training bottlenecks
- 🔄 **Reproducible Science**: Configuration-driven pipeline with unified preprocessing, evaluation, and analysis entrypoints


## 🆕 What’s Updated (v2)

Compared to earlier **v1**, we now add significant new features and insights:
- **Broader evaluation design**: fine-tuning strategies (**frozen-backbone / full-parameter / LoRA**) × task setups (**single-task / multi-task**) × classifier heads (**avg pooling / attention pooling / temporal-spatial-embedding aggregation**)
- **Diagnostic analyses beyond metrics**: gradient-space + representation-space tools for studying transfer and fine-tuning dynamics (e.g., gradient cosine similarity, subspace affinity, CKA, RSA)
- **New benchmark insights**: multi-task learning as a **regularizer** in data-scarce EEG, and **gradient conflicts / objective misalignment** as a pre-training efficiency bottleneck
- **Scaling**: performance does **not** simply follow “bigger is better”; compact EEG-specific inductive biases can outperform much larger models


## ✨ Key Features

### 🤖 **Foundation Model Support**
Comprehensive evaluation of state-of-the-art EEG foundation models:
- **BENDR** - Transformer with contrastive self-supervised learning
- **BIOT** - Biosignal transformer for cross-data learning  
- **CBraMod** - Criss-cross attention for spatio-temporal modeling
- **EEGPT** - Dual self-supervised universal representation learning
- **LaBraM** - Large brain model with vector quantization
- **CSBrain** - Brain-region-aware attention for EEG decoding
- **REVE** - 4D Fourier positional embedding + 19TB pre-training datasets

General time-series foundation models:
- **Mantis** - Token generation + ViT backbone for multivariate time series
- **MOMENT** - Time-series foundation model (T5 encoder backbone)

Additional classical model support:
- **EEGNet** - Well-Known compact CNN for EEG
- **EEGConformer** - Hybrid CNN-Transformer architecture for EEG analysis

### 📊 **Comprehensive Dataset Coverage**
| Paradigm | Datasets | Tasks |
|----------|----------|--------|
| **Motor Imagery** | BCIC-2a, PhysioMI, Mimul-11 | 3/4-class imagined movement classification |
| **Emotion Recognition** | SEED, SEED-V, SEED-VII | 3/5/7-class emotion state recognition |
| **Clinical Applications** | TUAB, TUEV, Siena, HMC, TUSL | Abnormal detection, seizure/event classification, sleep staging |
| **Cognitive & Neurodegenerative** | Things-EEG-2, Workload, ADFTD | Visual target detection, mental workload, AD classification |

**Note**: This repository provides the benchmark framework and evaluation code. **Datasets must be downloaded separately** from their original sources due to licensing restrictions.

### 🔧 **Advanced Evaluation Framework**
- **Fine-tuning strategies**: Frozen-backbone, full-parameter, and parameter-efficient adaptation (**LoRA**)
- **Task setups**: Single-task (one dataset at a time) and multi-task (mixture across all downstream tasks)
- **Classifier heads**: MLP with patch average pooling, MLP with attention pooling, and high-capacity aggregation over temporal/spatial/embedding dimensions
- **Standardized Preprocessing**: Unified pipeline (filtering, resampling, segmentation)
- **Robust Metrics**: Balanced accuracy, weighted F1, AUROC, AUC-PR, Cohen's Kappa

### 🎨 **Rich Visualization & Analysis**
- **t-SNE Embeddings**: Visualize learned feature representations
- **Integrated Gradients**: Understand model decision-making processes across different architectures
- **Neurophysiological Validation**: Ensure models focus on relevant brain regions
- **Gradient/Representation Dynamics (New)**: Two-stage pipeline (collect → visualize) for studying optimization dynamics across paradigms (e.g., scratch vs pretrained, pretrain vs finetune, multi-dataset joint)
- **Quantitative diagnostics**: gradient cosine similarity, gradient subspace affinity, CKA, and RSA for controlled comparisons across training settings and model components

## 📈 Benchmark Results

### Key Findings

🔍 **Generalization Gap (Frozen-backbone)**: Many models show limited out-of-the-box transfer, indicating pre-trained representations alone are often insufficient for novel downstream tasks.

🔄 **Multi-task as Regularizer**: Multi-task fine-tuning consistently alleviates overfitting in data-scarce EEG settings and improves cross-paradigm generalization.

🧩 **Classifier Head Matters (But is Task-dependent)**: Temporal/spatial/embedding aggregation tends to help most in motor imagery, while pooling heads remain competitive elsewhere.

🧠 **Fine-tuning Dynamics**: Layer-wise analyses suggest pre-training stabilizes Transformer backbones and shifts optimization burden toward input embedding/adaptation components.

⚙️ **Pre-training Efficiency Bottleneck**: Gradient conflicts between reconstruction-style objectives and downstream tasks indicate **objective misalignment**, limiting gains from simply scaling pre-training data.

📏 **Scaling Deviates from “Bigger is Better”**: Compact architectures with EEG-specific inductive biases can outperform significantly larger models.

### Sample Results
See paper for complete results, analysis and visualizations.

<div align="center">
  <img src="assets/img/result-1.png" alt="Sample Results Table 1" width="800">
</div>
<div align="center">
  <img src="assets/img/result-2.png" alt="Sample Results Table 2" width="800">
</div>

### Visualizations
<div align="center">
  <img src="assets/img/vis.png" alt="Benchmark Visualization Results" width="800">
</div>


## 📁 Project Structure

```
EEG-FM-Bench/
├── assets/                # Configuration templates & resources
│   └── conf/              #    YAML configs (analysis / baseline / preproc / s3)
├── baseline/              # Foundation model implementations  
│   ├── abstract/          #    Base class to be inherited for other models
│   ├── analysis/          #    Gradient/feature analysis toolkit
│   ├── bendr/             #    BENDR: Transformer + contrastive learning
│   ├── biot/              #    BIOT: Cross-data biosignal learning
│   ├── cbramod/           #    CBraMod: Criss-cross attention
│   ├── reve/              #    REVE: Fourier PE + 19TB datasets
│   ├── csbrain/           #    CSBrain: Brain-region-aware attention
│   ├── mantis/            #    Mantis: Token generation + ViT
│   ├── moment/            #    MOMENT: Time-series FM (T5 encoder)
│   ├── eegpt/             #    EEGPT: Dual self-supervised learning
│   ├── labram/            #    LaBraM: Vector quantized brain model
│   ├── eegnet/            #    EEGNet: Compact CNN baseline
│   └── conformer/         #    EEGConformer: Hybrid CNN-Transformer 
├── common/                # Shared utilities & configurations
├── data/                  # Data processing ecosystem
│   ├── dataset/           #    14 benchmark dataset definitions
│   └── processor/         #    Standardized preprocessing pipeline
├── plot/                  # Advanced visualization tools
├── scripts/               # Helper scripts (bash + slurm)
├── baseline_main.py       # Main training entry point for model training and evaluation
├── analysis_run.py         # Analysis stage-1 entry (collect gradients/features)
├── analysis_vis.py         # Analysis stage-2 entry (analysis visualization)
├── preproc.py             # Data preprocessing pipeline execution script
├── plot_vis.py            # Visualization generation (t-SNE, Grad-CAM, Integrated Gradients)
└── requirements.txt       # Python package dependencies 
```

## 🚀 Quick Start

### 📋 Prerequisites

```bash
# Clone the repository
git clone https://github.com/xw1216/EEG-FM-Bench.git
cd EEG-FM-Bench

# Please install torch by official command in https://pytorch.org/get-started/locally/
# torchaudio is not supported since torch 2.9
pip3 install torch torchaudio torchvision

# Install dependencies
pip install -r requirements.txt

# These packages are required in some scenarios but may cause conflicts with other packages
# braindecode : required by EEGNet and EEGConformer, but needs torch < 2.9
# moabb : required by BENDR
# captum : supports numpy > 2.0, install with --no-deps option
```

### ⚙️ Configuration Setup

Step 1: **Set project paths** (via environment variables; defaults to `./assets/...` if not set).

You can define the path in `./common/path.py` directly, or set environment variables in your shell profile for more flexibility.

For bash/zsh:
```bash
export EEGFM_PROJECT_ROOT=$PWD
export EEGFM_CONF_ROOT=$PWD/assets/conf
export EEGFM_RUN_ROOT=$PWD/assets/run
```

For PowerShell:
```powershell
$env:EEGFM_PROJECT_ROOT = (Get-Location).Path
$env:EEGFM_CONF_ROOT = "$env:EEGFM_PROJECT_ROOT/assets/conf"
$env:EEGFM_RUN_ROOT = "$env:EEGFM_PROJECT_ROOT/assets/run"
```

Step 2: **Configure your experiment** using YAML files under `assets/conf/` (values not assigned will be filled by the corresponding Pydantic config class):
```yaml
# Example: assets/conf/baseline/eegpt/eegpt_unified.yaml
model:
  pretrained_path: "/path/to/eegpt/weights"
data:
  batch_size: 128
  num_workers: 4
  datasets:
    tuab: 'finetune'
training:
  max_lr: 1e-4
  max_epochs: 50
log:
  run_dir: "/path/to/run"
```

### 🔄 Pipeline Execution

#### Step 1: Dataset Download & Preprocessing
```bash
# First, download datasets from their original sources (see Dataset Guide)
# Then preprocess with standardized pipeline
# Config file can be identified by absolute path or relative path to CONF_ROOT
python preproc.py conf_file=preproc/preproc_example.yaml
```

#### Step 2: Model Fine-tuning & Evaluation
```bash
# Fine-Tuning (examples for different models)
python baseline_main.py conf_file=baseline/eegpt/eegpt_unified.yaml model_type=eegpt

# List model types supported by the unified entrypoint
python baseline_main.py list-models
```

#### Step 3: Analysis & Visualization
```bash
# Generate t-SNE embeddings
python plot_vis.py t_sne assets/conf/baseline/csbrain/csbrain_unified.yaml plot/configs/example/tsne_config_csbrain.yaml

# Create integrated gradients analysis
python plot_vis.py integrated_gradients assets/conf/baseline/csbrain/csbrain_unified.yaml plot/configs/example/integrated_gradients_config_csbrain.yaml
```

#### Step 4: Gradient/Representation Analysis (New)

This analysis is a two-stage workflow:
1) **Stage-1 collection**: run lightweight training loops and save gradients/features to disk (no plotting at runtime)
2) **Stage-2 visualization**: generate figures from the saved tensors/metrics

```bash
# Stage-1: collect gradients/features
python analysis_run.py \
  --config assets/conf/analysis/analysis_example.yaml \
  --trainer-config assets/conf/baseline/csbrain/csbrain_unified.yaml

# Stage-2: visualize a single run/seed
# Tip: --data-dir can point to the run root (auto-discovered) or a specific seed directory.
python analysis_vis.py \
  --data-dir ./analysis_results/scratch_vs_pretrained_YYYYMMDD_HHMMSS
```

Supported paradigms (see `assets/conf/analysis/analysis_example.yaml`):
- `scratch_vs_pretrained`
- `pretrain_vs_finetune`
- `multi_dataset_joint`

Current analysis runner supports model types:
- `cbramod`, `labram`, `reve`, `csbrain`, `mantis`, `moment`

## 📊 Datasets

Our benchmark encompasses 14 carefully curated datasets spanning 10 canonical EEG paradigms. **All datasets must be downloaded separately from their original sources.**

<details>
<summary><b>🧠 Motor Imagery & Movement</b></summary>

- **BCIC-2a**: 4-class classification (left hand, right hand, feet, tongue)
- **PhysioMI**: 4-class motor imagery (left fist, right fist, both fists, feet)
- **Mimul-11**: 3-class upper extremity tasks (reaching, grasping, twisting)
</details>

<details>
<summary><b>😊 Emotion Recognition</b></summary>

- **SEED**: 3-class emotion recognition (sad, neutral, happy)
- **SEED-V**: 5-class emotion states (disgust, fear, sad, neutral, happy)
- **SEED-VII**: 7-class emotion recognition (disgust, fear, sad, neutral, happy, anger, surprise)
</details>

<details>
<summary><b>🏥 Clinical Applications</b></summary>

- **TUAB**: Binary abnormal EEG detection (abnormal vs normal)
- **TUEV**: 6-class epileptiform event classification (spike-wave, GPED, PLED, eye movement, artifact, background)
- **Siena**: Binary seizure detection (seizure vs healthy)
- **HMC**: 5-class sleep stage classification (wake, REM, N1, N2, N3)
- **TUSL**: 3-class slowing event classification (seizure, slow wave, background)
</details>

<details>
<summary><b>🧩 Cognitive</b></summary>

- **Things-EEG-2**: Binary visual target detection (target vs non-target)
- **Workload**: Binary mental workload assessment (arithmetic calculation vs resting)
- **ADFTD**: 3-class dementia classification (Alzheimer's Disease, Frontotemporal Dementia, healthy)
</details>

### 📥 Dataset Acquisition

**Each dataset must be downloaded separately from its original source.** This repository contains only the dataset loaders and preprocessing configurations - no actual data is distributed.

#### 🔍 Finding Dataset Information

Each of our 14 benchmark datasets has a corresponding Python file in `data/dataset/` that contains:

- **📖 Academic Citations**: Proper references for the original papers
- **🔗 Dataset Sources**: Information about where to find and request access to data  
- **⚙️ Preprocessing Configurations**: All technical parameters pre-configured
- **📁 Expected File Directory**: Required directory organization
- **📝 Usage Notes**: Special requirements and considerations

#### 🚀 General Acquisition Process

**Step 1: Explore Available Datasets**
```bash
# Browse all available dataset implementations
ls data/dataset/
find data/dataset/ -name "*.py"
```

**Step 2: Check Dataset Requirements**
```bash
# View dataset class documentation
python -c "
from data.dataset.workload import WorkloadConfig  # Example
conf = WorkloadConfig(name='finetune')
print(conf.description)
print(conf.citation)
"
```

**Step 3: Locate Original Sources**
- Many datasets require **individual applications** or **institutional access**
- **Check the orginal paper** for detailed descriptions and download method for each dataset
- Some datasets may require **data use agreements**

**Step 4: Download & Organize**
```
# Follow the directory structure specified in each dataset file
# Example structure (varies by dataset):
DATABASE_RAW_ROOT/
├── dataset_name/
│   ├── scan_dir (raw data dir)
│   ├── summary (statistical result caching)
│   └── other files
├── dataset_name/
└── ...
```

**Step 5: Configure Paths**
```bash
# Update preprocessing configuration with your data paths
vim assets/conf/preproc/preproc_example.yaml
# Edit the YAML file to add your downloaded datasets to preproc list
```

#### ⚠️ Important Considerations

- **No Direct Downloads**: Dataset scripts contain **source information**, not download links
- **Individual Licensing**: Each dataset has **unique terms and requirements**
- **Registration Often Required**: Many datasets need **approval before access**
- **Large File Sizes**: Plan for **several GBs per dataset**
- **Directory Structure**: Must **exactly match** the expectations in dataset files
- **Preprocessing Pipeline**: All parameters are **pre-configured** for consistency

**Some datasets may require additional steps** (e.g., converting formats, organizing files) - please refer to the implementation code in each dataset file for details. **Some `.gdf` and `.cnt` data files may require conversion to `.set` format to avoid compatibility issues on specific platforms (especially on Debian-based Linux distributions) and MNE versions**.


## 🏗️ Supported Models

### Foundation Models

| Model | Architecture | Key Innovation |
|-------|-------------|----------------|
| **BENDR** | Transformer + CNN | Contrastive learning from speech |
| **BIOT** | Channel-independent Transformer | Variable channel tokenization |
| **CBraMod** | Criss-cross Attention | dual-branch spatio-temporal modeling |
| **EEGPT** | Dual-branch Transformer | momentum latent feature alignment |
| **LaBraM** | Vector Quantized VAE + Transformer | Discrete neural codebook |
| **CSBrain** | Transformer | Brain-region-aware attention |
| **REVE** | Transformer | 4D Fourier positional embedding + 19TB datasets |
| **Mantis** | ViT-style Transformer | Token generation for multivariate time series |
| **MOMENT** | T5 Encoder | Time-series foundation model backbone |


### Classical Baselines
- **EEGNet**: Compact CNN for EEG classification
- **EEGConformer**: Hybrid CNN-Transformer architecture combining local feature extraction with global attention

**Note**: The unified entrypoint `baseline_main.py` runs *registered* models in `baseline/__init__.py`. If you want to run additional classical baselines through the same registry mechanism (disabled due to package conflicts), register their config/trainer there (see the commented examples).

## 🖥️ High-Performance Computing

### SLURM Integration

```bash
# Large-scale preprocessing (after downloading datasets)
sbatch scripts/slurm/preproc_submit.slurm conf_file=your_preproc_config.yaml

# Distributed model training (examples for different models)
sbatch scripts/slurm/baseline_submit.slurm conf_file=your_model_config.yaml model_type=eegpt
```

### Resource Requirements
- **Preprocessing**: 64GB RAM, 16~32 CPU cores
- **Training**: 1-8 A100 GPUs or better (depending on batch size)
- **Storage**: ~500GB for all datasets (processed, user must download separately)

## 📖 Documentation

### Configuration System

All experiments use YAML configuration files that must match the Pydantic structure defined in `common/config.py` and model-specific config class like `baseline\eegpt\eegpt_config.py`:

<details>
<summary><b>📋 Complete Configuration Example</b></summary>

```yaml
# Training pattern flags
seed: 42
master_port: 51001
multitask: true
model_type: 'eegpt'

# Data configuration
data:
  batch_size: 32
  num_workers: 1
  datasets:
    tuab: 'finetune'
    seed: 'finetune_sub_dependent'
    hmc: 'finetune'
    # ...


# EEGPT-specific model configuration
model:
  # Pretrained weights - each model will load from this checkpoint
  pretrained_path: null

  # Classifier head configuration
  classifier_head:
    head_type: 'avg_pool'  # Options: avg_pool, attention_pool, dual_stream_fusion, flatten_mlp
    
    # Common parameters
    hidden_dims: [128]
    dropout: 0.3

  # EEGPT architecture parameters
  patch_size: 64
  patch_stride: 32
  embed_num: 4
  embed_dim: 512
  depth: 8
  num_heads: 8
  mlp_ratio: 4.0

# Training configuration
training:
  max_epochs: 50
  weight_decay: 0.01
  max_grad_norm: 3.0

  # Optimizer settings
  max_lr: 5e-4
  encoder_lr_scale: 0.1    # Scale factor for encoder learning rate
  warmup_epochs: 5
  warmup_scale: 1e-2
  min_lr: 1e-6            # For CosineAnnealingLR

  # Training options
  freeze_encoder: false     # Whether to freeze encoder weights
  use_amp: true           # Use automatic mixed precision

  label_smoothing: 0.1    # Label smoothing factor

  # LoRA configuration
  lora:
    use_lora: false
    lora_r: 8

# Logging configuration
logging:
  experiment_name: "eegpt"
  run_dir: "assets/run"
  
  # Cloud logging configuration
  use_cloud: true
  
  project: 'eegpt'           # Project name (uses experiment_name if not specified)
  offline: false
  
  tags: ['eegpt', 'unified', 'full', "debug"]

  # Logging intervals
  log_step_interval: 1      # Log every N steps
  ckpt_interval: 5       # Evaluate every N epochs
```
</details>

### Advanced Usage

<details>
<summary><b>🔧 Custom Dataset Integration</b></summary>

To add a new dataset, create a file in `data/dataset/` and implement the required interface:

```python
@dataclass
class TemplateConfig(EEGConfig):
    name: str = 'finetune'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("0.0.0")
    description: Optional[str] = ""
    citation: Optional[str] = """
    bibtex Citation
    """

    filter_notch: float = 50.0
    is_notched: bool = False

    dataset_name: Optional[str] = 'template'
    task_type: DatasetTaskType = DatasetTaskType.UNKNOWN
    file_ext: str = 'edf'
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        '10_20': [
            'Fp1',
            'Fp2',
        ]
    })

    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    wnd_div_sec: int = 10
    suffix_path: str = 'template'
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ['class 1', 'class 2'])


class TemplateBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = TemplateConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        BUILDER_CONFIG_CLASS(name='finetune', is_finetune=True)
    ]

    def __init__(self, config_name='pretrain',**kwargs):
        super().__init__(config_name, **kwargs)
        self._load_meta_info()

    def _load_meta_info(self):
        pass

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        pass

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        pass

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        return [('default', 0, -1)]

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str):
        return self.config.montage[montage]

```
</details>

<details>
<summary><b>🤖 Custom Model Integration</b></summary>

To add a new foundation model, create a directory in `baseline/` and implement the interfaces in `baseline/abstract`:

```python
# baseline/your_model/your_model_config.py
class YourModelConfig(BaseModelArgs):
    """Model-specific configuration extending BaseModelArgs."""
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    # Add your model-specific parameters with type annotations


# baseline/your_model/model.py
class YourFoundationModel(nn.Module):
    """Your foundation model architecture."""
    
    def __init__(self, encoder, classifier, ):
        super().__init__()
        # Implement your architecture
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        pass


# baseline/your_model/your_model_trainer.py
class YourModelTrainer(AbstractTrainer):
    """Main trainer class for your model."""
    
    def __init__(self, config: YourModelConfig):
        super().__init__(config)
        
    def build_model(self) -> nn.Module:
        """Build the foundation model (include classifier) architecture."""
        return YourFoundationModel(...)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        pass


# baseline/your_model/your_model_adapter.py
# If your model uses different input data format, you should create an DatasetAdapter to do runtime data conversion
class YourModelDatasetAdapter(AbstractDatasetAdapter):
    def _setup_adapter(self):
      """Initialize specific adapter configurations."""
        self.model_name = 'your_model'
        super()._setup_adapter()

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, str, List[str], int]]:
      """Process a single sample according to model requirements."""
      pass

    def get_supported_channels(self) -> List[str]:
        """Return list of channels supported by your model"""
        return []

# Integrate DatasetAdapter into DataLaderFactory class
class YourModelDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> YourModelDatasetAdapter:
        return YourModelDatasetAdapter(dataset, dataset_names, dataset_configs) 


# baseline/__init__.py
# Register your own classes to Registry
ModelRegistry.register_model(
    model_type='your_model',
    config_class=YourModelConfig,
    adapter_class=YourModelDataLoaderFactory, # or None if no conversion needed
    trainer_class=YourModelTrainer
)
```

**Required Files:**
- `baseline/your_model/your_model_trainer.py` 
- `baseline/your_model/your_model_config.py`
- `baseline/your_model/model.py`
- `assets/conf/baseline/your_model/your_model_unified.yaml`

**For reference, see existing model implementations:**
- `baseline/conformer/` - EEGConformer for classic implementation example
- `baseline/eegpt/` - EEGPT for foundation model implementation example
</details>

## 🤝 Limitations & Contributing

We welcome contributions from the community!

### 🚨 Known Limitations

This project was initially developed for personal research purposes and implemented as a single-developer effort. While we've made it available to the community, please be aware of the following limitations:

- **🐛 Bugs & Issues**: As a personal project, you may encounter bugs or inconsistencies in the codebase. The overall design might not always follow best practices or feel convenient for all use cases.

- **🔧 Design Decisions**: Some architectural choices were made to solve specific research problems and may not generalize well to other scenarios. We acknowledge that the framework might need significant refactoring for broader adoption.

- **📦 Missing Model Implementations**: Some foundation models referenced in our paper have not released their official code or pre-trained weights. In these cases, we excluded models entirely when reliable implementation was not feasible.

- **⚡ Reproducibility Challenges**: Due to the above limitations, exact reproduction of all published results may not always be possible. We've done our best to document these cases clearly.

- **🏗️ Single Developer Limitations**: Code style, documentation quality, and API design may be inconsistent across different parts of the codebase.

**We greatly appreciate your understanding and encourage contributions to help improve these limitations!**

### How to Contribute
- 🐛 **Bug Reports**: Open an issue with reproduction steps - these are especially valuable given the current limitations
- 🚀 **Feature Requests**: Propose new models, datasets, or analysis tools  
- 📝 **Documentation**: Improve our guides and examples - documentation PRs are highly welcomed
- 🔬 **Research**: Share your findings and improvements
- 🔧 **Code Quality**: Help refactor and improve the overall codebase design
- 📦 **Model Implementations**: Contribute official implementations of missing foundation models


## 📚 Citations

If you use EEG-FM-Bench in your research, please cite our paper:

```bibtex
@misc{xiong2025eegfmbenchcomprehensivebenchmarksystematic,
      title={EEG-FM-Bench: A Comprehensive Benchmark for the Systematic Evaluation of EEG Foundation Models}, 
      author={Wei Xiong and Jiangtong Li and Jie Li and Kun Zhu},
      year={2025},
      eprint={2508.17742},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2508.17742}, 
}
```

### Model Citations

When using specific models, please also cite the original papers:

<details>
<summary><b>EEG Foundation Model Citations</b></summary>

```bibtex
@article{kostas2021bendr,
  title={BENDR: Using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data},
  author={Kostas, Demetres and Aroca-Ouellette, Stephane and Rudzicz, Frank},
  journal={Frontiers in Human Neuroscience},
  volume={15},
  pages={653659},
  year={2021},
  publisher={Frontiers Media SA}
}

@article{yang2023biot,
  title={Biot: Biosignal transformer for cross-data learning in the wild},
  author={Yang, Chaoqi and Westover, M and Sun, Jimeng},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={78240--78260},
  year={2023}
}

@article{wang2024cbramod,
  title={Cbramod: A criss-cross brain foundation model for eeg decoding},
  author={Wang, Jiquan and Zhao, Sha and Luo, Zhiling and Zhou, Yangxuan and Jiang, Haiteng and Li, Shijian and Li, Tao and Pan, Gang},
  journal={arXiv preprint arXiv:2412.07236},
  year={2024}
}

@article{wang2024eegpt,
  title={Eegpt: Pretrained transformer for universal and reliable representation of eeg signals},
  author={Wang, Guangyu and Liu, Wenchao and He, Yuhong and Xu, Cong and Ma, Lin and Li, Haifeng},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={39249--39280},
  year={2024}
}

@article{jiang2024labram,
  title={Large brain model for learning generic representations with tremendous EEG data in BCI},
  author={Jiang, Wei-Bang and Zhao, Li-Ming and Lu, Bao-Liang},
  journal={arXiv preprint arXiv:2405.18765},
  year={2024}
}

@article{Ouahidi2025reve,
  title={REVE: A Foundation Model for EEG - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects},
  author={Yassine El Ouahidi and Jonathan Lys and Philipp Th{\"o}lke and Nicolas Farrugia and Bastien Pasdeloup and Vincent Gripon and Karim Jerbi and Giulia Lioi},
  journal={ArXiv},
  year={2025},
  volume={abs/2510.21585},
}

@article{zhou2025csbrain,
  title={CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding},
  author={Zhou, Yuchen and Wu, Jiamin and Ren, Zichen and Yao, Zhouheng and Lu, Weiheng and Peng, Kunyu and Zheng, Qihao and Song, Chunfeng and Ouyang, Wanli and Gou, Chao},
  journal={arXiv preprint arXiv:2506.23075},
  year={2025}
}
```
</details>

<details>
<summary><b>General Time-Series Foundation Model Citations</b></summary>

```bibtex
@article{feofanov2025mantis,
  title={Mantis: Lightweight calibrated foundation model for user-friendly time series classification},
  author={Feofanov, Vasilii and Wen, Songkang and Alonso, Marius and Ilbert, Romain and Guo, Hongbo and Tiomoko, Malik and Pan, Lujia and Zhang, Jianfeng and Redko, Ievgen},
  journal={arXiv preprint arXiv:2502.15637},
  year={2025}
}

@article{goswami2024moment,
  title={Moment: A family of open time-series foundation models},
  author={Goswami, Mononito and Szafer, Konrad and Choudhry, Arjun and Cai, Yifu and Li, Shuo and Dubrawski, Artur},
  journal={arXiv preprint arXiv:2402.03885},
  year={2024}
}
```
</details>

<details>
<summary><b>Classical Baseline Citations</b></summary>

```bibtex
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  publisher={iOP Publishing}
}

@article{song2022eeg-conformer,
  title={EEG conformer: Convolutional transformer for EEG decoding and visualization},
  author={Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={31},
  pages={710--719},
  year={2022},
  publisher={IEEE}
}
```
</details>

## 📄 License

This project is licensed under the **Apache License 2.0**.
See the [LICENSE](LICENSE) file for complete terms and conditions.

**Important**: Individual datasets have their own licensing terms. Users must comply with all dataset-specific licenses when downloading and using the data.

## 🙏 Acknowledgments

- **Dataset Providers**: We thank all dataset creators for making their data publicly available. Please cite original dataset papers when using the data.
- **Foundation Model Authors**: Thanks for open-sourcing model implementations that enable fair comparison
- **Research Community**: The neuroscience and BCI communities for inspiration, feedback, and collaboration


---

<div align="center">

**🌟 Star us on GitHub if EEG-FM-Bench helps your research! 🌟**

[⬆ Back to Top](#eeg-fm-bench-a-comprehensive-benchmark-for-eeg-foundation-models)

</div>
