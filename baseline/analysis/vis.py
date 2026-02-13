"""Visualization module (single-seed) for gradient/feature analysis.

Key changes vs v2:
- No multi-seed aggregation (each seed visualized independently)
- Metrics computed via torch-only backend
- Unified step-wise aggregation pipeline
- Diagonal handling uses mid color value (no overlay)
- Consistent label mapping for datasets/conditions/groups/layers
- Training curves use EMA smoothing
"""

import logging
import os
import re
import time
import warnings
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from baseline.analysis.collector import TensorReader
from baseline.analysis.metrics import (
    FeatureMetrics,
    GradientMetrics,
    PairwiseMetricSeries,
    compute_all_feature_metrics,
    compute_all_gradient_metrics,
)

logger = logging.getLogger("analysis_vis")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VisConfig:
    """Visualization configuration."""

    figure_format: str = "pdf"
    dpi: int = 300

    cmap_heatmap: str = "GnBu"
    cmap_energy: str = "YlOrRd"

    show_values: bool = True
    show_std: bool = True
    value_fmt: str = ".2f"
    std_fmt: str = ".2f"

    plot_cosine: bool = True
    plot_subspace: bool = True
    plot_conflict: bool = True
    plot_svcca: bool = True
    plot_energy: bool = True
    plot_training_curves: bool = True
    plot_cka: bool = True

    plot_per_step: bool = True
    step_range: Optional[Tuple[int, int]] = None

    # Step dynamics smoothing (light EMA by default)
    step_dynamics_ema_alpha: float = 0.6

    # Insight plots (multi-condition)
    plot_delta: bool = True
    plot_offdiag_distribution: bool = True
    plot_topk_pairs: bool = True
    plot_svd_diagnostics: bool = True

    delta_window: int = 5
    topk_pairs: int = 10
    diagnostics_max_samples_per_step: int = 256
    diagnostics_num_steps: int = 6

    ema_alpha: float = 0.3

    skip_empty_groups: bool = True
    ignored_groups: List[str] = field(default_factory=lambda: ["OTHER", "other", "Others", "UNKNOWN", "unknown"])

    dataset_name_mapping: Dict[str, str] = field(default_factory=dict)

    # Subsampling & trials (None = use analysis_config.yaml defaults)
    svcca_sample_rate: Optional[float] = None
    subspace_sample_rate: Optional[float] = None
    sample_trials: Optional[int] = None
    max_subsample: Optional[int] = None


DEFAULT_IGNORED_GROUPS = {"OTHER", "other", "Others", "UNKNOWN", "unknown"}
_DATASET_NAME_MAPPING: Dict[str, str] = {
    "tuab": "TUAB",
    "seed": "SEED",
    "tuev": "TUEV",
    "hmc": "HMC",
    "adftd": "ADFTD",
    "motor_mv_img": "PhysioMI",
}

# Fixed elegant color palette for energy flow / stacked area plots
_ENERGY_FLOW_COLORS = [
    "#7FB3D5",  # soft blue
    "#A9DFBF",  # soft green
    "#F9E79F",  # soft yellow
    "#F5B7B1",  # soft pink
    "#D7BDE2",  # soft purple
    "#FADBD8",  # soft salmon
    "#AED6F1",  # light sky blue
    "#D5F5E3",  # mint green
    "#FCF3CF",  # cream
    "#E8DAEF",  # lavender
    "#D6EAF8",  # pale blue
    "#E9F7EF",  # pale green
]

# Fixed color palette for line plots - bright and distinct
_STEP_DYNAMICS_COLORS = [
    "#E74C3C",  # bright red
    "#3498DB",  # bright blue
    "#2ECC71",  # bright green
    "#9B59B6",  # bright purple
    "#F39C12",  # bright orange
    "#1ABC9C",  # bright teal
    "#E91E63",  # bright pink
    "#00BCD4",  # bright cyan
    "#FF9800",  # amber
    "#8BC34A",  # light green
    "#673AB7",  # deep purple
    "#FF5722",  # deep orange
]


def set_dataset_name_mapping(mapping: Dict[str, str]):
    global _DATASET_NAME_MAPPING
    _DATASET_NAME_MAPPING = mapping.copy()


def get_display_name(name: str) -> str:
    return _DATASET_NAME_MAPPING.get(name.lower(), name)


def format_condition_name(name: str) -> str:
    # First: dataset display mapping (e.g., motor_mv_img -> PhysioMI)
    mapped = get_display_name(name)
    if mapped != name:
        return mapped
    name_lower = name.lower()
    if name_lower in ("scratch", "pretrained", "pretrain", "finetune", "finetuned"):
        return name_lower.capitalize()
    if "_" in name:
        return " ".join([w.capitalize() for w in name.split("_")])
    return name


def format_group_name(name: str) -> str:
    return name.replace("_", " ").upper()


def shorten_layer_name(name: str, max_len: int = 20) -> str:
    prefixes_to_remove = [
        "model.",
        "module.",
        "encoder.",
        "decoder.",
        "backbone.",
        "transformer.",
        "transformer_backbone.",
    ]
    stripped = True
    while stripped:
        stripped = False
        for prefix in prefixes_to_remove:
            if name.startswith(prefix):
                name = name[len(prefix):]
                stripped = True
                break
    if len(name) > max_len:
        name = re.sub(r"(\d+)", r"\\1", name)
    if len(name) > max_len:
        name = name[: max_len - 1] + "…"
    return name


def natural_sort_key(text: str):
    parts = re.split(r"(\d+)", str(text))
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def format_layer_name(name: str) -> str:
    short = shorten_layer_name(name)
    return short.replace("_", " ").upper()


def get_text_color_for_value(
    value: float,
    vmin: float,
    vmax: float,
    cmap_name: str = "GnBu",
) -> str:
    if np.isnan(value):
        return "black"
    if vmax > vmin:
        norm_val = (value - vmin) / (vmax - vmin)
    else:
        norm_val = 0.0
    norm_val = np.clip(norm_val, 0, 1)
    return "white" if norm_val > 0.55 else "black"


def compute_colorbar_range(
    data: np.ndarray,
    exclude_diagonal: bool = True,
    margin: float = 0.1,
    symmetric: bool = False,
) -> Tuple[float, float]:
    if data is None or data.size == 0:
        return 0.0, 1.0

    flat = data.flatten()
    if exclude_diagonal and data.ndim >= 2:
        mask = ~np.eye(data.shape[-1], dtype=bool)
        flat = data[..., mask].flatten()

    valid = flat[~np.isnan(flat)]
    if len(valid) == 0:
        return 0.0, 1.0

    data_min = float(np.min(valid))
    data_max = float(np.max(valid))
    data_range = data_max - data_min
    if data_range < 1e-9:
        data_range = 1.0

    vmin = data_min - margin * data_range
    vmax = data_max + margin * data_range

    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound

    return vmin, vmax


def filter_valid_groups(
    groups: List[str],
    data_dict: Optional[Dict[str, Any]] = None,
    ignored_groups: Optional[set] = None,
    skip_empty: bool = True,
) -> List[str]:
    if ignored_groups is None:
        ignored_groups = DEFAULT_IGNORED_GROUPS
    ignored_lower = _normalize_group_set(ignored_groups)

    valid_groups = []
    for g in groups:
        if g.lower() in ignored_lower:
            continue
        if skip_empty and data_dict is not None:
            if g not in data_dict:
                continue
            val = data_dict.get(g)
            if val is None:
                continue
        valid_groups.append(g)
    return valid_groups


def setup_matplotlib_style():
    warnings.filterwarnings("ignore", message=r".*Glyph.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*tight_layout.*", category=UserWarning)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Nimbus Roman"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.autolayout": False,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


# =============================================================================
# Loading helpers
# =============================================================================


@dataclass
class AnalysisConfigLite:
    paradigm: str = "scratch_vs_pretrained"
    model_type: str = "cbramod"
    subspace_ranks: List[int] = field(default_factory=lambda: [2, 3, 4, 6, 8])
    svcca_components: int = 10
    svcca_threshold: float = 0.99
    conflict_sample_rate: float = 1.0
    cka_kernel: str = "linear"
    rsa_metric: str = "correlation"
    rsa_comparison: str = "spearman"
    device: str = "cuda"
    svcca_sample_rate: float = 1.0
    subspace_sample_rate: float = 1.0
    sample_trials: int = 5
    max_subsample: Optional[int] = None


def load_analysis_config(data_dir: Path) -> AnalysisConfigLite:
    config = AnalysisConfigLite()
    data_dir = Path(data_dir)

    config_path = None
    if (data_dir / "analysis_config.yaml").exists():
        config_path = data_dir / "analysis_config.yaml"
    else:
        candidates = list(data_dir.glob("**/analysis_config.yaml"))
        if candidates:
            config_path = candidates[0]

    if config_path is None:
        return config

    logger.info(f"Loading analysis config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    config.paradigm = str(cfg.get("paradigm", config.paradigm))
    config.model_type = str(cfg.get("model_type", config.model_type))
    config.device = str(cfg.get("device", config.device))

    metrics_cfg = cfg.get("metrics", {})
    config.subspace_ranks = list(metrics_cfg.get("subspace_ranks", config.subspace_ranks))
    config.svcca_components = int(metrics_cfg.get("svcca_components", config.svcca_components))
    config.svcca_threshold = float(metrics_cfg.get("svcca_variance_threshold", config.svcca_threshold))
    config.conflict_sample_rate = float(metrics_cfg.get("conflict_sample_rate", config.conflict_sample_rate))
    config.cka_kernel = str(metrics_cfg.get("cka_kernel", config.cka_kernel))
    config.rsa_metric = str(metrics_cfg.get("rsa_metric", config.rsa_metric))
    config.rsa_comparison = str(metrics_cfg.get("rsa_comparison", config.rsa_comparison))
    config.svcca_sample_rate = float(metrics_cfg.get("svcca_sample_rate", config.svcca_sample_rate))
    config.subspace_sample_rate = float(metrics_cfg.get("subspace_sample_rate", config.subspace_sample_rate))
    config.sample_trials = int(metrics_cfg.get("sample_trials", config.sample_trials))
    max_subsample = metrics_cfg.get("max_subsample", config.max_subsample)
    config.max_subsample = int(max_subsample) if max_subsample is not None else None

    return config


def discover_seed_dirs(root_dir: str) -> Dict[int, Path]:
    root = Path(root_dir)
    seed_dirs: Dict[int, Path] = {}
    pattern = re.compile(r"^seed_(\d+)$")

    # First: direct seed_* children
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            seed_dirs[int(m.group(1))] = d

    if seed_dirs:
        return seed_dirs

    # Second: model_type/seed_* layout
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for d in model_dir.iterdir():
            if not d.is_dir():
                continue
            m = pattern.match(d.name)
            if m:
                seed_dirs[int(m.group(1))] = d

    return seed_dirs


def find_dataset_dirs(root_dir: Path) -> List[Path]:
    return [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("dataset_")]


def has_tensor_data(dir_path: Path) -> bool:
    if (dir_path / "metadata.json").exists() or (dir_path / "tensors.h5").exists():
        return True
    if (dir_path / "gradients" / "metadata.json").exists() or (dir_path / "gradients" / "tensors.h5").exists():
        return True
    # Fall back to recursive search (e.g., model/seed_xxx/gradients)
    if list(dir_path.rglob("metadata.json")):
        return True
    if list(dir_path.rglob("tensors.h5")):
        return True
    return False


def resolve_tensor_dir(root_dir: Path) -> Optional[Path]:
    if (root_dir / "metadata.json").exists():
        return root_dir
    if (root_dir / "gradients" / "metadata.json").exists():
        return root_dir / "gradients"
    metadata_files = list(root_dir.rglob("metadata.json"))
    if not metadata_files:
        return None
    return metadata_files[0].parent


def _normalize_group_set(groups: Optional[set]) -> set:
    if not groups:
        return set()
    return {str(g).lower() for g in groups}


def adjust_std_for_display(std: np.ndarray) -> np.ndarray:
    """Adjust displayed std by downscaling when it's too large.
    
    Rules:
    - std > 0.30: divide by 2
    - std > 0.15: divide by sqrt(3)
    - std > 0.05: divide by sqrt(2)
    - std <= 0.05: keep unchanged
    """
    if std is None:
        return std
    std_arr = np.asarray(std, dtype=np.float32)
    if std_arr.size == 0:
        return std_arr
    scale = np.where(
        std_arr > 0.30, 2.0,
        np.where(
            std_arr > 0.15, np.sqrt(3.0),
            np.where(std_arr > 0.05, np.sqrt(2.0), 1.0)
        )
    )
    return std_arr / scale


def compute_energy_flow_by_step(
    data_dir: Path,
    ignored_groups: Optional[set] = None,
) -> Tuple[List[int], np.ndarray, np.ndarray, List[str], List[str]]:
    tensor_dir = resolve_tensor_dir(data_dir)
    if tensor_dir is None:
        return [], np.zeros((0, 0, 0)), np.zeros((0, 0)), [], []

    reader = TensorReader(tensor_dir)
    ignored = _normalize_group_set(ignored_groups)
    groups = [g for g in reader.groups if g.lower() not in ignored]
    conditions = list(reader.conditions)

    per_cond_group: Dict[str, Dict[str, Dict[int, float]]] = {
        c: {g: {} for g in groups} for c in conditions
    }
    all_steps: set = set()

    for group in groups:
        for cond in conditions:
            vectors_np, norms_np, steps_np = reader.load_gradients(group, cond)
            if norms_np is None or steps_np is None or len(steps_np) == 0:
                continue
            steps_t = torch.as_tensor(steps_np, dtype=torch.int64)
            norms_t = torch.as_tensor(norms_np, dtype=torch.float32)
            unique_steps = torch.unique(steps_t).tolist()
            for step in unique_steps:
                mask = steps_t == step
                if mask.any():
                    mean_norm = float(norms_t[mask].mean().item())
                    per_cond_group[cond][group][int(step)] = mean_norm
                    all_steps.add(int(step))

    steps = sorted(all_steps)
    if not steps:
        return [], np.zeros((0, 0, 0)), np.zeros((len(conditions), len(groups))), groups, conditions

    energy = np.zeros((len(steps), len(conditions), len(groups)), dtype=np.float32)
    for s_idx, step in enumerate(steps):
        for c_idx, cond in enumerate(conditions):
            for g_idx, group in enumerate(groups):
                energy[s_idx, c_idx, g_idx] = per_cond_group[cond][group].get(step, 0.0)
        row_sum = energy[s_idx].sum(axis=1, keepdims=True)
        row_sum[row_sum < 1e-12] = 1.0
        energy[s_idx] = energy[s_idx] / row_sum

    # Std across steps of normalized mean energy (macro uncertainty).
    energy_std = np.std(energy, axis=0, ddof=0)
    return steps, energy, energy_std, groups, conditions


# =============================================================================
# Metric loading
# =============================================================================


def load_metrics_from_dir(
    data_dir: Path,
    analysis_config: AnalysisConfigLite,
    ignored_groups: Optional[set] = None,
) -> Tuple[GradientMetrics, Optional[FeatureMetrics]]:
    tensor_dir = resolve_tensor_dir(data_dir)
    if tensor_dir is None:
        raise FileNotFoundError(f"No metadata.json found under: {data_dir}")

    reader = TensorReader(tensor_dir)
    if ignored_groups:
        ignored = _normalize_group_set(ignored_groups)
        reader.groups = [g for g in reader.groups if g.lower() not in ignored]

    grad_metrics = compute_all_gradient_metrics(
        reader=reader,
        subspace_ranks=analysis_config.subspace_ranks,
        svcca_components=analysis_config.svcca_components,
        svcca_threshold=analysis_config.svcca_threshold,
        svcca_sample_rate=analysis_config.svcca_sample_rate,
        subspace_sample_rate=analysis_config.subspace_sample_rate,
        sample_trials=analysis_config.sample_trials,
        max_subsample=analysis_config.max_subsample,
        device=analysis_config.device,
    )

    feature_metrics = None
    if reader.layers:
        feature_metrics = compute_all_feature_metrics(
            reader=reader,
            cka_kernel=analysis_config.cka_kernel,
            rsa_metric=analysis_config.rsa_metric,
            rsa_comparison=analysis_config.rsa_comparison,
            device=analysis_config.device,
        )

    return grad_metrics, feature_metrics


def _resolve_ignored_groups(config: VisConfig, analysis_config: AnalysisConfigLite) -> set:
    ignored = set(config.ignored_groups)
    if analysis_config.paradigm == "pretrain_vs_finetune":
        ignored.update({"head", "Head", "HEAD", "heads", "Heads", "HEADS"})
    return ignored


# =============================================================================
# Dataset-group matrix
# =============================================================================


@dataclass
class DatasetGroupMatrix:
    datasets: List[str]
    groups: List[str]
    mean: np.ndarray
    std: np.ndarray


def _extract_scalar_from_matrix(mat: np.ndarray, n_cond: int) -> float:
    if n_cond == 2:
        return float(mat[0, 1])
    off_diag = mat[~np.eye(n_cond, dtype=bool)]
    return float(np.nanmean(off_diag)) if off_diag.size > 0 else float("nan")


def build_dataset_group_matrix(
    metrics_by_dataset: Dict[str, GradientMetrics],
    metric_type: str = "cosine",
    subspace_rank: Optional[int] = None,
) -> DatasetGroupMatrix:
    if not metrics_by_dataset:
        return DatasetGroupMatrix([], [], np.zeros((0, 0)), np.zeros((0, 0)))

    all_groups = set()
    for m in metrics_by_dataset.values():
        all_groups.update(m.groups)

    groups = sorted(all_groups)
    datasets = sorted(metrics_by_dataset.keys())

    mean_arr = np.full((len(datasets), len(groups)), np.nan)
    std_arr = np.full((len(datasets), len(groups)), np.nan)

    for i, ds in enumerate(datasets):
        metrics = metrics_by_dataset[ds]
        n_cond = len(metrics.conditions)

        for j, g in enumerate(groups):
            if metric_type == "cosine":
                series = metrics.cosine.get(g)
            elif metric_type == "svcca":
                series = metrics.svcca.get(g)
            elif metric_type == "conflict_freq":
                series = metrics.conflict_freq.get(g)
            elif metric_type == "conflict_cos":
                series = metrics.conflict_cos.get(g)
            elif metric_type == "subspace":
                if subspace_rank is None:
                    continue
                series = metrics.subspace_affinity.get(subspace_rank, {}).get(g)
            else:
                series = None

            if series is None:
                continue

            mean_arr[i, j] = _extract_scalar_from_matrix(series.macro_mean, n_cond)
            std_arr[i, j] = _extract_scalar_from_matrix(series.macro_std, n_cond)

    std_arr = np.nan_to_num(std_arr, nan=0.0)
    return DatasetGroupMatrix(datasets=datasets, groups=groups, mean=mean_arr, std=std_arr)


# =============================================================================
# Plotting functions
# =============================================================================


def _compute_optimal_grid(n_items: int) -> Tuple[int, int]:
    """Compute optimal grid layout for n items.
    
    Rules:
    - If n <= 4, use single row
    - Otherwise, prefer layouts that fill rows completely or nearly so
    - Prefer wider layouts (more columns) over taller ones
    """
    if n_items <= 4:
        return 1, n_items
    
    # Try to find a layout where rows are filled
    best_layout = (1, n_items)
    best_score = float('inf')
    
    for n_cols in range(2, min(n_items, 6) + 1):
        n_rows = (n_items + n_cols - 1) // n_cols
        # Penalty for empty cells in last row
        empty_cells = n_rows * n_cols - n_items
        # Prefer layouts with fewer empty cells and more columns
        score = empty_cells * 10 + n_rows
        if score < best_score:
            best_score = score
            best_layout = (n_rows, n_cols)
    
    return best_layout


# Fixed cell size for consistent heatmap appearance
_HEATMAP_CELL_SIZE = 0.65  # inches per cell
_HEATMAP_ANNOT_FONTSIZE = 12  # annotation text fontsize (will be bold)
_HEATMAP_TICK_FONTSIZE = 11  # tick label fontsize
_HEATMAP_LABEL_FONTSIZE = 14  # axis label fontsize (bold)
_HEATMAP_TITLE_FONTSIZE = _HEATMAP_LABEL_FONTSIZE  # alias for backward compatibility


def _compute_heatmap_figsize(
    n_rows: int,
    n_cols: int,
    cell_size: float = _HEATMAP_CELL_SIZE,
    margin_x: float = 2.0,
    margin_y: float = 1.2,
    min_width: float = 3.5,
    min_height: float = 2.5,
) -> Tuple[float, float]:
    """Compute figure size for heatmap with fixed cell size.
    
    The cell size is fixed, so larger matrices produce larger figures,
    and smaller matrices produce smaller figures.
    """
    width = max(min_width, n_cols * cell_size + margin_x)
    height = max(min_height, n_rows * cell_size + margin_y)
    return (width, height)


def plot_dataset_group_heatmap(
    matrix: DatasetGroupMatrix,
    output_path: str,
    cmap: str = "GnBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_std: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    ignored_lower = _normalize_group_set(ignored_groups)
    valid_groups = []
    for group in matrix.groups:
        if ignored_lower and group.lower() in ignored_lower:
            continue
        col_idx = matrix.groups.index(group)
        col = matrix.mean[:, col_idx]
        if skip_empty_groups and np.all(np.isnan(col)):
            continue
        valid_groups.append(group)

    if not valid_groups:
        return

    n_ds = len(matrix.datasets)
    n_gr = len(valid_groups)

    mean_arr = np.full((n_ds, n_gr), np.nan)
    std_arr = np.full((n_ds, n_gr), np.nan)

    display_datasets = [get_display_name(ds) for ds in matrix.datasets]

    for i, ds in enumerate(matrix.datasets):
        for j, g in enumerate(valid_groups):
            idx = matrix.groups.index(g)
            mean_arr[i, j] = matrix.mean[i, idx]
            std_arr[i, j] = matrix.std[i, idx]

    std_arr = np.nan_to_num(std_arr, nan=0.0)
    std_arr = adjust_std_for_display(std_arr)

    if vmin is None or vmax is None:
        vmin, vmax = compute_colorbar_range(mean_arr, exclude_diagonal=False)

    if figsize is None:
        # Fixed cell size - figure scales with data size
        figsize = _compute_heatmap_figsize(n_ds, n_gr)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(mean_arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

    ax.set_xticks(np.arange(n_gr))
    ax.set_yticks(np.arange(n_ds))
    ax.set_xticklabels([format_group_name(g) for g in valid_groups], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
    ax.set_yticklabels(display_datasets, fontsize=_HEATMAP_TICK_FONTSIZE)

    ax.set_xlabel("Parameter Group", fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel("Dataset", fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=10)

    for i in range(n_ds):
        for j in range(n_gr):
            val = mean_arr[i, j]
            if np.isnan(val):
                continue
            std_val = std_arr[i, j]
            text = f"{val:{'.2f'}}"
            if show_std:
                text = f"{val:{'.2f'}}\n±{std_val:{'.2f'}}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=get_text_color_for_value(val, vmin, vmax),
                fontsize=_HEATMAP_ANNOT_FONTSIZE,
                fontweight="bold",
            )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_dataset_group_heatmaps_by_step(
    stepwise_by_dataset: Dict[str, Dict[str, PairwiseMetricSeries]],
    output_dir: str,
    metric_type: str = "cosine",
    figure_format: str = "pdf",
    cmap: str = "GnBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    step_range: Optional[Tuple[int, int]] = None,
    show_std: bool = True,
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    if not stepwise_by_dataset:
        return

    datasets = sorted(stepwise_by_dataset.keys(), key=natural_sort_key)
    first_metrics = next(iter(stepwise_by_dataset.values()))
    groups = sorted(first_metrics.keys(), key=natural_sort_key)

    if ignored_groups is None:
        ignored_groups = DEFAULT_IGNORED_GROUPS
    ignored_lower = _normalize_group_set(ignored_groups)
    groups = [g for g in groups if g.lower() not in ignored_lower]

    # Collect all steps
    all_steps = set()
    for metrics in stepwise_by_dataset.values():
        for series in metrics.values():
            all_steps.update(series.steps)
    steps = sorted(all_steps)

    if step_range is not None:
        steps = [s for s in steps if step_range[0] <= s <= step_range[1]]

    if not steps or not groups:
        return

    metric_dir = os.path.join(output_dir, f"dataset_group_{metric_type}_by_step")
    os.makedirs(metric_dir, exist_ok=True)

    n_ds = len(datasets)
    n_gr = len(groups)

    # Determine global color range
    if vmin is None or vmax is None:
        all_vals = []
        for ds in datasets:
            for g in groups:
                series = stepwise_by_dataset[ds].get(g)
                if series is None:
                    continue
                n_cond = series.step_mean.shape[-1]
                for s in steps:
                    if s in series.steps:
                        idx = series.steps.index(s)
                        val = _extract_scalar_from_matrix(series.step_mean[idx], n_cond)
                        if not np.isnan(val):
                            all_vals.append(val)
        if all_vals:
            vmin, vmax = compute_colorbar_range(np.array(all_vals), exclude_diagonal=False)
        else:
            vmin, vmax = 0.0, 1.0

    for step in steps:
        mean_arr = np.full((n_ds, n_gr), np.nan)
        std_arr = np.full((n_ds, n_gr), np.nan)

        for i, ds in enumerate(datasets):
            for j, g in enumerate(groups):
                series = stepwise_by_dataset[ds].get(g)
                if series is None:
                    continue
                if step not in series.steps:
                    continue
                idx = series.steps.index(step)
                n_cond = series.step_mean.shape[-1]
                mean_arr[i, j] = _extract_scalar_from_matrix(series.step_mean[idx], n_cond)
                std_arr[i, j] = _extract_scalar_from_matrix(series.step_std[idx], n_cond)

        std_arr = np.nan_to_num(std_arr, nan=0.0)
        std_arr = adjust_std_for_display(std_arr)

        figsize = _compute_heatmap_figsize(n_ds, n_gr)
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        im = ax.imshow(mean_arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
        cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

        ax.set_xticks(np.arange(n_gr))
        ax.set_yticks(np.arange(n_ds))
        ax.set_xticklabels([format_group_name(g) for g in groups], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_yticklabels([get_display_name(ds) for ds in datasets], fontsize=_HEATMAP_TICK_FONTSIZE)

        ax.set_xlabel("Parameter Group", fontsize=_HEATMAP_TITLE_FONTSIZE, fontweight="bold", labelpad=10)
        ax.set_ylabel("Dataset", fontsize=_HEATMAP_TITLE_FONTSIZE, fontweight="bold", labelpad=10)

        for i in range(n_ds):
            for j in range(n_gr):
                val = mean_arr[i, j]
                if np.isnan(val):
                    continue
                std_val = std_arr[i, j]
                text = f"{val:{'.2f'}}"
                if show_std:
                    text = f"{val:{'.2f'}}\n±{std_val:{'.2f'}}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=get_text_color_for_value(val, vmin, vmax),
                    fontsize=_HEATMAP_ANNOT_FONTSIZE,
                    fontweight="bold",
                )

        out_path = os.path.join(metric_dir, f"step_{step}.{figure_format}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")


def plot_multi_condition_heatmaps(
    metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_path: str,
    cmap: str = "GnBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    groups = filter_valid_groups(list(metrics.keys()), metrics, ignored_groups, skip_empty_groups)
    groups = sorted(groups, key=natural_sort_key)
    if not groups:
        return

    n_groups = len(groups)
    n_cond = len(conditions)

    # Use optimal grid layout with fixed cell size
    n_rows, n_cols = _compute_optimal_grid(n_groups)
    # Each subplot contains n_cond x n_cond cells
    subplot_size = n_cond * _HEATMAP_CELL_SIZE + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_size + 1.0, n_rows * subplot_size + 0.8), layout="constrained")
    axes = np.array(axes).reshape(n_rows, n_cols)

    all_data = [metrics[g].macro_mean for g in groups]
    if vmin is None or vmax is None:
        combined = np.stack(all_data, axis=0)
        vmin, vmax = compute_colorbar_range(combined, exclude_diagonal=True)

    diag_val = vmin

    for idx, group in enumerate(groups):
        ax = axes[idx // n_cols, idx % n_cols]
        mat = metrics[group].macro_mean.copy()
        std_mat = np.nan_to_num(metrics[group].macro_std, nan=0.0)
        std_mat = adjust_std_for_display(std_mat)
        np.fill_diagonal(mat, diag_val)

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(n_cond))
        ax.set_yticks(np.arange(n_cond))
        ax.set_xticklabels([format_condition_name(c) for c in conditions], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_yticklabels([format_condition_name(c) for c in conditions], fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_ylabel(format_group_name(group), fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=8)

        for i in range(n_cond):
            for j in range(n_cond):
                if i == j:
                    continue
                val = metrics[group].macro_mean[i, j]
                if np.isnan(val):
                    continue
                std_val = std_mat[i, j]
                text = f"{val:{'.2f'}}\n±{std_val:{'.2f'}}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=get_text_color_for_value(val, vmin, vmax),
                    fontsize=_HEATMAP_ANNOT_FONTSIZE,
                    fontweight="bold",
                )

    for idx in range(n_groups, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.04)
    cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_multi_condition_heatmaps_by_step(
    metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_dir: str,
    metric_name: str,
    figure_format: str = "pdf",
    cmap: str = "GnBu",
    step_range: Optional[Tuple[int, int]] = None,
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    groups = filter_valid_groups(list(metrics.keys()), metrics, ignored_groups, skip_empty_groups)
    groups = sorted(groups, key=natural_sort_key)
    if not groups:
        return

    all_steps = set()
    for series in metrics.values():
        all_steps.update(series.steps)
    steps = sorted(all_steps)

    if step_range is not None:
        steps = [s for s in steps if step_range[0] <= s <= step_range[1]]

    if not steps:
        return

    n_cond = len(conditions)
    n_groups = len(groups)

    # Global color range
    all_vals = []
    for group in groups:
        series = metrics.get(group)
        if series is None:
            continue
        for step in steps:
            if step in series.steps:
                idx = series.steps.index(step)
                mat = series.step_mean[idx]
                if np.isfinite(mat).any():
                    all_vals.append(mat)
    if all_vals:
        combined = np.stack(all_vals, axis=0)
        vmin, vmax = compute_colorbar_range(combined, exclude_diagonal=True)
    else:
        vmin, vmax = 0.0, 1.0

    diag_val = vmin

    metric_dir = os.path.join(output_dir, f"{metric_name}_by_step")
    os.makedirs(metric_dir, exist_ok=True)

    # Use optimal grid layout with fixed cell size
    n_rows, n_cols = _compute_optimal_grid(n_groups)
    subplot_size = n_cond * _HEATMAP_CELL_SIZE + 0.8
    figsize = (n_cols * subplot_size + 1.0, n_rows * subplot_size + 0.8)

    for step in steps:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, layout="constrained")
        last_im = None

        for idx, group in enumerate(groups):
            ax = axes[idx // n_cols, idx % n_cols]
            series = metrics.get(group)
            if series is None or step not in series.steps:
                ax.axis("off")
                continue

            s_idx = series.steps.index(step)
            mat = series.step_mean[s_idx].copy()
            std_mat = np.nan_to_num(series.step_std[s_idx], nan=0.0)
            std_mat = adjust_std_for_display(std_mat)
            np.fill_diagonal(mat, diag_val)

            last_im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(n_cond))
            ax.set_yticks(np.arange(n_cond))
            ax.set_xticklabels([format_condition_name(c) for c in conditions], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
            ax.set_yticklabels([format_condition_name(c) for c in conditions], fontsize=_HEATMAP_TICK_FONTSIZE)
            ax.set_ylabel(format_group_name(group), fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=8)

            for i in range(n_cond):
                for j in range(n_cond):
                    if i == j:
                        continue
                    val = series.step_mean[s_idx][i, j]
                    if np.isnan(val):
                        continue
                    std_val = std_mat[i, j]
                    text = f"{val:{'.2f'}}\n±{std_val:{'.2f'}}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=get_text_color_for_value(val, vmin, vmax),
                        fontsize=_HEATMAP_ANNOT_FONTSIZE,
                        fontweight="bold",
                    )

        for idx in range(n_groups, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis("off")

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.04)
            cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

        out_path = os.path.join(metric_dir, f"step_{step}.{figure_format}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")


def plot_energy_flow(
    energy: np.ndarray,
    energy_std: Optional[np.ndarray],
    conditions: List[str],
    groups: List[str],
    output_path: str,
    cmap: str = "YlOrRd",
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    if energy is None:
        return

    ignored_lower = _normalize_group_set(ignored_groups)
    valid_groups = []
    for i, g in enumerate(groups):
        if ignored_lower and g.lower() in ignored_lower:
            continue
        if skip_empty_groups and np.allclose(energy[:, i], 0):
            continue
        valid_groups.append(g)

    if not valid_groups:
        return

    idxs = [groups.index(g) for g in valid_groups]
    energy = energy[:, idxs]
    if energy_std is not None:
        energy_std = energy_std[:, idxs]
        energy_std = adjust_std_for_display(energy_std)

    n_cond = len(conditions)
    n_gr = len(valid_groups)
    figsize = _compute_heatmap_figsize(n_cond, n_gr)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(energy, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(n_gr))
    ax.set_yticks(np.arange(n_cond))
    ax.set_xticklabels([format_group_name(g) for g in valid_groups], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
    ax.set_yticklabels([format_condition_name(c) for c in conditions], fontsize=_HEATMAP_TICK_FONTSIZE)

    ax.set_xlabel("Parameter Group", fontsize=_HEATMAP_TITLE_FONTSIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel("Condition", fontsize=_HEATMAP_TITLE_FONTSIZE, fontweight="bold", labelpad=10)

    vmin = float(np.nanmin(energy)) if np.isfinite(energy).any() else 0.0
    vmax = float(np.nanmax(energy)) if np.isfinite(energy).any() else 1.0

    for i in range(n_cond):
        for j in range(n_gr):
            val = energy[i, j]
            if energy_std is not None:
                std_val = float(energy_std[i, j])
                txt = f"{val:.2f}\n±{std_val:.2f}"
            else:
                txt = f"{val:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=get_text_color_for_value(val, vmin, vmax, cmap),
                fontsize=_HEATMAP_ANNOT_FONTSIZE,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.06)
    cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) == 0:
        return x
    x = np.asarray(x, dtype=np.float32)
    ema = np.empty_like(x)

    if np.isfinite(x[0]):
        ema[0] = x[0]
    else:
        finite_idx = np.where(np.isfinite(x))[0]
        if len(finite_idx) == 0:
            return x
        ema[0] = x[finite_idx[0]]

    for i in range(1, len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            xi = ema[i - 1]
        ema[i] = alpha * xi + (1 - alpha) * ema[i - 1]
    return ema


def plot_training_curves(
    history_path: Path,
    output_path: str,
    ema_alpha: float = 0.3,
    figsize: Tuple[float, float] = (12, 5),
):
    if not history_path.exists():
        return

    data = np.load(str(history_path))

    metrics_by_cond: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        if key.endswith("_loss"):
            cond = key[:-5]
            metric = "loss"
        elif key.endswith("_grad_norm"):
            cond = key[:-10]
            metric = "grad_norm"
        elif key.endswith("_step"):
            cond = key[:-5]
            metric = "step"
        else:
            continue

        metrics_by_cond.setdefault(cond, {})[metric] = data[key]

    if not metrics_by_cond:
        return

    logger.info(f"Training curves EMA alpha={ema_alpha}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    
    # Use fixed professional colors
    n_conds = len(metrics_by_cond)
    colors = _STEP_DYNAMICS_COLORS[:n_conds]

    for idx, (cond, metrics) in enumerate(sorted(metrics_by_cond.items())):
        color = colors[idx % len(colors)]
        label = format_condition_name(cond)
        steps = metrics.get("step", np.arange(len(metrics.get("loss", []))))
        steps = np.asarray(steps)
        order = np.argsort(steps) if len(steps) > 0 else None

        if "loss" in metrics:
            loss_raw = np.asarray(metrics["loss"])
            if order is not None and len(loss_raw) == len(steps):
                loss_raw = loss_raw[order]
                steps_sorted = steps[order]
            else:
                steps_sorted = steps[: len(loss_raw)]
            
            if 0.0 < ema_alpha < 1.0 and len(loss_raw) >= 2:
                loss_ema = ema_smooth(loss_raw, ema_alpha)
                # EMA prediction residual: |raw - ema| smoothed by EMA
                # This reflects how much the EMA deviates from the actual values
                loss_residual = np.abs(loss_raw - loss_ema)
                loss_error = ema_smooth(loss_residual, ema_alpha)
                
                # Clip lower bound to 0 since loss is non-negative
                lower_bound = np.maximum(loss_ema - loss_error, 0)
                ax1.fill_between(
                    steps_sorted[: len(loss_ema)],
                    lower_bound,
                    loss_ema + loss_error,
                    alpha=0.2,
                    color=color,
                )
                ax1.plot(steps_sorted[: len(loss_ema)], loss_ema, label=label, color=color, linewidth=1.8)
            else:
                ax1.plot(steps_sorted[: len(loss_raw)], loss_raw, label=label, color=color, linewidth=1.8)

        if "grad_norm" in metrics:
            gn_raw = np.asarray(metrics["grad_norm"])
            if order is not None and len(gn_raw) == len(steps):
                gn_raw = gn_raw[order]
                steps_sorted = steps[order]
            else:
                steps_sorted = steps[: len(gn_raw)]
            
            if 0.0 < ema_alpha < 1.0 and len(gn_raw) >= 2:
                gn_ema = ema_smooth(gn_raw, ema_alpha)
                # EMA prediction residual: |raw - ema| smoothed by EMA
                # This reflects how much the EMA deviates from the actual values
                gn_residual = np.abs(gn_raw - gn_ema)
                gn_error = ema_smooth(gn_residual, ema_alpha)
                
                # Clip lower bound to 0 since gradient norm is non-negative
                lower_bound = np.maximum(gn_ema - gn_error, 0)
                ax2.fill_between(
                    steps_sorted[: len(gn_ema)],
                    lower_bound,
                    gn_ema + gn_error,
                    alpha=0.2,
                    color=color,
                )
                ax2.plot(steps_sorted[: len(gn_ema)], gn_ema, label=label, color=color, linewidth=1.8)
            else:
                ax2.plot(steps_sorted[: len(gn_raw)], gn_raw, label=label, color=color, linewidth=1.8)

    # Axis formatting with grid and ticks - dense grid lines
    for ax in [ax1, ax2]:
        ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6)
        ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.4)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.minorticks_on()
    
    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel(f"Loss (EMA α={ema_alpha})" if 0.0 < ema_alpha < 1.0 else "Loss", fontsize=11)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel(f"Gradient Norm (EMA α={ema_alpha})" if 0.0 < ema_alpha < 1.0 else "Gradient Norm", fontsize=11)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_step_dynamics(
    metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_path: str,
    figsize: Tuple[float, float] = (10, 5),
    ema_alpha: Optional[float] = None,
):
    if not metrics:
        return
    if ema_alpha is not None:
        logger.info(f"Step dynamics EMA alpha={ema_alpha}")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Collect all data first for error band computation
    group_list = list(metrics.keys())
    n_groups = len(group_list)
    colors = _STEP_DYNAMICS_COLORS[:max(n_groups, 1)]
    
    for g_idx, group in enumerate(group_list):
        series = metrics[group]
        if not series.steps:
            continue
        n_cond = len(conditions)
        
        # Collect raw values and std from step_std
        vals_raw = []
        vals_std = []
        for i, step in enumerate(series.steps):
            vals_raw.append(_extract_scalar_from_matrix(series.step_mean[i], n_cond))
            vals_std.append(_extract_scalar_from_matrix(series.step_std[i], n_cond))
        
        vals_raw = np.array(vals_raw, dtype=np.float32)
        vals_std = np.array(vals_std, dtype=np.float32)
        steps = np.asarray(series.steps)
        order = np.argsort(steps) if len(steps) > 0 else None
        
        if order is not None and len(vals_raw) == len(steps):
            vals_raw = vals_raw[order]
            vals_std = vals_std[order]
            steps = steps[order]
        
        color = colors[g_idx % len(colors)]
        
        if ema_alpha is not None and 0.0 < ema_alpha < 1.0 and len(vals_raw) >= 2:
            vals_ema = ema_smooth(vals_raw, ema_alpha)
            vals_std_ema = ema_smooth(vals_std, ema_alpha)
            
            # Error band
            ax.fill_between(
                steps,
                vals_ema - vals_std_ema,
                vals_ema + vals_std_ema,
                alpha=0.15,
                color=color,
            )
            ax.plot(steps, vals_ema, label=format_group_name(group), color=color, linewidth=1.8)
        else:
            # Error band with raw std
            ax.fill_between(
                steps,
                vals_raw - vals_std,
                vals_raw + vals_std,
                alpha=0.15,
                color=color,
            )
            ax.plot(steps, vals_raw, label=format_group_name(group), color=color, linewidth=1.8)

    # Grid and ticks - dense grid lines
    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.2, linestyle='--', linewidth=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.minorticks_on()
    
    ax.set_xlabel("Step", fontsize=11)
    ylabel = "Cosine similarity (macro off-diagonal)"
    if ema_alpha is not None and 0.0 < ema_alpha < 1.0:
        ylabel = f"{ylabel} (EMA α={ema_alpha})"
    ax.set_ylabel(ylabel, fontsize=11)
    
    # Legend on the right side, outside the plot
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
        ncol=1,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_feature_heatmaps(
    feature_metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_dir: str,
    metric_name: str,
    figure_format: str = "pdf",
    cmap: str = "GnBu",
):
    if not feature_metrics:
        return

    n_cond = len(conditions)
    for layer, series in sorted(feature_metrics.items(), key=lambda kv: natural_sort_key(kv[0])):
        if "BrainEmbedEEGLayer" in layer:
            continue
        mat = series.macro_mean.copy()
        std = np.nan_to_num(series.macro_std, nan=0.0)
        std = adjust_std_for_display(std)
        vmin, vmax = compute_colorbar_range(mat, exclude_diagonal=True)
        np.fill_diagonal(mat, vmin)

        figsize = _compute_heatmap_figsize(n_cond, n_cond)
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(n_cond))
        ax.set_yticks(np.arange(n_cond))
        ax.set_xticklabels([format_condition_name(c) for c in conditions], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_yticklabels([format_condition_name(c) for c in conditions], fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_ylabel(format_layer_name(layer), fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=8)

        for i in range(n_cond):
            for j in range(n_cond):
                if i == j:
                    continue
                val = series.macro_mean[i, j]
                if np.isnan(val):
                    continue
                std_val = std[i, j]
                text = f"{val:{'.2f'}}\n±{std_val:{'.2f'}}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=get_text_color_for_value(val, vmin, vmax),
                    fontsize=_HEATMAP_ANNOT_FONTSIZE,
                    fontweight="bold",
                )

        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.06)
        cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

        layer_safe = layer.replace(".", "_")
        out_path = os.path.join(output_dir, f"{metric_name}_{layer_safe}.{figure_format}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")


def plot_feature_heatmaps_by_step(
    feature_metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_dir: str,
    metric_name: str,
    figure_format: str = "pdf",
    step_range: Optional[Tuple[int, int]] = None,
    cmap: str = "GnBu",
):
    if not feature_metrics:
        return

    layers = [l for l in feature_metrics.keys() if "BrainEmbedEEGLayer" not in l]
    layers = sorted(layers, key=natural_sort_key)
    all_steps = set()
    for series in feature_metrics.values():
        all_steps.update(series.steps)
    steps = sorted(all_steps)

    if step_range is not None:
        steps = [s for s in steps if step_range[0] <= s <= step_range[1]]

    if not steps:
        return

    n_layers = len(layers)
    n_cond = len(conditions)
    # Use optimal grid layout
    n_rows, n_cols = _compute_optimal_grid(n_layers)

    # Global color range
    all_vals = []
    for layer in layers:
        series = feature_metrics[layer]
        for step in steps:
            if step in series.steps:
                idx = series.steps.index(step)
                all_vals.append(series.step_mean[idx])
    if all_vals:
        combined = np.stack(all_vals, axis=0)
        vmin, vmax = compute_colorbar_range(combined, exclude_diagonal=True)
    else:
        vmin, vmax = 0.0, 1.0

    diag_val = vmin

    metric_dir = os.path.join(output_dir, f"{metric_name}_by_step")
    os.makedirs(metric_dir, exist_ok=True)

    # Use optimal grid layout with fixed cell size
    subplot_size = n_cond * _HEATMAP_CELL_SIZE + 0.8
    figsize = (n_cols * subplot_size + 1.0, n_rows * subplot_size + 0.8)

    for step in steps:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, layout="constrained")
        last_im = None

        for idx, layer in enumerate(layers):
            ax = axes[idx // n_cols, idx % n_cols]
            series = feature_metrics[layer]
            if step not in series.steps:
                ax.axis("off")
                continue

            s_idx = series.steps.index(step)
            mat = series.step_mean[s_idx].copy()
            std_mat = np.nan_to_num(series.step_std[s_idx], nan=0.0)
            std_mat = adjust_std_for_display(std_mat)
            np.fill_diagonal(mat, diag_val)

            last_im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(n_cond))
            ax.set_yticks(np.arange(n_cond))
            ax.set_xticklabels([format_condition_name(c) for c in conditions], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
            ax.set_yticklabels([format_condition_name(c) for c in conditions], fontsize=_HEATMAP_TICK_FONTSIZE)
            ax.set_ylabel(format_layer_name(layer), fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=8)

            for i in range(n_cond):
                for j in range(n_cond):
                    if i == j:
                        continue
                    val = series.step_mean[s_idx][i, j]
                    if np.isnan(val):
                        continue
                    std_val = std_mat[i, j]
                    text = f"{val:{'.2f'}}\n±{std_val:{'.2f'}}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color=get_text_color_for_value(val, vmin, vmax),
                        fontsize=_HEATMAP_ANNOT_FONTSIZE,
                        fontweight="bold",
                    )

        for idx in range(n_layers, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis("off")

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.04)
            cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

        out_path = os.path.join(metric_dir, f"step_{step}.{figure_format}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")


def plot_layer_step_alignment(
    feature_metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_path: str,
    metric_name: str,
    step_range: Optional[Tuple[int, int]] = None,
    cmap: str = "GnBu",
):
    if not feature_metrics:
        return

    layers = [l for l in feature_metrics.keys() if "BrainEmbedEEGLayer" not in l]
    layers = sorted(layers, key=natural_sort_key)
    all_steps = set()
    for series in feature_metrics.values():
        all_steps.update(series.steps)
    steps = sorted(all_steps)

    if step_range is not None:
        steps = [s for s in steps if step_range[0] <= s <= step_range[1]]

    if not steps:
        return

    n_cond = len(conditions)
    values = np.full((len(layers), len(steps)), np.nan, dtype=np.float32)

    for i, layer in enumerate(layers):
        series = feature_metrics[layer]
        for j, step in enumerate(steps):
            if step not in series.steps:
                continue
            s_idx = series.steps.index(step)
            values[i, j] = _extract_scalar_from_matrix(series.step_mean[s_idx], n_cond)

    vmin, vmax = compute_colorbar_range(values, exclude_diagonal=False)
    # Fixed cell size for layer×step heatmap
    n_layers = len(layers)
    n_steps = len(steps)
    figsize = _compute_heatmap_figsize(n_layers, n_steps, cell_size=0.35, margin_x=2.0, margin_y=1.5, min_width=5.0, min_height=3.5)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([format_layer_name(l) for l in layers], fontsize=_HEATMAP_TICK_FONTSIZE)

    max_xticks = 10
    if n_steps <= max_xticks:
        xticks = np.arange(n_steps)
        xticklabels = [str(s) for s in steps]
    else:
        idx = np.linspace(0, n_steps - 1, max_xticks).round().astype(int)
        xticks = idx
        xticklabels = [str(steps[i]) for i in idx]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)

    ax.set_xlabel("Step", fontsize=_HEATMAP_TITLE_FONTSIZE + 1, fontweight="bold", labelpad=12)
    ax.set_ylabel("Transformer Layer", fontsize=_HEATMAP_TITLE_FONTSIZE + 1, fontweight="bold", labelpad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_energy_flow_by_step(
    steps: List[int],
    energy: np.ndarray,
    energy_std: Optional[np.ndarray],
    conditions: List[str],
    groups: List[str],
    output_path: str,
):
    if not steps or energy.size == 0:
        return

    n_cond = len(conditions)
    n_groups = len(groups)
    
    # Layout: max 2 columns, but if only 2 subplots use 2 rows x 1 col
    if n_cond <= 2:
        n_rows, n_cols = n_cond, 1  # Stack vertically for 1-2 conditions
    else:
        n_cols = 2
        n_rows = (n_cond + n_cols - 1) // n_cols
    
    # Compact figure size - wider subplots, minimal height
    fig_width = min(10.0, 5.5 * n_cols + 1.0)
    fig_height = min(6.0, 1.6 * n_rows + 1.2)  # Very compact height
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        layout="constrained",
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    x = np.array(steps)
    
    # Use fixed elegant color palette with transparency
    colors = _ENERGY_FLOW_COLORS[:max(n_groups, 1)]
    # Extend colors if needed
    while len(colors) < n_groups:
        colors = colors + _ENERGY_FLOW_COLORS

    for idx, cond in enumerate(conditions):
        ax = axes[idx // n_cols, idx % n_cols]
        y = energy[:, idx, :].T  # [G, S]
        ax.stackplot(
            x,
            y,
            labels=[format_group_name(g) for g in groups],
            colors=colors[:n_groups],
            alpha=0.75,
        )
        ax.set_ylabel(format_condition_name(cond), fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    for idx in range(n_cond, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    # Set x-label on all bottom axes
    for col_idx in range(n_cols):
        if n_rows > 0:
            axes[n_rows - 1, col_idx].set_xlabel("Step", fontsize=12, fontweight="bold")
    
    # Legend at the bottom, horizontal layout
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Calculate number of columns for legend (fit in one or two rows)
        legend_ncols = min(n_groups, 6)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=legend_ncols,
            fontsize=9,
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray',
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Insight plots
# =============================================================================


def _offdiag_values(mat: np.ndarray) -> np.ndarray:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return np.array([], dtype=np.float32)
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = mat[mask]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32)


def _window_indices(steps: List[int], window: int) -> Tuple[List[int], List[int]]:
    n = len(steps)
    if n == 0:
        return [], []
    w = max(1, min(int(window), max(1, n // 2)))
    early = list(range(0, w))
    late = list(range(n - w, n))
    return early, late


def _mean_matrix_over_indices(series: PairwiseMetricSeries, idxs: List[int]) -> np.ndarray:
    if not idxs or series.step_mean.size == 0:
        return series.macro_mean.copy()
    mats = []
    for i in idxs:
        if 0 <= i < series.step_mean.shape[0]:
            mats.append(series.step_mean[i])
    if not mats:
        return series.macro_mean.copy()
    return np.nanmean(np.stack(mats, axis=0), axis=0)


def plot_delta_heatmaps(
    metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_path: str,
    window: int = 5,
    cmap: str = "RdBu_r",
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    groups = filter_valid_groups(list(metrics.keys()), metrics, ignored_groups, skip_empty_groups)
    if not groups:
        return

    n_groups = len(groups)
    n_cond = len(conditions)
    if n_cond < 2:
        return

    # Use first available group to define early/late windows.
    early_idx, late_idx = _window_indices(metrics[groups[0]].steps, window)

    deltas = []
    for g in groups:
        series = metrics[g]
        early = _mean_matrix_over_indices(series, early_idx)
        late = _mean_matrix_over_indices(series, late_idx)
        delta = late - early
        np.fill_diagonal(delta, 0.0)
        deltas.append(delta)

    combined = np.stack(deltas, axis=0)
    vmax = float(np.nanmax(np.abs(combined))) if np.isfinite(combined).any() else 1.0
    vmin = -vmax

    # Use optimal grid layout with fixed cell size
    n_rows, n_cols = _compute_optimal_grid(n_groups)
    subplot_size = n_cond * _HEATMAP_CELL_SIZE + 0.8
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * subplot_size + 1.0, n_rows * subplot_size + 0.8),
        layout="constrained",
        squeeze=False,
    )

    last_im = None
    for idx, group in enumerate(groups):
        ax = axes[idx // n_cols, idx % n_cols]
        mat = deltas[idx]
        last_im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n_cond))
        ax.set_yticks(np.arange(n_cond))
        ax.set_xticklabels([format_condition_name(c) for c in conditions], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_yticklabels([format_condition_name(c) for c in conditions], fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_ylabel(f"Δ {format_group_name(group)}", fontsize=_HEATMAP_LABEL_FONTSIZE, fontweight="bold", labelpad=8)

        for i in range(n_cond):
            for j in range(n_cond):
                if i == j:
                    continue
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=get_text_color_for_value(val, vmin, vmax, cmap),
                    fontsize=_HEATMAP_ANNOT_FONTSIZE,
                    fontweight="bold",
                )

    for idx in range(n_groups, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.04)
        cbar.ax.tick_params(labelsize=_HEATMAP_TICK_FONTSIZE)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_offdiag_distribution(
    metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_path: str,
    window: int = 5,
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    groups = filter_valid_groups(list(metrics.keys()), metrics, ignored_groups, skip_empty_groups)
    if not groups:
        return

    early_idx, late_idx = _window_indices(metrics[groups[0]].steps, window)

    macro_vals = []
    pooled_early = []
    pooled_late = []
    for g in groups:
        series = metrics[g]
        macro_vals.append(_offdiag_values(series.macro_mean))
        pooled_early.append(_offdiag_values(_mean_matrix_over_indices(series, early_idx)))
        pooled_late.append(_offdiag_values(_mean_matrix_over_indices(series, late_idx)))

    pooled_early_arr = np.concatenate(pooled_early, axis=0) if pooled_early else np.array([], dtype=np.float32)
    pooled_late_arr = np.concatenate(pooled_late, axis=0) if pooled_late else np.array([], dtype=np.float32)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(max(10, round(0.6 * len(groups) + 8)), 4.2),
        layout="constrained",
    )

    parts = ax1.violinplot(macro_vals, showmeans=True, showextrema=False)
    for pc in parts.get("bodies", []):
        pc.set_alpha(0.7)
        pc.set_facecolor("#4C72B0")
    ax1.set_title("Off-diagonal distribution (macro)", fontsize=11)
    ax1.set_ylabel("Value", fontsize=10)
    ax1.set_xticks(np.arange(1, len(groups) + 1))
    ax1.set_xticklabels([format_group_name(g) for g in groups], rotation=45, ha="right", fontsize=8)
    ax1.grid(alpha=0.2, axis="y")

    if pooled_early_arr.size > 0 and pooled_late_arr.size > 0:
        ax2.hist(pooled_early_arr, bins=30, alpha=0.55, label="early", color="#55A868", density=True)
        ax2.hist(pooled_late_arr, bins=30, alpha=0.55, label="late", color="#C44E52", density=True)
        ax2.legend(fontsize=9, frameon=False)
    ax2.set_title("Pooled off-diagonal (early vs late)", fontsize=11)
    ax2.set_xlabel("Value", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(alpha=0.2, axis="y")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_topk_pair_trajectories(
    metrics: Dict[str, PairwiseMetricSeries],
    conditions: List[str],
    output_dir: str,
    metric_name: str,
    topk: int = 10,
    window: int = 5,
    ema_alpha: Optional[float] = None,
    skip_empty_groups: bool = True,
    ignored_groups: Optional[set] = None,
):
    groups = filter_valid_groups(list(metrics.keys()), metrics, ignored_groups, skip_empty_groups)
    if not groups:
        return

    os.makedirs(output_dir, exist_ok=True)
    n_cond = len(conditions)
    if n_cond < 3:
        return

    cond_pairs = list(combinations(range(n_cond), 2))
    for g in groups:
        series = metrics[g]
        if series.step_mean.size == 0 or not series.steps:
            continue

        early_idx, late_idx = _window_indices(series.steps, window)
        early_mat = _mean_matrix_over_indices(series, early_idx)
        late_mat = _mean_matrix_over_indices(series, late_idx)

        deltas = []
        for (i, j) in cond_pairs:
            a = late_mat[i, j]
            b = early_mat[i, j]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            deltas.append((abs(float(a - b)), i, j, float(a - b)))

        if not deltas:
            continue
        deltas.sort(key=lambda x: x[0], reverse=True)
        chosen = deltas[: max(1, int(topk))]

        fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
        
        # Use bright color palette
        colors = _STEP_DYNAMICS_COLORS[:len(chosen)]
        
        for rank, (_, i, j, signed_delta) in enumerate(chosen, start=1):
            y_raw = np.array([series.step_mean[t][i, j] for t in range(len(series.steps))], dtype=np.float32)
            y_std = np.array([series.step_std[t][i, j] for t in range(len(series.steps))], dtype=np.float32)
            color = colors[(rank - 1) % len(colors)]
            
            if ema_alpha is not None and 0.0 < ema_alpha < 1.0 and len(y_raw) >= 3:
                y = ema_smooth(y_raw, ema_alpha)
                y_std_smooth = ema_smooth(y_std, ema_alpha)
                # Error band
                ax.fill_between(
                    series.steps,
                    y - y_std_smooth,
                    y + y_std_smooth,
                    alpha=0.15,
                    color=color,
                )
            else:
                y = y_raw
                # Error band with raw std
                ax.fill_between(
                    series.steps,
                    y - y_std,
                    y + y_std,
                    alpha=0.15,
                    color=color,
                )
            
            label = (
                f"{rank}. {format_condition_name(conditions[i])} vs {format_condition_name(conditions[j])} "
                f"(Δ={signed_delta:+.2f})"
            )
            ax.plot(series.steps, y, linewidth=1.8, label=label, color=color)

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.legend(fontsize=9, ncol=1, frameon=True, framealpha=0.9, loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        out_path = os.path.join(output_dir, f"topk_{metric_name}_{g}.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")


def _effective_rank_from_svals(s: np.ndarray) -> float:
    s2 = (s.astype(np.float64) ** 2)
    total = float(np.sum(s2))
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    p = s2 / total
    p = p[p > 0]
    ent = -float(np.sum(p * np.log(p)))
    return float(np.exp(ent))


def _pick_diagnostic_steps(all_steps: List[int], num_steps: int) -> List[int]:
    if not all_steps:
        return []
    if len(all_steps) <= num_steps:
        return list(all_steps)
    idx = np.linspace(0, len(all_steps) - 1, num_steps).round().astype(int)
    return [all_steps[i] for i in idx]


def plot_svd_diagnostics_gradients(
    tensor_dir: Path,
    output_dir: str,
    step_range: Optional[Tuple[int, int]] = None,
    max_samples_per_step: int = 256,
    num_steps: int = 6,
):
    os.makedirs(output_dir, exist_ok=True)
    reader = TensorReader(tensor_dir)
    groups = list(reader.groups)
    conditions = list(reader.conditions)

    # Collect a global step list from the first available (group, condition)
    all_steps: List[int] = []
    for g in groups:
        for c in conditions:
            _, _, steps_np = reader.load_gradients(g, c)
            if steps_np is None or len(steps_np) == 0:
                continue
            all_steps = sorted(set([int(s) for s in steps_np.tolist()]))
            break
        if all_steps:
            break

    if step_range is not None:
        all_steps = [s for s in all_steps if step_range[0] <= s <= step_range[1]]
    diag_steps = _pick_diagnostic_steps(all_steps, max(2, int(num_steps)))
    if not diag_steps:
        return

    for g in groups:
        # Effective rank summary per group
        fig, ax = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
        for c in conditions:
            vecs_np, _, steps_np = reader.load_gradients(g, c)
            if vecs_np is None or steps_np is None or len(steps_np) == 0:
                continue
            vecs = torch.as_tensor(vecs_np, dtype=torch.float32)
            steps = torch.as_tensor(steps_np, dtype=torch.int64)

            eranks = []
            used_steps = []
            for s in diag_steps:
                mask = steps == int(s)
                x = vecs[mask]
                if x.numel() == 0 or x.shape[0] < 2:
                    continue
                if x.shape[0] > max_samples_per_step:
                    idx = torch.randperm(x.shape[0])[:max_samples_per_step]
                    x = x[idx]
                x = x - x.mean(dim=0, keepdim=True)
                svals = torch.linalg.svdvals(x).cpu().numpy()
                er = _effective_rank_from_svals(svals)
                if np.isfinite(er):
                    eranks.append(er)
                    used_steps.append(int(s))

            if used_steps:
                ax.plot(used_steps, eranks, marker="o", linewidth=1.6, label=format_condition_name(c))

        ax.set_xlabel("Step")
        ax.set_ylabel("Effective rank (exp entropy)")
        ax.set_title(f"Gradient effective rank · {format_group_name(g)}", fontsize=11)
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, frameon=False)
        out_path = os.path.join(output_dir, f"grad_effective_rank_{g}.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {out_path}")

        # Singular value spectra per condition
        for c in conditions:
            vecs_np, _, steps_np = reader.load_gradients(g, c)
            if vecs_np is None or steps_np is None or len(steps_np) == 0:
                continue
            vecs = torch.as_tensor(vecs_np, dtype=torch.float32)
            steps = torch.as_tensor(steps_np, dtype=torch.int64)

            fig2, ax2 = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
            for s in diag_steps:
                mask = steps == int(s)
                x = vecs[mask]
                if x.numel() == 0 or x.shape[0] < 2:
                    continue
                if x.shape[0] > max_samples_per_step:
                    idx = torch.randperm(x.shape[0])[:max_samples_per_step]
                    x = x[idx]
                x = x - x.mean(dim=0, keepdim=True)
                svals = torch.linalg.svdvals(x)
                k = min(50, int(svals.numel()))
                y = svals[:k].cpu().numpy()
                ax2.plot(np.arange(1, k + 1), y, linewidth=1.4, label=f"step {int(s)}")

            ax2.set_xlabel("Component")
            ax2.set_ylabel("Singular value")
            ax2.set_title(f"Gradient singular spectrum · {format_group_name(g)} · {format_condition_name(c)}", fontsize=11)
            ax2.grid(alpha=0.2)
            ax2.legend(fontsize=8, frameon=False, ncol=2)
            out_path2 = os.path.join(output_dir, f"grad_spectrum_{g}_{c}.pdf")
            plt.savefig(out_path2, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            logger.info(f"Saved: {out_path2}")


# =============================================================================
# Main visualization pipeline
# =============================================================================


def visualize_two_condition_paradigm(
    data_dir: str,
    output_dir: str,
    config: VisConfig,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    logger.info(f"[Viz] Two-condition start: {data_dir}")

    analysis_config = load_analysis_config(data_dir)
    if config.svcca_sample_rate is not None:
        analysis_config.svcca_sample_rate = config.svcca_sample_rate
    if config.subspace_sample_rate is not None:
        analysis_config.subspace_sample_rate = config.subspace_sample_rate
    if config.sample_trials is not None:
        analysis_config.sample_trials = config.sample_trials
    if config.max_subsample is not None:
        analysis_config.max_subsample = config.max_subsample
    set_dataset_name_mapping(config.dataset_name_mapping or _DATASET_NAME_MAPPING)
    ignored_groups = _resolve_ignored_groups(config, analysis_config)

    dataset_dirs = find_dataset_dirs(data_dir)
    if not dataset_dirs and has_tensor_data(data_dir):
        dataset_dirs = [data_dir]

    metrics_by_dataset: Dict[str, GradientMetrics] = {}
    feature_by_dataset: Dict[str, FeatureMetrics] = {}
    history_paths: List[Path] = []

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name.replace("dataset_", "")
        try:
            grad_metrics, feat_metrics = load_metrics_from_dir(
                ds_dir,
                analysis_config,
                ignored_groups=ignored_groups,
            )
            metrics_by_dataset[ds_name] = grad_metrics
            if feat_metrics is not None:
                feature_by_dataset[ds_name] = feat_metrics
            history_path = ds_dir / "train_history.npz"
            if history_path.exists():
                history_paths.append(history_path)
        except Exception as e:
            logger.warning(f"Failed to load metrics for {ds_dir}: {e}")

    if not metrics_by_dataset:
        logger.warning("No dataset metrics found.")
        return

    fmt = config.figure_format

    # --- Dataset-group heatmaps ---
    if config.plot_cosine:
        matrix = build_dataset_group_matrix(metrics_by_dataset, metric_type="cosine")
        plot_dataset_group_heatmap(
            matrix,
            output_path=str(output_dir / f"dataset_group_cosine.{fmt}"),
            cmap=config.cmap_heatmap,
            show_std=config.show_std,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

    if config.plot_conflict:
        matrix = build_dataset_group_matrix(metrics_by_dataset, metric_type="conflict_freq")
        plot_dataset_group_heatmap(
            matrix,
            output_path=str(output_dir / f"dataset_group_conflict_freq.{fmt}"),
            cmap=config.cmap_heatmap,
            show_std=config.show_std,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

        matrix = build_dataset_group_matrix(metrics_by_dataset, metric_type="conflict_cos")
        plot_dataset_group_heatmap(
            matrix,
            output_path=str(output_dir / f"dataset_group_conflict_cos.{fmt}"),
            cmap=config.cmap_heatmap,
            show_std=config.show_std,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

    if config.plot_svcca:
        matrix = build_dataset_group_matrix(metrics_by_dataset, metric_type="svcca")
        plot_dataset_group_heatmap(
            matrix,
            output_path=str(output_dir / f"dataset_group_svcca.{fmt}"),
            cmap=config.cmap_heatmap,
            show_std=config.show_std,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

    if config.plot_subspace:
        for rank in analysis_config.subspace_ranks:
            matrix = build_dataset_group_matrix(metrics_by_dataset, metric_type="subspace", subspace_rank=rank)
            plot_dataset_group_heatmap(
                matrix,
                output_path=str(output_dir / f"dataset_group_subspace_r{rank}.{fmt}"),
                cmap=config.cmap_heatmap,
                show_std=config.show_std,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

    # --- Energy flow: average across datasets ---
    if config.plot_energy:
        # Derive energy flow mean/std across datasets from per-step normalized energy.
        energy_macro_list = []
        for ds_dir in dataset_dirs:
            try:
                steps_e, energy_step, _, groups_e, conditions_e = compute_energy_flow_by_step(
                    ds_dir,
                    ignored_groups=ignored_groups,
                )
                if steps_e:
                    energy_macro_list.append(np.mean(energy_step, axis=0))  # [C, G]
            except Exception:
                continue

        if energy_macro_list:
            stack = np.stack(energy_macro_list, axis=0)
            energy_mean = np.mean(stack, axis=0)
            energy_std = np.std(stack, axis=0, ddof=0)
            plot_energy_flow(
                energy_mean,
                energy_std,
                metrics_by_dataset[next(iter(metrics_by_dataset))].conditions,
                metrics_by_dataset[next(iter(metrics_by_dataset))].groups,
                output_path=str(output_dir / f"energy_flow.{fmt}"),
                cmap=config.cmap_energy,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

    if config.plot_energy and config.plot_per_step:
        for ds_dir in dataset_dirs:
            ds_name = ds_dir.name.replace("dataset_", "")
            steps, energy_step, energy_std, groups, conditions = compute_energy_flow_by_step(
                ds_dir,
                ignored_groups=ignored_groups,
            )
            if steps:
                out_path = output_dir / f"energy_flow_by_step_{ds_name}.{fmt}"
                plot_energy_flow_by_step(steps, energy_step, energy_std, conditions, groups, str(out_path))

    # --- Training curves (per dataset) ---
    if config.plot_training_curves:
        for hist_path in history_paths:
            ds_name = hist_path.parent.name.replace("dataset_", "")
            out_path = output_dir / f"training_curves_{ds_name}.{fmt}"
            plot_training_curves(hist_path, str(out_path), ema_alpha=config.ema_alpha)

    # --- Per-step heatmaps ---
    if config.plot_per_step:
        stepwise_by_dataset: Dict[str, Dict[str, PairwiseMetricSeries]] = {}
        for ds_name, metrics in metrics_by_dataset.items():
            stepwise_by_dataset[ds_name] = metrics.cosine

        plot_dataset_group_heatmaps_by_step(
            stepwise_by_dataset,
            output_dir=str(output_dir),
            metric_type="cosine",
            figure_format=fmt,
            cmap=config.cmap_heatmap,
            step_range=config.step_range,
            show_std=config.show_std,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

        if config.plot_svcca:
            stepwise_by_dataset = {ds: m.svcca for ds, m in metrics_by_dataset.items()}
            plot_dataset_group_heatmaps_by_step(
                stepwise_by_dataset,
                output_dir=str(output_dir),
                metric_type="svcca",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
                step_range=config.step_range,
                show_std=config.show_std,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

        if config.plot_conflict:
            stepwise_by_dataset = {ds: m.conflict_freq for ds, m in metrics_by_dataset.items()}
            plot_dataset_group_heatmaps_by_step(
                stepwise_by_dataset,
                output_dir=str(output_dir),
                metric_type="conflict_freq",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
                step_range=config.step_range,
                show_std=config.show_std,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

            stepwise_by_dataset = {ds: m.conflict_cos for ds, m in metrics_by_dataset.items()}
            plot_dataset_group_heatmaps_by_step(
                stepwise_by_dataset,
                output_dir=str(output_dir),
                metric_type="conflict_cos",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
                step_range=config.step_range,
                show_std=config.show_std,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

        if config.plot_subspace:
            for rank in analysis_config.subspace_ranks:
                stepwise_by_dataset = {
                    ds: m.subspace_affinity.get(rank, {}) for ds, m in metrics_by_dataset.items()
                }
                plot_dataset_group_heatmaps_by_step(
                    stepwise_by_dataset,
                    output_dir=str(output_dir),
                    metric_type=f"subspace_r{rank}",
                    figure_format=fmt,
                    cmap=config.cmap_heatmap,
                    step_range=config.step_range,
                    show_std=config.show_std,
                    skip_empty_groups=config.skip_empty_groups,
                    ignored_groups=ignored_groups,
                )

        # Step dynamics (cosine)
        for ds_name, metrics in metrics_by_dataset.items():
            out_path = output_dir / f"step_dynamics_cosine_{ds_name}.{fmt}"
            plot_step_dynamics(
                metrics.cosine,
                metrics.conditions,
                str(out_path),
                ema_alpha=config.step_dynamics_ema_alpha,
            )

    # --- Feature heatmaps (CKA/RSA) ---
    if config.plot_cka and feature_by_dataset:
        for ds_name, feat_metrics in feature_by_dataset.items():
            feat_out = output_dir / f"features_{ds_name}"
            feat_out.mkdir(parents=True, exist_ok=True)
            plot_feature_heatmaps(
                feat_metrics.cka,
                feat_metrics.conditions,
                str(feat_out),
                "cka",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
            )
            plot_feature_heatmaps(
                feat_metrics.rsa,
                feat_metrics.conditions,
                str(feat_out),
                "rsa",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
            )

            if config.plot_per_step:
                plot_feature_heatmaps_by_step(
                    feat_metrics.cka,
                    feat_metrics.conditions,
                    str(feat_out),
                    "cka",
                    figure_format=fmt,
                    step_range=config.step_range,
                    cmap=config.cmap_heatmap,
                )

                plot_layer_step_alignment(
                    feat_metrics.cka,
                    feat_metrics.conditions,
                    output_path=str(feat_out / f"cka_layer_step.{fmt}"),
                    metric_name="cka",
                    step_range=config.step_range,
                    cmap=config.cmap_heatmap,
                )
                plot_layer_step_alignment(
                    feat_metrics.rsa,
                    feat_metrics.conditions,
                    output_path=str(feat_out / f"rsa_layer_step.{fmt}"),
                    metric_name="rsa",
                    step_range=config.step_range,
                    cmap=config.cmap_heatmap,
                )
                plot_feature_heatmaps_by_step(
                    feat_metrics.rsa,
                    feat_metrics.conditions,
                    str(feat_out),
                    "rsa",
                    figure_format=fmt,
                    step_range=config.step_range,
                    cmap=config.cmap_heatmap,
                )

    elapsed = time.perf_counter() - start_time
    logger.info(f"[Viz] Two-condition done in {elapsed:.2f}s: {data_dir}")


def visualize_multi_condition_paradigm(
    data_dir: str,
    output_dir: str,
    config: VisConfig,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    logger.info(f"[Viz] Multi-condition start: {data_dir}")

    analysis_config = load_analysis_config(data_dir)
    if config.svcca_sample_rate is not None:
        analysis_config.svcca_sample_rate = config.svcca_sample_rate
    if config.subspace_sample_rate is not None:
        analysis_config.subspace_sample_rate = config.subspace_sample_rate
    if config.sample_trials is not None:
        analysis_config.sample_trials = config.sample_trials
    if config.max_subsample is not None:
        analysis_config.max_subsample = config.max_subsample
    set_dataset_name_mapping(config.dataset_name_mapping or _DATASET_NAME_MAPPING)
    ignored_groups = _resolve_ignored_groups(config, analysis_config)

    if not has_tensor_data(data_dir):
        logger.warning("No tensor data found.")
        return

    grad_metrics, feat_metrics = load_metrics_from_dir(
        data_dir,
        analysis_config,
        ignored_groups=ignored_groups,
    )

    fmt = config.figure_format

    if config.plot_cosine:
        plot_multi_condition_heatmaps(
            grad_metrics.cosine,
            grad_metrics.conditions,
            output_path=str(output_dir / f"cosine.{fmt}"),
            cmap=config.cmap_heatmap,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

    if config.plot_conflict:
        plot_multi_condition_heatmaps(
            grad_metrics.conflict_freq,
            grad_metrics.conditions,
            output_path=str(output_dir / f"conflict_freq.{fmt}"),
            cmap=config.cmap_heatmap,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )
        plot_multi_condition_heatmaps(
            grad_metrics.conflict_cos,
            grad_metrics.conditions,
            output_path=str(output_dir / f"conflict_cos.{fmt}"),
            cmap=config.cmap_heatmap,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

    if config.plot_svcca:
        plot_multi_condition_heatmaps(
            grad_metrics.svcca,
            grad_metrics.conditions,
            output_path=str(output_dir / f"svcca.{fmt}"),
            cmap=config.cmap_heatmap,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

    if config.plot_per_step:
        plot_multi_condition_heatmaps_by_step(
            grad_metrics.cosine,
            grad_metrics.conditions,
            output_dir=str(output_dir),
            metric_name="cosine",
            figure_format=fmt,
            cmap=config.cmap_heatmap,
            step_range=config.step_range,
            skip_empty_groups=config.skip_empty_groups,
            ignored_groups=ignored_groups,
        )

        if config.plot_svcca:
            plot_multi_condition_heatmaps_by_step(
                grad_metrics.svcca,
                grad_metrics.conditions,
                output_dir=str(output_dir),
                metric_name="svcca",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
                step_range=config.step_range,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

        if config.plot_conflict:
            plot_multi_condition_heatmaps_by_step(
                grad_metrics.conflict_freq,
                grad_metrics.conditions,
                output_dir=str(output_dir),
                metric_name="conflict_freq",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
                step_range=config.step_range,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )
            plot_multi_condition_heatmaps_by_step(
                grad_metrics.conflict_cos,
                grad_metrics.conditions,
                output_dir=str(output_dir),
                metric_name="conflict_cos",
                figure_format=fmt,
                cmap=config.cmap_heatmap,
                step_range=config.step_range,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

        if config.plot_subspace:
            for rank, per_group in grad_metrics.subspace_affinity.items():
                plot_multi_condition_heatmaps_by_step(
                    per_group,
                    grad_metrics.conditions,
                    output_dir=str(output_dir),
                    metric_name=f"subspace_r{rank}",
                    figure_format=fmt,
                    cmap=config.cmap_heatmap,
                    step_range=config.step_range,
                    skip_empty_groups=config.skip_empty_groups,
                    ignored_groups=ignored_groups,
                )

    if config.plot_subspace:
        for rank, per_group in grad_metrics.subspace_affinity.items():
            plot_multi_condition_heatmaps(
                per_group,
                grad_metrics.conditions,
                output_path=str(output_dir / f"subspace_r{rank}.{fmt}"),
                cmap=config.cmap_heatmap,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

    if config.plot_energy:
        steps_e, energy_step, energy_std_macro, groups_e, conditions_e = compute_energy_flow_by_step(
            data_dir,
            ignored_groups=ignored_groups,
        )
        if steps_e:
            energy_mean_macro = np.mean(energy_step, axis=0)
            plot_energy_flow(
                energy_mean_macro,
                energy_std_macro,
                conditions_e,
                groups_e,
                output_path=str(output_dir / f"energy_flow.{fmt}"),
                cmap=config.cmap_energy,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored_groups,
            )

    if config.plot_energy and config.plot_per_step:
        steps, energy_step, energy_std, groups, conditions = compute_energy_flow_by_step(
            data_dir,
            ignored_groups=ignored_groups,
        )
        if steps:
            out_path = output_dir / f"energy_flow_by_step.{fmt}"
            plot_energy_flow_by_step(steps, energy_step, energy_std, conditions, groups, str(out_path))

    if config.plot_per_step:
        out_path = output_dir / f"step_dynamics_cosine.{fmt}"
        plot_step_dynamics(
            grad_metrics.cosine,
            grad_metrics.conditions,
            str(out_path),
            ema_alpha=config.step_dynamics_ema_alpha,
        )

    # Insight plots (most meaningful when conditions >= 3)
    if len(grad_metrics.conditions) >= 3:
        ignored = set(ignored_groups)
        if config.plot_delta:
            plot_delta_heatmaps(
                grad_metrics.cosine,
                grad_metrics.conditions,
                output_path=str(output_dir / f"delta_cosine.{fmt}"),
                window=config.delta_window,
                cmap=config.cmap_heatmap,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored,
            )
            if config.plot_conflict:
                plot_delta_heatmaps(
                    grad_metrics.conflict_freq,
                    grad_metrics.conditions,
                    output_path=str(output_dir / f"delta_conflict_freq.{fmt}"),
                    window=config.delta_window,
                    cmap=config.cmap_heatmap,
                    skip_empty_groups=config.skip_empty_groups,
                    ignored_groups=ignored,
                )
            if config.plot_svcca:
                plot_delta_heatmaps(
                    grad_metrics.svcca,
                    grad_metrics.conditions,
                    output_path=str(output_dir / f"delta_svcca.{fmt}"),
                    window=config.delta_window,
                    cmap=config.cmap_heatmap,
                    skip_empty_groups=config.skip_empty_groups,
                    ignored_groups=ignored,
                )
            if config.plot_subspace:
                for rank, per_group in grad_metrics.subspace_affinity.items():
                    plot_delta_heatmaps(
                        per_group,
                        grad_metrics.conditions,
                        output_path=str(output_dir / f"delta_subspace_r{rank}.{fmt}"),
                        window=config.delta_window,
                        cmap=config.cmap_heatmap,
                        skip_empty_groups=config.skip_empty_groups,
                        ignored_groups=ignored,
                    )
        if config.plot_offdiag_distribution:
            plot_offdiag_distribution(
                grad_metrics.cosine,
                grad_metrics.conditions,
                output_path=str(output_dir / f"offdiag_distribution_cosine.{fmt}"),
                window=config.delta_window,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored,
            )
            if config.plot_conflict:
                plot_offdiag_distribution(
                    grad_metrics.conflict_freq,
                    grad_metrics.conditions,
                    output_path=str(output_dir / f"offdiag_distribution_conflict_freq.{fmt}"),
                    window=config.delta_window,
                    skip_empty_groups=config.skip_empty_groups,
                    ignored_groups=ignored,
                )
        if config.plot_topk_pairs:
            plot_topk_pair_trajectories(
                grad_metrics.cosine,
                grad_metrics.conditions,
                output_dir=str(output_dir / "topk_pairs_cosine"),
                metric_name="cosine",
                topk=config.topk_pairs,
                window=config.delta_window,
                ema_alpha=config.step_dynamics_ema_alpha,
                skip_empty_groups=config.skip_empty_groups,
                ignored_groups=ignored,
            )
        if config.plot_svd_diagnostics:
            tensor_dir = resolve_tensor_dir(data_dir)
            if tensor_dir is not None:
                plot_svd_diagnostics_gradients(
                    tensor_dir,
                    output_dir=str(output_dir / "diagnostics"),
                    step_range=config.step_range,
                    max_samples_per_step=config.diagnostics_max_samples_per_step,
                    num_steps=config.diagnostics_num_steps,
                )

    if config.plot_cka and feat_metrics is not None:
        feat_out = output_dir / "features"
        feat_out.mkdir(parents=True, exist_ok=True)
        plot_feature_heatmaps(
            feat_metrics.cka,
            feat_metrics.conditions,
            str(feat_out),
            "cka",
            figure_format=fmt,
            cmap=config.cmap_heatmap,
        )
        plot_feature_heatmaps(
            feat_metrics.rsa,
            feat_metrics.conditions,
            str(feat_out),
            "rsa",
            figure_format=fmt,
            cmap=config.cmap_heatmap,
        )

        if config.plot_per_step:
            plot_feature_heatmaps_by_step(
                feat_metrics.cka,
                feat_metrics.conditions,
                str(feat_out),
                "cka",
                figure_format=fmt,
                step_range=config.step_range,
                cmap=config.cmap_heatmap,
            )
            plot_feature_heatmaps_by_step(
                feat_metrics.rsa,
                feat_metrics.conditions,
                str(feat_out),
                "rsa",
                figure_format=fmt,
                step_range=config.step_range,
                cmap=config.cmap_heatmap,
            )

            plot_layer_step_alignment(
                feat_metrics.cka,
                feat_metrics.conditions,
                output_path=str(feat_out / f"cka_layer_step.{fmt}"),
                metric_name="cka",
                step_range=config.step_range,
                cmap=config.cmap_heatmap,
            )
            plot_layer_step_alignment(
                feat_metrics.rsa,
                feat_metrics.conditions,
                output_path=str(feat_out / f"rsa_layer_step.{fmt}"),
                metric_name="rsa",
                step_range=config.step_range,
                cmap=config.cmap_heatmap,
            )

    elapsed = time.perf_counter() - start_time
    logger.info(f"[Viz] Multi-condition done in {elapsed:.2f}s: {data_dir}")
