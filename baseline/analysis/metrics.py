"""Torch-native metric computation module for gradient and feature analysis.

This module replaces numpy-based implementations with torch-only computation
and enforces a standardized step-wise aggregation pipeline:
1) Group vectors by step
2) Compute sample-wise metrics with torch
3) Aggregate per-step mean/std
4) Aggregate macro mean/std across steps

Outputs are converted to numpy arrays for downstream visualization only.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import logging
import time

import numpy as np
import torch

from baseline.analysis.collector import TensorReader


logger = logging.getLogger("metrics")


# =============================================================================
# Helpers
# =============================================================================


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(arr: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=torch.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return v / norm


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PairwiseMetricSeries:
    """Pairwise metric statistics across steps.

    Shapes:
        step_mean/step_std: [S, C, C]
        macro_mean/macro_std: [C, C]
    """

    steps: List[int] = field(default_factory=list)
    step_mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    step_std: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    macro_mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    macro_std: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class GradientMetrics:
    groups: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    cosine: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)
    svcca: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)
    conflict_freq: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)
    conflict_cos: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)
    conflict_angle: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)
    subspace_affinity: Dict[int, Dict[str, PairwiseMetricSeries]] = field(default_factory=dict)
    energy_flow: Optional[np.ndarray] = None


@dataclass
class FeatureMetrics:
    layers: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)

    cka: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)
    rsa: Dict[str, PairwiseMetricSeries] = field(default_factory=dict)


# =============================================================================
# Step grouping
# =============================================================================


def group_by_step(vectors: torch.Tensor, steps: torch.Tensor) -> Dict[int, torch.Tensor]:
    if vectors.numel() == 0 or steps.numel() == 0:
        return {}
    steps = steps.to(dtype=torch.int64)
    unique_steps = torch.unique(steps).tolist()
    grouped: Dict[int, torch.Tensor] = {}
    for step in unique_steps:
        mask = steps == step
        grouped[int(step)] = vectors[mask]
    return grouped


def _common_steps(step_groups: Dict[str, Dict[int, torch.Tensor]]) -> List[int]:
    if not step_groups:
        return []
    step_sets = [set(g.keys()) for g in step_groups.values()]
    if not step_sets:
        return []
    common = set.intersection(*step_sets)
    return sorted(common)


def _align_step_tensors(
    step_groups: Dict[str, Dict[int, torch.Tensor]],
    conditions: List[str],
    steps: List[int],
) -> Tuple[List[int], List[torch.Tensor]]:
    aligned_steps: List[int] = []
    aligned_tensors: List[torch.Tensor] = []

    for step in steps:
        tensors = [step_groups[c].get(step) for c in conditions]
        if any(t is None or t.numel() == 0 for t in tensors):
            continue
        min_n = min(t.shape[0] for t in tensors)
        if min_n <= 0:
            continue
        stacked = torch.stack([t[:min_n] for t in tensors], dim=0)  # [C, N, D]
        aligned_steps.append(step)
        aligned_tensors.append(stacked)

    return aligned_steps, aligned_tensors


# =============================================================================
# Core Metric Computations (Torch)
# =============================================================================


def compute_cosine_samples(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute sample-wise cosine matrices.

    Args:
        v: [C, N, D]
        eps: Small constant for numerical stability

    Returns:
        cos_samples: [N, C, C]
    """
    v_norm = _normalize(v, eps=eps)
    cos_samples = torch.einsum("cnd,knd->nck", v_norm, v_norm)
    return cos_samples


def compute_conflict_from_cos(
    cos_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute conflict statistics from sample-wise cosine.

    Returns:
        freq_mean, freq_std, cos_mean, cos_std, angle_mean, angle_std (all [C, C])
    """
    cos_mean = cos_samples.mean(dim=0)
    cos_std = cos_samples.std(dim=0, unbiased=False)

    conflict_mask = cos_samples < 0.0
    freq = conflict_mask.float()
    freq_mean = freq.mean(dim=0)
    freq_std = freq.std(dim=0, unbiased=False)

    angles = torch.arccos(cos_samples.clamp(-1.0, 1.0)) * (180.0 / torch.pi)
    angle_sum = torch.where(conflict_mask, angles, torch.zeros_like(angles))
    angle_count = conflict_mask.sum(dim=0).clamp_min(1)
    angle_mean = angle_sum.sum(dim=0) / angle_count

    angle_var = torch.where(
        conflict_mask,
        (angles - angle_mean) ** 2,
        torch.zeros_like(angles),
    ).sum(dim=0) / angle_count
    angle_std = torch.sqrt(angle_var)

    return freq_mean, freq_std, cos_mean, cos_std, angle_mean, angle_std


def _svd_reduce(
    x: torch.Tensor,
    n_components: int,
    variance_threshold: float,
) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = var.sum().clamp_min(1e-12)
    cumsum = torch.cumsum(var, dim=0) / total
    k_var = int(torch.searchsorted(cumsum, variance_threshold).item()) + 1
    k = max(1, min(n_components, k_var if k_var > 0 else len(s)))
    basis = vh[:k].T
    return x @ basis


def _invsqrt_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(eps)
    invsqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    return invsqrt


def compute_svcca_pair(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n_components: int,
    variance_threshold: float,
) -> float:
    n = min(x1.shape[0], x2.shape[0])
    if n < 2:
        return float("nan")
    x1 = x1[:n]
    x2 = x2[:n]

    z1 = _svd_reduce(x1, n_components, variance_threshold)
    z2 = _svd_reduce(x2, n_components, variance_threshold)
    if z1.numel() == 0 or z2.numel() == 0:
        return float("nan")

    z1 = z1 - z1.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)

    denom = max(1, n - 1)
    cxx = (z1.T @ z1) / denom
    cyy = (z2.T @ z2) / denom
    cxy = (z1.T @ z2) / denom

    cxx = cxx + 1e-6 * torch.eye(cxx.shape[0], device=cxx.device)
    cyy = cyy + 1e-6 * torch.eye(cyy.shape[0], device=cyy.device)

    wx = _invsqrt_psd(cxx)
    wy = _invsqrt_psd(cyy)
    t = wx @ cxy @ wy

    s = torch.linalg.svdvals(t)
    if s.numel() == 0:
        return float("nan")
    return float(torch.clamp(s.mean(), 0.0, 1.0).item())


def compute_subspace_affinity_pair(
    x1: torch.Tensor,
    x2: torch.Tensor,
    rank: int,
) -> float:
    n = min(x1.shape[0], x2.shape[0])
    if n < 2:
        return float("nan")
    x1 = x1[:n] - x1[:n].mean(dim=0, keepdim=True)
    x2 = x2[:n] - x2[:n].mean(dim=0, keepdim=True)

    _, _, vh1 = torch.linalg.svd(x1, full_matrices=False)
    _, _, vh2 = torch.linalg.svd(x2, full_matrices=False)

    k = max(1, min(rank, vh1.shape[0], vh2.shape[0]))
    b1 = vh1[:k].T
    b2 = vh2[:k].T

    affinity = torch.linalg.norm(b1.T @ b2, ord="fro") ** 2 / k
    return float(torch.clamp(affinity, 0.0, 1.0).item())


def compute_cka_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = "linear",
) -> float:
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return float("nan")
    x = x[:n]
    y = y[:n]

    if kernel == "linear":
        kx = x @ x.T
        ky = y @ y.T
    else:
        dxx = torch.cdist(x, x, p=2) ** 2
        dyy = torch.cdist(y, y, p=2) ** 2
        sigma2 = torch.median(dxx).clamp_min(1e-12)
        kx = torch.exp(-dxx / (2 * sigma2))
        ky = torch.exp(-dyy / (2 * sigma2))

    one_n = torch.ones(n, n, device=x.device) / n
    kx_c = kx - one_n @ kx - kx @ one_n + one_n @ kx @ one_n
    ky_c = ky - one_n @ ky - ky @ one_n + one_n @ ky @ one_n

    hsic_xy = (kx_c * ky_c).sum() / ((n - 1) ** 2)
    hsic_xx = (kx_c * kx_c).sum() / ((n - 1) ** 2)
    hsic_yy = (ky_c * ky_c).sum() / ((n - 1) ** 2)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return float("nan")
    return float(torch.clamp(hsic_xy / denom, 0.0, 1.0).item())


def _rank_data(x: torch.Tensor) -> torch.Tensor:
    """Compute rank data for Spearman correlation (ties handled by average rank)."""
    n = x.numel()
    if n == 0:
        return torch.empty((0,), device=x.device, dtype=torch.float32)

    sorted_vals, sorted_idx = torch.sort(x)
    ranks = torch.empty(n, device=x.device, dtype=torch.float32)

    _, counts = torch.unique_consecutive(sorted_vals, return_counts=True)
    counts_f = counts.to(dtype=torch.float32)
    ends = torch.cumsum(counts_f, dim=0) - 1.0
    starts = ends - counts_f + 1.0
    avg_ranks = (starts + ends) / 2.0
    expanded = torch.repeat_interleave(avg_ranks, counts)
    ranks[sorted_idx] = expanded
    return ranks


def compute_rsa_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: str = "correlation",
    comparison: str = "spearman",
) -> float:
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return float("nan")
    x = x[:n]
    y = y[:n]

    if metric == "correlation":
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        x = _normalize(x)
        y = _normalize(y)
        rdm_x = 1 - (x @ x.T)
        rdm_y = 1 - (y @ y.T)
    else:
        rdm_x = torch.cdist(x, x, p=2)
        rdm_y = torch.cdist(y, y, p=2)

    triu = torch.triu_indices(n, n, offset=1, device=x.device)
    vec_x = rdm_x[triu[0], triu[1]]
    vec_y = rdm_y[triu[0], triu[1]]

    if comparison == "spearman":
        vec_x = _rank_data(vec_x)
        vec_y = _rank_data(vec_y)

    vx = vec_x - vec_x.mean()
    vy = vec_y - vec_y.mean()
    denom = torch.linalg.norm(vx) * torch.linalg.norm(vy) + 1e-8
    return float((vx @ vy / denom).item())


# =============================================================================
# Aggregation utilities
# =============================================================================


def _finalize_pairwise_series(
    steps: List[int],
    step_means: List[torch.Tensor],
    step_stds: List[torch.Tensor],
) -> PairwiseMetricSeries:
    if not steps:
        return PairwiseMetricSeries()

    step_mean = torch.stack(step_means, dim=0)  # [S, C, C]
    step_std = torch.stack(step_stds, dim=0)

    macro_mean = step_mean.mean(dim=0)
    macro_std = step_mean.std(dim=0, unbiased=False)

    return PairwiseMetricSeries(
        steps=steps,
        step_mean=to_numpy(step_mean),
        step_std=to_numpy(step_std),
        macro_mean=to_numpy(macro_mean),
        macro_std=to_numpy(macro_std),
    )


def _get_subsample_indices(
    n: int,
    sample_rate: float,
    max_samples: Optional[int] = None,
    min_samples: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    k = int(n * sample_rate)
    if max_samples is not None:
        k = min(k, max_samples)
    k = max(min_samples, k)
    k = min(k, n)
    return torch.randperm(n, device=device)[:k]


# =============================================================================
# High-Level API
# =============================================================================


def compute_all_gradient_metrics(
    reader: TensorReader,
    subspace_ranks: List[int],
    svcca_components: int = 10,
    svcca_threshold: float = 0.99,
    svcca_sample_rate: float = 1.0,
    subspace_sample_rate: float = 1.0,
    sample_trials: int = 5,
    max_subsample: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> GradientMetrics:
    device = get_device(device)

    groups = list(reader.groups)
    conditions = list(reader.conditions)
    metrics = GradientMetrics(groups=groups, conditions=conditions)

    start_total = time.perf_counter()
    logger.info(
        "[Grad] Start metrics: groups=%d, conditions=%d, svcca_sr=%.2f, subspace_sr=%.2f, trials=%d",
        len(groups),
        len(conditions),
        svcca_sample_rate,
        subspace_sample_rate,
        sample_trials,
    )

    energy_matrix = torch.zeros((len(conditions), len(groups)), device=device)

    for g_idx, group in enumerate(groups):
        group_start = time.perf_counter()
        logger.info("[Grad] Group %d/%d: %s", g_idx + 1, len(groups), group)
        # Load per-condition data
        data_by_cond: Dict[str, torch.Tensor] = {}
        steps_by_cond: Dict[str, torch.Tensor] = {}
        norms_by_cond: Dict[str, torch.Tensor] = {}

        for c_idx, cond in enumerate(conditions):
            vectors_np, norms_np, steps_np = reader.load_gradients(group, cond)
            if vectors_np is None or steps_np is None:
                data_by_cond[cond] = torch.empty((0, reader.grad_projection_dim), device=device)
                steps_by_cond[cond] = torch.empty((0,), device=device, dtype=torch.int64)
                norms_by_cond[cond] = torch.empty((0,), device=device)
                continue

            vectors = to_tensor(vectors_np, device=device)
            steps = to_tensor(steps_np, device=device).to(torch.int64)
            norms = to_tensor(norms_np, device=device) if norms_np is not None else torch.empty((0,), device=device)

            data_by_cond[cond] = vectors
            steps_by_cond[cond] = steps
            norms_by_cond[cond] = norms

            if norms.numel() > 0:
                energy_matrix[c_idx, g_idx] = norms.mean()

        # Group by step
        step_groups = {c: group_by_step(data_by_cond[c], steps_by_cond[c]) for c in conditions}
        common_steps = _common_steps(step_groups)
        aligned_steps, aligned_tensors = _align_step_tensors(step_groups, conditions, common_steps)
        if not aligned_steps:
            logger.warning("[Grad] Group %s: no aligned steps", group)
            continue

        metrics.steps = aligned_steps

        sample_count = int(aligned_tensors[0].shape[1]) if aligned_tensors else 0
        logger.info("[Grad] Group %s: aligned_steps=%d, samples/step=%d", group, len(aligned_steps), sample_count)

        # Per-step metrics
        cosine_step_means = []
        cosine_step_stds = []
        conflict_freq_means = []
        conflict_freq_stds = []
        conflict_cos_means = []
        conflict_cos_stds = []
        conflict_angle_means = []
        conflict_angle_stds = []
        svcca_step_means = []
        svcca_step_stds = []

        subspace_step_means: Dict[int, List[torch.Tensor]] = {r: [] for r in subspace_ranks}
        subspace_step_stds: Dict[int, List[torch.Tensor]] = {r: [] for r in subspace_ranks}

        for step_idx, v_step in enumerate(aligned_tensors):
            if step_idx % 10 == 0 or step_idx == len(aligned_tensors) - 1:
                logger.info(
                    "[Grad] Group %s: step %d/%d (global_step=%s)",
                    group,
                    step_idx + 1,
                    len(aligned_tensors),
                    aligned_steps[step_idx],
                )
            # Cosine (sample-wise)
            cos_samples = compute_cosine_samples(v_step)
            cos_mean = cos_samples.mean(dim=0)
            cos_std = cos_samples.std(dim=0, unbiased=False)
            cosine_step_means.append(cos_mean)
            cosine_step_stds.append(cos_std)

            # Conflict stats
            freq_mean, freq_std, cos_mean2, cos_std2, angle_mean, angle_std = compute_conflict_from_cos(cos_samples)
            conflict_freq_means.append(freq_mean)
            conflict_freq_stds.append(freq_std)
            conflict_cos_means.append(cos_mean2)
            conflict_cos_stds.append(cos_std2)
            conflict_angle_means.append(angle_mean)
            conflict_angle_stds.append(angle_std)

            # SVCCA (pairwise) with subsampling
            c = len(conditions)
            n_samples = v_step.shape[1]
            svcca_trials = []
            svcca_n_trials = max(1, sample_trials) if svcca_sample_rate < 1.0 or max_subsample else 1

            for _ in range(svcca_n_trials):
                idx = _get_subsample_indices(
                    n_samples,
                    sample_rate=svcca_sample_rate,
                    max_samples=max_subsample,
                    device=device,
                )
                v_sub = v_step[:, idx, :]
                svcca_mat = torch.zeros((c, c), device=device)
                for i in range(c):
                    svcca_mat[i, i] = 1.0
                    for j in range(i + 1, c):
                        val = compute_svcca_pair(v_sub[i], v_sub[j], svcca_components, svcca_threshold)
                        svcca_mat[i, j] = val
                        svcca_mat[j, i] = val
                svcca_trials.append(svcca_mat)

            svcca_stack = torch.stack(svcca_trials, dim=0)
            svcca_step_means.append(svcca_stack.mean(dim=0))
            svcca_step_stds.append(svcca_stack.std(dim=0, unbiased=False))

            # Subspace affinity per rank with subsampling
            subspace_n_trials = max(1, sample_trials) if subspace_sample_rate < 1.0 or max_subsample else 1
            for rank in subspace_ranks:
                trial_mats = []
                for _ in range(subspace_n_trials):
                    idx = _get_subsample_indices(
                        n_samples,
                        sample_rate=subspace_sample_rate,
                        max_samples=max_subsample,
                        device=device,
                    )
                    v_sub = v_step[:, idx, :]
                    sub_mat = torch.zeros((c, c), device=device)
                    for i in range(c):
                        sub_mat[i, i] = 1.0
                        for j in range(i + 1, c):
                            val = compute_subspace_affinity_pair(v_sub[i], v_sub[j], rank)
                            sub_mat[i, j] = val
                            sub_mat[j, i] = val
                    trial_mats.append(sub_mat)

                sub_stack = torch.stack(trial_mats, dim=0)
                subspace_step_means[rank].append(sub_stack.mean(dim=0))
                subspace_step_stds[rank].append(sub_stack.std(dim=0, unbiased=False))

        metrics.cosine[group] = _finalize_pairwise_series(aligned_steps, cosine_step_means, cosine_step_stds)
        metrics.conflict_freq[group] = _finalize_pairwise_series(aligned_steps, conflict_freq_means, conflict_freq_stds)
        metrics.conflict_cos[group] = _finalize_pairwise_series(aligned_steps, conflict_cos_means, conflict_cos_stds)
        metrics.conflict_angle[group] = _finalize_pairwise_series(aligned_steps, conflict_angle_means, conflict_angle_stds)
        metrics.svcca[group] = _finalize_pairwise_series(aligned_steps, svcca_step_means, svcca_step_stds)

        for rank in subspace_ranks:
            if rank not in metrics.subspace_affinity:
                metrics.subspace_affinity[rank] = {}
            metrics.subspace_affinity[rank][group] = _finalize_pairwise_series(
                aligned_steps, subspace_step_means[rank], subspace_step_stds[rank]
            )

        group_elapsed = time.perf_counter() - group_start
        logger.info("[Grad] Group %s done in %.2fs", group, group_elapsed)

    # Energy flow normalization
    if energy_matrix.numel() > 0:
        energy_sum = energy_matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
        energy_flow = energy_matrix / energy_sum
        metrics.energy_flow = to_numpy(energy_flow)

    total_elapsed = time.perf_counter() - start_total
    logger.info("[Grad] Metrics complete in %.2fs", total_elapsed)
    return metrics


def compute_all_feature_metrics(
    reader: TensorReader,
    cka_kernel: str = "linear",
    rsa_metric: str = "correlation",
    rsa_comparison: str = "spearman",
    device: Optional[Union[str, torch.device]] = None,
) -> FeatureMetrics:
    device = get_device(device)
    layers = [l for l in list(reader.layers) if "BrainEmbedEEGLayer" not in l]
    conditions = list(reader.conditions)

    metrics = FeatureMetrics(layers=layers, conditions=conditions)

    if len(layers) != len(reader.layers):
        logger.info(
            "[Feat] Filtered layers containing BrainEmbedEEGLayer: %d removed",
            len(reader.layers) - len(layers),
        )

    start_total = time.perf_counter()
    logger.info(
        "[Feat] Start metrics: layers=%d, conditions=%d, cka=%s, rsa=%s/%s",
        len(layers),
        len(conditions),
        cka_kernel,
        rsa_metric,
        rsa_comparison,
    )

    for layer_idx, layer in enumerate(layers):
        layer_start = time.perf_counter()
        logger.info("[Feat] Layer %d/%d: %s", layer_idx + 1, len(layers), layer)
        data_by_cond: Dict[str, torch.Tensor] = {}
        steps_by_cond: Dict[str, torch.Tensor] = {}

        for cond in conditions:
            vectors_np, steps_np = reader.load_features(layer, cond)
            if vectors_np is None or steps_np is None:
                data_by_cond[cond] = torch.empty((0, reader.feature_projection_dim), device=device)
                steps_by_cond[cond] = torch.empty((0,), device=device, dtype=torch.int64)
                continue

            vectors = to_tensor(vectors_np, device=device)
            steps = to_tensor(steps_np, device=device).to(torch.int64)
            data_by_cond[cond] = vectors
            steps_by_cond[cond] = steps

        step_groups = {c: group_by_step(data_by_cond[c], steps_by_cond[c]) for c in conditions}
        common_steps = _common_steps(step_groups)
        aligned_steps, aligned_tensors = _align_step_tensors(step_groups, conditions, common_steps)
        if not aligned_steps:
            logger.warning("[Feat] Layer %s: no aligned steps", layer)
            continue

        metrics.steps = aligned_steps

        cka_step_means = []
        cka_step_stds = []
        rsa_step_means = []
        rsa_step_stds = []

        for step_idx, v_step in enumerate(aligned_tensors):
            if step_idx % 10 == 0 or step_idx == len(aligned_tensors) - 1:
                logger.info(
                    "[Feat] Layer %s: step %d/%d (global_step=%s)",
                    layer,
                    step_idx + 1,
                    len(aligned_tensors),
                    aligned_steps[step_idx],
                )
            c = len(conditions)
            cka_mat = torch.zeros((c, c), device=device)
            rsa_mat = torch.zeros((c, c), device=device)
            for i in range(c):
                cka_mat[i, i] = 1.0
                rsa_mat[i, i] = 1.0
                for j in range(i + 1, c):
                    cka_val = compute_cka_pair(v_step[i], v_step[j], kernel=cka_kernel)
                    rsa_val = compute_rsa_pair(v_step[i], v_step[j], metric=rsa_metric, comparison=rsa_comparison)
                    cka_mat[i, j] = cka_val
                    cka_mat[j, i] = cka_val
                    rsa_mat[i, j] = rsa_val
                    rsa_mat[j, i] = rsa_val

            cka_step_means.append(cka_mat)
            cka_step_stds.append(torch.zeros_like(cka_mat))
            rsa_step_means.append(rsa_mat)
            rsa_step_stds.append(torch.zeros_like(rsa_mat))

        metrics.cka[layer] = _finalize_pairwise_series(aligned_steps, cka_step_means, cka_step_stds)
        metrics.rsa[layer] = _finalize_pairwise_series(aligned_steps, rsa_step_means, rsa_step_stds)

        layer_elapsed = time.perf_counter() - layer_start
        logger.info("[Feat] Layer %s done in %.2fs", layer, layer_elapsed)

    total_elapsed = time.perf_counter() - start_total
    logger.info("[Feat] Metrics complete in %.2fs", total_elapsed)
    return metrics
