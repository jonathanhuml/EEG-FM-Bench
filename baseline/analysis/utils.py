"""Shared utilities for gradient analysis.

This module centralizes generic helpers that were previously in
`baseline.labram.grad_utils` so that all gradient analysis code
can depend on `analysis.grad` only.
"""


import os
import struct
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any,  Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# -----------------------------
# Seeds & basic utils
# -----------------------------


def set_seeds(seed: int, deterministic: bool = False) -> None:
    import random

    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def ensure_dir(p: str | Path) -> None:
    d = Path(p)
    if d.is_dir():
        return
    d.mkdir(parents=True, exist_ok=True)


def ensure_dir_of(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def strip_all_suffixes(p: Path) -> str:
    name = p.name
    for suf in p.suffixes:
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def make_run_base_dir(
    script_path: str | Path,
    out_dir: Optional[str],
    run_name: Optional[str],
    prefix: str,
) -> Path:
    script_dir = Path(script_path).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(out_dir) if out_dir else (script_dir / "results")
    name = run_name or f"{prefix}_{ts}"
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def round_robin_batches(loaders: Sequence[Iterable[Any]]):
    iters = [iter(ld) for ld in loaders]
    idx = 0
    while True:
        if not iters:
            return
        it = iters[idx % len(iters)]
        try:
            yield next(it)
            idx += 1
        except StopIteration:
            iters.pop(idx % len(iters))
            if not iters:
                return
            idx = idx % len(iters)


# -----------------------------
# Hashing projector & vectors
# -----------------------------


def _deterministic_hash(seed: int, key: str, length: int) -> int:
    """Generate deterministic hash using struct-based mixing.
    
    Uses a combination of the seed, key, and length to produce a
    deterministic integer that's consistent across Python runs
    (unlike Python's built-in hash() which uses PYTHONHASHSEED).
    
    This avoids the collision issues of md5 for numerical data while
    maintaining determinism.
    """
    # Create a bytes representation
    key_bytes = key.encode('utf-8')
    # Pack as: seed (8 bytes), length (8 bytes), key length (4 bytes), key bytes
    data = struct.pack('>qqI', seed, length, len(key_bytes)) + key_bytes
    
    # Use a simple mixing function (FNV-1a style)
    h = 0xcbf29ce484222325  # FNV offset basis
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff  # FNV prime, mask to 64 bits
    
    return h & 0xffffffffffffffff  # Return positive 64-bit integer


class HashingProjector:
    """Count-Sketch style hashing projection with per-(key,L) cached buckets/signs.

    - Works on CPU/GPU tensors
    - Deterministic across runs for the same (seed, key, length)
    - Uses numpy's random generator with deterministic seeding for better distribution
    """

    def __init__(self, proj_dim: int, seed: int, max_cache_entries: int = 256):
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.max_cache_entries = int(max_cache_entries)
        self._cache: "OrderedDict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def _get_hash(self, device: torch.device, key: str, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.proj_dim <= 0:
            raise ValueError("proj_dim must be > 0 to use HashingProjector")

        d = int(self.proj_dim)
        k = (str(key), int(length))
        cached = self._cache.get(k)

        if cached is not None:
            self._cache.move_to_end(k)
            buckets_cpu, signs_cpu = cached
            return buckets_cpu.to(device), signs_cpu.to(device)

        # Use deterministic hash for seed mixing
        mixed_seed = _deterministic_hash(self.seed, key, length)
        
        # Use numpy's random generator for better distribution
        rng = np.random.Generator(np.random.PCG64(mixed_seed))
        
        # Generate bucket assignments and signs
        buckets_np = rng.integers(0, max(1, d), size=length, dtype=np.int64)
        signs_np = rng.choice([-1.0, 1.0], size=length).astype(np.float32)
        
        buckets_cpu = torch.from_numpy(buckets_np)
        signs_cpu = torch.from_numpy(signs_np)
        self._cache[k] = (buckets_cpu, signs_cpu)
        if self.max_cache_entries > 0:
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)

        return buckets_cpu.to(device), signs_cpu.to(device)

    @torch.no_grad()
    def project_and_norm(self, v: torch.Tensor, key: str = "default") -> torch.Tensor:
        """Project a gradient vector to lower dimension and normalize.
        
        Args:
            v: Input gradient vector (any shape, will be flattened)
            key: Cache key for hash mapping (use param group name)
            
        Returns:
            If proj_dim > 0: projected and L2-normalized vector of shape [proj_dim].
            If proj_dim <= 0: L2-normalized original vector (flattened).
        """
        if self.proj_dim <= 0:
            v = v.detach().to(torch.float32).flatten()
            norm = torch.norm(v)
            return v / (norm + 1e-12) if norm > 1e-8 else torch.zeros_like(v)

        v = v.detach().to(torch.float32).flatten()
        device = v.device
        l = int(v.numel())

        if l == 0:
            return torch.zeros(self.proj_dim, dtype=torch.float32, device=device)
        
        # Check for near-zero input
        v_norm = torch.norm(v)
        if v_norm < 1e-8:
            return torch.zeros(self.proj_dim, dtype=torch.float32, device=device)
        
        # Normalize input first to reduce numerical accumulation error
        v_normalized = v / (v_norm + 1e-12)

        buckets, signs = self._get_hash(device, key, l)
        out = torch.zeros(self.proj_dim, dtype=torch.float32, device=device)
        out.scatter_add_(0, buckets, signs * v_normalized)
        
        # Final normalization
        out_norm = torch.norm(out)
        if out_norm < 1e-12:
            return torch.zeros(self.proj_dim, dtype=torch.float32, device=device)
        return out / out_norm


def flatten_from_grads(
    grads: Sequence[Optional[torch.Tensor]],
    sizes: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for g, sz in zip(grads, sizes):
        if g is None:
            vectors.append(torch.zeros(sz, dtype=torch.float32, device=device))
        else:
            vectors.append(g.detach().reshape(-1).to(torch.float32))

    return torch.cat(vectors)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.flatten().to(torch.float32)
    bf = b.flatten().to(torch.float32)
    na = torch.norm(af)
    nb = torch.norm(bf)
    denom = na * nb
    if denom < eps:
        return 0.0
    val = torch.dot(af, bf) / denom
    return float(torch.clamp(val, -1.0, 1.0).item())


def ema_series(values: List[float], beta: float = 0.9) -> List[float]:
    out: List[float] = []
    m: Optional[float] = None
    b = float(beta)
    for v in values:
        m = (b * m + (1 - b) * float(v)) if m is not None else float(v)
        out.append(float(m))
    return out


# -----------------------------
# Plotting utilities
# -----------------------------

def export_group_matrix_csv(
    out_path: str | Path,
    matrices: Dict[str, np.ndarray],
    axis_names: Sequence[str],
    fmt: str = ".6f",
) -> None:
    if not matrices:
        return
    ensure_dir_of(str(out_path))
    labels = [str(n).upper() for n in axis_names]
    try:
        import csv as _csv
    except Exception:
        return

    with open(out_path, "w", newline="") as f:
        writer = _csv.writer(f)
        header = ["axis"] + labels
        for g, mat in matrices.items():
            arr = np.asarray(mat, dtype=np.float32)
            writer.writerow([f"group={g}"])
            writer.writerow(header)
            for i, axis in enumerate(labels):
                row_vals: List[str] = []
                for j in range(len(labels)):
                    if i < arr.shape[0] and j < arr.shape[1]:
                        val = float(arr[i, j])
                        row_vals.append(format(val, fmt) if np.isfinite(val) else "")
                    else:
                        row_vals.append("")
                writer.writerow([axis] + row_vals)
            writer.writerow([])


__all__ = [
    "set_seeds",
    "ensure_dir",
    "strip_all_suffixes",
    "make_run_base_dir",
    "round_robin_batches",
    "HashingProjector",
    "flatten_from_grads",
    "cosine",
    "ema_series",
    "ensure_dir_of",
    "export_group_matrix_csv",
]
