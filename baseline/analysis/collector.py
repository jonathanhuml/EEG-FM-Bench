"""Unified tensor collector for gradient and feature analysis.

This module provides a unified interface for collecting:
1. Gradient vectors during training
2. Intermediate feature activations

Key design principles:
- Streaming to disk: Avoid memory explosion for long runs
- Unified interface: Same API for gradients and features
- Efficient storage: HDF5 with chunked writing
- Thread-safe: Support for multi-process data loading

Data is organized hierarchically:
    /gradients/{group_name}/{condition_name}/
        - vectors: [N, projection_dim] projected gradients
        - raw_norms: [N] original L2 norms
        - steps: [N] training step indices
        
    /features/{layer_name}/{condition_name}/
        - vectors: [N, projection_dim] projected features
        - steps: [N] training step indices
        
    /metadata/
        - config: JSON serialized config
        - groups: list of group names
        - conditions: list of condition names
        - layers: list of layer names
"""

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple, Union

import torch
import numpy as np

from baseline.analysis.utils import HashingProjector
from baseline.analysis.grouper import EncoderParamGrouper


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CollectorConfig:
    """Configuration for TensorCollector (for convenience, not required)."""
    output_path: str = "./collected_data"
    storage_format: str = "hdf5"  # "hdf5" or "npz"
    groups: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    max_memory_samples: int = 500
    projection_dim: int = 1024
    projection_seed: int = 42
    flush_interval: int = 100
    buffer_size: int = 100


@dataclass
class CollectionStats:
    """Statistics for collected data."""
    count: int = 0
    mean_norm: float = 0.0
    std_norm: float = 0.0
    m2: float = 0.0
    min_norm: float = float('inf')
    max_norm: float = float('-inf')
    
    def update(self, norm: float):
        """Update statistics with new norm value."""
        if not np.isfinite(norm):
            return
        self.count += 1
        delta = norm - self.mean_norm
        self.mean_norm += delta / self.count
        # Welford's online variance (track M2)
        delta2 = norm - self.mean_norm
        self.m2 += delta * delta2
        self.std_norm = float(np.sqrt(self.m2 / (self.count - 1))) if self.count > 1 else 0.0
        self.min_norm = min(self.min_norm, norm)
        self.max_norm = max(self.max_norm, norm)


@dataclass
class TensorBuffer:
    """Buffer for accumulating tensors before writing to disk."""
    vectors: Deque[torch.Tensor] = field(default_factory=lambda: deque())
    raw_norms: Deque[float] = field(default_factory=lambda: deque())
    steps: Deque[int] = field(default_factory=lambda: deque())
    stats: CollectionStats = field(default_factory=CollectionStats)
    
    def add(
        self,
        vector: torch.Tensor,
        raw_norm: float,
        step: int,
        track_norms: bool = True,
    ):
        """Add a tensor to the buffer."""
        self.vectors.append(vector.detach().cpu())
        self.raw_norms.append(raw_norm)
        self.steps.append(step)
        if track_norms:
            self.stats.update(raw_norm)
    
    def __len__(self) -> int:
        return len(self.vectors)
    
    def clear(self):
        """Clear the buffer."""
        self.vectors.clear()
        self.raw_norms.clear()
        self.steps.clear()
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert buffer to numpy arrays."""
        if len(self.vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        vectors = torch.stack(list(self.vectors), dim=0).numpy()
        raw_norms = np.array(list(self.raw_norms), dtype=np.float32)
        steps = np.array(list(self.steps), dtype=np.int32)
        return vectors, raw_norms, steps


# =============================================================================
# Unified Tensor Collector
# =============================================================================


class TensorCollector:
    """Unified collector for gradient and feature tensors.
    
    This class handles:
    1. Gradient collection from model parameters
    2. Feature collection from forward hooks
    3. Streaming writes to disk
    4. Memory-efficient sliding window
    
    Usage:
        ```python
        collector = TensorCollector(
            output_path="./analysis/run_001",
            groups=["attention", "ffn", "embed"],
            conditions=["scratch", "pretrained"],
            projection_dim=1024
        )
        
        # Collect gradients
        collector.collect_gradients(
            model=model,
            grouper=grouper,
            condition="scratch",
            step=100
        )
        
        # Collect features
        collector.collect_features(
            features={"layer_0": tensor, "layer_1": tensor},
            condition="scratch", 
            step=100
        )
        
        # Flush to disk periodically
        if step % 100 == 0:
            collector.flush()
        
        # Finalize at end
        collector.close()
        ```
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        groups: List[str],
        conditions: List[str],
        layers: Optional[List[str]] = None,
        projection_dim: int = 1024,
        projection_seed: int = 42,
        feature_projection_dim: Optional[int] = None,
        feature_projection_seed: Optional[int] = None,
        track_raw_norms: bool = True,
        buffer_size: int = 100,
        max_memory_samples: int = 500,
        use_hdf5: bool = True,
    ):
        """Initialize tensor collector.
        
        Args:
            output_path: Directory for output files
            groups: Parameter group names for gradient collection
            conditions: Condition names (datasets or training phases)
            layers: Layer names for feature collection (None = no features)
            projection_dim: Dimension for random projection
            projection_seed: Random seed for projector
            buffer_size: Samples to buffer before writing to disk
            max_memory_samples: Maximum samples to keep in memory (sliding window)
            use_hdf5: Whether to use HDF5 (True) or NPZ (False) format
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.groups = list(groups)
        self.conditions = list(conditions)
        self.layers = list(layers) if layers else []
        
        self.grad_projection_dim = projection_dim
        self.grad_projection_seed = projection_seed
        self.feature_projection_dim = feature_projection_dim or projection_dim
        self.feature_projection_seed = feature_projection_seed or projection_seed
        self.buffer_size = buffer_size
        self.max_memory_samples = max_memory_samples
        self.use_hdf5 = use_hdf5
        self.track_raw_norms = track_raw_norms
        
        # Projectors for dimensionality reduction
        self.grad_projector = HashingProjector(self.grad_projection_dim, self.grad_projection_seed)
        self.feat_projector = HashingProjector(self.feature_projection_dim, self.feature_projection_seed)

        # To Disk
        # Buffers for gradients: {group: {condition: TensorBuffer}}
        self.grad_buffers: Dict[str, Dict[str, TensorBuffer]] = {
            g: {c: TensorBuffer() for c in self.conditions}
            for g in self.groups
        }
        
        # Buffers for features: {layer: {condition: TensorBuffer}}
        self.feat_buffers: Dict[str, Dict[str, TensorBuffer]] = {
            l: {c: TensorBuffer() for c in self.conditions}
            for l in self.layers
        }

        # Real time
        # In-memory data for analysis (sliding window)
        self.grad_data: Dict[str, Dict[str, Deque[torch.Tensor]]] = {
            g: {c: deque(maxlen=max_memory_samples) for c in self.conditions}
            for g in self.groups
        }
        self.grad_norms: Dict[str, Dict[str, Deque[float]]] = {
            g: {c: deque(maxlen=max_memory_samples) for c in self.conditions}
            for g in self.groups
        }
        
        self.feat_data: Dict[str, Dict[str, Deque[torch.Tensor]]] = {
            l: {c: deque(maxlen=max_memory_samples) for c in self.conditions}
            for l in self.layers
        }
        
        # HDF5 file handle
        self._h5file = None
        self._h5lock = threading.Lock()
        
        # Tracking
        self.current_step = 0
        self.total_grad_samples = 0
        self.total_feat_samples = 0
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage files."""
        if self.use_hdf5:
            self._init_hdf5()
        else:
            # NPZ files will be created on first flush
            pass
        
        # Save metadata
        metadata = {
            "groups": self.groups,
            "conditions": self.conditions,
            "layers": self.layers,
            "projection_dim": self.grad_projection_dim,
            "grad_projection_dim": self.grad_projection_dim,
            "grad_projection_seed": self.grad_projection_seed,
            "feature_projection_dim": self.feature_projection_dim,
            "feature_projection_seed": self.feature_projection_seed,
            "track_raw_norms": self.track_raw_norms,
            "created": datetime.now().isoformat(),
        }
        with open(self.output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _init_hdf5(self):
        """Initialize HDF5 file with datasets."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 storage. Install with: pip install h5py")
        
        h5_path = self.output_path / "tensors.h5"
        # Use append mode and replace existing groups to avoid dataset name conflicts
        self._h5file = h5py.File(h5_path, "a")
        if "gradients" in self._h5file:
            del self._h5file["gradients"]
        if "features" in self._h5file:
            del self._h5file["features"]
        
        # Create groups for gradients
        grad_grp = self._h5file.create_group("gradients")
        for g in self.groups:
            g_grp = grad_grp.create_group(g)
            for c in self.conditions:
                c_grp = g_grp.create_group(c)
                # Create resizable datasets
                c_grp.create_dataset(
                    "vectors",
                    shape=(0, self.grad_projection_dim),
                    maxshape=(None, self.grad_projection_dim),
                    dtype=np.float32,
                    chunks=(min(100, self.buffer_size), self.grad_projection_dim),
                )
                c_grp.create_dataset(
                    "raw_norms",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                    chunks=(min(1000, self.buffer_size * 10),),
                )
                c_grp.create_dataset(
                    "steps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int32,
                    chunks=(min(1000, self.buffer_size * 10),),
                )
        
        # Create groups for features
        if self.layers:
            feat_grp = self._h5file.create_group("features")
            for l in self.layers:
                l_grp = feat_grp.create_group(l.replace(".", "_"))  # HDF5 doesn't like dots
                for c in self.conditions:
                    c_grp = l_grp.create_group(c)
                    c_grp.create_dataset(
                        "vectors",
                        shape=(0, self.feature_projection_dim),
                        maxshape=(None, self.feature_projection_dim),
                        dtype=np.float32,
                        chunks=(min(100, self.buffer_size), self.feature_projection_dim),
                    )
                    c_grp.create_dataset(
                        "steps",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=(min(1000, self.buffer_size * 10),),
                    )
    
    def collect_gradients(
        self,
        grouper: EncoderParamGrouper,
        condition: str,
        step: int,
    ):
        """Collect gradients from model parameters.
        
        Call this AFTER loss.backward() but BEFORE optimizer.step().
        
        Args:
            grouper: Parameter grouper for categorization
            condition: Condition name (e.g., "scratch", "pretrained")
            step: Current training step
        """
        if condition not in self.conditions:
            return
        
        self.current_step = step
        
        # Collect gradients by group
        group_grads: Dict[str, List[torch.Tensor]] = {g: [] for g in self.groups}
        
        for name, group, param in grouper.iter_grouped_params():
            if group.name in self.groups and param.grad is not None:
                group_grads[group.name].append(param.grad.detach())
        
        # Flatten and project each group
        for g in self.groups:
            grads = group_grads.get(g, [])
            
            if not grads:
                continue
            
            # Flatten all gradients in group
            flat = torch.cat([grad.flatten() for grad in grads])
            self._add_projected(
                kind="grad",
                name=g,
                condition=condition,
                flat=flat,
                step=step,
            )
        
        # Check if flush needed
        if self._should_flush():
            self.flush()
    
    def collect_features(
        self,
        features: Dict[str, torch.Tensor],
        condition: str,
        step: int,
    ):
        """Collect intermediate features.
        
        Args:
            features: Dict mapping layer_name to feature tensor [batch, ...]
            condition: Condition name
            step: Current training step
        """
        if condition not in self.conditions:
            return
        
        for layer_name, feat in features.items():
            if layer_name not in self.layers:
                continue
            
            # Handle different feature shapes
            feat = feat.detach().cpu()
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            
            # Process each sample in batch
            batch_size = feat.shape[0]
            for i in range(batch_size):
                v = feat[i].flatten().to(torch.float32)
                self._add_projected(
                    kind="feat",
                    name=layer_name,
                    condition=condition,
                    flat=v,
                    step=step,
                )
        
        if self._should_flush():
            self.flush()
    
    def add_gradient(
        self,
        group: str,
        condition: str,
        vector: Union[np.ndarray, torch.Tensor],
        step: int,
    ):
        """Add a single gradient vector (simplified API).
        
        This is a convenience method for adding pre-computed gradient vectors.
        Use this when you've already concatenated and flattened gradients.
        
        Args:
            group: Parameter group name
            condition: Condition name
            vector: Flattened gradient vector (numpy array or torch tensor)
            step: Training step
        """
        if group not in self.groups or condition not in self.conditions:
            return
        
        # Accept both numpy and tensor, avoid unnecessary conversions
        if isinstance(vector, np.ndarray):
            flat = torch.from_numpy(vector).float()
        else:
            flat = vector.float()
        
        self._add_projected(
            kind="grad",
            name=group,
            condition=condition,
            flat=flat,
            step=step,
        )
        
        if self._should_flush():
            self.flush()
    
    def add_feature(
        self,
        layer: str,
        condition: str,
        vector: np.ndarray,
        step: int,
    ):
        """Add a single feature vector (simplified API).
        
        Args:
            layer: Layer name
            condition: Condition name
            vector: Flattened feature vector
            step: Training step
        """
        if layer not in self.layers or condition not in self.conditions:
            return
        
        flat = torch.from_numpy(vector).float()
        self._add_projected(
            kind="feat",
            name=layer,
            condition=condition,
            flat=flat,
            step=step,
        )
        
        if self._should_flush():
            self.flush()
    
    def _should_flush(self) -> bool:
        """Check if buffers should be flushed to disk."""
        for g in self.groups:
            for c in self.conditions:
                if len(self.grad_buffers[g][c]) >= self.buffer_size:
                    return True
        
        for l in self.layers:
            for c in self.conditions:
                if len(self.feat_buffers[l][c]) >= self.buffer_size:
                    return True
        
        return False

    def _add_projected(
        self,
        kind: str,
        name: str,
        condition: str,
        flat: torch.Tensor,
        step: int,
    ):
        if kind == "grad":
            buffers = self.grad_buffers
            data_store = self.grad_data
            norms_store = self.grad_norms
            key = name
        elif kind == "feat":
            buffers = self.feat_buffers
            data_store = self.feat_data
            norms_store = None
            key = name
        else:
            raise ValueError(f"Unknown kind: {kind}")

        raw_norm = float(torch.norm(flat).item()) if self.track_raw_norms else 0.0
        if kind == "grad":
            projected = self.grad_projector.project_and_norm(flat, key=key)
        else:
            projected = self.feat_projector.project_and_norm(flat, key=key)

        buffers[name][condition].add(
            projected,
            raw_norm,
            step,
            track_norms=self.track_raw_norms,
        )
        data_store[name][condition].append(projected)
        if norms_store is not None and self.track_raw_norms:
            # noinspection PyUnresolvedReferences
            norms_store[name][condition].append(raw_norm)

        if kind == "grad":
            self.total_grad_samples += 1
        else:
            self.total_feat_samples += 1
    
    def flush(self):
        """Flush all buffers to disk."""
        if self.use_hdf5:
            self._flush_hdf5()
        else:
            self._flush_npz()
    
    def _flush_hdf5(self):
        """Flush buffers to HDF5 file."""
        if self._h5file is None:
            return
        
        with self._h5lock:
            # Flush gradient buffers
            for g in self.groups:
                for c in self.conditions:
                    buf = self.grad_buffers[g][c]
                    if len(buf) == 0:
                        continue
                    
                    vectors, raw_norms, steps = buf.to_arrays()
                    
                    ds_path = f"gradients/{g}/{c}"
                    vec_ds = self._h5file[f"{ds_path}/vectors"]
                    norm_ds = self._h5file[f"{ds_path}/raw_norms"]
                    step_ds = self._h5file[f"{ds_path}/steps"]
                    
                    # Resize and append
                    old_len = vec_ds.shape[0]
                    new_len = old_len + len(vectors)
                    
                    vec_ds.resize((new_len, self.grad_projection_dim))
                    norm_ds.resize((new_len,))
                    step_ds.resize((new_len,))
                    
                    vec_ds[old_len:new_len] = vectors
                    norm_ds[old_len:new_len] = raw_norms
                    step_ds[old_len:new_len] = steps

                    buf.clear()
            
            # Flush feature buffers
            for l in self.layers:
                l_safe = l.replace(".", "_")
                for c in self.conditions:
                    buf = self.feat_buffers[l][c]
                    if len(buf) == 0:
                        continue
                    
                    vectors, _, steps = buf.to_arrays()
                    
                    ds_path = f"features/{l_safe}/{c}"
                    vec_ds = self._h5file[f"{ds_path}/vectors"]
                    step_ds = self._h5file[f"{ds_path}/steps"]
                    
                    old_len = vec_ds.shape[0]
                    new_len = old_len + len(vectors)
                    
                    vec_ds.resize((new_len, self.feature_projection_dim))
                    step_ds.resize((new_len,))
                    
                    vec_ds[old_len:new_len] = vectors
                    step_ds[old_len:new_len] = steps
                    
                    buf.clear()
            
            self._h5file.flush()
    
    def _flush_npz(self):
        """Flush buffers to NPZ files."""
        grad_dir = self.output_path / "gradients"
        feat_dir = self.output_path / "features"
        
        # Flush gradients
        for g in self.groups:
            for c in self.conditions:
                buf = self.grad_buffers[g][c]
                if len(buf) == 0:
                    continue
                
                vectors, raw_norms, steps = buf.to_arrays()
                
                out_dir = grad_dir / g / c
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Append to existing or create new
                chunk_idx = len(list(out_dir.glob("chunk_*.npz")))
                np.savez_compressed(
                    out_dir / f"chunk_{chunk_idx:04d}.npz",
                    vectors=vectors,
                    raw_norms=raw_norms,
                    steps=steps,
                )
                
                buf.clear()
        
        # Flush features
        for l in self.layers:
            l_safe = l.replace(".", "_")
            for c in self.conditions:
                buf = self.feat_buffers[l][c]
                if len(buf) == 0:
                    continue
                
                vectors, _, steps = buf.to_arrays()
                
                out_dir = feat_dir / l_safe / c
                out_dir.mkdir(parents=True, exist_ok=True)
                
                chunk_idx = len(list(out_dir.glob("chunk_*.npz")))
                np.savez_compressed(
                    out_dir / f"chunk_{chunk_idx:04d}.npz",
                    vectors=vectors,
                    steps=steps,
                )
                
                buf.clear()
    
    def get_gradient_data(
        self,
        group: str,
        condition: str,
    ) -> Optional[torch.Tensor]:
        """Get in-memory gradient data for analysis.
        
        Returns:
            Tensor of shape [N, projection_dim] or None
        """
        data = self.grad_data.get(group, {}).get(condition, None)
        if data is None or len(data) == 0:
            return None
        return torch.stack(list(data), dim=0)
    
    def get_gradient_norms(
        self,
        group: str,
        condition: str,
    ) -> List[float]:
        """Get in-memory gradient norms."""
        norms = self.grad_norms.get(group, {}).get(condition, None)
        if norms is None:
            return []
        return list(norms)
    
    def get_feature_data(
        self,
        layer: str,
        condition: str,
    ) -> Optional[torch.Tensor]:
        """Get in-memory feature data for analysis.
        
        Returns:
            Tensor of shape [N, projection_dim] or None
        """
        data = self.feat_data.get(layer, {}).get(condition, None)
        if data is None or len(data) == 0:
            return None
        return torch.stack(list(data), dim=0)
    
    def get_collection_stats(self) -> Dict[str, Dict[str, CollectionStats]]:
        """Get collection statistics for all groups/conditions."""
        stats = {}
        
        for g in self.groups:
            stats[g] = {}
            for c in self.conditions:
                stats[g][c] = self.grad_buffers[g][c].stats
        
        return stats
    
    def save_metrics(
        self,
        step: int,
        metrics: Dict[str, Any],
    ):
        """Save per-step metrics to JSONL file.
        
        Args:
            step: Training step
            metrics: Dict of metric name -> value
        """
        metrics_path = self.output_path / "metrics.jsonl"
        
        record = {"step": step, **metrics}
        
        # Convert numpy/torch types
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj
        
        record = convert(record)
        
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def close(self):
        """Flush remaining data and close files."""
        self.flush()
        
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
        
        # Save final statistics
        stats = self.get_collection_stats()
        stats_dict = {
            g: {
                c: {
                    "count": s.count,
                    "mean_norm": s.mean_norm,
                    "std_norm": s.std_norm,
                    "min_norm": s.min_norm if s.min_norm != float('inf') else 0,
                    "max_norm": s.max_norm,
                }
                for c, s in cond_stats.items()
            }
            for g, cond_stats in stats.items()
        }
        
        with open(self.output_path / "collection_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
    
    def finalize(self):
        """Alias for close() - flush and close all files."""
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Data Reader (for post-processing)
# =============================================================================


class TensorReader:
    """Reader for collected tensor data.
    
    Usage:
        ```python
        reader = TensorReader("./analysis/run_001")
        
        # Get gradient data
        grads = reader.load_gradients("attention", "scratch")

        metrics = reader.load_metrics()
        ```
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """Initialize reader.
        
        Args:
            data_path: Path to analysis output directory
        """
        self.data_path = Path(data_path)
        
        # Load metadata
        with open(self.data_path / "metadata.json") as f:
            self.metadata = json.load(f)
        
        self.groups = self.metadata["groups"]
        self.conditions = self.metadata["conditions"]
        self.layers = self.metadata.get("layers", [])
        self.grad_projection_dim = self.metadata.get("grad_projection_dim", self.metadata.get("projection_dim"))
        self.feature_projection_dim = self.metadata.get("feature_projection_dim", self.grad_projection_dim)
        self.projection_dim = self.grad_projection_dim
        # Removed raw gradient loading API and metadata usage
        
        # Check for HDF5 or NPZ
        self.use_hdf5 = (self.data_path / "tensors.h5").exists()
    
    def load_gradients(
        self,
        group: str,
        condition: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load gradient data for a group/condition.
        
        Returns:
            (vectors, raw_norms, steps) tuple of arrays
        """
        if self.use_hdf5:
            return self._load_gradients_hdf5(group, condition)
        else:
            return self._load_gradients_npz(group, condition)
    
    def _load_gradients_hdf5(
        self,
        group: str,
        condition: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load gradients from HDF5."""
        return self._load_hdf5_group(
            group_path=f"gradients/{group}/{condition}",
            include_norms=True,
        )
    
    def _load_gradients_npz(
        self,
        group: str,
        condition: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load gradients from NPZ chunks."""
        return self._load_npz_chunks(
            chunk_dir=self.data_path / "gradients" / group / condition,
            include_norms=True,
        )
    
    def load_features(
        self,
        layer: str,
        condition: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load feature data for a layer/condition.
        
        Returns:
            (vectors, steps) tuple of arrays
        """
        layer_safe = layer.replace(".", "_")
        
        if self.use_hdf5:
            vectors, _, steps = self._load_hdf5_group(
                group_path=f"features/{layer_safe}/{condition}",
                include_norms=False,
            )
            return vectors, steps
        else:
            vectors, _, steps = self._load_npz_chunks(
                chunk_dir=self.data_path / "features" / layer_safe / condition,
                include_norms=False,
            )
            return vectors, steps

    def load_metrics(self) -> List[Dict[str, Any]]:
        """Load all metrics from JSONL file.
        
        Returns:
            List of metric records
        """
        metrics_path = self.data_path / "metrics.jsonl"
        
        if not metrics_path.exists():
            return []
        
        records = []
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        return records
    
    def load_collection_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load collection statistics."""
        stats_path = self.data_path / "collection_stats.json"
        
        if not stats_path.exists():
            return {}
        
        with open(stats_path) as f:
            return json.load(f)
    
    def iter_gradient_data(
        self,
    ) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over all gradient data.
        
        Yields:
            (group, condition, vectors, raw_norms, steps) tuples
        """
        for group in self.groups:
            for condition in self.conditions:
                vectors, raw_norms, steps = self.load_gradients(group, condition)
                if len(vectors) > 0:
                    yield group, condition, vectors, raw_norms, steps

    def iter_feature_data(
        self,
    ) -> Iterator[Tuple[str, str, np.ndarray, np.ndarray]]:
        """Iterate over all feature data.
        
        Yields:
            (layer, condition, vectors, steps) tuples
        """
        for layer in self.layers:
            for condition in self.conditions:
                vectors, steps = self.load_features(layer, condition)
                if len(vectors) > 0:
                    yield layer, condition, vectors, steps

    def _load_hdf5_group(
        self,
        group_path: str,
        include_norms: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import h5py

        with h5py.File(self.data_path / "tensors.h5", "r") as f:
            if group_path not in f:
                if include_norms:
                    return np.array([]), np.array([]), np.array([])
                return np.array([]), np.array([]), np.array([])
            vectors = f[f"{group_path}/vectors"][:]
            steps = f[f"{group_path}/steps"][:]
            if include_norms:
                raw_norms = f[f"{group_path}/raw_norms"][:]
            else:
                raw_norms = np.array([])

        return vectors, raw_norms, steps

    @staticmethod
    def _load_npz_chunks(
        chunk_dir: Path,
        include_norms: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not chunk_dir.exists():
            return np.array([]), np.array([]), np.array([])

        all_vectors: List[np.ndarray] = []
        all_norms: List[np.ndarray] = []
        all_steps: List[np.ndarray] = []

        for chunk_path in sorted(chunk_dir.glob("chunk_*.npz")):
            data = np.load(chunk_path)
            all_vectors.append(data["vectors"])
            all_steps.append(data["steps"])
            if include_norms:
                all_norms.append(data["raw_norms"])

        if not all_vectors:
            return np.array([]), np.array([]), np.array([])

        vectors = np.concatenate(all_vectors, axis=0)
        steps = np.concatenate(all_steps, axis=0)
        norms = np.concatenate(all_norms, axis=0) if include_norms else np.array([])
        return vectors, norms, steps

