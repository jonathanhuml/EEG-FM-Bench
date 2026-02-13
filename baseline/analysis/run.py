"""Unified Gradient Analysis Entry Script ().

This module provides a unified interface for gradient analysis of EEG foundation models.
Major improvements over v1:
- Three experimental paradigms with clear separation
- No runtime plotting - data saved to HDF5/NPZ for offline visualization
- Unified tensor collection with memory-efficient streaming
- Clean separation of concerns: collect → compute → visualize

Supported Models:
- CBraMod (Criss-Cross Attention)
- LaBraM (VQ-NSP Tokenizer + ViT)
- REVE (4D Fourier Positional Embedding + Denoising)
- CSBrain (Cross-Scale Brain Region Attention)
- Mantis (Multi-scale Attention)
- Moment (Time-series Foundation Model)

Experimental Paradigms:
1. SCRATCH_VS_PRETRAINED: Compare from-scratch vs pretrained initialization finetuning
2. PRETRAIN_VS_FINETUNE: Compare pretrain (reconstruction) vs finetune (classification) gradients
3. MULTI_DATASET_JOINT: Analyze cross-dataset gradient conflicts in joint training

Usage:
    python run_analysis.py --config analysis_config.yaml
    python run_analysis.py --model csbrain --paradigm scratch_vs_pretrained
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import datasets
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from baseline.abstract.config import AbstractConfig
from baseline.abstract.trainer import AbstractTrainer

# New modular components
from baseline.analysis.config import (
    AnalysisConfig,
    ExperimentParadigm,
    GroupingStrategy,
    MODEL_INNOVATION_GROUPS,
)
from baseline.analysis.collector import TensorCollector

# Model trainers and configs
from baseline.cbramod.cbramod_trainer import CBraModTrainer
from baseline.cbramod.cbramod_config import CBraModConfig
from baseline.labram.labram_trainer import LabramTrainer
from baseline.labram.labram_config import LabramConfig
from baseline.reve.reve_trainer import ReveTrainer
from baseline.reve.reve_config import ReveConfig
from baseline.csbrain.csbrain_trainer import CSBrainTrainer
from baseline.csbrain.csbrain_config import CSBrainConfig
from baseline.mantis.mantis_trainer import MantisTrainer
from baseline.mantis.mantis_config import MantisConfig
from baseline.moment.moment_trainer import MomentTrainer
from baseline.moment.moment_config import MomentConfig

# Parameter groupers
from baseline.utils.param_groupers import PARAM_GROUPER_REGISTRY
from baseline.analysis.grouper import EncoderParamGrouper


logger = logging.getLogger("analysis_run")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# =============================================================================
# Registry
# =============================================================================


TRAINER_REGISTRY: Dict[str, Tuple[Type[AbstractTrainer], Type[AbstractConfig]]] = {
    "cbramod":  (CBraModTrainer,    CBraModConfig   ),
    "labram":   (LabramTrainer,     LabramConfig    ),
    "reve":     (ReveTrainer,       ReveConfig      ),
    "csbrain":  (CSBrainTrainer,    CSBrainConfig   ),
    "mantis":   (MantisTrainer,     MantisConfig    ),
    "moment":   (MomentTrainer,     MomentConfig    ),
}


# =============================================================================
# Utility Functions
# =============================================================================


def build_trainer(
    model_type: str,
    cfg: AbstractConfig,
    device: str,
) -> AbstractTrainer:
    """Build trainer from model type and config."""
    trainer_cls, _ = TRAINER_REGISTRY[model_type]
    trainer = trainer_cls(cfg)
    trainer.setup_device(device)
    trainer.setup_analysis_mode()
    return trainer


def load_trainer_config(
    model_type: str,
    cfg_path: Optional[str],
) -> AbstractConfig:
    """Load trainer configuration from YAML with overrides."""
    cfg_cls: Type[AbstractConfig] = TRAINER_REGISTRY[model_type][1]
    
    base_cfg = cfg_cls()
    merged = OmegaConf.create(base_cfg.model_dump())
    
    if cfg_path:
        file_cfg = OmegaConf.load(cfg_path)
        merged = OmegaConf.merge(merged, file_cfg)
    
    config_dict = OmegaConf.to_container(merged, resolve=True)
    return cfg_cls.model_validate(config_dict)


def get_param_grouper(
    model_type: str,
    model: nn.Module,
    strategy: GroupingStrategy = GroupingStrategy.BY_MODULE_TYPE,
    innovation_groups: Optional[Dict[str, List[str]]] = None,
) -> EncoderParamGrouper:
    """Get parameter grouper for model."""
    grouper_cls = PARAM_GROUPER_REGISTRY.get(model_type)
    grouper = grouper_cls(model)
    
    # Apply innovation-based grouping if requested
    if strategy == GroupingStrategy.BY_MODEL_INNOVATION and innovation_groups:
        grouper.set_innovation_groups(innovation_groups)
        print(grouper.get_innovation_group_summary())
    
    return grouper


def move_batch_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Move batch tensors to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.deterministic = True


# =============================================================================
# Base Runner
# =============================================================================


class BaseRunner:
    """Base class for all analysis runners."""
    
    def __init__(
        self,
        config: AnalysisConfig,
        seed: Optional[int] = None,
        shared_run_dir: Optional[str] = None,
    ):
        self.config = config
        self.device = config.device
        self.output_dir = config.output.output_dir
        self.current_seed = seed if seed is not None else config.seed
        
        # Create output directory
        # If shared_run_dir is provided (multi-seed run), use it as parent
        # Otherwise create a new timestamped directory
        if shared_run_dir is not None:
            # Multi-seed run: use shared root with seed subdirectory
            self.shared_run_dir = shared_run_dir
            self.run_dir = os.path.join(
                shared_run_dir,
                config.model_type.value,
                f"seed_{self.current_seed}",
            )
        else:
            # Single-seed run: create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.shared_run_dir = os.path.join(
                self.output_dir,
                f"{config.paradigm.value}_{timestamp}",
            )
            self.run_dir = os.path.join(
                self.shared_run_dir,
                config.model_type.value,
            )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Save config
        self._save_config()
    
    def _setup_file_logging(self):
        """Setup file handler for logging."""
        log_path = os.path.join(self.run_dir, "console.log")
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add to root logger and analysis logger
        logging.getLogger().addHandler(file_handler)
        logger.addHandler(file_handler)
        self._file_handler = file_handler
        
        logger.info(f"Logging to: {log_path}")
        logger.info(f"Seed: {self.current_seed}")
        logger.info(f"Run directory: {self.run_dir}")
    
    def _cleanup_file_logging(self):
        """Remove file handler after run."""
        if hasattr(self, '_file_handler'):
            logging.getLogger().removeHandler(self._file_handler)
            logger.removeHandler(self._file_handler)
            self._file_handler.close()
    
    def _save_config(self):
        """Save analysis configuration."""
        config_path = os.path.join(self.run_dir, "analysis_config.yaml")
        OmegaConf.save(
            config=OmegaConf.create(self.config.model_dump(mode="json")),
            f=config_path,
        )
    
    def _create_collector(
        self,
        groups: List[str],
        conditions: List[str],
        output_path: str,
        layers: Optional[List[str]] = None,
    ) -> TensorCollector:
        """Create tensor collector for data saving."""
        max_samples = self.config.gradient.max_samples_per_condition
        if self.config.feature.enabled:
            max_samples = max(max_samples, self.config.feature.max_samples_per_condition)
        
        # Log collector configuration
        layers_list = layers or []
        logger.info("TensorCollector Configuration:")
        logger.info(f"  Output path: {output_path}")
        logger.info(f"  Groups ({len(groups)}): {groups}")
        logger.info(f"  Conditions ({len(conditions)}): {conditions}")
        logger.info(f"  Feature layers ({len(layers_list)}):")
        for i, layer in enumerate(layers_list):
            logger.info(f"    [{i}] {layer}")
        logger.info(f"  Gradient projection dim: {self.config.gradient.projection_dim}")
        logger.info(f"  Feature projection dim: {self.config.feature.projection_dim}")
        logger.info(f"  Max samples per condition: {max_samples}")
        
        return TensorCollector(
            output_path=output_path,
            groups=groups,
            conditions=conditions,
            layers=layers or [],
            projection_dim=self.config.gradient.projection_dim,
            projection_seed=self.config.gradient.projection_seed,
            max_memory_samples=max_samples,
            buffer_size=100,
            use_hdf5=(self.config.output.save_format == "hdf5"),
            feature_projection_dim=self.config.feature.projection_dim,
            feature_projection_seed=self.config.feature.projection_seed,
            track_raw_norms=self.config.gradient.track_raw_norms,
        )

    def _auto_detect_feature_layers(self, model: nn.Module) -> List[str]:
        """Auto-detect feature layers from model structure.
        
        This method intelligently detects hookable layers by:
        1. Looking for transformer block patterns (highest priority)
        2. Looking for attention/FFN output layers
        3. Handling Sequential/ModuleList containers by preferring parent modules
        4. Ensuring no nested/duplicate layers are selected
        
        Supported models and their key layer patterns:
        - labram: encoder.blocks.{i} (NeuralTransformer blocks)
        - cbramod: encoder.layers.{i} (TransformerEncoder layers with criss-cross attn)
        - csbrain: encoder.inter_region_attn, encoder.inter_window_attn (cross-scale attention)
        - reve: encoder.transformer.layers.{i} (TransformerBackbone layers with 4D Fourier PE)
        - mantis: encoder.vit_unit.transformer.layers.{i} (ViT transformer layers)
        - moment: encoder.encoder.block.{i} (T5 encoder blocks)
        """
        import re
        
        layers: List[str] = []
        
        # Container types that should be skipped in favor of their parents
        container_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        
        # Model-specific patterns for transformer blocks
        # Priority order: specific patterns first, then generic fallbacks
        block_patterns = [
            # LaBraM: NeuralTransformer with blocks
            "blocks.",
            # CBraMod: TransformerEncoder with criss-cross attention
            "encoder.layers.",
            # CSBrain: Cross-scale attention modules
            "inter_region_attn", "inter_window_attn",
            # REVE: TransformerBackbone layers
            "layers.",
            # Mantis: ViT transformer layers
            "vit_unit.transformer.layers.", "tokgen_unit",
            # Moment: T5 encoder blocks
            "encoder.block.", "encoder.encoder.block.",
            # Generic fallbacks
            "transformer.layers", "encoder_layers",
        ]
        
        # REVE-specific pattern: transformer.layers.N where N is a number
        # These are ModuleLists containing [Attention, FeedForward]. ModuleList
        # has no forward, so we must hook a real submodule (prefer FFN).
        reve_layer_pattern = re.compile(r"^encoder\.transformer\.layers\.\d+$")
        
        # Output layer patterns (attention/FFN outputs)
        output_patterns = [
            "attn.proj", "attn.to_out", "self_attn_s", "self_attn_t",
            "mlp.fc2", "ffn.linear2", "net.3",  # FFN output layers
            "final_layer", "fc_norm",  # Final normalization layers
        ]
        
        # Patterns that indicate we should prefer the parent module
        # (e.g., if we match "net.3", prefer the FeedForward module containing it)
        prefer_parent_patterns = [
            r"\.net\.\d+$",  # Sequential layers like net.0, net.3
            r"\.\d+$",       # Pure numeric indices in ModuleList
        ]
        
        block_layers = []
        output_layers = []
        
        # Build name -> module mapping
        name_to_module = {name: mod for name, mod in model.named_modules()}
        
        for name, module in model.named_modules():
            # Skip very short names or leaf modules without children that are trivial
            if len(name) < 3:
                continue
            
            # Special handling for REVE transformer layers
            # These are ModuleLists; hook a real submodule with forward
            if reve_layer_pattern.match(name):
                ff_name = f"{name}.1"
                attn_name = f"{name}.0"
                preferred = None
                if ff_name in name_to_module:
                    preferred = ff_name
                elif attn_name in name_to_module:
                    preferred = attn_name
                else:
                    preferred = name
                if not any(preferred.startswith(existing + ".") for existing in block_layers):
                    block_layers.append(preferred)
                continue
            
            # Skip pure container types - we want their children or parents
            if isinstance(module, container_types):
                # But check if the container's parent is a meaningful module
                # (we'll add it via the parent logic below)
                continue
            
            # Check for block patterns
            is_block = any(p in name for p in block_patterns)
            # Ensure we capture complete blocks, not sub-components
            is_complete_block = is_block and not any(
                sub in name for sub in [".norm", ".drop", ".act", ".bias", ".weight"]
            )
            
            if is_complete_block:
                # Avoid duplicate/nested blocks - only add if not a child of existing
                if not any(name.startswith(existing + ".") for existing in block_layers):
                    block_layers.append(name)
            
            # Check for output layer patterns
            is_output = any(p in name for p in output_patterns)
            if is_output and isinstance(module, (nn.Linear, nn.Module)):
                # Check if we should prefer the parent module
                should_use_parent = any(
                    re.search(pattern, name) for pattern in prefer_parent_patterns
                )
                
                if should_use_parent:
                    # Find the parent module that's not a container
                    parts = name.split(".")
                    for i in range(len(parts) - 1, 0, -1):
                        parent_name = ".".join(parts[:i])
                        if parent_name in name_to_module:
                            parent_mod = name_to_module[parent_name]
                            if not isinstance(parent_mod, container_types):
                                # Use parent instead
                                if parent_name not in output_layers:
                                    output_layers.append(parent_name)
                                break
                else:
                    output_layers.append(name)
        
        # Prefer block layers over output layers
        layers = block_layers if block_layers else output_layers
        
        # Filter out duplicates and sort by depth
        seen = set()
        unique_layers = []
        for layer in sorted(layers, key=lambda x: (x.count("."), x)):
            # Skip if this is a child of an already added layer
            if not any(layer.startswith(parent + ".") for parent in seen):
                unique_layers.append(layer)
                seen.add(layer)
        
        layers = unique_layers
        
        # Limit number of layers
        if len(layers) > self.config.feature.auto_detect_max_layers:
            step = max(1, len(layers) // self.config.feature.auto_detect_max_layers)
            layers = layers[::step]
        
        # Fallback to encoder if nothing found
        if not layers:
            # Try to find the main encoder module
            for name, _ in model.named_modules():
                if name in ["encoder", "backbone", "transformer"]:
                    layers = [name]
                    break
        
        final_layers = layers or ["encoder"]
        
        # Log auto-detected layers
        logger.info("Feature Layer Auto-Detection:")
        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Block layers found: {len(block_layers)}")
        logger.info(f"  Output layers found: {len(output_layers)}")
        logger.info(f"  Final detected layers ({len(final_layers)}):")
        for i, layer in enumerate(final_layers):
            logger.info(f"    [{i}] {layer}")
        
        return final_layers

    def _log_param_grouper_info(self, grouper: 'EncoderParamGrouper', label: str = ""):
        """Log detailed parameter grouper information."""
        prefix = f"[{label}] " if label else ""
        logger.info(f"{prefix}Parameter Grouper Summary:")
        logger.info(f"{prefix}  Groups: {grouper.group_names}")
        
        total_params = 0
        for group_name in grouper.group_names:
            params_in_group = grouper.get_params_in_group(group_name)
            param_count = grouper.get_group_param_count(group_name)
            total_params += param_count
            logger.info(f"{prefix}  [{group_name}] {len(params_in_group)} parameters, {param_count:,} total params")
            
            # Log first few parameter names for verification
            if params_in_group[:3]:
                for pname in params_in_group[:3]:
                    logger.debug(f"{prefix}    - {pname}")
                if len(params_in_group) > 3:
                    logger.debug(f"{prefix}    ... and {len(params_in_group) - 3} more")
        
        logger.info(f"{prefix}  Total grouped parameters: {total_params:,}")
    
    def _log_hook_registration_info(
        self,
        model: nn.Module,
        requested_layers: List[str],
        registered_layers: List[str],
        label: str = "",
        fallback_info: Optional[Dict[str, str]] = None,
    ):
        """Log detailed hook registration information.
        
        Args:
            model: The model being hooked
            requested_layers: Original layer names/patterns requested
            registered_layers: Actually registered layer names
            label: Optional prefix for log messages
            fallback_info: Dict mapping original name -> fallback name for parent fallbacks
        """
        prefix = f"[{label}] " if label else ""
        fallback_info = fallback_info or {}
        
        logger.info(f"{prefix}Feature Hook Registration:")
        logger.info(f"{prefix}  Requested layers: {len(requested_layers)}")
        logger.info(f"{prefix}  Registered layers: {len(registered_layers)}")
        
        # Log registered layers with fallback info
        for layer_name in registered_layers:
            # Check if this was a fallback registration
            original = [k for k, v in fallback_info.items() if v == layer_name]
            if original:
                logger.info(f"{prefix}    ✓ {layer_name} (fallback for: {', '.join(original)})")
            else:
                logger.info(f"{prefix}    ✓ {layer_name}")
        
        # Log failed registrations (those not in registered_layers and not in fallback values)
        registered_set = set(registered_layers)
        fallback_originals = set(fallback_info.keys())
        
        failed = []
        for layer in requested_layers:
            # Skip patterns - they may match zero modules legitimately
            if "*" in layer or "?" in layer or layer.startswith("r:"):
                continue
            # Skip if directly registered or had a fallback
            if layer in registered_set or layer in fallback_originals:
                continue
            failed.append(layer)
        
        if failed:
            logger.warning(f"{prefix}  Failed to register ({len(failed)}):")
            for layer_name in failed:
                # Try to find why it failed
                found = False
                for name, module in model.named_modules():
                    if name == layer_name:
                        found = True
                        logger.warning(f"{prefix}    ✗ {layer_name} (module type: {type(module).__name__})")
                        break
                if not found:
                    logger.warning(f"{prefix}    ✗ {layer_name} (NOT FOUND in model)")
        
        # Summary comparison
        logger.info(f"{prefix}  Summary: {len(requested_layers)} requested -> {len(registered_layers)} registered")
        if fallback_info:
            logger.info(f"{prefix}  Fallbacks applied: {len(fallback_info)}")
            for orig, actual in fallback_info.items():
                logger.info(f"{prefix}    '{orig}' -> '{actual}'")
    
    def _find_parent_module(
        self,
        model: nn.Module,
        layer_name: str,
    ) -> Optional[Tuple[str, nn.Module]]:
        """Find the nearest parent module that has a forward method.
        
        When a layer is inside a container (Sequential, ModuleList), this finds
        the container or a meaningful parent that can be hooked instead.
        
        Args:
            model: The root model
            layer_name: The target layer name that couldn't be found
            
        Returns:
            Tuple of (parent_name, parent_module) or None if not found
        """
        # Build module name -> module mapping
        name_to_module = {name: mod for name, mod in model.named_modules()}
        
        # Container types that wrap other modules
        container_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        base_forward = nn.Module.forward
        unimpl_forward = getattr(nn.Module, "_forward_unimplemented", None)
        
        # Try progressively shorter prefixes (parent modules)
        parts = layer_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_name = ".".join(parts[:i])
            if parent_name in name_to_module:
                parent_mod = name_to_module[parent_name]
                # Prefer non-trivial modules that have their own forward
                # Skip pure containers without custom logic
                if not isinstance(parent_mod, container_types):
                    return parent_name, parent_mod
                forward_impl = type(parent_mod).forward
                if forward_impl is base_forward:
                    continue
                if unimpl_forward is not None and forward_impl is unimpl_forward:
                    continue
                return parent_name, parent_mod
        
        # Try root module (empty string name) as last resort
        # This handles cases like "net.5" when model is FeedForward with self.net
        if "" in name_to_module:
            root_mod = name_to_module[""]
            if not isinstance(root_mod, container_types):
                # Return empty string as name, but caller should handle this
                # by using the model's class name or a default identifier
                return "", root_mod
        
        return None

    def _find_module_by_pattern(
        self,
        model: nn.Module,
        pattern: str,
    ) -> List[Tuple[str, nn.Module]]:
        """Find modules matching a pattern (supports wildcards and regex).
        
        Patterns:
        - Exact match: "encoder.blocks.0"
        - Wildcard: "encoder.blocks.*" matches all blocks
        - Suffix match: "*.attn" matches all attention modules
        - Regex (if starts with 'r:'): "r:encoder\\.blocks\\.[0-9]+"
        
        Args:
            model: The root model
            pattern: The pattern to match
            
        Returns:
            List of (name, module) tuples
        """
        import fnmatch
        import re
        
        matches = []
        
        # Check if it's a regex pattern
        if pattern.startswith("r:"):
            regex = re.compile(pattern[2:])
            for name, module in model.named_modules():
                if regex.fullmatch(name):
                    matches.append((name, module))
        else:
            # Use fnmatch for glob-style patterns
            for name, module in model.named_modules():
                if fnmatch.fnmatch(name, pattern):
                    matches.append((name, module))
        
        return matches

    def _register_feature_hooks(
        self,
        model: nn.Module,
        layers: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle], List[str]]:
        """Register forward hooks to capture intermediate features.
        
        This method handles several edge cases:
        1. Exact match: Registers hook on the exact layer name
        2. Pattern match: If layer contains wildcards, matches multiple modules
        3. Parent fallback: If layer not found, tries to register on parent module
        4. Container penetration: For Sequential/ModuleList, can register on
           the container or find meaningful children
        
        Args:
            model: The model to register hooks on
            layers: List of layer names or patterns to hook
            
        Returns:
            Tuple of:
            - current_features: Dict to be populated with features during forward
            - hooks: List of hook handles (for cleanup)
            - registered_layers: List of ACTUALLY registered layer names 
              (use this for TensorCollector initialization to avoid name mismatch)
        """
        current_features: Dict[str, torch.Tensor] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []
        registered_layers: List[str] = []
        fallback_info: Dict[str, str] = {}  # original -> actual registered

        def make_hook(name: str):
            def hook(module, input, output):
                def _extract_tensor(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj
                    if isinstance(obj, (list, tuple)):
                        for item in obj:
                            t = _extract_tensor(item)
                            if t is not None:
                                return t
                        return None
                    if isinstance(obj, dict):
                        for key in [
                            "last_hidden_state",
                            "hidden_states",
                            "embeddings",
                            "features",
                            "x",
                            "logits",
                        ]:
                            if key in obj:
                                t = _extract_tensor(obj[key])
                                if t is not None:
                                    return t
                        for value in obj.values():
                            t = _extract_tensor(value)
                            if t is not None:
                                return t
                        return None
                    for attr in [
                        "last_hidden_state",
                        "hidden_states",
                        "embeddings",
                        "features",
                        "x",
                        "logits",
                    ]:
                        if hasattr(obj, attr):
                            t = _extract_tensor(getattr(obj, attr))
                            if t is not None:
                                return t
                    return None

                tensor = _extract_tensor(output)
                if tensor is not None:
                    current_features[name] = tensor.detach()
            return hook

        # Build name -> module mapping for fast lookup
        name_to_module = {name: mod for name, mod in model.named_modules()}
        
        # Track already registered to avoid duplicates
        already_registered: set = set()
        
        for layer in layers:
            # Check if this is a pattern (contains wildcards or regex prefix)
            is_pattern = "*" in layer or "?" in layer or layer.startswith("r:")
            
            if is_pattern:
                # Pattern matching mode
                matched = self._find_module_by_pattern(model, layer)
                for name, module in matched:
                    if name not in already_registered:
                        hooks.append(module.register_forward_hook(make_hook(name)))
                        registered_layers.append(name)
                        already_registered.add(name)
                if not matched:
                    logger.warning(f"Pattern '{layer}' matched no modules")
            elif layer in name_to_module:
                # Exact match
                if layer not in already_registered:
                    module = name_to_module[layer]
                    hooks.append(module.register_forward_hook(make_hook(layer)))
                    registered_layers.append(layer)
                    already_registered.add(layer)
            else:
                # Try parent fallback
                parent_info = self._find_parent_module(model, layer)
                if parent_info is not None:
                    parent_name, parent_mod = parent_info
                    # Handle root module (empty string name)
                    if parent_name == "":
                        parent_name = f"_root_{type(parent_mod).__name__}"
                    if parent_name not in already_registered:
                        hooks.append(parent_mod.register_forward_hook(make_hook(parent_name)))
                        registered_layers.append(parent_name)
                        already_registered.add(parent_name)
                        fallback_info[layer] = parent_name
                        logger.info(
                            f"Layer '{layer}' not found, registered parent '{parent_name}' "
                            f"(type: {type(parent_mod).__name__})"
                        )
                else:
                    logger.warning(f"Layer '{layer}' not found and no suitable parent")
        
        # Log registration summary
        self._log_hook_registration_info(
            model, layers, registered_layers, fallback_info=fallback_info
        )

        return current_features, hooks, registered_layers

    def _remove_feature_hooks(
        self,
        hooks: List[torch.utils.hooks.RemovableHandle],
        current_features: Dict[str, torch.Tensor],
    ):
        """Remove all feature hooks and clear cached features."""
        for handle in hooks:
            handle.remove()
        hooks.clear()
        current_features.clear()

    def _collect_feature_batches(
        self,
        trainer: AbstractTrainer,
        collector: TensorCollector,
        current_features: Dict[str, torch.Tensor],
        batches: List[Dict[str, Any]],
        condition: str,
        step: int,
    ):
        """Run feature probes on fixed batches and collect features."""
        if not batches:
            return

        was_training = trainer.model.training
        trainer.model.eval()
        try:
            with torch.no_grad():
                for batch in batches:
                    current_features.clear()
                    batch = move_batch_to_device(batch, self.device)
                    _ = trainer.model(batch)
                    if current_features:
                        collector.collect_features(current_features, condition, step)
        finally:
            # Always clear features after collection to avoid memory leaks
            current_features.clear()
            if was_training:
                trainer.model.train()

    def _resolve_num_steps(self, train_loader: DataLoader) -> int:
        """Resolve number of steps from config (num_steps > num_epochs > loader length)."""
        if self.config.training.num_steps is not None:
            return self.config.training.num_steps
        if self.config.training.num_epochs is not None:
            return int(self.config.training.num_epochs) * len(train_loader)
        return len(train_loader)

    def _get_probe_splits(self) -> List[Tuple[str, datasets.NamedSplit]]:
        """Get evaluation splits used for probing."""
        return [
            ("validation", datasets.Split.VALIDATION),
            ("test", datasets.Split.TEST),
        ]

    def _create_probe_loaders(
        self,
        trainer: AbstractTrainer,
        dataset_name: str,
        ds_config: Any,
    ) -> Dict[str, DataLoader]:
        """Create probe loaders for validation/test splits."""
        loaders: Dict[str, DataLoader] = {}
        for split_name, split in self._get_probe_splits():
            try:
                loader, _ = trainer.create_single_dataloader(dataset_name, ds_config, split)
                loaders[split_name] = loader
            except Exception as e:
                logger.warning(f"Probe split '{split_name}' unavailable for {dataset_name}: {e}")
        return loaders

    def _sample_probe_batches(
        self,
        loader: DataLoader,
        max_batches: int,
    ) -> List[Dict[str, Any]]:
        """Sample fixed probe batches from a loader (cyclic if needed)."""
        if max_batches <= 0:
            return []
        batches: List[Dict[str, Any]] = []
        while len(batches) < max_batches:
            for _, batch in enumerate(loader):
                batches.append(batch)
                if len(batches) >= max_batches:
                    break
        return batches

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute global L2 norm of current gradients without clipping."""
        total = torch.zeros(1, device=self.device)
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum()
        return float(torch.sqrt(total).item())

    def _maybe_unfreeze_encoder_for_analysis(
        self,
        trainer: AbstractTrainer,
        current_step: int,
    ):
        """Unfreeze encoder parameters after analysis warmup steps.

        Some trainers (e.g., REVE) implement warmup-freeze by setting
        encoder params to requires_grad=False and only unfreeze in epoch-based
        training loops. Analysis loops are step-based, so we explicitly unfreeze
        once warmup_steps is reached.
        """
        cfg = getattr(trainer, "cfg", None)
        if cfg is None:
            return
        if getattr(cfg.training, "freeze_encoder", False):
            return
        if not getattr(cfg.training, "warmup_freeze_encoder", False):
            return
        if not getattr(trainer, "warmup_freeze_state", False):
            return
        if current_step < self.config.training.warmup_steps:
            return

        # Prefer trainer-specific unfreeze if available
        if hasattr(trainer, "unfreeze_encoder"):
            trainer.unfreeze_encoder()
        else:
            for _, param in trainer.model.named_parameters():
                param.requires_grad = True
            trainer.warmup_freeze_state = False
            logger.info("Encoder parameters unfrozen for analysis")

    def _reset_scheduler_for_analysis(
        self,
        trainer: AbstractTrainer,
        train_loader: DataLoader,
        num_steps: int,
    ):
        """Reset LR scheduler to use analysis step budget instead of epochs."""
        if trainer.optimizer is None:
            return

        total_steps = max(1, int(num_steps))
        warmup_steps = int(max(0, self.config.training.warmup_steps))
        warmup_steps = int(min(warmup_steps, total_steps))

        lrs = [p["lr"] for p in trainer.optimizer.param_groups]

        if trainer.cfg.training.lr_schedule == "onecycle":
            trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                trainer.optimizer,
                max_lr=lrs,
                total_steps=total_steps,
                pct_start=trainer.cfg.training.pct_start,
            )
            return

        if trainer.cfg.training.lr_schedule == "cosine":
            if total_steps <= warmup_steps:
                trainer.scheduler = torch.optim.lr_scheduler.LinearLR(
                    trainer.optimizer,
                    start_factor=trainer.cfg.training.warmup_scale,
                    end_factor=1.0,
                    total_iters=total_steps,
                )
                return

            warm_scheduler = torch.optim.lr_scheduler.LinearLR(
                trainer.optimizer,
                start_factor=trainer.cfg.training.warmup_scale,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=trainer.cfg.training.min_lr,
            )
            trainer.scheduler = torch.optim.lr_scheduler.SequentialLR(
                trainer.optimizer,
                schedulers=[warm_scheduler, cos_scheduler],
                milestones=[warmup_steps],
            )
            return

        if trainer.cfg.training.lr_schedule == "reduce_on_plateau":
            # For reduce_on_plateau, use a simple warmup + constant LR schedule for analysis
            # The actual ReduceLROnPlateau requires validation metrics, which is not suitable for analysis
            if warmup_steps > 0:
                trainer.scheduler = torch.optim.lr_scheduler.LinearLR(
                    trainer.optimizer,
                    start_factor=trainer.cfg.training.min_lr / trainer.cfg.training.max_lr,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
            else:
                # No warmup, use constant LR (dummy scheduler that does nothing)
                trainer.scheduler = torch.optim.lr_scheduler.ConstantLR(
                    trainer.optimizer,
                    factor=1.0,
                    total_iters=total_steps,
                )
            return

        raise NotImplementedError("Unknown learning rate schedule")

    def _probe_gradients_and_metrics(
        self,
        trainer: AbstractTrainer,
        grouper: EncoderParamGrouper,
        collector: TensorCollector,
        condition: str,
        step: int,
        batches: List[Dict[str, Any]],
        dataset_name: str,
        split_name: str,
    ):
        """Probe gradients on fixed batches and save metrics (eval/test only)."""
        if not batches:
            return

        was_training = trainer.model.training
        trainer.model.eval()

        labels_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        loss_list: List[float] = []
        grad_norms: List[float] = []

        try:
            for batch in batches:
                trainer.optimizer.zero_grad()
                batch = move_batch_to_device(batch, self.device)
                labels = batch.get("label")

                with torch.set_grad_enabled(True):
                    logits, loss = trainer.train_step(batch, labels)

                    # Probe should not use GradScaler to avoid repeated unscale_ calls
                    loss.backward()

                # Use collector.collect_gradients for consistency with other code paths
                collector.collect_gradients(
                    grouper=grouper,
                    condition=condition,
                    step=step,
                )

                grad_norm = self._compute_grad_norm(trainer.model)
                grad_norms.append(grad_norm)

                if self.config.output.save_metrics_jsonl and labels is not None:
                    labels_list.append(labels.detach().to(torch.int32).cpu())
                    logits_list.append(logits.detach().to(torch.float32).cpu())
                    loss_list.append(float(loss.detach().item()))

                trainer.optimizer.zero_grad()

            if self.config.output.save_metrics_jsonl and labels_list:
                all_labels = torch.cat(labels_list, dim=0)
                all_logits = torch.cat(logits_list, dim=0)
                mean_loss = float(np.mean(loss_list)) if loss_list else 0.0
                mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

                metrics = trainer._calculate_metrics_for_dataset(
                    labels=all_labels,
                    logits=all_logits,
                    ds_name=dataset_name,
                    prefix=f"probe/{condition}/{split_name}",
                    loss=mean_loss,
                )
                metrics[f"{dataset_name}/probe/{condition}/{split_name}/grad_norm"] = mean_grad_norm
                collector.save_metrics(step=step, metrics=metrics)
        finally:
            if was_training:
                trainer.model.train()
    
    def run(self) -> Dict[str, Any]:
        """Run the analysis. Override in subclasses."""
        raise NotImplementedError


# =============================================================================
# Paradigm 1: Scratch vs Pretrained
# =============================================================================

class ScratchVsPretrainedRunner(BaseRunner):
    """Runner for comparing from-scratch vs pretrained finetuning."""
    
    def __init__(
        self,
        config: AnalysisConfig,
        trainer_config: AbstractConfig,
        seed: Optional[int] = None,
        shared_run_dir: Optional[str] = None,
    ):
        super().__init__(config, seed=seed, shared_run_dir=shared_run_dir)
        self.trainer_config = trainer_config
        
        # Get dataset names
        self.dataset_names = (
            list(config.datasets.keys())
            if config.datasets else list(trainer_config.data.datasets.keys())
        )
        if not self.dataset_names:
            raise ValueError("No datasets specified for analysis")
        
        # Get pretrained checkpoint
        self.pretrained_ckpt = (
            config.pretrained_checkpoint or
            getattr(trainer_config.model, "pretrained_path", None)
        )
        if not self.pretrained_ckpt:
            logger.warning("No pretrained checkpoint specified")
    
    def _run_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Run analysis for a single dataset."""
        ds_config = self.trainer_config.data.datasets.get(dataset_name)
        if ds_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_dir = os.path.join(self.run_dir, f"dataset_{dataset_name}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create configs for scratch and pretrained
        scratch_cfg = self.trainer_config.model_copy(deep=True)
        pretrained_cfg = self.trainer_config.model_copy(deep=True)
        
        scratch_cfg.multitask = False
        pretrained_cfg.multitask = False
        
        scratch_cfg.model.pretrained_path = None
        pretrained_cfg.model.pretrained_path = self.pretrained_ckpt
        
        scratch_cfg.data.datasets = {dataset_name: ds_config}
        pretrained_cfg.data.datasets = {dataset_name: ds_config}
        
        # Build trainers
        scratch_trainer = build_trainer(
            self.config.model_type.value, scratch_cfg, self.device
        )
        pretrained_trainer = build_trainer(
            self.config.model_type.value, pretrained_cfg, self.device
        )
        
        scratch_trainer.collect_dataset_info(mixed=False, ds_name=dataset_name)
        pretrained_trainer.collect_dataset_info(mixed=False, ds_name=dataset_name)
        
        scratch_model = scratch_trainer.setup_model()
        pretrained_model = pretrained_trainer.setup_model()
        
        train_loader, _ = scratch_trainer.create_single_dataloader(
            dataset_name, ds_config, split=datasets.Split.TRAIN
        )

        # Probe loaders from eval/test splits (do not use train split for probes)
        probe_loaders = self._create_probe_loaders(
            scratch_trainer, dataset_name, ds_config
        )
        
        scratch_trainer.setup_optimizer_and_scheduler(scratch_model, train_loader)
        pretrained_trainer.setup_optimizer_and_scheduler(pretrained_model, train_loader)
        
        # Get parameter groups (use separate groupers per model instance)
        scratch_grouper = get_param_grouper(
            self.config.model_type.value,
            scratch_trainer.model,
            self.config.grouping_strategy,
            MODEL_INNOVATION_GROUPS.get(self.config.model_type.value),
        )
        pretrained_grouper = get_param_grouper(
            self.config.model_type.value,
            pretrained_trainer.model,
            self.config.grouping_strategy,
            MODEL_INNOVATION_GROUPS.get(self.config.model_type.value),
        )
        
        # Log parameter grouper details
        self._log_param_grouper_info(scratch_grouper, f"{dataset_name}/scratch")
        self._log_param_grouper_info(pretrained_grouper, f"{dataset_name}/pretrained")
        
        groups = list(scratch_grouper.group_names)
        if pretrained_grouper.group_names != scratch_grouper.group_names:
            extra_groups = [
                g for g in pretrained_grouper.group_names if g not in groups
            ]
            if extra_groups:
                groups.extend(extra_groups)
            logger.warning(
                "Scratch and pretrained group names differ; using union of groups."
            )
        conditions = ["scratch", "pretrained"]
        
        # Probe batches: sample from validation/test splits and merge into one list
        # (we don't distinguish splits during probing, just need diverse eval data)
        probe_batch_count = max(1, self.config.gradient.probe_batches_per_condition)
        if self.config.feature.enabled:
            probe_batch_count = max(
                probe_batch_count, self.config.feature.probe_batches_per_condition
            )
        # Sample from each split and merge
        probe_batches: List[Dict[str, Any]] = []
        for split_name, loader in probe_loaders.items():
            split_batches = self._sample_probe_batches(loader, probe_batch_count)
            probe_batches.extend(split_batches)
            logger.info(f"  Sampled {len(split_batches)} batches from {split_name}")
        logger.info(f"  Total probe batches: {len(probe_batches)}")

        # Feature collection setup
        feature_layers: List[str] = []
        actual_feature_layers: List[str] = []  # Actually registered layers
        scratch_features: Dict[str, torch.Tensor] = {}
        scratch_hooks: List[torch.utils.hooks.RemovableHandle] = []
        pretrained_features: Dict[str, torch.Tensor] = {}
        pretrained_hooks: List[torch.utils.hooks.RemovableHandle] = []
        if self.config.feature.enabled:
            logger.info("=" * 50)
            logger.info("Feature Collection Setup")
            logger.info("=" * 50)
            
            # Step 1: Get requested layers (from config or auto-detect)
            if self.config.feature.feature_layers:
                feature_layers = self.config.feature.feature_layers
                logger.info(f"Using configured feature_layers: {feature_layers}")
            else:
                logger.info("Auto-detecting feature layers...")
                feature_layers = self._auto_detect_feature_layers(scratch_trainer.model)
            
            # Step 2: Register hooks on both models
            logger.info(f"\\n[Scratch Model] Registering hooks...")
            scratch_features, scratch_hooks, scratch_registered = self._register_feature_hooks(
                scratch_trainer.model, feature_layers
            )
            logger.info(f"\\n[Pretrained Model] Registering hooks...")
            pretrained_features, pretrained_hooks, pretrained_registered = self._register_feature_hooks(
                pretrained_trainer.model, feature_layers
            )
            
            # Step 3: Use union of actually registered layers for collector
            actual_feature_layers = list(dict.fromkeys(
                scratch_registered + pretrained_registered
            ))
            
            # Step 4: Log summary
            logger.info("\\n" + "-" * 50)
            logger.info("Feature Layer Summary:")
            logger.info(f"  Requested: {len(feature_layers)} layers")
            logger.info(f"  Scratch registered: {len(scratch_registered)} layers")
            logger.info(f"  Pretrained registered: {len(pretrained_registered)} layers")
            logger.info(f"  Final (union): {len(actual_feature_layers)} layers")
            if set(actual_feature_layers) != set(feature_layers):
                logger.info("  Note: Actual layers differ from requested (fallbacks applied)")
            logger.info("-" * 50)

        # Create collector
        collector = self._create_collector(
            groups=groups,
            conditions=conditions,
            output_path=os.path.join(dataset_dir, "gradients"),
            layers=actual_feature_layers,  # Use actually registered layers
        )
        
        # Training history
        history = {
            "scratch": {"loss": [], "grad_norm": [], "step": []},
            "pretrained": {"loss": [], "grad_norm": [], "step": []},
        }
        
        # Training loop
        num_steps = self._resolve_num_steps(train_loader)
        self._reset_scheduler_for_analysis(scratch_trainer, train_loader, num_steps)
        self._reset_scheduler_for_analysis(pretrained_trainer, train_loader, num_steps)
        
        logger.info(
            f"Starting {dataset_name}: {num_steps} steps, "
            f"collect_interval={self.config.gradient.collect_interval}"
        )
        
        step = 0
        while step < num_steps:
            for _, batch in enumerate(train_loader):
                if step >= num_steps:
                    break

                current_step = step + 1

                # Ensure encoder is unfrozen after warmup for analysis
                self._maybe_unfreeze_encoder_for_analysis(scratch_trainer, current_step)
                self._maybe_unfreeze_encoder_for_analysis(pretrained_trainer, current_step)

                # Train scratch
                sc_loss, sc_grad, _ = scratch_trainer.finetune_one_batch(batch)

                history["scratch"]["loss"].append(sc_loss)
                history["scratch"]["grad_norm"].append(sc_grad)
                history["scratch"]["step"].append(current_step)

                # Train pretrained
                pt_loss, pt_grad, _ = pretrained_trainer.finetune_one_batch(batch)

                history["pretrained"]["loss"].append(pt_loss)
                history["pretrained"]["grad_norm"].append(pt_grad)
                history["pretrained"]["step"].append(current_step)

                should_probe = (
                    current_step >= self.config.training.warmup_steps
                    and current_step % self.config.gradient.collect_interval == 0
                )
                if should_probe:
                    logger.info(
                        f"[{dataset_name}] Step {current_step}/{num_steps} | "
                        f"Scratch: loss={sc_loss:.4f}, grad={sc_grad:.4f} | "
                        f"Pretrained: loss={pt_loss:.4f}, grad={pt_grad:.4f}"
                    )
                    # Probe gradients on merged batches (no split distinction)
                    self._probe_gradients_and_metrics(
                        scratch_trainer,
                        scratch_grouper,
                        collector,
                        condition="scratch",
                        step=current_step,
                        batches=probe_batches,
                        dataset_name=dataset_name,
                        split_name="probe",  # unified name since splits are merged
                    )
                    self._probe_gradients_and_metrics(
                        pretrained_trainer,
                        pretrained_grouper,
                        collector,
                        condition="pretrained",
                        step=current_step,
                        batches=probe_batches,
                        dataset_name=dataset_name,
                        split_name="probe",
                    )

                    # Collect features on merged batches
                    if self.config.feature.enabled and feature_layers:
                        self._collect_feature_batches(
                            scratch_trainer,
                            collector,
                            scratch_features,
                            probe_batches,
                            "scratch",
                            current_step,
                        )
                        self._collect_feature_batches(
                            pretrained_trainer,
                            collector,
                            pretrained_features,
                            probe_batches,
                            "pretrained",
                            current_step,
                        )

                    if (self.config.output.checkpoint_interval > 0 and
                        current_step % self.config.output.checkpoint_interval == 0):
                        collector.flush()

                step += 1
        
        # Finalize and save
        collector.finalize()

        if self.config.feature.enabled and feature_layers:
            self._remove_feature_hooks(scratch_hooks, scratch_features)
            self._remove_feature_hooks(pretrained_hooks, pretrained_features)
        
        # Save training history
        history_path = os.path.join(dataset_dir, "train_history.npz")
        np.savez(
            history_path,
            **{f"{k}_{m}": np.array(v) for k, hist in history.items() for m, v in hist.items()}
        )
        
        logger.info(f"Completed {dataset_name}, saved to {dataset_dir}")
        
        return {
            "dataset": dataset_name,
            "output_dir": dataset_dir,
            "num_steps": num_steps,
            "groups": groups,
            "conditions": conditions,
        }
    
    def run(self) -> Dict[str, Any]:
        """Run scratch vs pretrained analysis for all datasets.
        
        When multiple datasets are specified, this will run analysis on each
        dataset independently. Cross-dataset consistency analysis should be
        done in post-processing via visualize.py.
        
        Returns:
            Dict with per-dataset results
        """
        logger.info("=" * 60)
        logger.info("Starting SCRATCH_VS_PRETRAINED analysis")
        logger.info("=" * 60)
        
        try:
            results = {}
            for dataset_name in self.dataset_names:
                results[dataset_name] = self._run_single_dataset(dataset_name)
            
            return results
        finally:
            self._cleanup_file_logging()


# =============================================================================
# Paradigm 2: Pretrain vs Finetune
# =============================================================================


class PretrainVsFinetuneRunner(BaseRunner):
    """Runner for comparing pretrain (reconstruction) vs finetune (classification) gradients.
    
    This paradigm answers: Do pretrain and finetune optimize in the same direction?
    Both start from scratch, but pretrain uses reconstruction loss while finetune
    uses classification loss.
    """
    
    def __init__(
        self,
        config: AnalysisConfig,
        trainer_config: AbstractConfig,
        seed: Optional[int] = None,
        shared_run_dir: Optional[str] = None,
    ):
        super().__init__(config, seed=seed, shared_run_dir=shared_run_dir)
        self.trainer_config = trainer_config
        
        # Get dataset names
        self.dataset_names = (
            list(config.datasets.keys())
            if config.datasets else list(trainer_config.data.datasets.keys())
        )
        if not self.dataset_names:
            raise ValueError("No datasets specified for analysis")
        
        # Masking config for pretrain
        self.mask_ratio = config.masking.mask_ratio
        self.mask_strategy = config.masking.mask_strategy
    
    def _run_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Run pretrain vs finetune analysis for a single dataset."""
        ds_config = self.trainer_config.data.datasets.get(dataset_name)
        if ds_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_dir = os.path.join(self.run_dir, f"dataset_{dataset_name}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create single trainer (both pretrain and finetune use same model)
        cfg = self.trainer_config.model_copy(deep=True)
        cfg.multitask = False
        cfg.model.pretrained_path = None  # Start from scratch for both
        cfg.data.datasets = {dataset_name: ds_config}
        
        # Build two trainers with same init (for fair comparison)
        set_seeds(self.config.seed)
        pretrain_trainer = build_trainer(
            self.config.model_type.value, cfg, self.device
        )
        pretrain_trainer.collect_dataset_info(mixed=False, ds_name=dataset_name)
        pretrain_model = pretrain_trainer.setup_model()
        
        set_seeds(self.config.seed)  # Same initialization
        finetune_trainer = build_trainer(
            self.config.model_type.value, cfg, self.device
        )
        finetune_trainer.collect_dataset_info(mixed=False, ds_name=dataset_name)
        finetune_model = finetune_trainer.setup_model()
        
        train_loader, _ = pretrain_trainer.create_single_dataloader(
            dataset_name, ds_config, split=datasets.Split.TRAIN
        )
        
        pretrain_trainer.setup_optimizer_and_scheduler(pretrain_model, train_loader)
        finetune_trainer.setup_optimizer_and_scheduler(finetune_model, train_loader)
        
        # Get parameter groups (use separate groupers per model instance)
        pretrain_grouper = get_param_grouper(
            self.config.model_type.value,
            pretrain_trainer.model,
            self.config.grouping_strategy,
            MODEL_INNOVATION_GROUPS.get(self.config.model_type.value),
        )
        finetune_grouper = get_param_grouper(
            self.config.model_type.value,
            finetune_trainer.model,
            self.config.grouping_strategy,
            MODEL_INNOVATION_GROUPS.get(self.config.model_type.value),
        )
        
        # Log parameter grouper details
        self._log_param_grouper_info(pretrain_grouper, f"{dataset_name}/pretrain")
        self._log_param_grouper_info(finetune_grouper, f"{dataset_name}/finetune")
        
        groups = list(pretrain_grouper.group_names)
        if finetune_grouper.group_names != pretrain_grouper.group_names:
            extra_groups = [
                g for g in finetune_grouper.group_names if g not in groups
            ]
            if extra_groups:
                groups.extend(extra_groups)
            logger.warning(
                "Pretrain and finetune group names differ; using union of groups."
            )
        conditions = ["pretrain", "finetune"]
        
        # Feature collection setup (optional)
        feature_layers: List[str] = []
        actual_feature_layers: List[str] = []  # Actually registered layers
        pretrain_features: Dict[str, torch.Tensor] = {}
        pretrain_hooks: List[torch.utils.hooks.RemovableHandle] = []
        finetune_features: Dict[str, torch.Tensor] = {}
        finetune_hooks: List[torch.utils.hooks.RemovableHandle] = []
        probe_batches: List[Dict[str, Any]] = []

        if self.config.feature.enabled:
            logger.info("=" * 50)
            logger.info("Feature Collection Setup")
            logger.info("=" * 50)
            
            # Step 1: Get requested layers (from config or auto-detect)
            if self.config.feature.feature_layers:
                feature_layers = self.config.feature.feature_layers
                logger.info(f"Using configured feature_layers: {feature_layers}")
            else:
                logger.info("Auto-detecting feature layers...")
                feature_layers = self._auto_detect_feature_layers(pretrain_trainer.model)
            
            # Step 2: Register hooks on both models
            logger.info(f"\\n[Pretrain Model] Registering hooks...")
            pretrain_features, pretrain_hooks, pretrain_registered = self._register_feature_hooks(
                pretrain_trainer.model, feature_layers
            )
            logger.info(f"\\n[Finetune Model] Registering hooks...")
            finetune_features, finetune_hooks, finetune_registered = self._register_feature_hooks(
                finetune_trainer.model, feature_layers
            )
            
            # Step 3: Use union of actually registered layers for collector
            actual_feature_layers = list(dict.fromkeys(
                pretrain_registered + finetune_registered
            ))
            
            # Step 4: Log summary
            logger.info("\\n" + "-" * 50)
            logger.info("Feature Layer Summary:")
            logger.info(f"  Requested: {len(feature_layers)} layers")
            logger.info(f"  Pretrain registered: {len(pretrain_registered)} layers")
            logger.info(f"  Finetune registered: {len(finetune_registered)} layers")
            logger.info(f"  Final (union): {len(actual_feature_layers)} layers")
            if set(actual_feature_layers) != set(feature_layers):
                logger.info("  Note: Actual layers differ from requested (fallbacks applied)")
            logger.info("-" * 50)
            
            probe_batches = self._sample_probe_batches(
                train_loader,
                max(1, self.config.feature.probe_batches_per_condition),
            )

        # Create collector
        collector = self._create_collector(
            groups=groups,
            conditions=conditions,
            output_path=os.path.join(dataset_dir, "gradients"),
            layers=actual_feature_layers,  # Use actually registered layers
        )
        
        # Training history
        history = {
            "pretrain": {"loss": [], "grad_norm": [], "step": []},
            "finetune": {"loss": [], "grad_norm": [], "step": []},
        }
        
        # Training loop
        num_steps = self._resolve_num_steps(train_loader)
        self._reset_scheduler_for_analysis(pretrain_trainer, train_loader, num_steps)
        self._reset_scheduler_for_analysis(finetune_trainer, train_loader, num_steps)
        
        logger.info(
            f"Starting PRETRAIN_VS_FINETUNE for {dataset_name}: {num_steps} steps"
        )
        
        step = 0
        while step < num_steps:
            for _, batch in enumerate(train_loader):
                if step >= num_steps:
                    break

                # Current step (1-indexed for gradient collection)
                current_step = step + 1

                # Ensure encoder is unfrozen after warmup for analysis
                self._maybe_unfreeze_encoder_for_analysis(pretrain_trainer, current_step)
                self._maybe_unfreeze_encoder_for_analysis(finetune_trainer, current_step)

                # Gradient collection hook - uses captured current_step from outer scope
                def make_collect_hook(condition, collect_step, hook_grouper):
                    def hook(model, step_num, batch_data):
                        # Use the captured collect_step instead of step_num
                        # step_num from trainer.current_step may not be accurate
                        if collect_step < self.config.training.warmup_steps:
                            return
                        if collect_step % self.config.gradient.collect_interval != 0:
                            return
                        # Use collector.collect_gradients for consistency
                        collector.collect_gradients(
                            grouper=hook_grouper,
                            condition=condition,
                            step=collect_step,
                        )
                    return hook

                # Pretrain step (reconstruction)
                pre_loss, pre_grad = pretrain_trainer.pretrain_one_batch_for_analysis(
                    batch,
                    mask_ratio=self.mask_ratio,
                    mask_strategy=self.mask_strategy,
                    pre_step_hook=make_collect_hook(
                        "pretrain",
                        current_step,
                        pretrain_grouper,
                    ),
                )
                history["pretrain"]["loss"].append(pre_loss)
                history["pretrain"]["grad_norm"].append(pre_grad)
                history["pretrain"]["step"].append(current_step)

                # Finetune step (classification)
                ft_loss, ft_grad, _ = finetune_trainer.finetune_one_batch(
                    batch,
                    pre_step_hook=make_collect_hook(
                        "finetune",
                        current_step,
                        finetune_grouper,
                    )
                )
                history["finetune"]["loss"].append(ft_loss)
                history["finetune"]["grad_norm"].append(ft_grad)
                history["finetune"]["step"].append(current_step)

                if current_step % self.config.gradient.collect_interval == 0:
                    logger.info(
                        f"[{dataset_name}] Step {current_step}/{num_steps} | "
                        f"Pretrain: loss={pre_loss:.4f}, grad={pre_grad:.4f} | "
                        f"Finetune: loss={ft_loss:.4f}, grad={ft_grad:.4f}"
                    )
                    # Feature collection (with warmup check)
                    should_collect_features = (
                        self.config.feature.enabled 
                        and feature_layers
                        and current_step >= self.config.training.warmup_steps
                    )
                    if should_collect_features:
                        self._collect_feature_batches(
                            pretrain_trainer,
                            collector,
                            pretrain_features,
                            probe_batches,
                            "pretrain",
                            current_step,
                        )
                        self._collect_feature_batches(
                            finetune_trainer,
                            collector,
                            finetune_features,
                            probe_batches,
                            "finetune",
                            current_step,
                        )

                if (self.config.output.checkpoint_interval > 0 and
                    current_step % self.config.output.checkpoint_interval == 0):
                    collector.flush()

                step += 1
        
        # Finalize and save
        collector.finalize()

        if self.config.feature.enabled and feature_layers:
            self._remove_feature_hooks(pretrain_hooks, pretrain_features)
            self._remove_feature_hooks(finetune_hooks, finetune_features)
        
        # Save training history
        history_path = os.path.join(dataset_dir, "train_history.npz")
        np.savez(
            history_path,
            **{f"{k}_{m}": np.array(v) for k, hist in history.items() for m, v in hist.items()}
        )
        
        logger.info(f"Completed {dataset_name}, saved to {dataset_dir}")
        
        return {
            "dataset": dataset_name,
            "output_dir": dataset_dir,
            "num_steps": num_steps,
            "groups": groups,
            "conditions": conditions,
        }
    
    def run(self) -> Dict[str, Any]:
        """Run pretrain vs finetune analysis for all datasets.
        
        When multiple datasets are specified, this will run analysis on each
        dataset independently. Cross-dataset consistency analysis should be
        done in post-processing via visualize.py.
        
        Returns:
            Dict with per-dataset results
        """
        logger.info("=" * 60)
        logger.info("Starting PRETRAIN_VS_FINETUNE analysis")
        logger.info("=" * 60)
        
        try:
            results = {}
            for dataset_name in self.dataset_names:
                results[dataset_name] = self._run_single_dataset(dataset_name)
            
            # Note: Cross-dataset analysis moved to visualize.py for post-processing
            if len(self.dataset_names) >= 2:
                logger.info("Multiple datasets detected. Use visualize.py --cross-dataset for consistency analysis.")
            
            return results
        finally:
            self._cleanup_file_logging()


# =============================================================================
# Paradigm 3: Multi-Dataset Joint
# =============================================================================


class MultiDatasetJointRunner(BaseRunner):
    """Runner for analyzing gradient conflicts in multi-dataset joint training."""
    
    def __init__(
        self,
        config: AnalysisConfig,
        trainer_config: AbstractConfig,
        seed: Optional[int] = None,
        shared_run_dir: Optional[str] = None,
    ):
        super().__init__(config, seed=seed, shared_run_dir=shared_run_dir)
        self.trainer_config = trainer_config
        
        # Get dataset names
        self.dataset_names = (
            list(config.datasets.keys())
            if config.datasets else list(trainer_config.data.datasets.keys())
        )
        if len(self.dataset_names) < 2:
            raise ValueError("Multi-dataset analysis requires at least 2 datasets")
        
        # Pretrained checkpoint (optional)
        self.pretrained_ckpt = config.pretrained_checkpoint
    
    def _round_robin_iter(
        self,
        loaders: Dict[str, DataLoader],
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Infinite round-robin iterator over datasets."""
        iters = {name: enumerate(loader) for name, loader in loaders.items()}
        names = list(loaders.keys())
        idx = 0
        
        while True:
            name = names[idx % len(names)]
            try:
                _, batch = next(iters[name])
            except StopIteration:
                iters[name] = enumerate(loaders[name])
                _, batch = next(iters[name])
            
            yield name, batch
            idx += 1
    
    def run(self) -> Dict[str, Any]:
        """Run multi-dataset gradient conflict analysis."""
        logger.info("=" * 60)
        logger.info("Starting MULTI_DATASET_JOINT analysis")
        logger.info("=" * 60)
        
        # Create config for joint training
        cfg = self.trainer_config.model_copy(deep=True)
        cfg.multitask = True
        if self.pretrained_ckpt:
            cfg.model.pretrained_path = self.pretrained_ckpt
        
        # Filter datasets
        cfg.data.datasets = {
            k: v for k, v in cfg.data.datasets.items()
            if k in self.dataset_names
        }
        
        # Build trainer
        trainer = build_trainer(
            self.config.model_type.value, cfg, self.device
        )
        trainer.collect_dataset_info(mixed=True)
        model = trainer.setup_model()
        
        # Create per-dataset train loaders
        loaders: Dict[str, DataLoader] = {}
        for ds_name, ds_conf in cfg.data.datasets.items():
            loader, _ = trainer.create_single_dataloader(
                ds_name, ds_conf, datasets.Split.TRAIN
            )
            loaders[ds_name] = loader

        # Create eval/test probe loaders per dataset
        probe_loaders_by_ds: Dict[str, Dict[str, DataLoader]] = {}
        for ds_name, ds_conf in cfg.data.datasets.items():
            probe_loaders_by_ds[ds_name] = self._create_probe_loaders(
                trainer, ds_name, ds_conf
            )
        
        # Use first loader for optimizer setup
        trainer.setup_optimizer_and_scheduler(model, next(iter(loaders.values())))
        
        # Get parameter groups
        grouper = get_param_grouper(
            self.config.model_type.value,
            trainer.model,
            self.config.grouping_strategy,
            MODEL_INNOVATION_GROUPS.get(self.config.model_type.value),
        )
        
        # Log parameter grouper details
        self._log_param_grouper_info(grouper, "joint_training")
        
        groups = grouper.group_names
        conditions = self.dataset_names
        
        # Probe batches: for each dataset, sample from validation/test and merge
        # (we don't distinguish splits during probing, just need diverse eval data)
        probe_batch_count = max(1, self.config.gradient.probe_batches_per_condition)
        if self.config.feature.enabled:
            probe_batch_count = max(
                probe_batch_count, self.config.feature.probe_batches_per_condition
            )
        # Per-dataset merged probe batches
        probe_batches_by_ds: Dict[str, List[Dict[str, Any]]] = {}
        for ds_name, split_loaders in probe_loaders_by_ds.items():
            probe_batches_by_ds[ds_name] = []
            for split_name, loader in split_loaders.items():
                split_batches = self._sample_probe_batches(loader, probe_batch_count)
                probe_batches_by_ds[ds_name].extend(split_batches)
                logger.info(f"  [{ds_name}] Sampled {len(split_batches)} batches from {split_name}")
            logger.info(f"  [{ds_name}] Total probe batches: {len(probe_batches_by_ds[ds_name])}")

        # Feature collection setup (optional)
        feature_layers: List[str] = []
        actual_feature_layers: List[str] = []  # Actually registered layers
        current_features: Dict[str, torch.Tensor] = {}
        feature_hooks: List[torch.utils.hooks.RemovableHandle] = []
        if self.config.feature.enabled:
            logger.info("=" * 50)
            logger.info("Feature Collection Setup")
            logger.info("=" * 50)
            
            # Step 1: Get requested layers (from config or auto-detect)
            if self.config.feature.feature_layers:
                feature_layers = self.config.feature.feature_layers
                logger.info(f"Using configured feature_layers: {feature_layers}")
            else:
                logger.info("Auto-detecting feature layers...")
                feature_layers = self._auto_detect_feature_layers(trainer.model)
            
            # Step 2: Register hooks
            logger.info(f"\\n[Joint Model] Registering hooks...")
            current_features, feature_hooks, actual_feature_layers = self._register_feature_hooks(
                trainer.model, feature_layers
            )
            
            # Step 3: Log summary
            logger.info("\\n" + "-" * 50)
            logger.info("Feature Layer Summary:")
            logger.info(f"  Requested: {len(feature_layers)} layers")
            logger.info(f"  Registered: {len(actual_feature_layers)} layers")
            if set(actual_feature_layers) != set(feature_layers):
                logger.info("  Note: Actual layers differ from requested (fallbacks applied)")
            logger.info("-" * 50)

        # Create collector
        collector = self._create_collector(
            groups=groups,
            conditions=conditions,
            output_path=os.path.join(self.run_dir, "gradients"),
            layers=actual_feature_layers,  # Use actually registered layers
        )
        
        # Training history
        history = {
            ds: {"loss": [], "grad_norm": [], "step": []}
            for ds in self.dataset_names
        }
        
        # Training loop with round-robin
        num_steps = self._resolve_num_steps(next(iter(loaders.values())))
        self._reset_scheduler_for_analysis(
            trainer,
            next(iter(loaders.values())),
            num_steps,
        )
        probe_interval = self.config.gradient.collect_interval
        rr_iter = self._round_robin_iter(loaders)
        
        logger.info(
            f"Starting joint training: {num_steps} steps, "
            f"probe_interval={probe_interval}, datasets={self.dataset_names}"
        )
        
        for step in range(num_steps):
            ds_name, batch = next(rr_iter)

            # Ensure encoder is unfrozen after warmup for analysis
            self._maybe_unfreeze_encoder_for_analysis(trainer, step + 1)
            
            # Normal training step
            loss, grad_norm, _ = trainer.finetune_one_batch(batch)
            history[ds_name]["loss"].append(loss)
            history[ds_name]["grad_norm"].append(grad_norm)
            history[ds_name]["step"].append(step + 1)
            
            # Gradient probe at intervals (eval/test only)
            should_probe = (
                (step + 1) >= self.config.training.warmup_steps
                and (step + 1) % probe_interval == 0
            )
            if should_probe:
                logger.info(f"Step {step + 1}/{num_steps} - Running probe")
                
                # Probe each dataset (using merged batches, no split distinction)
                for probe_ds in self.dataset_names:
                    batches = probe_batches_by_ds.get(probe_ds, [])
                    if batches:
                        self._probe_gradients_and_metrics(
                            trainer,
                            grouper,
                            collector,
                            condition=probe_ds,
                            step=step + 1,
                            batches=batches,
                            dataset_name=probe_ds,
                            split_name="probe",  # unified name since splits are merged
                        )

                # Feature collection (using merged batches)
                if self.config.feature.enabled and feature_layers:
                    for probe_ds in self.dataset_names:
                        batches = probe_batches_by_ds.get(probe_ds, [])
                        if batches:
                            self._collect_feature_batches(
                                trainer,
                                collector,
                                current_features,
                                batches,
                                probe_ds,
                                step + 1,
                            )

            if (self.config.output.checkpoint_interval > 0 and
                (step + 1) % self.config.output.checkpoint_interval == 0):
                collector.flush()
        
        # Finalize and save
        collector.finalize()

        if self.config.feature.enabled and feature_layers:
            self._remove_feature_hooks(feature_hooks, current_features)
        
        # Save training history
        history_path = os.path.join(self.run_dir, "train_history.npz")
        np.savez(
            history_path,
            **{
                f"{ds}_{m}": np.array(v)
                for ds, hist in history.items()
                for m, v in hist.items()
            }
        )
        
        logger.info(f"Completed multi-dataset analysis, saved to {self.run_dir}")
        
        self._cleanup_file_logging()
        
        return {
            "output_dir": self.run_dir,
            "num_steps": num_steps,
            "datasets": self.dataset_names,
            "groups": groups,
            "conditions": conditions,
        }


# =============================================================================
# Main Entry Point
# =============================================================================


def load_analysis_config(
    config_path: Optional[str],
) -> AnalysisConfig:
    """Load analysis configuration from YAML with overrides."""
    base_cfg = AnalysisConfig()
    merged = OmegaConf.create(base_cfg.model_dump(mode="json"))
    
    if config_path:
        file_cfg = OmegaConf.load(config_path)
        merged = OmegaConf.merge(merged, file_cfg)

    config_dict = OmegaConf.to_container(merged, resolve=True)
    return AnalysisConfig.model_validate(config_dict)


def run_analysis_single(
    config: AnalysisConfig,
    trainer_config: AbstractConfig,
    seed: Optional[int] = None,
    shared_run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run gradient analysis for a single seed.
    
    Args:
        config: Analysis configuration
        trainer_config: Trainer configuration
        seed: Optional seed override (for multi-seed runs)
        shared_run_dir: Shared root directory for multi-seed runs
        
    Returns:
        Results dictionary
    """
    if config.paradigm == ExperimentParadigm.SCRATCH_VS_PRETRAINED:
        runner = ScratchVsPretrainedRunner(
            config, trainer_config, seed=seed, shared_run_dir=shared_run_dir
        )
    elif config.paradigm == ExperimentParadigm.PRETRAIN_VS_FINETUNE:
        runner = PretrainVsFinetuneRunner(
            config, trainer_config, seed=seed, shared_run_dir=shared_run_dir
        )
    elif config.paradigm == ExperimentParadigm.MULTI_DATASET_JOINT:
        runner = MultiDatasetJointRunner(
            config, trainer_config, seed=seed, shared_run_dir=shared_run_dir
        )
    else:
        raise ValueError(f"Unknown paradigm: {config.paradigm}")
    
    return runner.run()


def run_analysis(
    config: AnalysisConfig,
    trainer_config: AbstractConfig,
) -> Dict[str, Any]:
    """Run gradient analysis based on paradigm.
    
    If config.seeds is specified, runs analysis for each seed and aggregates results.
    Otherwise, runs a single analysis with config.seed.
    
    Directory structure:
        Single seed:  output_dir/{paradigm}_{timestamp}/{model_type}/...
        Multi-seed:   output_dir/{paradigm}_{timestamp}/{model_type}/seed_{N}/...
    
    Args:
        config: Analysis configuration
        trainer_config: Trainer configuration
        
    Returns:
        Results dictionary. For multi-seed runs, includes per-seed results.
    """
    seeds = config.seeds or [config.seed]
    
    if len(seeds) == 1:
        # Single seed run
        set_seeds(seeds[0])
        return run_analysis_single(config, trainer_config, seed=None, shared_run_dir=None)
    
    # Multi-seed runs: create shared root directory ONCE at start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_run_dir = os.path.join(
        config.output.output_dir,
        f"{config.paradigm.value}_{timestamp}",
    )
    os.makedirs(shared_run_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"Running multi-seed analysis with seeds: {seeds}")
    logger.info(f"Shared run directory: {shared_run_dir}")
    logger.info("=" * 60)
    
    all_results = {}
    for i, seed in enumerate(seeds):
        logger.info(f"[Seed {i+1}/{len(seeds)}] Starting analysis with seed={seed}")
        set_seeds(seed)

        try:
            results = run_analysis_single(
                config, trainer_config, seed=seed, shared_run_dir=shared_run_dir
            )
            all_results[f"seed_{seed}"] = results
            logger.info(f"[Seed {i+1}/{len(seeds)}] Completed successfully")
        except Exception as e:
            logger.error(f"[Seed {i+1}/{len(seeds)}] Failed with error: {e}")
            all_results[f"seed_{seed}"] = {"error": str(e)}
    
    # Summary
    successful = sum(1 for k, v in all_results.items() if "error" not in v)
    logger.info(f"Multi-seed analysis complete: {successful}/{len(seeds)} successful")
    logger.info(f"All results saved under: {shared_run_dir}")
    
    # Store shared run dir in results for visualization
    all_results["shared_run_dir"] = shared_run_dir
    
    return all_results
