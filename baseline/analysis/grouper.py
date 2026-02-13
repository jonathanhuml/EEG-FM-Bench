"""Parameter grouper base class for gradient analysis.

This module provides the abstract base class for grouping model parameters
into semantically meaningful groups for gradient analysis.

Parameter groups are essential for:
1. Analyzing which parts of the model receive the most gradient energy
2. Detecting gradient conflicts between datasets at different model components
3. Visualizing energy flow from datasets to parameter groups (Sankey diagrams)

Typical parameter groups:
- patch_embed / embed: Embedding layers (input projections)
- attention / attn: Self-attention components (Q, K, V, output projections)
- ffn / mlp: Feed-forward / MLP layers
- norm: Normalization layers (LayerNorm, BatchNorm)
- head: Task-specific heads (classification, reconstruction)
- other: Parameters that don't fit other categories
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

import torch.nn as nn


class LayerType(Enum):
    """Layer type for backward compatibility with old GradStore."""
    EMBED = auto()
    ATTENTION = auto()
    FFN = auto()
    NORM = auto()
    OTHER = auto()


class ParamGroupType(Enum):
    """Canonical parameter group types."""
    EMBED = auto()        # Embedding / patch embedding layers
    ATTENTION = auto()    # Self-attention components
    FFN = auto()          # Feed-forward / MLP layers
    NORM = auto()         # Normalization layers
    HEAD = auto()         # Task-specific heads
    OTHER = auto()        # Uncategorized parameters


@dataclass
class ParamGroup:
    """A group of related parameters."""
    name: str
    group_type: ParamGroupType
    param_names: List[str] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        """Ensure param_names is a list."""
        if not isinstance(self.param_names, list):
            self.param_names = list(self.param_names)


class EncoderParamGrouper(ABC):
    """Abstract base class for grouping encoder parameters.
    
    Each model should implement its own ParamGrouper that understands
    the model's architecture and can correctly categorize parameters.
    
    Usage:
        ```python
        grouper = CBraModParamGrouper(model)
        groups = grouper.get_groups()
        
        for name, param in model.named_parameters():
            group = grouper.get_param_group(name)
            print(f"{name} -> {group.name}")
        ```
    """
    
    # Subclasses should define these patterns
    EMBED_PATTERNS: List[str] = []      # Patterns matching embedding layers
    ATTENTION_PATTERNS: List[str] = []  # Patterns matching attention layers
    FFN_PATTERNS: List[str] = []        # Patterns matching FFN/MLP layers
    NORM_PATTERNS: List[str] = []       # Patterns matching normalization layers
    HEAD_PATTERNS: List[str] = []       # Patterns matching task heads
    
    def __init__(self, model: nn.Module):
        """Initialize grouper with model.
        
        Args:
            model: The model to group parameters for
        """
        self.model = model
        self._groups: Dict[str, ParamGroup] = {}
        self._param_to_group: Dict[str, str] = {}
        self._innovation_groups: Optional[Dict[str, List[str]]] = None
        self._grouped_params_cache: Optional[Dict[str, List[nn.Parameter]]] = None
        self._setup_groups()
        print(self.summary())
    
    @abstractmethod
    def _setup_groups(self):
        """Set up parameter groups based on model architecture.
        
        Subclasses must implement this to:
        1. Create ParamGroup objects for each semantic group
        2. Populate self._groups with group_name -> ParamGroup mapping
        3. Populate self._param_to_group with param_name -> group_name mapping
        """
        raise NotImplementedError
    
    @staticmethod
    def match_pattern(param_name: str, patterns: List[str]) -> bool:
        """Check if parameter name matches any pattern.
        
        Args:
            param_name: Full parameter name (e.g., "encoder.layer.0.attn.qkv.weight")
            patterns: List of substring patterns to match
            
        Returns:
            True if any pattern matches
        """
        name_lower = param_name.lower()
        return any(p.lower() in name_lower for p in patterns)
    
    def _infer_group_type(self, param_name: str) -> ParamGroupType:
        """Infer group type from parameter name using patterns.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Inferred ParamGroupType
        """
        if self.match_pattern(param_name, self.EMBED_PATTERNS):
            return ParamGroupType.EMBED
        if self.match_pattern(param_name, self.ATTENTION_PATTERNS):
            return ParamGroupType.ATTENTION
        if self.match_pattern(param_name, self.FFN_PATTERNS):
            return ParamGroupType.FFN
        if self.match_pattern(param_name, self.NORM_PATTERNS):
            return ParamGroupType.NORM
        if self.match_pattern(param_name, self.HEAD_PATTERNS):
            return ParamGroupType.HEAD
        return ParamGroupType.OTHER
    
    def get_groups(self) -> Dict[str, ParamGroup]:
        """Get all parameter groups.
        
        Returns:
            Dict mapping group names to ParamGroup objects
        """
        return self._groups.copy()
    
    def get_group_names(self) -> List[str]:
        """Get ordered list of group names.
        
        Returns:
            List of group names in canonical order
        """
        # Return in canonical order
        type_order = [
            ParamGroupType.EMBED,
            ParamGroupType.ATTENTION,
            ParamGroupType.FFN,
            ParamGroupType.NORM,
            ParamGroupType.HEAD,
            ParamGroupType.OTHER,
        ]
        
        groups_by_type: Dict[ParamGroupType, List[str]] = {t: [] for t in type_order}
        for name, group in self._groups.items():
            groups_by_type[group.group_type].append(name)
        
        result = []
        for t in type_order:
            result.extend(sorted(groups_by_type[t]))
        return result
    
    def get_param_group(self, param_name: str) -> Optional[ParamGroup]:
        """Get the group for a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            ParamGroup object or None if not found
        """
        group_name = self._param_to_group.get(param_name)
        return self._groups.get(group_name) if group_name else None
    
    def get_param_group_name(self, param_name: str) -> str:
        """Get the group name for a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Group name or "other" if not found
        """
        return self._param_to_group.get(param_name, "other")
    
    def iter_grouped_params(
        self
    ) -> Iterator[Tuple[str, ParamGroup, nn.Parameter]]:
        """Iterate over parameters with their groups.
        
        Yields:
            (param_name, ParamGroup, parameter) tuples
        """
        for name, param in self.model.named_parameters():
            group = self.get_param_group(name)
            if group is not None:
                yield name, group, param
    
    def get_params_by_group(
        self,
        group_name: str
    ) -> Dict[str, nn.Parameter]:
        """Get all parameters in a group.
        
        Args:
            group_name: Name of the group
            
        Returns:
            Dict mapping param names to parameters
        """
        group = self._groups.get(group_name)
        if group is None:
            return {}
        
        result = {}
        for name, param in self.model.named_parameters():
            if name in group.param_names:
                result[name] = param
        return result

    def get_params_in_group(self, group_name: str) -> List[str]:
        """Get parameter names in a group.

        Args:
            group_name: Name of the group

        Returns:
            List of parameter names in the group
        """
        group = self._groups.get(group_name)
        if group is None:
            return []
        return list(group.param_names)
    
    def get_group_param_count(self, group_name: str) -> int:
        """Get number of parameters in a group.
        
        Args:
            group_name: Name of the group
            
        Returns:
            Total number of scalar parameters
        """
        params = self.get_params_by_group(group_name)
        return sum(p.numel() for p in params.values())
    
    def summary(self) -> str:
        """Get summary of parameter groups.
        
        Returns:
            Human-readable summary string
        """
        lines = ["Parameter Groups Summary:", "=" * 50]
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        for group_name in self.get_group_names():
            group = self._groups[group_name]
            count = self.get_group_param_count(group_name)
            pct = 100 * count / total_params if total_params > 0 else 0
            lines.append(
                f"  {group_name:20s} [{group.group_type.name:10s}]: "
                f"{count:>12,d} params ({pct:5.1f}%)"
            )
        
        lines.append("=" * 50)
        lines.append(f"  {'TOTAL':20s} {'':10s}  {total_params:>12,d} params")
        
        return "\n".join(lines)
    
    def to_layer_types(self) -> Dict[str, LayerType]:
        """Convert groups to LayerType mapping for GradStore (backward compatibility).
        
        Returns:
            Dict mapping param_name to LayerType
        """
        
        type_mapping = {
            ParamGroupType.EMBED: LayerType.EMBED,
            ParamGroupType.ATTENTION: LayerType.ATTENTION,
            ParamGroupType.FFN: LayerType.FFN,
            ParamGroupType.NORM: LayerType.NORM,
            ParamGroupType.HEAD: LayerType.OTHER,
            ParamGroupType.OTHER: LayerType.OTHER,
        }
        
        result = {}
        for name, group in self._groups.items():
            layer_type = type_mapping.get(group.group_type, LayerType.OTHER)
            for param_name in group.param_names:
                result[param_name] = layer_type
        
        return result
    
    # =========================================================================
    # Innovation-based grouping support
    # =========================================================================
    
    def set_innovation_groups(self, innovation_groups: Dict[str, List[str]]):
        """Set innovation-based parameter groups.
        
        This allows grouping parameters by model-specific architectural innovations
        (e.g., "criss_cross_attn" for CBraMod, "vq_tokenizer" for LaBraM).
        
        Args:
            innovation_groups: Dict mapping innovation group names to parameter
                             name patterns. E.g.:
                             {
                                 "cross_scale_attn": ["cross_scale", "region_attn"],
                                 "vq_tokenizer": ["vq", "codebook", "quantize"],
                             }
        """
        self._innovation_groups = innovation_groups
        self._grouped_params_cache = None  # Invalidate cache
    
    @property
    def group_names(self) -> List[str]:
        """Get list of current group names (considering innovation groups if set)."""
        if self._innovation_groups:
            return list(self._innovation_groups.keys()) + ["other_innovation"]
        return self.get_group_names()
    
    @property
    def grouped_params(self) -> Dict[str, List[nn.Parameter]]:
        """Get parameters grouped by current grouping strategy.
        
        Returns:
            Dict mapping group names to lists of parameters
        """
        if self._grouped_params_cache is not None:
            return self._grouped_params_cache
        
        if self._innovation_groups:
            result = self._build_innovation_grouped_params()
        else:
            result = self._build_type_grouped_params()
        
        self._grouped_params_cache = result
        return result
    
    def _build_type_grouped_params(self) -> Dict[str, List[nn.Parameter]]:
        """Build grouped params using type-based grouping."""
        result: Dict[str, List[nn.Parameter]] = {
            name: [] for name in self.get_group_names()
        }
        
        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            group_name = self.get_param_group_name(param_name)
            if group_name in result:
                result[group_name].append(param)
            else:
                if "other" not in result:
                    result["other"] = []
                result["other"].append(param)
        
        return result
    
    def _build_innovation_grouped_params(self) -> Dict[str, List[nn.Parameter]]:
        """Build grouped params using innovation-based grouping."""
        if not self._innovation_groups:
            return self._build_type_grouped_params()
        
        result: Dict[str, List[nn.Parameter]] = {
            name: [] for name in self._innovation_groups.keys()
        }
        result["other_innovation"] = []
        
        assigned_params = set()
        
        # Match parameters to innovation groups
        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            matched = False
            for group_name, patterns in self._innovation_groups.items():
                if self.match_pattern(param_name, patterns):
                    result[group_name].append(param)
                    assigned_params.add(param_name)
                    matched = True
                    break
            
            if not matched:
                result["other_innovation"].append(param)
        
        # Remove empty groups
        result = {k: v for k, v in result.items() if v}
        
        return result
    
    def get_innovation_group_summary(self) -> str:
        """Get summary of innovation-based groups.
        
        Returns:
            Human-readable summary of innovation groups
        """
        if not self._innovation_groups:
            return "No innovation groups set. Using type-based grouping."
        
        lines = ["Innovation-based Parameter Groups:", "=" * 50]
        
        grouped = self.grouped_params
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        for group_name in list(self._innovation_groups.keys()) + ["other_innovation"]:
            if group_name not in grouped:
                continue
            params = grouped[group_name]
            count = sum(p.numel() for p in params)
            pct = 100 * count / total_params if total_params > 0 else 0
            patterns = self._innovation_groups.get(group_name, ["(unmatched)"])
            lines.append(
                f"  {group_name:25s}: {count:>12,d} params ({pct:5.1f}%) "
                f"[patterns: {', '.join(patterns[:3])}{'...' if len(patterns) > 3 else ''}]"
            )
        
        lines.append("=" * 50)
        lines.append(f"  {'TOTAL':25s}  {total_params:>12,d} params")
        
        return "\n".join(lines)


class DefaultParamGrouper(EncoderParamGrouper):
    """Default parameter grouper using common naming patterns.
    
    This grouper uses heuristic pattern matching and works for most
    transformer-based models. For model-specific grouping, subclass
    EncoderParamGrouper and implement _setup_groups.
    """
    
    EMBED_PATTERNS = [
        'embed', 'patch_embed', 'cls_token', 'pos_embed',
        'position', 'input_proj', 'token_embed',
    ]
    
    ATTENTION_PATTERNS = [
        'attn', 'attention', 'self_attn', 'mha',
        'q_proj', 'k_proj', 'v_proj', 'out_proj',
        'qkv', 'query', 'key', 'value',
    ]
    
    FFN_PATTERNS = [
        'mlp', 'ffn', 'feed_forward', 'fc1', 'fc2',
        'linear1', 'linear2', 'dense', 'intermediate',
    ]
    
    NORM_PATTERNS = [
        'norm', 'ln', 'layer_norm', 'layernorm',
        'batch_norm', 'bn', 'group_norm',
    ]
    
    HEAD_PATTERNS = [
        'head', 'classifier', 'cls_head', 'output_proj',
        'decoder', 'prediction',
    ]
    
    def _setup_groups(self):
        """Set up groups using pattern matching."""
        # Initialize groups
        group_types = [
            ("embed", ParamGroupType.EMBED, "Embedding layers"),
            ("attention", ParamGroupType.ATTENTION, "Self-attention layers"),
            ("ffn", ParamGroupType.FFN, "Feed-forward layers"),
            ("norm", ParamGroupType.NORM, "Normalization layers"),
            ("head", ParamGroupType.HEAD, "Task heads"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in group_types:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        # Assign parameters to groups
        for param_name, _ in self.model.named_parameters():
            gtype = self._infer_group_type(param_name)
            
            # Map ParamGroupType to group name
            type_to_name = {
                ParamGroupType.EMBED: "embed",
                ParamGroupType.ATTENTION: "attention",
                ParamGroupType.FFN: "ffn",
                ParamGroupType.NORM: "norm",
                ParamGroupType.HEAD: "head",
                ParamGroupType.OTHER: "other",
            }
            
            group_name = type_to_name[gtype]
            self._groups[group_name].param_names.append(param_name)
            self._param_to_group[param_name] = group_name
