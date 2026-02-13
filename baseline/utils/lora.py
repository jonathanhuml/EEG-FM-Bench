"""
LoRA (Low-Rank Adaptation) implementation for baseline models.

This module provides a PyTorch native implementation of LoRA that can be applied
to nn.Linear and nn.MultiheadAttention layers in transformer-based models.

Features:
- Support for nn.Linear layers
- Support for nn.MultiheadAttention (in_proj and out_proj)
- Two scope levels: "full" (all layers) and "transformer" (only Transformer blocks)
- Automatic target module detection based on model type

Reference: https://arxiv.org/abs/2106.09685
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("baseline")


#################################################################################
#                              LoRA Core Layers                                  #
#################################################################################


class LoRALayer(nn.Module):
    """
    Base LoRA layer that can be applied to any linear transformation.
    
    LoRA decomposes weight updates as: W' = W + BA
    where B ∈ R^(d×r) and A ∈ R^(r×k), with rank r << min(d, k)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA matrices using Kaiming uniform for A and zeros for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA contribution: x @ A^T @ B^T * scaling"""
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This wraps an existing nn.Linear layer and adds LoRA adaptation.
    The original weights are frozen by default.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.merge_weights = merge_weights
        self.merged = False
        
        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        else:
            return self.base_layer(x) + self.lora(x)
    
    def merge(self):
        """Merge LoRA weights into base layer weights."""
        if not self.merged:
            # W' = W + B @ A * scaling
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base layer weights."""
        if self.merged:
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False
    
    @property
    def weight(self) -> torch.Tensor:
        """Get effective weight (base + LoRA if not merged)."""
        if self.merged:
            return self.base_layer.weight
        else:
            delta_w = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            return self.base_layer.weight + delta_w
    
    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Get bias from base layer."""
        return self.base_layer.bias


class LoRAMultiheadAttention(nn.Module):
    """
    MultiheadAttention with LoRA adaptation.
    
    This wraps an existing nn.MultiheadAttention and adds LoRA to:
    - in_proj (Q, K, V projection combined or separate)
    - out_proj (output projection)
    
    The original weights are frozen by default.
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        base_layer: nn.MultiheadAttention,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_in_proj: bool = True,
        lora_out_proj: bool = True,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.embed_dim: int = base_layer.embed_dim
        self.num_heads: int = base_layer.num_heads
        self.head_dim: int = base_layer.head_dim
        self.batch_first: bool = base_layer.batch_first
        self.merged: bool = False
        
        self.lora_in_proj = None
        self.lora_out_proj = None
        
        # Freeze base layer weights
        if base_layer.in_proj_weight is not None:
            base_layer.in_proj_weight.requires_grad = False
        if base_layer.in_proj_bias is not None:
            base_layer.in_proj_bias.requires_grad = False
        if base_layer.q_proj_weight is not None:
            base_layer.q_proj_weight.requires_grad = False
        if base_layer.k_proj_weight is not None:
            base_layer.k_proj_weight.requires_grad = False
        if base_layer.v_proj_weight is not None:
            base_layer.v_proj_weight.requires_grad = False
        if base_layer.bias_k is not None:
            base_layer.bias_k.requires_grad = False
        if base_layer.bias_v is not None:
            base_layer.bias_v.requires_grad = False
        base_layer.out_proj.weight.requires_grad = False
        if base_layer.out_proj.bias is not None:
            base_layer.out_proj.bias.requires_grad = False
        
        # Create LoRA for in_proj (combined QKV projection)
        if lora_in_proj and base_layer.in_proj_weight is not None:
            # in_proj_weight shape: (3 * embed_dim, embed_dim)
            self.lora_in_proj = LoRALayer(
                in_features=self.embed_dim,
                out_features=3 * self.embed_dim,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        
        # Create LoRA for out_proj
        if lora_out_proj:
            self.lora_out_proj = LoRALayer(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

    def _get_in_proj_weight(self) -> Optional[torch.Tensor]:
        if self.lora_in_proj is None or self.base_layer.in_proj_weight is None:
            return self.base_layer.in_proj_weight
        delta_w = (self.lora_in_proj.lora_B @ self.lora_in_proj.lora_A) * self.lora_in_proj.scaling
        return self.base_layer.in_proj_weight + delta_w

    def merge(self):
        """Merge LoRA weights into base layer weights."""
        if self.merged:
            return
        if self.lora_in_proj is not None and self.base_layer.in_proj_weight is not None:
            delta_w = (self.lora_in_proj.lora_B @ self.lora_in_proj.lora_A) * self.lora_in_proj.scaling
            self.base_layer.in_proj_weight.data += delta_w
        if self.lora_out_proj is not None:
            delta_w = (self.lora_out_proj.lora_B @ self.lora_out_proj.lora_A) * self.lora_out_proj.scaling
            self.base_layer.out_proj.weight.data += delta_w
        self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from base layer weights."""
        if not self.merged:
            return
        if self.lora_in_proj is not None and self.base_layer.in_proj_weight is not None:
            delta_w = (self.lora_in_proj.lora_B @ self.lora_in_proj.lora_A) * self.lora_in_proj.scaling
            self.base_layer.in_proj_weight.data -= delta_w
        if self.lora_out_proj is not None:
            delta_w = (self.lora_out_proj.lora_B @ self.lora_out_proj.lora_A) * self.lora_out_proj.scaling
            self.base_layer.out_proj.weight.data -= delta_w
        self.merged = False
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with LoRA adaptation."""
        if self.merged:
            return self.base_layer(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        # Match MultiheadAttention behavior for batch_first
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        in_proj_weight = self._get_in_proj_weight()
        use_separate_proj_weight = self.base_layer.in_proj_weight is None

        kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=self.base_layer.in_proj_bias,
            bias_k=self.base_layer.bias_k,
            bias_v=self.base_layer.bias_v,
            add_zero_attn=self.base_layer.add_zero_attn,
            dropout_p=self.base_layer.dropout,
            out_proj_weight=self.base_layer.out_proj.weight,
            out_proj_bias=self.base_layer.out_proj.bias,
            training=self.base_layer.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )

        if use_separate_proj_weight:
            kwargs.update(
                use_separate_proj_weight=True,
                q_proj_weight=self.base_layer.q_proj_weight,
                k_proj_weight=self.base_layer.k_proj_weight,
                v_proj_weight=self.base_layer.v_proj_weight,
            )

        try:
            attn_output, attn_weights = F.multi_head_attention_forward(
                **kwargs,
                is_causal=is_causal,
            )
        except TypeError:
            attn_output, attn_weights = F.multi_head_attention_forward(
                **kwargs,
            )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # Add LoRA contribution for out_proj
        if self.lora_out_proj is not None:
            attn_output = attn_output + self.lora_out_proj(attn_output)

        return attn_output, attn_weights
    
    def __getattr__(self, name: str):
        """Forward attribute access to base layer for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


# Common LoRA target module patterns for different model architectures
COMMON_TARGET_MODULES = {
    # Attention projections (most common patterns)
    "qkv": r".*\.qkv$",  # LaBraM style: combined QKV projection
    "q_proj": r".*\.q_proj$",  # Separate Q projection
    "k_proj": r".*\.k_proj$",  # Separate K projection
    "v_proj": r".*\.v_proj$",  # Separate V projection
    "o_proj": r".*\.o_proj$",  # Output projection (Time-MoE)
    "proj": r".*\.proj$",  # General projection (EEGPT)
    "out_proj": r".*\.out_proj$",  # Output projection
    
    # FFN layers
    "fc1": r".*\.fc1$",  # First FFN layer
    "fc2": r".*\.fc2$",  # Second FFN layer
    "gate_proj": r".*\.gate_proj$",  # Gate projection (SwiGLU)
    "up_proj": r".*\.up_proj$",  # Up projection (SwiGLU)
    "down_proj": r".*\.down_proj$",  # Down projection (SwiGLU)
    
    # MLP layers
    "linear1": r".*\.linear1$",  # CBraMod style FFN
    "linear2": r".*\.linear2$",  # CBraMod style FFN
    
    # MultiheadAttention (for CBraMod etc.)
    "self_attn": r".*\.self_attn$",  # Self attention modules
    "self_attn_s": r".*\.self_attn_s$",  # Spatial self attention (CBraMod)
    "self_attn_t": r".*\.self_attn_t$",  # Temporal self attention (CBraMod)
    
    # MoE expert layers
    "expert_gate": r".*\.experts\.\d+\.gate_proj$",  # MoE expert gate
    "expert_up": r".*\.experts\.\d+\.up_proj$",  # MoE expert up
    "expert_down": r".*\.experts\.\d+\.down_proj$",  # MoE expert down
}

# Patterns to identify Transformer block/layer modules (for scope filtering)
TRANSFORMER_BLOCK_PATTERNS = [
    r"^blocks\.\d+\..*",              # Top-level blocks: blocks.0.attn
    r".*\.blocks\.\d+\..*",           # LaBraM, EEGPT style: model.blocks.0.attn
    r"^layers\.\d+\..*",              # Top-level layers: layers.0.self_attn
    r".*\.layers\.\d+\..*",           # Time-MoE, Reve, CBraMod style: model.layers.0.self_attn
    r".*\.encoder\.layers\.\d+\..*",  # Some encoders: encoder.layers.0
    r".*\.decoder\.layers\.\d+\..*",  # Decoder layers
    r".*\.encoder\.block\.\d+\..*",   # T5 encoder blocks: encoder.block.0.layer.0.SelfAttention
    r".*\.decoder\.block\.\d+\..*",   # T5 decoder blocks (if any)
    r".*\.block\.\d+\..*",            # Generic T5-style blocks: block.0.layer.0
    r".*\.DenseReluDense\..*",         # T5 FFN submodules
    r".*\.layer\.1\..*",               # FFN layer in T5 blocks
    r".*Block\d*\..*",                # Named blocks: Block0, TransformerBlock
    r".*Layer\d*\..*",                # Named layers: Layer0, DecoderLayer
    r".*\.transformer\.\d+\..*",      # Transformer indexed layers
    r".*\.attn\..*",                  # Attention sub-modules
    r".*\.mlp\..*",                   # MLP sub-modules in blocks
    r".*\.ffn.*\..*",                 # FFN sub-modules
]

# Default target modules for common use cases
DEFAULT_ATTENTION_MODULES = ["qkv", "q_proj", "k_proj", "v_proj", "o_proj", "proj", "out_proj"]
DEFAULT_FFN_MODULES = ["fc1", "fc2", "gate_proj", "up_proj", "down_proj", "linear1", "linear2"]
DEFAULT_MHA_MODULES = ["self_attn", "self_attn_s", "self_attn_t"]
DEFAULT_ALL_MODULES = DEFAULT_ATTENTION_MODULES + DEFAULT_FFN_MODULES + DEFAULT_MHA_MODULES


def get_target_module_pattern(module_names: List[str]) -> str:
    """
    Convert module names to regex pattern.
    
    Args:
        module_names: List of module name patterns (can be from COMMON_TARGET_MODULES keys
                     or custom regex patterns)
    
    Returns:
        Combined regex pattern for matching module names
    """
    patterns = []
    for name in module_names:
        if name in COMMON_TARGET_MODULES:
            patterns.append(COMMON_TARGET_MODULES[name])
        else:
            # Treat as custom pattern - add .* prefix if not already a regex
            if not name.startswith(".*"):
                patterns.append(f".*\\.{re.escape(name)}$")
            else:
                patterns.append(name)
    
    return "|".join(patterns) if patterns else ""


def is_in_transformer_block(module_path: str) -> bool:
    """
    Check if a module path is inside a Transformer block/layer.
    
    Args:
        module_path: Full path to the module (e.g., "encoder.blocks.0.attn.qkv")
    
    Returns:
        True if the module is inside a Transformer block
    """
    for pattern in TRANSFORMER_BLOCK_PATTERNS:
        if re.match(pattern, module_path):
            return True
    return False


def find_target_modules(
    model: nn.Module,
    target_modules: Union[List[str], str],
    exclude_modules: Optional[List[str]] = None,
    scope: str = "transformer",
) -> Tuple[Dict[str, nn.Linear], Dict[str, nn.MultiheadAttention]]:
    """
    Find all Linear and MultiheadAttention modules that match the target patterns.
    
    Args:
        model: The model to search
        target_modules: List of module name patterns or regex string
        exclude_modules: List of module name patterns to exclude
        scope: "full" (all matching layers) or "transformer" (only in Transformer blocks)
    
    Returns:
        Tuple of (dict of Linear layers, dict of MultiheadAttention layers)
    """
    if isinstance(target_modules, list):
        pattern = get_target_module_pattern(target_modules)
    else:
        pattern = target_modules
    
    exclude_pattern = None
    if exclude_modules:
        if isinstance(exclude_modules, list):
            exclude_pattern = get_target_module_pattern(exclude_modules)
        else:
            exclude_pattern = exclude_modules
    
    linear_layers = {}
    mha_layers = {}
    
    for name, module in model.named_modules():
        # Check scope filter
        if scope == "transformer" and not is_in_transformer_block(name):
            continue
        
        # Check exclude pattern
        if exclude_pattern and re.match(exclude_pattern, name):
            continue
        
        # Check if matches target pattern
        if not re.match(pattern, name):
            continue
        
        if isinstance(module, nn.Linear):
            linear_layers[name] = module
        elif isinstance(module, nn.MultiheadAttention):
            mha_layers[name] = module
    
    return linear_layers, mha_layers


def inject_lora(
    model: nn.Module,
    target_modules: Union[List[str], str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    exclude_modules: Optional[List[str]] = None,
    scope: str = "transformer",
    verbose: bool = True,
) -> Tuple[nn.Module, List[str]]:
    """
    Inject LoRA layers into the model's target modules.
    
    This function replaces specified nn.Linear and nn.MultiheadAttention layers
    with LoRA-enabled versions, freezing the original weights and adding 
    trainable LoRA parameters.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns (from COMMON_TARGET_MODULES keys
                       or custom patterns) or a regex string
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout rate for LoRA layers
        exclude_modules: List of module name patterns to exclude
        scope: "full" (all matching layers) or "transformer" (only in Transformer blocks)
        verbose: Whether to log injection details
    
    Returns:
        Tuple of (modified model, list of injected module paths)
    
    Example:
        # Inject LoRA into attention layers (transformer scope)
        model, lora_modules = inject_lora(
            model,
            target_modules=["qkv", "proj"],  # Use predefined patterns
            r=8,
            lora_alpha=16,
            scope="transformer",  # Only in Transformer blocks
        )
        
        # Inject LoRA into all layers (full scope)
        model, lora_modules = inject_lora(
            model,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            r=8,
            scope="full",  # All matching layers
        )
    """
    # Find target modules (both Linear and MHA)
    linear_layers, mha_layers = find_target_modules(
        model, target_modules, exclude_modules, scope
    )
    
    if not linear_layers and not mha_layers:
        if verbose:
            logger.warning(f"No matching modules found for patterns: {target_modules} (scope={scope})")
        return model, []
    
    injected_modules = []
    
    # Replace each Linear layer with LoRALinear
    for module_path, linear_layer in linear_layers.items():
        lora_layer = LoRALinear(
            base_layer=linear_layer,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Navigate to parent module and replace
        path_parts = module_path.split(".")
        parent = model
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, path_parts[-1], lora_layer)
        injected_modules.append(module_path)
    
    # Replace each MultiheadAttention layer with LoRAMultiheadAttention
    for module_path, mha_layer in mha_layers.items():
        lora_mha = LoRAMultiheadAttention(
            base_layer=mha_layer,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        # Navigate to parent module and replace
        path_parts = module_path.split(".")
        parent = model
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, path_parts[-1], lora_mha)
        injected_modules.append(module_path)
    
    if verbose:
        total_lora_params = count_lora_parameters(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"LoRA injection complete:")
        logger.info(f"  - Scope: {scope}")
        logger.info(f"  - Injected {len(linear_layers)} Linear modules")
        logger.info(f"  - Injected {len(mha_layers)} MultiheadAttention modules")
        logger.info(f"  - LoRA rank: {r}, alpha: {lora_alpha}")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - LoRA parameters: {total_lora_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        if verbose and len(injected_modules) <= 20:
            logger.info(f"  - Modules: {injected_modules}")
    
    return model, injected_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from the model."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora.lora_A, module.lora.lora_B])
        elif isinstance(module, LoRAMultiheadAttention):
            if module.lora_in_proj is not None:
                lora_params.extend([module.lora_in_proj.lora_A, module.lora_in_proj.lora_B])
            if module.lora_out_proj is not None:
                lora_params.extend([module.lora_out_proj.lora_A, module.lora_out_proj.lora_B])
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.
    
    Args:
        model: Model with LoRA layers
    
    Returns:
        State dict with only LoRA parameters (lora_A and lora_B)
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora.lora_A"] = module.lora.lora_A.data.clone()
            lora_state_dict[f"{name}.lora.lora_B"] = module.lora.lora_B.data.clone()
        elif isinstance(module, LoRAMultiheadAttention):
            if module.lora_in_proj is not None:
                lora_state_dict[f"{name}.lora_in_proj.lora_A"] = module.lora_in_proj.lora_A.data.clone()
                lora_state_dict[f"{name}.lora_in_proj.lora_B"] = module.lora_in_proj.lora_B.data.clone()
            if module.lora_out_proj is not None:
                lora_state_dict[f"{name}.lora_out_proj.lora_A"] = module.lora_out_proj.lora_A.data.clone()
                lora_state_dict[f"{name}.lora_out_proj.lora_B"] = module.lora_out_proj.lora_B.data.clone()
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load LoRA parameters from state dict.
    
    Args:
        model: Model with LoRA layers
        lora_state_dict: State dict containing LoRA parameters
        strict: Whether to strictly enforce that the keys match
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    missing_keys = []
    unexpected_keys = list(lora_state_dict.keys())
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora.lora_A"
            b_key = f"{name}.lora.lora_B"
            
            if a_key in lora_state_dict:
                module.lora.lora_A.data.copy_(lora_state_dict[a_key])
                unexpected_keys.remove(a_key)
            elif strict:
                missing_keys.append(a_key)
            
            if b_key in lora_state_dict:
                module.lora.lora_B.data.copy_(lora_state_dict[b_key])
                unexpected_keys.remove(b_key)
            elif strict:
                missing_keys.append(b_key)
        
        elif isinstance(module, LoRAMultiheadAttention):
            if module.lora_in_proj is not None:
                in_a_key = f"{name}.lora_in_proj.lora_A"
                in_b_key = f"{name}.lora_in_proj.lora_B"

                if in_a_key in lora_state_dict:
                    module.lora_in_proj.lora_A.data.copy_(lora_state_dict[in_a_key])
                    unexpected_keys.remove(in_a_key)
                elif strict:
                    missing_keys.append(in_a_key)

                if in_b_key in lora_state_dict:
                    module.lora_in_proj.lora_B.data.copy_(lora_state_dict[in_b_key])
                    unexpected_keys.remove(in_b_key)
                elif strict:
                    missing_keys.append(in_b_key)

            if module.lora_out_proj is not None:
                out_a_key = f"{name}.lora_out_proj.lora_A"
                out_b_key = f"{name}.lora_out_proj.lora_B"

                if out_a_key in lora_state_dict:
                    module.lora_out_proj.lora_A.data.copy_(lora_state_dict[out_a_key])
                    unexpected_keys.remove(out_a_key)
                elif strict:
                    missing_keys.append(out_a_key)

                if out_b_key in lora_state_dict:
                    module.lora_out_proj.lora_B.data.copy_(lora_state_dict[out_b_key])
                    unexpected_keys.remove(out_b_key)
                elif strict:
                    missing_keys.append(out_b_key)
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Error loading LoRA state dict:\n"
            f"  Missing keys: {missing_keys}\n"
            f"  Unexpected keys: {unexpected_keys}"
        )
    
    return missing_keys, unexpected_keys


def count_lora_parameters(model: nn.Module) -> int:
    """Count total number of LoRA parameters in the model."""
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora.lora_A.numel() + module.lora.lora_B.numel()
        elif isinstance(module, LoRAMultiheadAttention):
            if module.lora_in_proj is not None:
                total += module.lora_in_proj.lora_A.numel() + module.lora_in_proj.lora_B.numel()
            if module.lora_out_proj is not None:
                total += module.lora_out_proj.lora_A.numel() + module.lora_out_proj.lora_B.numel()
    return total


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base layer weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
        elif isinstance(module, LoRAMultiheadAttention):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights from base layer weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
        elif isinstance(module, LoRAMultiheadAttention):
            module.unmerge()


def set_lora_trainable(model: nn.Module, trainable: bool = True):
    """Set LoRA parameters trainable/frozen."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora.lora_A.requires_grad = trainable
            module.lora.lora_B.requires_grad = trainable
        elif isinstance(module, LoRAMultiheadAttention):
            if module.lora_in_proj is not None:
                module.lora_in_proj.lora_A.requires_grad = trainable
                module.lora_in_proj.lora_B.requires_grad = trainable
            if module.lora_out_proj is not None:
                module.lora_out_proj.lora_A.requires_grad = trainable
                module.lora_out_proj.lora_B.requires_grad = trainable


def freeze_non_lora_parameters(model: nn.Module):
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False


def print_lora_modules(model: nn.Module):
    """Print all LoRA modules in the model."""
    print("LoRA Modules:")
    print("-" * 60)
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"  {name}: Linear in={module.in_features}, out={module.out_features}, "
                  f"r={module.lora.r}, alpha={module.lora.lora_alpha}")
        elif isinstance(module, LoRAMultiheadAttention):
            in_r = module.lora_in_proj.r if module.lora_in_proj is not None else None
            in_alpha = module.lora_in_proj.lora_alpha if module.lora_in_proj is not None else None
            print(f"  {name}: MHA embed_dim={module.embed_dim}, num_heads={module.num_heads}, "
                  f"in_proj_r={in_r}, in_proj_alpha={in_alpha}")


# Model-specific target module configurations
MODEL_LORA_TARGETS = {
    "labram": {
        "default": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],  # Attention layers
        "attention": ["attn.qkv", "attn.proj"],
        "ffn": ["mlp.fc1", "mlp.fc2"],
        "full": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],  # Attention + FFN
    },
    "eegpt": {
        "default": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        "attention": ["attn.qkv", "attn.proj"],
        "ffn": ["mlp.fc1", "mlp.fc2"],
        "full": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    },
    "cbramod": {
        # CBraMod uses nn.MultiheadAttention
        "default": ["self_attn_s", "self_attn_t", "linear1", "linear2"],  # MHA + FFN
        "attention": ["self_attn_s", "self_attn_t"],  # MHA only
        "ffn": ["linear1", "linear2"],  # FFN only
        "full": ["self_attn_s", "self_attn_t", "linear1", "linear2"],
    },
    "bendr": {
        # BENDR contextualizer uses nn.TransformerEncoderLayer with self_attn (MHA) + linear1/linear2 (FFN)
        "default": ["self_attn", "linear1", "linear2"],  # MHA + FFN
        "attention": ["self_attn"],  # MHA only
        "ffn": ["linear1", "linear2"],  # FFN only
        "full": ["self_attn", "linear1", "linear2"],
    },
    "biot": {
        "default": ["to_q", "to_k", "to_v", "to_out", "w1", "w2"],
        "attention": ["to_q", "to_k", "to_v", "to_out"],
        "ffn": ["w1", "w2"],
        "full": ["to_q", "to_k", "to_v", "to_out", "w1", "w2"],
    },
    "conformer": {
        "default": ["__unsupported__"],  # No local model implementation in this repo
        "full": ["__unsupported__"],
    },
    "csbrain": {
        "default": ["inter_region_attn", "inter_window_attn", "linear1", "linear2"],
        "attention": ["inter_region_attn", "inter_window_attn"],
        "ffn": ["linear1", "linear2"],
        "full": ["inter_region_attn", "inter_window_attn", "linear1", "linear2"],
    },
    "reve": {
        "default": ["to_qkv", "to_out", "net.1", "net.3"],
        "attention": ["to_qkv", "to_out"],
        "ffn": ["net.1", "net.3"],
        "full": ["to_qkv", "to_out", "net.1", "net.3"],
    },
    "mantis": {
        # Mantis ViT unit uses custom Transformer with Attention (to_qkv, to_out) + FeedForward (net)
        "default": ["to_qkv", "to_out.0", "net.0", "net.3"],  # Attention layers
        "attention": ["to_qkv", "to_out.0"],
        "ffn": ["net.0", "net.3"],  # FeedForward Linear layers (indices in Sequential)
        "full": ["to_qkv", "to_out.0", "net.0", "net.3"],
    },
    "moment": {
        # Moment uses T5 encoder with q, k, v, o projections and wi/wo FFN
        "default": ["q", "k", "v", "o", "wo", "wi_0", "wi_1"],  # T5 attention projections
        "attention": ["q", "k", "v", "o"],
        "ffn": ["wi", "wo", "wi_0", "wi_1"],  # T5 FFN layers
        "full": ["q", "k", "v", "o", "wo", "wi_0", "wi_1"],
    },
}


def get_model_lora_targets(model_type: str, target_type: str = "default") -> List[str]:
    """
    Get recommended LoRA target modules for a specific model type.
    
    Args:
        model_type: Model type identifier (e.g., "labram", "eegpt")
        target_type: Type of targets ("default", "full", "attention", "ffn")
    
    Returns:
        List of target module patterns
    """
    if model_type not in MODEL_LORA_TARGETS:
        logger.warning(f"No predefined LoRA targets for model type: {model_type}, using defaults")
        return DEFAULT_ATTENTION_MODULES
    
    targets = MODEL_LORA_TARGETS[model_type]
    if target_type not in targets:
        logger.warning(f"Target type '{target_type}' not found for {model_type}, using 'default'")
        target_type = "default"
    
    return targets[target_type]
