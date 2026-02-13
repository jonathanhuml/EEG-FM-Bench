"""Model-specific parameter groupers for gradient analysis.

This module provides ParamGrouper implementations for each EEG foundation model:
- CBraModParamGrouper: For CBraMod (Criss-Cross Transformer)
- LaBraModParamGrouper: For LaBraM (Large Brain Model)
- REVEParamGrouper: For REVE (4D Fourier PE Transformer)
- CSBrainParamGrouper: For CSBrain (Cross-Scale Transformer)

These groupers correctly categorize model parameters into semantic groups
for gradient analysis.
"""

import re
from typing import Any, Dict, List, Optional

import torch.nn as nn

from baseline.analysis.grouper import (
    EncoderParamGrouper,
    ParamGroup,
    ParamGroupType,
)
from baseline.analysis.config import MODEL_INNOVATION_GROUPS


class CBraModParamGrouper(EncoderParamGrouper):
    """Parameter grouper for CBraMod model.
    
    CBraMod architecture:
    - patch_embedding: PatchEmbedding with conv layers and spectral projection
        - proj_in: Conv2d layers for spatial embedding
        - spectral_proj: Linear for frequency domain features
        - positional_encoding: Conv2d for position encoding
    - encoder: TransformerEncoder with criss-cross attention layers
        - layers[i].self_attn_s: Spatial attention (MultiheadAttention)
        - layers[i].self_attn_t: Temporal attention (MultiheadAttention)
        - layers[i].linear1/linear2: FFN layers
        - layers[i].norm1/norm2: LayerNorm
    - proj_out: Output projection (reconstruction head)
    """
    
    EMBED_PATTERNS = [
        'patch_embedding', 'proj_in', 'spectral_proj', 
        'positional_encoding', 'mask_encoding',
    ]
    
    ATTENTION_PATTERNS = [
        'self_attn_s', 'self_attn_t', 'attn',
    ]
    
    FFN_PATTERNS = [
        'linear1', 'linear2', 'ffn',
    ]
    
    NORM_PATTERNS = [
        'norm1', 'norm2', 'norm', 'ln',
    ]
    
    HEAD_PATTERNS = [
        'proj_out', 'head', 'classifier',
    ]
    
    def _setup_groups(self):
        """Set up groups based on CBraMod architecture."""
        # Define semantic groups
        groups_def = [
            ("patch_embed", ParamGroupType.EMBED, "Patch embedding (conv + spectral)"),
            ("spatial_attn", ParamGroupType.ATTENTION, "Spatial attention (self_attn_s)"),
            ("temporal_attn", ParamGroupType.ATTENTION, "Temporal attention (self_attn_t)"),
            ("ffn", ParamGroupType.FFN, "Feed-forward layers"),
            ("norm", ParamGroupType.NORM, "Layer normalization"),
            ("head", ParamGroupType.HEAD, "Output projection / task head"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in groups_def:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        # Categorize each parameter
        for param_name, _ in self.model.named_parameters():
            name_lower = param_name.lower()
            
            # Check specific patterns in order of priority
            if 'patch_embedding' in name_lower or 'proj_in' in name_lower or 'spectral_proj' in name_lower:
                group = "patch_embed"
            elif 'self_attn_s' in name_lower:
                group = "spatial_attn"
            elif 'self_attn_t' in name_lower:
                group = "temporal_attn"
            elif 'linear1' in name_lower or 'linear2' in name_lower:
                group = "ffn"
            elif 'norm' in name_lower:
                group = "norm"
            elif 'proj_out' in name_lower or 'head' in name_lower or 'classifier' in name_lower:
                group = "head"
            else:
                group = "other"
            
            self._groups[group].param_names.append(param_name)
            self._param_to_group[param_name] = group


class LaBraModParamGrouper(EncoderParamGrouper):
    """Parameter grouper for LaBraM model.
    
    LaBraM architecture (NeuralTransformer):
    - patch_embed: TemporalConv (for encoder) or PatchEmbed (for decoder)
        - conv1/conv2/conv3: Temporal convolutions
        - norm1/norm2/norm3: GroupNorm
    - pos_embed: Learnable positional embedding
    - time_embed: Temporal embedding
    - cls_token: Classification token
    - blocks[i]: Transformer Block
        - norm1/norm2: LayerNorm
        - attn: Attention (qkv, proj, q_norm, k_norm)
        - mlp: Mlp (fc1, fc2)
        - gamma_1/gamma_2: LayerScale parameters
    - norm: Final LayerNorm
    - head: Classification head
    """
    
    EMBED_PATTERNS = [
        'patch_embed', 'temporal_conv', 'conv1', 'conv2', 'conv3',
        'pos_embed', 'time_embed', 'cls_token', 'mask_token',
    ]
    
    ATTENTION_PATTERNS = [
        'attn', 'qkv', 'proj', 'q_norm', 'k_norm', 'q_bias', 'v_bias',
        'relative_position',
    ]
    
    FFN_PATTERNS = [
        'mlp', 'fc1', 'fc2',
    ]
    
    NORM_PATTERNS = [
        'norm1', 'norm2', 'norm', 'ln',
    ]
    
    # LayerScale gamma_1 -> attention, gamma_2 -> mlp (NOT norm)
    LAYERSCALE_ATTENTION_PATTERNS = ['gamma_1']
    LAYERSCALE_FFN_PATTERNS = ['gamma_2']
    
    HEAD_PATTERNS = [
        'head', 'classifier', 'lm_head',
    ]
    
    def _setup_groups(self):
        """Set up groups based on LaBraM architecture.
        
        Note on LayerScale (gamma_1, gamma_2):
        - gamma_1 scales attention output → belongs to attention group
        - gamma_2 scales MLP output → belongs to mlp group
        This follows the semantic meaning of LayerScale: it adjusts the
        contribution of each sublayer (attention or FFN) rather than
        being a normalization parameter.
        """
        groups_def = [
            ("temporal_embed", ParamGroupType.EMBED, "Temporal convolution embedding"),
            ("pos_embed", ParamGroupType.EMBED, "Position and cls token embeddings"),
            ("attention", ParamGroupType.ATTENTION, "Self-attention layers + LayerScale gamma_1"),
            ("mlp", ParamGroupType.FFN, "MLP / Feed-forward layers + LayerScale gamma_2"),
            ("norm", ParamGroupType.NORM, "Layer normalization (pre-norm)"),
            ("head", ParamGroupType.HEAD, "Classification / LM head"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in groups_def:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        for param_name, _ in self.model.named_parameters():
            name_lower = param_name.lower()
            
            # Temporal conv embedding
            if any(p in name_lower for p in ['temporal_conv', 'patch_embed.conv']):
                group = "temporal_embed"
            # Positional embeddings and tokens
            elif any(p in name_lower for p in ['pos_embed', 'time_embed', 'cls_token', 'mask_token']):
                group = "pos_embed"
            # LayerScale gamma_1 scales attention output -> attention group
            elif 'gamma_1' in name_lower:
                group = "attention"
            # LayerScale gamma_2 scales MLP output -> mlp group
            elif 'gamma_2' in name_lower:
                group = "mlp"
            # Attention layers (excluding norm which may contain 'attn')
            elif 'attn' in name_lower and 'norm' not in name_lower:
                group = "attention"
            # MLP layers
            elif 'mlp' in name_lower or ('fc1' in name_lower) or ('fc2' in name_lower):
                group = "mlp"
            # Normalization (only actual norm layers, not LayerScale)
            elif 'norm' in name_lower:
                group = "norm"
            # Head
            elif 'head' in name_lower or 'classifier' in name_lower:
                group = "head"
            else:
                group = "other"
            
            self._groups[group].param_names.append(param_name)
            self._param_to_group[param_name] = group


class REVEParamGrouper(EncoderParamGrouper):
    """Parameter grouper for REVE model.
    
    REVE architecture:
    - to_patch_embedding: Linear patch projection
    - fourier4d: FourierEmb4D (4D Fourier positional embedding, no learnable params)
    - mlp4d: MLP for position embedding (Linear + GELU + LayerNorm)
    - ln: LayerNorm after position embedding
    - transformer: TransformerBackbone
        - layers[i][0]: Attention (norm, to_qkv, to_out)
        - layers[i][1]: FeedForward (net.0=RMSNorm, net.1=Linear, net.3=Linear)
    - final_layer: Output projection
    - cls_query_token: Attention pooling query
    
    Uses regex patterns for more robust matching of nested modules.
    """
    
    # Regex patterns for better matching
    # Embedding: to_patch_embedding, mlp4d.*, cls_query_token
    EMBED_RE = re.compile(r'(to_patch_embedding|mlp4d|cls_query_token)', re.IGNORECASE)
    
    # Attention: layers.X.0.to_qkv, layers.X.0.to_out
    # Pattern: transformer.layers.N.0.(to_qkv|to_out)
    ATTENTION_RE = re.compile(r'layers\.\d+\.0\.(to_qkv|to_out)', re.IGNORECASE)
    
    # FFN: layers.X.1.net.1 (first linear), layers.X.1.net.3 (second linear)
    # Pattern: transformer.layers.N.1.net.(1|3)
    FFN_RE = re.compile(r'layers\.\d+\.1\.net\.[13]\.', re.IGNORECASE)
    
    # Norm: layers.X.0.norm (attention norm), layers.X.1.net.0 (FFN RMSNorm)
    # Also: top-level ln (post-embedding norm)
    NORM_RE = re.compile(r'(layers\.\d+\.0\.norm|layers\.\d+\.1\.net\.0\.|^ln\.)', re.IGNORECASE)
    
    # Head: final_layer
    HEAD_RE = re.compile(r'(final_layer|head|classifier)', re.IGNORECASE)
    
    EMBED_PATTERNS = [
        'to_patch_embedding', 'patch_embed', 'mlp4d', 'cls_query_token',
    ]
    
    ATTENTION_PATTERNS = [
        'to_qkv', 'to_out', 'attend',
    ]
    
    FFN_PATTERNS = [
        'ff', 'feedforward', 'net.1', 'net.3',  # FeedForward linear layers
    ]
    
    NORM_PATTERNS = [
        'norm', 'rms', 'ln', 'net.0',  # RMSNorm in attention and FFN
    ]
    
    HEAD_PATTERNS = [
        'final_layer', 'head', 'classifier',
    ]
    
    def _setup_groups(self):
        """Set up groups based on REVE architecture.
        
        Uses compiled regex patterns for more robust matching of
        the nested module structure in REVE's TransformerBackbone.
        """
        groups_def = [
            ("patch_embed", ParamGroupType.EMBED, "Patch and position embedding"),
            ("attention", ParamGroupType.ATTENTION, "Self-attention (QKV and output)"),
            ("ffn", ParamGroupType.FFN, "Feed-forward layers"),
            ("norm", ParamGroupType.NORM, "RMSNorm / LayerNorm"),
            ("head", ParamGroupType.HEAD, "Final output layer"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in groups_def:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        for param_name, _ in self.model.named_parameters():
            # Use regex matching for more robust classification
            
            # Embedding layers
            if self.EMBED_RE.search(param_name):
                group = "patch_embed"
            # Head / final layer (check before attention to avoid false matches)
            elif self.HEAD_RE.search(param_name):
                group = "head"
            # Attention QKV and output (layers.X.0.to_qkv, layers.X.0.to_out)
            elif self.ATTENTION_RE.search(param_name):
                group = "attention"
            # FFN layers (layers.X.1.net.1, layers.X.1.net.3)
            elif self.FFN_RE.search(param_name):
                group = "ffn"
            # Normalization (attention norm, FFN RMSNorm, post-embed ln)
            elif self.NORM_RE.search(param_name):
                group = "norm"
            # Fallback: top-level ln not captured by regex
            elif param_name.startswith('ln.') or param_name == 'ln':
                group = "norm"
            else:
                group = "other"
            
            self._groups[group].param_names.append(param_name)
            self._param_to_group[param_name] = group


class CSBrainParamGrouper(EncoderParamGrouper):
    """Parameter grouper for CSBrain model.
    
    CSBrain architecture:
    - patch_embedding: PatchEmbedding
        - proj_in: Conv2d layers
        - spectral_proj: Linear for FFT features
        - positional_encoding: Conv2d
    - tem_embed: TemEmbedEEGLayer (multiscale temporal convolutions)
    - brain_embed: BrainEmbedEEGLayer (region-specific convolutions)
    - encoder: CSBrainTransformerEncoder
        - layers[i]: CSBrainTransformerEncoderLayer
            - inter_region_attn: Region attention
            - inter_window_attn: Window attention
            - global_fc: Global projection
            - linear1/linear2: FFN
            - norm1/norm2/norm3: LayerNorm
    - proj_out: Output projection
    
    Note: User requested BrainEmbedEEGLayer + temporal → patch_embed
    """
    
    EMBED_PATTERNS = [
        'patch_embedding', 'tem_embed', 'brain_embed',
        'temembed', 'brainembed',
        'tembedeeglayer', 'brainembedeeglayer',
        'proj_in', 'spectral_proj', 'positional_encoding',
    ]
    
    ATTENTION_PATTERNS = [
        'inter_region_attn', 'inter_window_attn', 'global_fc', 'attn',
    ]
    
    FFN_PATTERNS = [
        'linear1', 'linear2', 'ffn', 'fc',
    ]
    
    NORM_PATTERNS = [
        'norm1', 'norm2', 'norm3', 'norm', 'ln',
    ]
    
    HEAD_PATTERNS = [
        'proj_out', 'head', 'classifier',
    ]
    
    def _setup_groups(self):
        """Set up groups based on CSBrain architecture."""
        # Per user: BrainEmbedEEGLayer + temporal → patch_embed
        groups_def = [
            ("patch_embed", ParamGroupType.EMBED, "Patch + Brain + Temporal embedding"),
            ("region_attn", ParamGroupType.ATTENTION, "Inter-region attention"),
            ("window_attn", ParamGroupType.ATTENTION, "Inter-window attention"),
            ("ffn", ParamGroupType.FFN, "Feed-forward layers"),
            ("norm", ParamGroupType.NORM, "Layer normalization"),
            ("head", ParamGroupType.HEAD, "Output projection / task head"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in groups_def:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        for param_name, _ in self.model.named_parameters():
            name_lower = param_name.lower()
            
            # All embedding related (patch, brain, temporal) → patch_embed
            if any(p in name_lower for p in [
                'patch_embedding', 'tem_embed', 'brain_embed',
                'proj_in', 'spectral_proj', 'positional_encoding',
                'mask_encoding', 'region_blocks', 'convs'
            ]):
                group = "patch_embed"
            # Region attention
            elif 'inter_region_attn' in name_lower:
                group = "region_attn"
            # Window attention and global FC
            elif 'inter_window_attn' in name_lower or 'global_fc' in name_lower:
                group = "window_attn"
            # FFN
            elif 'linear1' in name_lower or 'linear2' in name_lower:
                group = "ffn"
            # Normalization
            elif 'norm' in name_lower:
                group = "norm"
            # Head
            elif 'proj_out' in name_lower or 'head' in name_lower or 'classifier' in name_lower:
                group = "head"
            else:
                group = "other"
            
            self._groups[group].param_names.append(param_name)
            self._param_to_group[param_name] = group


class MantisParamGrouper(EncoderParamGrouper):
    """Parameter grouper for Mantis model.
    
    Mantis architecture:
    - tokgen_unit: TokenGeneratorUnit
        - convs: 1D convolutions for time series
        - layer_norms: LayerNorm for conv outputs
        - scalar_encoders: MultiScaledScalarEncoder
            - mlp: Linear layers for scalar stats
        - linear_encoder: Final token projection
    - vit_unit: ViTUnit
        - pos_encoder: PositionalEncoding (no learnable params in sinusoidal)
        - cls_token: Learnable CLS token
        - transformer: Transformer (Attention + FeedForward)
            - layers[i]: (norm1, attn, norm2, ff)
                - attn: Attention (to_qkv, to_out)
                - ff: FeedForward (net.0=Linear, net.2=Linear)
    - prj: Output projection (LayerNorm + Linear)
    """
    
    def _setup_groups(self):
        """Set up groups based on Mantis architecture."""
        groups_def = [
            ("tokgen_conv", ParamGroupType.EMBED, "Token generator convolutions"),
            ("tokgen_scalar", ParamGroupType.EMBED, "Scalar encoder MLPs"),
            ("cls_token", ParamGroupType.EMBED, "CLS token"),
            ("attention", ParamGroupType.ATTENTION, "ViT attention layers"),
            ("ffn", ParamGroupType.FFN, "ViT feed-forward layers"),
            ("norm", ParamGroupType.NORM, "Layer normalization"),
            ("head", ParamGroupType.HEAD, "Output projection"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in groups_def:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        for param_name, _ in self.model.named_parameters():
            name_lower = param_name.lower()
            
            # Token generator convolutions
            if 'tokgen_unit' in name_lower and 'conv' in name_lower:
                group = "tokgen_conv"
            # Scalar encoders
            elif 'scalar_encoder' in name_lower or 'scalar_encoders' in name_lower or 'linear_encoder' in name_lower:
                group = "tokgen_scalar"
            # CLS token
            elif 'cls_token' in name_lower:
                group = "cls_token"
            # ViT attention (to_qkv, to_out)
            elif 'to_qkv' in name_lower or 'to_out' in name_lower:
                group = "attention"
            # ViT FFN (in FeedForward.net)
            elif 'vit_unit' in name_lower and '.net.' in name_lower:
                group = "ffn"
            # Normalization
            elif 'norm' in name_lower or 'layer_norm' in name_lower:
                group = "norm"
            # Output projection
            elif 'prj' in name_lower or 'head' in name_lower or 'classifier' in name_lower:
                group = "head"
            else:
                group = "other"
            
            self._groups[group].param_names.append(param_name)
            self._param_to_group[param_name] = group


class MomentParamGrouper(EncoderParamGrouper):
    """Parameter grouper for Moment model.
    
    Moment architecture (based on T5 encoder):
    - normalizer: RevIN (affine params if enabled)
    - tokenizer: Patching (no learnable params)
    - patch_embedding: PatchEmbedding
        - value_embedding: Linear projection
        - position_embedding: Learnable positional params
    - encoder: T5EncoderModel
        - block[i]: T5Block
            - layer[0]: T5LayerSelfAttention
                - SelfAttention: q, k, v, o projections
                - layer_norm
            - layer[1]: T5LayerFF
                - DenseReluDense or DenseGatedGeluDense: wi, wo
                - layer_norm
    - head: Task-specific head
    """
    
    def _setup_groups(self):
        """Set up groups based on Moment architecture."""
        groups_def = [
            ("patch_embed", ParamGroupType.EMBED, "Patch and position embedding"),
            ("attention", ParamGroupType.ATTENTION, "T5 self-attention (q, k, v, o)"),
            ("ffn", ParamGroupType.FFN, "T5 feed-forward (wi, wo)"),
            ("norm", ParamGroupType.NORM, "Layer normalization"),
            ("head", ParamGroupType.HEAD, "Task-specific head"),
            ("other", ParamGroupType.OTHER, "Other parameters"),
        ]
        
        for name, gtype, desc in groups_def:
            self._groups[name] = ParamGroup(
                name=name,
                group_type=gtype,
                param_names=[],
                description=desc,
            )
        
        for param_name, _ in self.model.named_parameters():
            name_lower = param_name.lower()
            
            # Patch embedding
            if (
                'patch_embedding' in name_lower
                or 'value_embedding' in name_lower
                or 'position_embedding' in name_lower
                or 'embed_tokens' in name_lower
                or 'token_embedding' in name_lower
            ):
                group = "patch_embed"
            # T5 self-attention layers (q, k, v, o)
            elif any(f'.{proj}.' in name_lower or f'.{proj}_' in name_lower 
                    for proj in ['q', 'k', 'v', 'o']):
                group = "attention"
            elif 'selfattention' in name_lower and 'norm' not in name_lower:
                group = "attention"
            # T5 FFN layers (wi, wo, wi_0, wi_1 for gated variants)
            elif any(p in name_lower for p in ['wi', 'wo', 'densereludense', 'densegeludense']):
                group = "ffn"
            # Normalization
            elif 'norm' in name_lower or 'layer_norm' in name_lower:
                group = "norm"
            # Head / output
            elif 'head' in name_lower or 'classifier' in name_lower or 'linear' in name_lower:
                # Check if it's in the head module, not encoder
                if 'encoder' not in name_lower:
                    group = "head"
                else:
                    group = "other"
            elif 'normalizer' in name_lower:
                # RevIN affine params
                group = "other"
            else:
                group = "other"
            
            self._groups[group].param_names.append(param_name)
            self._param_to_group[param_name] = group


# Registry for easy lookup
PARAM_GROUPER_REGISTRY: Dict[str, type] = {
    "cbramod": CBraModParamGrouper,
    "labram": LaBraModParamGrouper,
    "reve": REVEParamGrouper,
    "csbrain": CSBrainParamGrouper,
    "mantis": MantisParamGrouper,
    "moment": MomentParamGrouper,
}


def get_param_grouper(model_name: str, model: nn.Module) -> EncoderParamGrouper:
    """Get appropriate parameter grouper for a model.
    
    Args:
        model_name: Name of the model ("cbramod", "labram", "reve", "csbrain")
        model: The model instance
        
    Returns:
        Configured parameter grouper
    """
    from baseline.analysis.grouper import DefaultParamGrouper
    
    grouper_cls = PARAM_GROUPER_REGISTRY.get(model_name.lower())
    if grouper_cls is None:
        # Fall back to default grouper
        return DefaultParamGrouper(model)
    return grouper_cls(model)


def verify_grouper_coverage(
    model: nn.Module,
    grouper: EncoderParamGrouper,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Verify that a parameter grouper covers all model parameters.
    
    This is a testing/debugging utility to ensure:
    1. All parameters are assigned to exactly one group
    2. No parameters are missed (ungrouped)
    3. No parameters are duplicated across groups
    4. The 'other' group is reasonably small
    
    Args:
        model: The model to verify
        grouper: The parameter grouper to verify
        verbose: If True, print detailed information
        
    Returns:
        Dictionary with verification results:
            - 'all_covered': bool, True if all parameters are covered
            - 'no_duplicates': bool, True if no parameters appear in multiple groups
            - 'ungrouped': List of parameter names not in any group
            - 'duplicated': List of parameter names in multiple groups
            - 'group_stats': Dict with per-group statistics
            - 'other_ratio': float, ratio of parameters in 'other' group
    """
    results = {
        'all_covered': True,
        'no_duplicates': True,
        'ungrouped': [],
        'duplicated': [],
        'group_stats': {},
        'other_ratio': 0.0,
    }
    
    # Get all model parameter names
    all_param_names = set(name for name, _ in model.named_parameters())
    
    # Get all grouped parameter names
    grouped_params = set()
    param_to_groups = {}  # Track which groups each param belongs to
    
    groups = grouper.get_group_names()
    total_params = 0
    other_params = 0
    
    for group_name in groups:
        param_names = grouper.get_params_in_group(group_name)
        group_param_count = 0
        
        for name in param_names:
            if name in param_to_groups:
                param_to_groups[name].append(group_name)
            else:
                param_to_groups[name] = [group_name]
            
            # Count parameters
            if name in all_param_names:
                param = dict(model.named_parameters())[name]
                group_param_count += param.numel()
                total_params += param.numel()
        
        grouped_params.update(param_names)
        
        results['group_stats'][group_name] = {
            'count': len(param_names),
            'param_count': group_param_count,
        }
        
        if group_name == 'other':
            other_params = group_param_count
    
    # Check for ungrouped parameters
    ungrouped = all_param_names - grouped_params
    if ungrouped:
        results['all_covered'] = False
        results['ungrouped'] = sorted(list(ungrouped))
    
    # Check for duplicated parameters (in multiple groups)
    duplicated = [name for name, groups_list in param_to_groups.items() if len(groups_list) > 1]
    if duplicated:
        results['no_duplicates'] = False
        results['duplicated'] = sorted(duplicated)
    
    # Calculate 'other' ratio
    if total_params > 0:
        results['other_ratio'] = other_params / total_params
    
    # Print verbose information
    if verbose:
        print("=" * 60)
        print("Parameter Grouper Verification Report")
        print("=" * 60)
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Grouper: {grouper.__class__.__name__}")
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Total parameter tensors: {len(all_param_names)}")
        
        print("\n--- Group Statistics ---")
        for group_name in groups:
            stats = results['group_stats'][group_name]
            print(f"  {group_name:15s}: {stats['count']:4d} tensors, {stats['param_count']:12,} params")
        
        print(f"\n--- Coverage ---")
        print(f"  All covered: {results['all_covered']}")
        print(f"  No duplicates: {results['no_duplicates']}")
        print(f"  'other' ratio: {results['other_ratio']:.2%}")
        
        if results['ungrouped']:
            print(f"\n--- Ungrouped Parameters ({len(results['ungrouped'])}) ---")
            for name in results['ungrouped'][:10]:  # Show first 10
                print(f"    - {name}")
            if len(results['ungrouped']) > 10:
                print(f"    ... and {len(results['ungrouped']) - 10} more")
        
        if results['duplicated']:
            print(f"\n--- Duplicated Parameters ({len(results['duplicated'])}) ---")
            for name in results['duplicated'][:10]:
                groups_str = ", ".join(param_to_groups[name])
                print(f"    - {name} -> [{groups_str}]")
        
        # Warning if 'other' ratio is high
        if results['other_ratio'] > 0.1:
            print(f"\n⚠️  Warning: 'other' group contains {results['other_ratio']:.1%} of parameters.")
            print("    Consider adding more specific grouping rules.")
        
        print("=" * 60)
    
    return results


def _layer_index_group_name(param_name: str) -> str:
    """Infer layer index group name from parameter name."""
    patterns = [r"layers\.(\d+)", r"layer\.(\d+)", r"encoder\.layers\.(\d+)"]
    for pat in patterns:
        match = re.search(pat, param_name)
        if match:
            return f"layer_{match.group(1)}"
    return "other_layer"


def _innovation_patterns_for_model(model_name: str) -> Dict[str, List[str]]:
    groups = MODEL_INNOVATION_GROUPS.get(model_name, {})
    return {name: patterns for name, (_, patterns) in groups.items()}


def _assign_innovation_group(param_name: str, patterns: Dict[str, List[str]]) -> str:
    for group_name, pats in patterns.items():
        if EncoderParamGrouper.match_pattern(param_name, pats):
            return group_name
    return "other_innovation"


def test_all_groupers(
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Test all registered parameter groupers under three grouping strategies.

    Strategies:
        1) by_module_type (default grouper)
        2) by_layer_index
        3) by_model_innovation
    """
    import traceback
    import contextlib

    results: Dict[str, Dict[str, Any]] = {}

    def _run_tests():
        for model_name, grouper_cls in PARAM_GROUPER_REGISTRY.items():
            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing {model_name} with {grouper_cls.__name__}")
                print('='*80)

            try:
                # Build model via the AbstractTrainer-based trainers
                if model_name == "cbramod":
                    from baseline.cbramod.cbramod_config import CBraModConfig
                    from baseline.cbramod.cbramod_trainer import CBraModTrainer
                    trainer = CBraModTrainer(CBraModConfig())

                elif model_name == "labram":
                    from baseline.labram.labram_config import LabramConfig
                    from baseline.labram.labram_trainer import LabramTrainer
                    trainer = LabramTrainer(LabramConfig())

                elif model_name == "reve":
                    from baseline.reve.reve_config import ReveConfig
                    from baseline.reve.reve_trainer import ReveTrainer
                    trainer = ReveTrainer(ReveConfig())

                elif model_name == "csbrain":
                    from baseline.csbrain.csbrain_config import CSBrainConfig
                    from baseline.csbrain.csbrain_trainer import CSBrainTrainer
                    trainer = CSBrainTrainer(CSBrainConfig())

                elif model_name == "mantis":
                    from baseline.mantis.mantis_config import MantisConfig
                    from baseline.mantis.mantis_trainer import MantisTrainer
                    trainer = MantisTrainer(MantisConfig())

                elif model_name == "moment":
                    from baseline.moment.moment_config import MomentConfig
                    from baseline.moment.moment_trainer import MomentTrainer
                    trainer = MomentTrainer(MomentConfig())

                else:
                    if verbose:
                        print(f"  Skipping unknown model: {model_name}")
                    continue

                trainer.setup_device("cpu")
                trainer.setup_analysis_mode()
                model = trainer.setup_model()

                model_results: Dict[str, Any] = {}

                # Strategy 1: by_module_type
                grouper = grouper_cls(model)
                model_results["by_module_type"] = verify_grouper_coverage(model, grouper, verbose=False)
                if verbose:
                    print("\n[by_module_type] parameter assignments:")
                    for name, _ in model.named_parameters():
                        print(f"  {name} -> {grouper.get_param_group_name(name)}")

                # Strategy 2: by_layer_index
                if verbose:
                    print("\n[by_layer_index] parameter assignments:")
                layer_groups: Dict[str, List[str]] = {}
                for name, _ in model.named_parameters():
                    group = _layer_index_group_name(name)
                    layer_groups.setdefault(group, []).append(name)
                    if verbose:
                        print(f"  {name} -> {group}")
                model_results["by_layer_index"] = {
                    "groups": {k: len(v) for k, v in layer_groups.items()},
                }

                # Strategy 3: by_model_innovation
                patterns = _innovation_patterns_for_model(model_name)
                if patterns:
                    if verbose:
                        print("\n[by_model_innovation] parameter assignments:")
                    innovation_groups: Dict[str, List[str]] = {}
                    for name, _ in model.named_parameters():
                        group = _assign_innovation_group(name, patterns)
                        innovation_groups.setdefault(group, []).append(name)
                        if verbose:
                            print(f"  {name} -> {group}")
                    model_results["by_model_innovation"] = {
                        "groups": {k: len(v) for k, v in innovation_groups.items()},
                    }
                else:
                    model_results["by_model_innovation"] = {"error": "No innovation patterns"}

                results[model_name] = model_results

            except Exception as e:
                if verbose:
                    print(f"  Failed to test {model_name}: {e}")
                    traceback.print_exc()
                results[model_name] = {'error': str(e)}

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                _run_tests()
    else:
        _run_tests()

    return results
