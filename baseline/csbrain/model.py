"""
CSBrain Model wrapper for unified training.

This module provides a unified interface for CSBrain that:
1. Adapts the official CSBrain model to work with variable channel configurations
2. Supports dynamic brain region configuration based on input channels
3. Integrates with the unified multi-head classifier framework
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from baseline.csbrain.utils import generate_area_config, _get_clones

logger = logging.getLogger("baseline")


def _weights_init(m):
    """Initialize weights for linear and conv layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



class PatchEmbedding(nn.Module):
    """Patch embedding with spectral and positional encoding."""
    
    def __init__(self, in_dim: int, out_dim: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Positional encoding via depth-wise convolution
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model, out_channels=d_model, 
                kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                groups=d_model
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        
        # Patch projection
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        
        # Spectral projection
        self.spectral_proj = nn.Sequential(
            nn.Linear(d_model // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, channels, patches, patch_size)
            mask: Optional mask tensor
        """
        bz, ch_num, patch_num, patch_size = x.shape
        
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding
        
        # Reshape for convolution
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)
        
        # Spectral embedding
        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, mask_x.shape[1] // 2 + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb
        
        # Positional embedding
        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        patch_emb = patch_emb + positional_embedding
        
        return patch_emb


class TemEmbedEEGLayer(nn.Module):
    """Temporal embedding layer with multiscale convolutions."""
    
    def __init__(self, dim_in: int, dim_out: int, kernel_sizes: List[tuple], stride: int = 1):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=dim_in, out_channels=dim_scale, 
                kernel_size=(kt, 1), stride=(stride, 1), padding=((kt - 1) // 2, 0)
            )
            for (kt,), dim_scale in zip(kernel_sizes, dim_scales)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, time, d_model)
        """
        batch, chans, time, d_model = x.shape
        x = x.view(batch * chans, d_model, time, 1)
        
        fmaps = [conv(x) for conv in self.convs]

        assert all(f.shape[2] == time for f in fmaps), "Time dimension mismatch after convolutions!"

        x = torch.cat(fmaps, dim=1)
        x = x.view(batch, chans, time, -1)
        
        return x


class BrainEmbedEEGLayer(nn.Module):
    """Brain region embedding layer with region-specific convolutions."""
    
    def __init__(self, dim_in: int = 200, dim_out: int = 200, total_regions: int = 5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.total_regions = total_regions
        
        kernel_sizes = [1, 3, 5]
        dim_scales = [dim_out // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        dim_scales.append(dim_out - sum(dim_scales))
        
        self.region_blocks = nn.ModuleDict({
            f"region_{i}": nn.ModuleList([
                nn.Conv2d(
                    in_channels=dim_in, out_channels=dim_scale,
                    kernel_size=(k, 1), padding=(0, 0), groups=1
                ) for k, dim_scale in zip(kernel_sizes, dim_scales)
            ])
            for i in range(total_regions)
        })

    def forward(self, x, area_config: Dict):
        """
        Args:
            x: Input tensor of shape (batch, channels, time, features)
            area_config: Dictionary with brain region configuration
        """
        batch, chans, t, f = x.shape
        device = x.device
        output = torch.zeros((batch, chans, t, self.dim_out), device=device)
        
        for region_key, region_info in area_config.items():
            if region_key not in self.region_blocks:
                continue
            
            channel_slice = region_info['slice']
            n_electrodes = region_info['channels']
            
            x_region = x[:, channel_slice, :, :]
            x_trans = x_region.permute(0, 2, 1, 3).reshape(-1, n_electrodes, f)
            x_trans = x_trans.permute(0, 2, 1).unsqueeze(-1)
            
            fmap_outputs = []
            # noinspection PyTypeChecker
            for conv, k in zip(self.region_blocks[region_key], [1, 3, 5]):
                pad_size = (k - 1) // 2
                if n_electrodes == 1:
                    x_padded = torch.nn.functional.pad(x_trans, (0, 0, pad_size, pad_size), mode='constant', value=0)
                else:
                    x_padded = torch.nn.functional.pad(x_trans, (0, 0, pad_size, pad_size), mode='circular')
                fmap_outputs.append(conv(x_padded))
            
            fmap_cat = torch.cat(fmap_outputs, dim=1)
            fmap_out = fmap_cat.squeeze(-1).permute(0, 2, 1).reshape(batch, t, n_electrodes, self.dim_out)
            fmap_out = fmap_out.permute(0, 2, 1, 3)
            output[:, channel_slice, :, :] = fmap_out
        
        return output


class RegionAttentionMaskBuilder:
    """Build attention masks for region-aware attention."""
    
    def __init__(self, num_channels: int, area_config: Dict, device=None):
        self.num_channels = num_channels
        self.area_config = area_config
        self.device = device
        
        self.region_indices_dict = self._process_region_indices()
        self.attention_mask = self._build_attention_mask()

    def _process_region_indices(self):
        region_indices_dict = {}
        for region_name, region_info in self.area_config.items():
            region_slice = region_info['slice']
            if isinstance(region_slice, slice):
                start = region_slice.start or 0
                stop = region_slice.stop
                step = region_slice.step or 1
                region_indices = list(range(start, stop, step))
            else:
                region_indices = list(region_slice)
            region_indices_dict[region_name] = region_indices
        return region_indices_dict

    def _build_attention_mask(self):
        device = self.device if self.device is not None else torch.device('cpu')
        region_attn_mask = torch.ones(self.num_channels, self.num_channels, device=device) * float('-inf')
        
        num_groups = max(len(indices) for indices in self.region_indices_dict.values()) if self.region_indices_dict else 1
        groups = [[] for _ in range(num_groups)]
        
        for g in range(num_groups):
            for region_name, region_indices in self.region_indices_dict.items():
                n_electrodes = len(region_indices)
                if n_electrodes == 0:
                    continue
                electrode_idx = region_indices[g % n_electrodes]
                groups[g].append(electrode_idx)
        
        for g, group_electrodes in enumerate(groups):
            for idx1 in group_electrodes:
                for idx2 in group_electrodes:
                    region_attn_mask[idx1, idx2] = 0
        
        return region_attn_mask

    def get_mask(self):
        return self.attention_mask

    def get_region_indices(self):
        return self.region_indices_dict


class CSBrainTransformerEncoder(nn.Module):
    """CSBrain transformer encoder with multiple encoder layers."""
    
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers: nn.ModuleList = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        area_config: Dict,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None
    ) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, area_config, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class CSBrainTransformerEncoderLayer(nn.Module):
    """CSBrain transformer encoder layer with region-aware attention."""
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = nn.functional.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        bias: bool = True,
        area_config: Dict = None,
        sorted_indices: List[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        
        # Attention layers
        self.inter_region_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first
        )
        self.inter_window_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first
        )
        
        self.global_fc = nn.Linear(d_model, d_model, bias=bias)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        if isinstance(activation, str):
            activation = getattr(nn.functional, activation, nn.functional.relu)
        self.activation = activation
        
        self.area_config = area_config
        self.sorted_indices = sorted_indices
        self.mask_builder = None
        self.region_attn_mask = None
        self.region_indices_dict = None
        
        # Cache for dynamic masks (area_config_hash -> (mask, indices))
        self._dynamic_mask_cache = {}
        
        # Build static mask if area_config provided at init
        if area_config is not None and len(area_config) > 0:
            total_channels = sum(
                len(range(info['slice'].start or 0, info['slice'].stop, info['slice'].step or 1))
                if isinstance(info['slice'], slice) else len(info['slice'])
                for info in area_config.values()
            )
            self.mask_builder = RegionAttentionMaskBuilder(total_channels, area_config)
            self.region_attn_mask = self.mask_builder.get_mask()
            self.region_indices_dict = self.mask_builder.get_region_indices()

    def forward(
        self, 
        src: torch.Tensor, 
        area_config: Optional[Dict] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = src
        x = x + self._inter_window_attention(self.norm1(x), src_mask, src_key_padding_mask)
        
        if self.region_attn_mask is None and area_config is not None:
            x = x + self._inter_region_attention_dynamic(self.norm2(x), area_config, src_mask, src_key_padding_mask)
        else:
            x = x + self._inter_region_attention_static(self.norm2(x), src_mask, src_key_padding_mask)
        
        x = x + self._ff_block(self.norm3(x))
        return x

    def _inter_region_attention_static(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Static region attention using pre-built mask."""
        if self.region_attn_mask is None or self.region_indices_dict is None:
            raise ValueError("No initialized region attention mask or region indices dictionary")
        
        batch, chans, T, F = x.shape
        
        x_reshaped = x.permute(0, 2, 1, 3)
        x_flat = x_reshaped.reshape(batch * T, chans, F)
        
        # Compute global features per region
        region_global_features = {}
        for region_name, region_indices in self.region_indices_dict.items():
            region_x = x[:, region_indices, :, :]
            region_global = region_x.mean(dim=1, keepdim=True)
            region_global_features[region_name] = region_global
        
        global_features = torch.zeros_like(x_flat)
        for region_name, region_indices in self.region_indices_dict.items():
            region_global = region_global_features[region_name]
            region_global = region_global.permute(0, 2, 1, 3)
            region_global = region_global.reshape(batch * T, 1, F)
            for idx in region_indices:
                global_features[:, idx:idx + 1, :] = region_global
        
        global_features = self.global_fc(global_features)
        x_enhanced = x_flat + global_features
        
        region_attn_mask = self.region_attn_mask.to(x.device)
        
        attn_output = self.inter_region_attn(
            x_enhanced, x_enhanced, x_enhanced,
            attn_mask=region_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        
        attn_output = attn_output.reshape(batch, T, chans, F).permute(0, 2, 1, 3)
        return self.dropout1(attn_output)

    def _inter_region_attention_dynamic(
        self, 
        x: torch.Tensor, 
        area_config: Dict,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dynamic region attention building mask on-the-fly with caching.
        
        This method is used when area_config is not provided at initialization
        but passed dynamically during forward pass (e.g., multi-dataset training).
        To improve efficiency, masks are cached based on the area_config structure.
        """
        batch, chans, T, Fea = x.shape
        
        # Create a hashable key from area_config
        config_key = self._get_area_config_hash(area_config)
        
        # Check cache
        if config_key in self._dynamic_mask_cache:
            region_attn_mask, region_indices_dict = self._dynamic_mask_cache[config_key]
            region_attn_mask = region_attn_mask.to(x.device)
        else:
            # Build mask and indices dynamically
            mask_builder = RegionAttentionMaskBuilder(chans, area_config, device=x.device)
            region_attn_mask = mask_builder.get_mask()
            region_indices_dict = mask_builder.get_region_indices()
            
            # Cache for future use
            self._dynamic_mask_cache[config_key] = (region_attn_mask.cpu(), region_indices_dict)
        
        x_reshaped = x.permute(0, 2, 1, 3)
        x_flat = x_reshaped.reshape(batch * T, chans, Fea)
        
        # Compute global features per region
        region_global_features = {}
        for region_name, region_indices in region_indices_dict.items():
            region_x = x[:, region_indices, :, :]
            region_global = region_x.mean(dim=1, keepdim=True)
            region_global_features[region_name] = region_global
        
        global_features = torch.zeros_like(x_flat)
        for region_name, region_indices in region_indices_dict.items():
            region_global = region_global_features[region_name]
            region_global = region_global.permute(0, 2, 1, 3)
            region_global = region_global.reshape(batch * T, 1, Fea)
            for idx in region_indices:
                global_features[:, idx:idx + 1, :] = region_global
        
        global_features = self.global_fc(global_features)
        x_enhanced = x_flat + global_features
        
        attn_output = self.inter_region_attn(
            x_enhanced, x_enhanced, x_enhanced,
            attn_mask=region_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        
        attn_output = attn_output.reshape(batch, T, chans, Fea).permute(0, 2, 1, 3)
        return self.dropout1(attn_output)
    
    def _get_area_config_hash(self, area_config: Dict) -> str:
        """Create a hashable key from area_config for caching."""
        items = []
        for region_name in sorted(area_config.keys()):
            region_info = area_config[region_name]
            region_slice = region_info['slice']
            if isinstance(region_slice, slice):
                items.append(f"{region_name}:{region_slice.start}:{region_slice.stop}:{region_slice.step}")
            else:
                items.append(f"{region_name}:{tuple(region_slice)}")
        return "|".join(items)

    def _inter_window_attention(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, chans, T, Fea = x.shape
        window_size = min(T, 5)
        num_windows = T // window_size
        original_T = T
        
        if T % window_size != 0:
            pad_length = window_size - (T % window_size)
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_length))
            T = T + pad_length
            num_windows = T // window_size
        
        x = x.view(batch, chans, num_windows, window_size, Fea)
        x = x.permute(0, 3, 1, 2, 4)
        x = x.reshape(batch * window_size * chans, num_windows, Fea)
        
        temporal_attn_mask = None
        if attn_mask is not None:
            if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() == 2:
                temporal_attn_mask = torch.triu(
                    torch.ones(num_windows, num_windows, device=x.device) * float('-inf'),
                    diagonal=1
                )
        
        x = self.inter_window_attn(
            x, x, x,
            attn_mask=temporal_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        
        x = x.reshape(batch, window_size, chans, num_windows, Fea)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.reshape(batch, chans, T, Fea)
        
        if T != original_T:
            x = x[:, :, :original_T, :]
        
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, Fea = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * T, C, Fea)
        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x_reshaped))))
        x_ff = x_ff.reshape(B, T, C, Fea).permute(0, 2, 1, 3)
        return self.dropout3(x_ff)


class CSBrain(nn.Module):
    """
    CSBrain: Brain-region-aware EEG Foundation Model.
    
    This is a unified version that supports dynamic channel configurations
    and brain region assignments for multi-dataset joint training.
    """
    
    def __init__(
        self, 
        in_dim: int = 200, 
        out_dim: int = 200, 
        d_model: int = 200, 
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        tem_embed_kernel_sizes: List[tuple] = None,
        brain_regions: List[int] = None,
        sorted_indices: List[int] = None,
    ):
        super().__init__()
        
        if tem_embed_kernel_sizes is None:
            tem_embed_kernel_sizes = [(1,), (3,), (5,)]
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model)
        
        # Temporal embedding (names match original checkpoint)
        self.TemEmbed_kernel_sizes = tem_embed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(
            dim_in=in_dim, dim_out=out_dim, 
            kernel_sizes=tem_embed_kernel_sizes, stride=1
        )
        
        # Brain region configuration
        self.brain_regions = brain_regions if brain_regions is not None else []
        self.area_config = generate_area_config(sorted(self.brain_regions)) if brain_regions else {}
        self.sorted_indices = sorted_indices if sorted_indices is not None else []
        
        # Brain region embedding (name matches original checkpoint)
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        
        # Transformer encoder
        encoder_layer = CSBrainTransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True,
            area_config=self.area_config, sorted_indices=self.sorted_indices,
            activation=nn.functional.gelu
        )
        self.encoder = CSBrainTransformerEncoder(
            encoder_layer, num_layers=n_layer, enable_nested_tensor=False
        )
        
        # Output projection
        # self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.proj_out = nn.Identity()

        self.apply(_weights_init)
        
        # Feature storage for analysis
        self.features_by_layer = []
        self.input_features = []

    def forward(self, x: torch.Tensor, area_config: Dict = None, mask=None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, patches, patch_size)
            area_config: Brain region configuration dictionary (optional, uses self.area_config if None)
            mask: Optional mask
            
        Returns:
            Output tensor of shape (batch, channels, patches, out_dim)
        """
        # Use internal area_config if not provided
        if area_config is None:
            area_config = self.area_config
        
        # Apply sorted indices if available
        if len(self.sorted_indices) > 0:
            x = x[:, self.sorted_indices, :, :]
        
        # Patch embedding
        patch_emb = self.patch_embedding(x, mask)
        
        # Apply transformer layers with temporal and brain embeddings
        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, area_config) + patch_emb
            patch_emb = self.encoder.layers[layer_idx](patch_emb, area_config)
        
        out = self.proj_out(patch_emb)
        return out
