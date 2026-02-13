"""
Mantis Model Implementation.

This module provides a local implementation of the Mantis 8M architecture,
adapted from the official implementation for integration with our framework.

Reference: Mantis official - architecture/architecture.py
"""

import math
import torch
from torch import nn
from einops import rearrange, repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin


# TokenGeneratorUnit components

class ScalarEncoder(nn.Module):
    def __init__(self, k, hidden_dim):
        super(ScalarEncoder, self).__init__()
        self.w = torch.nn.Parameter(torch.rand(
            (1, hidden_dim), dtype=torch.float, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(
            (1, hidden_dim), dtype=torch.float, requires_grad=True))
        self.k = k
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape=hidden_dim, eps=1e-15)

    def forward(self, x):
        z = x * self.w + self.k * self.b
        y = self.layer_norm(z)
        return y


class MultiScaledScalarEncoder(nn.Module):
    def __init__(self, scales, hidden_dim, epsilon):
        """
        A multi-scaled encoding of a scalar variable:
        https://arxiv.org/pdf/2310.07402.pdf

        Parameters
        ----------
        scales: list, default=None
            List of scales. By default, initialized as [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4].
        hidden_dim: int, default=32
            Hidden dimension of a scalar encoder.
        epsilon: float, default=1.1
            A constant term used to tolerate the computational error in computation of scale weights.
        """
        super(MultiScaledScalarEncoder, self).__init__()
        self.register_buffer('scales', torch.tensor(scales))
        self.epsilon = epsilon
        self.encoders = nn.ModuleList(
            [ScalarEncoder(k, hidden_dim) for k in scales])

    def forward(self, x):
        alpha = abs(1 / torch.log(torch.matmul(abs(x), 1 /
                    self.scales.reshape(1, -1)) + self.epsilon))
        alpha = alpha / torch.sum(alpha, dim=-1, keepdim=True)
        alpha = torch.unsqueeze(alpha, dim=-1)
        y = [encoder(x) for encoder in self.encoders]
        y = torch.stack(y, dim=-2)
        y = torch.sum(y * alpha, dim=-2)
        return y


class LinearEncoder(nn.Module):
    """Simple linear projection encoder."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features, eps=1e-15)
    
    def forward(self, x):
        return self.norm(self.linear(x))


class Convolution(nn.Module):
    """1D Convolution for time series processing."""
    
    def __init__(self, kernel_size, out_channels, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size, 
                             padding=padding, dilation=dilation)
    
    def forward(self, x):
        # x: (batch, 1, seq_len)
        return self.conv(x).transpose(1, 2)  # (batch, seq_len, out_channels)


class TokenGeneratorUnit(nn.Module):
    """Generates tokens from time series by combining temporal convolutions and scalar statistics."""
    
    def __init__(self, hidden_dim, num_patches, patch_window_size, scalar_scales, 
                 hidden_dim_scalar_enc, epsilon_scalar_enc):
        super().__init__()
        self.num_patches = num_patches
        # Scales each time-series w.r.t. its mean and std
        self._eps = 1e-5

        # Token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        kernel_size = patch_window_size + 1 if patch_window_size % 2 == 0 else patch_window_size
        
        self.convs = nn.ModuleList([
            Convolution(kernel_size=kernel_size, out_channels=hidden_dim, dilation=1)
            for _ in range(num_ts_feats)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=self._eps)
            for _ in range(num_ts_feats)
        ])

        # Token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 2  # mean + std
        
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for _ in range(num_scalar_stats)
        ])

        # Final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * num_ts_feats, hidden_dim)

    def _ts_scaler(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return (x - mean) / (std + self._eps)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, seq_len)
        Returns:
            tokens: (batch, num_patches, hidden_dim)
        """
        with torch.no_grad():
            # Compute statistics for each patch
            x_patched = x.reshape(x.shape[0], self.num_patches, -1)
            mean_patched = x_patched.mean(dim=-1, keepdim=True)
            std_patched = x_patched.std(dim=-1, keepdim=True)
            statistics = [mean_patched, std_patched]

        # Encode scalar statistics
        scalar_embeddings = [self.scalar_encoders[i](statistics[i]) 
                           for i in range(len(statistics))]

        # Apply convolution for original ts and its diff
        ts_var_embeddings = []
        
        # Diff
        with torch.no_grad():
            diff_x = torch.diff(x, n=1, dim=2)
            diff_x = torch.nn.functional.pad(diff_x, (0, 1))
        
        embedding = self.convs[0](self._ts_scaler(diff_x))
        ts_var_embeddings.append(embedding)
        
        # Original ts
        embedding = self.convs[1](self._ts_scaler(x))
        ts_var_embeddings.append(embedding)

        # Split ts_var_embeddings into patches
        patched_ts_var_embeddings = []
        for i, embedding in enumerate(ts_var_embeddings):
            embedding = self.layer_norms[i](embedding)
            embedding = embedding.reshape(embedding.shape[0], self.num_patches, -1, embedding.shape[2])
            embedding = embedding.mean(dim=2)
            patched_ts_var_embeddings.append(embedding)

        # Concatenate and project
        x_embeddings = torch.cat([
            torch.cat(patched_ts_var_embeddings, dim=-1),
            torch.cat(scalar_embeddings, dim=-1)
        ], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings


# ViTUnit components

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoder.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor of shape ``[seq_len, batch_size, d_model]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PreNorm(nn.Module):
    """
    Layer Normalization before a layer.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    The attention block in a transformer.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer layer.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTUnit(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_patches,
            depth,
            heads,
            mlp_dim,
            dim_head,
            dropout,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)

        self.cls_token = nn.Parameter(torch.randn(hidden_dim))

        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)

        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)

        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)


class Mantis8M(nn.Module, PyTorchModelHubMixin):
    """
    Mantis 8M time series foundation model.
    
    A univariate time series model that uses Token Generation + ViT architecture.
    For multivariate data, each channel is processed independently.
    """
    
    def __init__(
        self, 
        seq_len=512, 
        hidden_dim=256, 
        num_patches=32, 
        scalar_scales=None, 
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1, 
        transf_depth=6, 
        transf_num_heads=8, 
        transf_mlp_dim=512, 
        transf_dim_head=128,
        transf_dropout=0.1,
        pre_training=False
    ):
        super().__init__()
        assert seq_len % num_patches == 0, 'seq_len must be multiple of num_patches'
        
        patch_window_size = seq_len // num_patches
        
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.seq_len = seq_len
        self.pre_training = pre_training

        self.tokgen_unit = TokenGeneratorUnit(
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            patch_window_size=patch_window_size,
            scalar_scales=scalar_scales,
            hidden_dim_scalar_enc=hidden_dim_scalar_enc,
            epsilon_scalar_enc=epsilon_scalar_enc
        )
        
        self.vit_unit = ViTUnit(
            hidden_dim=hidden_dim, 
            num_patches=num_patches, 
            depth=transf_depth,
            heads=transf_num_heads, 
            mlp_dim=transf_mlp_dim, 
            dim_head=transf_dim_head,
            dropout=transf_dropout,
        )

        self.prj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, seq_len) - single channel time series
        Returns:
            embedding: (batch, hidden_dim)
        """
        x_embeddings = self.tokgen_unit(x)
        vit_out = self.vit_unit(x_embeddings)
        
        if self.pre_training:
            return self.prj(vit_out)
        else:
            return vit_out
