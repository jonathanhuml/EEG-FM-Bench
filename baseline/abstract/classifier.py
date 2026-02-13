"""
Classification heads for baseline models.
Supports multiple classification head types: AVG_POOL, ATTENTION_POOL, DUAL_STREAM_FUSION, FLATTEN_MLP.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import RMSNorm

from baseline.abstract.config import ClassifierHeadType, ClassifierHeadConfig
from baseline.utils.common import Conv1dWithConstraint
from data.processor.wrapper import get_dataset_montage


###############################################################################
#                          Rotary Positional Encoding                          #
###############################################################################

class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding for attention mechanisms."""
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 1024):
        super().__init__()
        self.dim: int = dim
        self.base = base
        self.max_seq_len_cached = max_seq_len

        arange = torch.arange(0, self.dim, 2)
        inv_freq = 1.0 / (self.base ** (arange.float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        emb = self._precompute()
        self.register_buffer("emb_cached", emb, persistent=False)

    def _precompute(self) -> Tensor:
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = torch.stack(
            [
                emb.cos()[None, None, :, :],
                emb.sin()[None, None, :, :],
            ],
            dim=0,
        )
        return emb

    def forward(self, ref_tensor: Tensor, seq_len: int) -> Tensor:
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            # noinspection PyAttributeOutsideInit
            self.emb_cached = self._precompute()

        return self.emb_cached[:, :, :, :seq_len, ...].to(ref_tensor.device)

    def reset_parameters(self):
        self.inv_freq[...] = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)
        )
        emb = self._precompute()
        self.emb_cached[...] = emb


def _rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_seq_emb(*args, enc: Tensor, seq_len: int):
    """Apply rotary positional embedding to input tensors."""
    cos = enc[0, 0, 0, :seq_len, :].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = enc[1, 0, 0, :seq_len, :].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]

    target_dtype = args[0].dtype
    if cos.dtype != target_dtype:
        cos = cos.to(dtype=target_dtype)
        sin = sin.to(dtype=target_dtype)

    out = []
    for arg in args:
        out.append((arg * cos) + (_rotate_half(arg) * sin))
    return out[0] if len(out) == 1 else out


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat key-value heads for grouped-query attention."""
    bs, seq_len, n_head, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x
        .unsqueeze(3)
        .expand(bs, seq_len, n_head, n_rep, head_dim)
        .reshape(bs, seq_len, n_head * n_rep, head_dim)
    )


###############################################################################
#                          Attention Modules                                   #
###############################################################################

class CrossAttention(nn.Module):
    """Cross attention module for classification heads."""
    def __init__(
            self,
            dim: int,
            head_dim: int,
            n_head: int,
            n_kv_head: int,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.heads_per_group = self.n_head // self.n_kv_head

        self.proj_q = nn.Linear(dim, n_head * head_dim, bias=False)
        self.proj_k = nn.Linear(dim, n_kv_head * head_dim, bias=False)
        self.proj_v = nn.Linear(dim, n_kv_head * head_dim, bias=False)

        self.proj_out = nn.Linear(n_head * head_dim, dim, bias=False)
        self.proj_dropout = nn.Dropout(self.dropout_rate)

    def forward(
            self,
            q: Tensor,
            kv: Tensor,
            q_rope: Optional[Tensor] = None,
            kv_rope: Optional[Tensor] = None,
    ) -> Tensor:
        bs, q_len, _ = q.shape
        _, kv_len, _ = kv.shape

        x_q = self.proj_q(q)
        x_k = self.proj_k(kv)
        x_v = self.proj_v(kv)

        out_shape = (bs, q_len, self.n_head * self.head_dim)

        x_q = x_q.view(bs, q_len, self.n_head, self.head_dim)
        x_k = x_k.view(bs, kv_len, self.n_kv_head, self.head_dim)
        x_v = x_v.view(bs, kv_len, self.n_kv_head, self.head_dim)

        if q_rope is not None:
            x_q = apply_rotary_seq_emb(x_q, enc=q_rope, seq_len=q_len)
        if kv_rope is not None:
            x_k = apply_rotary_seq_emb(x_k, enc=kv_rope, seq_len=kv_len)

        x_k = _repeat_kv(x_k, self.heads_per_group)
        x_v = _repeat_kv(x_v, self.heads_per_group)

        x_q, x_k, x_v = map(lambda e: e.transpose(1, 2), (x_q, x_k, x_v))

        attn_dropout_p = self.dropout_rate if self.training and self.dropout_rate > 0 else 0.0
        out = torch.nn.functional.scaled_dot_product_attention(
            x_q, x_k, x_v,
            dropout_p=attn_dropout_p,
        )

        out = out.transpose(1, 2).contiguous()
        out = self.proj_out(out.reshape(out_shape))
        out = self.proj_dropout(out)

        return out

    def reset_parameters(self):
        init_std = self.dim ** -0.5
        for m in [self.proj_q, self.proj_k, self.proj_v]:
            nn.init.trunc_normal_(m.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        nn.init.trunc_normal_(self.proj_out.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int = 256,
            ffn_dim_multiplier: Optional[float] = None,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dropout_rate = dropout_rate

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.linear_in = nn.Linear(dim, hidden_dim, bias=False)
        self.linear_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear_out(self.activation(self.linear_gate(x)) * self.linear_in(x)))

    def reset_parameters(self):
        init_std = self.dim ** -0.5
        for m in [self.linear_in, self.linear_gate, self.linear_out]:
            nn.init.trunc_normal_(m.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)


class CrossBlock(nn.Module):
    """Cross attention block with feed-forward network."""
    def __init__(
            self,
            dim: int,
            attn: CrossAttention,
            ffn: FeedForward,
    ):
        super().__init__()
        self.dim = dim
        self.attn = attn
        self.ffn = ffn

        self.attn_norm_q = RMSNorm(dim)
        self.attn_norm_kv = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
            self,
            q: Tensor,
            kv: Tensor,
            q_rope: Optional[Tensor] = None,
            kv_rope: Optional[Tensor] = None,
    ) -> tuple[Tensor, None]:
        h = q + self.attn(self.attn_norm_q(q), self.attn_norm_kv(kv), q_rope=q_rope, kv_rope=kv_rope)
        out = h + self.ffn(self.ffn_norm(h))
        return out, None

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.ffn.reset_parameters()
        self.attn_norm_q.reset_parameters()
        self.attn_norm_kv.reset_parameters()
        self.ffn_norm.reset_parameters()


###############################################################################
#                             Routers                                         #
###############################################################################

class DynamicChannelConvRouter(nn.Module):
    def __init__(self, ds_conf: dict[str, str], target_channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds_conf = ds_conf
        self.target_channel = target_channel

        self.montage_dict: dict[str, int] = dict()
        self.collect_montage()

        self.conv_router = nn.ModuleDict()
        for mont_name, mont_len in self.montage_dict.items():
            self.add_conv(mont_name, mont_len)

    def collect_montage(self):
        for ds_name, conf_name in self.ds_conf.items():
            montages = get_dataset_montage(dataset_name=ds_name, config_name=conf_name)
            for mont_name, montage in montages.items():
                self.montage_dict.update({mont_name: len(montage)})

    def add_conv(self, mont_name: str, mont_len: int):
        self.conv_router[mont_name] = Conv1dWithConstraint(
                mont_len, self.target_channel, 1, max_norm=1
            )

    def forward(self, x: Tensor, mont_name: str) -> Tensor:
        if mont_name not in self.conv_router.keys():
            raise ValueError(f"Head '{mont_name}' not found. Available heads: {list(self.conv_router.keys())}")

        x = self.conv_router[mont_name](x)
        return x


class DynamicTemporalConvRouter(nn.Module):
    def __init__(
        self,
        ds_shapes: dict[str, tuple[int, int, int]],
        target_seq_len: int,
        temporal_kernel_size: int = 9,
        temporal_dilation: int = 1,
        search_radius: int = 3,
    ):
        super().__init__()
        self.ds_shapes = ds_shapes
        self.target_seq_len = target_seq_len
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilation = temporal_dilation
        self.search_radius = search_radius

        self.temporal_router = nn.ModuleDict()

        for mont_name, (n_pts, n_chs, _) in self.ds_shapes.items():
            n_pts = n_pts
            n_chs = n_chs

            # If <= target: skip convolution; align with interpolate in forward
            if n_pts <= self.target_seq_len:
                self.temporal_router[mont_name] = nn.Identity()
                continue

            stride, kernel_size, padding = self._calculate_conv_params(n_pts, self.target_seq_len)

            conv = nn.Conv1d(
                in_channels=n_chs,
                out_channels=n_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=self.temporal_dilation,
                padding=padding,
                groups=n_chs,  # depthwise
                bias=False,
            )

            # Anti-aliasing: initialize with an averaging kernel (simple/stable; can learn a better filter)
            with torch.no_grad():
                conv.weight.fill_(1.0 / kernel_size)

            self.temporal_router[mont_name] = conv

    def forward(self, x: Tensor, mont_name: str) -> Tensor:
        # x: (B, C, L)
        if mont_name not in self.temporal_router:
            raise ValueError(f"Unknown montage '{mont_name}'. Available: {list(self.temporal_router.keys())}")

        x = self.temporal_router[mont_name](x)

        if x.size(-1) != self.target_seq_len:
            x = nn.functional.interpolate(x, size=self.target_seq_len, mode="linear", align_corners=False)

        return x

    @staticmethod
    def _out_len(l_in: int, k: int, s: int, p: int, d: int) -> int:
        return (l_in + 2 * p - d * (k - 1) - 1) // s + 1

    @staticmethod
    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def _calculate_conv_params(self, input_len: int, target_len: int) -> Tuple[int, int, int]:
        """
        Return (stride, kernel_size, padding).

        Uses lexicographic comparison of a tuple cost while keeping the cost type consistent
        (Tuple[float, ...]) to avoid IDE/type-checker warnings.
        """
        if input_len <= target_len:
            k = max(3, (int(self.temporal_kernel_size) | 1))
            d = int(self.temporal_dilation)
            p = (d * (k - 1)) // 2
            return 1, int(k), int(p)

        d = int(self.temporal_dilation)
        base_k = int(self.temporal_kernel_size)
        radius = int(self.search_radius)

        ratio = input_len / float(target_len)
        base_s = max(1, int(round(ratio)))

        s_min = max(1, base_s - radius)
        s_max = max(1, base_s + radius)

        best_cost: Optional[Tuple[float, float, float, float, float]] = None
        best: Optional[Tuple[int, int, int, int]] = None  # (s, k, p, out_len)

        for s in range(s_min, s_max + 1):
            # Increase kernel upper bound with stride; try to keep k >= s (anti-aliasing)
            k_min = max(3, base_k - 2 * radius)
            k_max = max(base_k + 2 * radius, 2 * s + 1)

            kernel_candidates = [k for k in range(k_min, k_max + 1) if (k % 2 == 1 and k <= input_len)]
            if not kernel_candidates:
                kernel_candidates = [min(input_len, max(3, base_k | 1))]

            for k in kernel_candidates:
                t = target_len
                a = s * (t - 1) - input_len + d * (k - 1) + 1
                p0 = max(0, self.ceil_div(a, 2))

                for p in range(max(0, p0 - radius), p0 + radius + 1):
                    out_len = self._out_len(input_len, k, s, p, d)
                    if out_len <= 0:
                        continue

                    len_err: float = float(abs(out_len - target_len))
                    alias_pen: float = 0.0 if k >= s else float(s - k)  # penalty when kernel < stride
                    stride_err: float = float(abs(s - ratio))
                    kernel_err: float = float(abs(k - base_k))
                    pad_pen: float = float(p)

                    cost: Tuple[float, float, float, float, float] = (
                        len_err,  # priority 1: output length closest to target
                        alias_pen,  # priority 2: reduce aliasing
                        stride_err,  # priority 3: stride close to ratio
                        kernel_err,  # keep kernel close to base_k
                        pad_pen,  # prefer smaller padding
                    )

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best = (s, k, p, out_len)

        if best is None:
            s = max(1, base_s)
            k = min(input_len, max(3, base_k | 1))
            p = (d * (k - 1)) // 2
            return int(s), int(k), int(p)

        s, k, p, _ = best
        return int(s), int(k), int(p)


###############################################################################
#                          Classification Heads                                #
###############################################################################

class AvgPoolClassificationHead(nn.Module):
    """
    Classification head using global average pooling.
    Input: [B, T, C, D] -> Output: [B, n_class]
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dims: list[int],
            n_class: int,
            dropout: float = 0.3,
            activation=nn.ELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_class = n_class

        # Global pooling to handle variable sequence lengths
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        layers = []
        input_dim = embed_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        # Add final output layer
        layers.append(nn.Linear(input_dim, n_class))

        # Combine all layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor, capture_features: bool = False) -> Tensor:
        """
        Args:
            x: Features of shape [B, T, C, D]
            capture_features: Whether to return features for t-SNE
        Returns:
            logits: [B, n_class]
        """
        b, t, c, d = x.shape
        # Reshape to [B, D, T*C] for pooling
        x = x.reshape(b, -1, d).transpose(1, 2)
        # Global average pooling: [B, D, T*C] -> [B, D, 1] -> [B, D]
        x = self.global_pool(x).squeeze(-1)

        # Fetch features for t-sne
        if capture_features:
            return x

        # Classification
        logits = self.mlp(x)
        return logits


class AttentionPoolHead(nn.Module):
    """
    Attention pooling classification head (Reve-style).
    Uses learnable query token to attend to all positions.
    Input: [B, T, C, D] -> Output: [B, n_class]
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dims: list[int],
            n_class: int,
            n_head: int = 4,
            head_dim: int = 64,
            dropout: float = 0.3,
            activation=nn.ELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = head_dim

        # Learnable query token
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.scale = head_dim ** -0.5

        # Build MLP
        layers = []
        input_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_class))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor, capture_features: bool = False) -> Tensor:
        """
        Args:
            x: Features of shape [B, T, C, D]
            capture_features: Whether to return features for t-SNE
        Returns:
            logits: [B, n_class]
        """
        b, t, c, d = x.shape
        # Flatten to [B, T*C, D]
        x = x.reshape(b, t * c, d)

        # Expand query: [1, 1, D] -> [B, 1, D]
        query = self.cls_query_token.expand(b, -1, -1)

        # Attention: [B, 1, D] x [B, D, T*C] -> [B, 1, T*C]
        attention_scores = torch.matmul(query, x.transpose(-1, -2)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Weighted sum: [B, 1, T*C] x [B, T*C, D] -> [B, 1, D] -> [B, D]
        pooled = torch.matmul(attention_weights, x).squeeze(1)

        if capture_features:
            return pooled

        logits = self.mlp(pooled)
        return logits


class DualStreamFusionHead(nn.Module):
    """
    Dual stream fusion classification head (Former-style).
    Uses cross-attention to pool temporal and channel dimensions.
    Input: [B, T, C, D] -> Output: [B, n_class]
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dims: list[int],
            n_class: int,
            mode: str = "dual",  # "time_first", "channel_first", "dual"
            n_head: int = 4,
            head_dim: int = 64,
            use_rope: bool = True,
            rope_theta: float = 10000.0,
            max_seq_len: int = 1024,
            dropout: float = 0.3,
            activation=nn.ELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        self.n_head = n_head
        self.head_dim = head_dim
        self.use_rope = use_rope

        n_kv_head = n_head  # Use same number for simplicity

        if use_rope:
            self.rope_pos_enc = RotaryPositionalEncoding(head_dim, rope_theta, max_seq_len)

        def create_cross_block():
            return CrossBlock(
                dim=embed_dim,
                attn=CrossAttention(embed_dim, head_dim, n_head, n_kv_head, dropout),
                ffn=FeedForward(embed_dim, embed_dim, dropout_rate=dropout),
            )

        dual = (mode == "dual")
        if dual:
            self.fusion_norm = RMSNorm(2 * embed_dim)
            self.fusion_head = nn.Linear(2 * embed_dim, embed_dim)

        if dual or mode == "time_first":
            self.time_first_query = nn.Parameter(torch.randn(2, 1, embed_dim))
            self.time_first_branch = nn.ModuleList([create_cross_block() for _ in range(2)])

        if dual or mode == "channel_first":
            self.ch_first_query = nn.Parameter(torch.randn(2, 1, embed_dim))
            self.ch_first_branch = nn.ModuleList([create_cross_block() for _ in range(2)])

        # Build MLP
        layers = []
        input_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_class))
        self.mlp = nn.Sequential(*layers)

    def _forward_time_first(self, x: Tensor) -> Tensor:
        b, t, c, d = x.shape

        rope = self.rope_pos_enc(x, seq_len=t) if self.use_rope else None
        x_time_context = x.permute(0, 2, 1, 3).reshape(b * c, t, d)
        q_time = self.time_first_query[0].expand(b * c, -1, -1)
        pooled_time, _ = self.time_first_branch[0](q=q_time, kv=x_time_context, kv_rope=rope)

        x_channel_context = pooled_time.view(b, c, d)
        q_channel = self.time_first_query[1].expand(b, -1, -1)
        pooled_channel, _ = self.time_first_branch[1](q=q_channel, kv=x_channel_context)

        return pooled_channel.squeeze(1)

    def _forward_channel_first(self, x: Tensor) -> Tensor:
        b, t, c, d = x.shape

        x_channel_context = x.reshape(b * t, c, d)
        q_channel = self.ch_first_query[0].expand(b * t, -1, -1)
        pooled_channel, _ = self.ch_first_branch[0](q=q_channel, kv=x_channel_context)

        rope = self.rope_pos_enc(x, seq_len=t) if self.use_rope else None
        x_time_context = pooled_channel.view(b, t, d)
        q_time = self.ch_first_query[1].expand(b, -1, -1)
        pooled_time, _ = self.ch_first_branch[1](q=q_time, kv=x_time_context, kv_rope=rope)

        return pooled_time.squeeze(1)

    def forward(self, x: Tensor, capture_features: bool = False) -> Tensor:
        """
        Args:
            x: Features of shape [B, T, C, D]
            capture_features: Whether to return features for t-SNE
        Returns:
            logits: [B, n_class]
        """
        if self.mode == 'time_first':
            pooled = self._forward_time_first(x)
        elif self.mode == 'channel_first':
            pooled = self._forward_channel_first(x)
        elif self.mode == 'dual':
            out_t_branch = self._forward_time_first(x)
            out_ch_branch = self._forward_channel_first(x)
            fused = torch.cat((out_t_branch, out_ch_branch), dim=1)
            fused = self.fusion_norm(fused)
            pooled = self.fusion_head(fused)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

        if capture_features:
            return pooled

        logits = self.mlp(pooled)
        return logits


class FlattenMLPHead(nn.Module):
    """
    Flatten MLP classification head (CSBrain-style).
    Flattens all dimensions and passes through a fixed 3-layer MLP.
    Input: [B, T, C, D] -> Output: [B, n_class]

    The MLP architecture is fixed based on dataset shape (n_patches, n_channels, dim):
    - Layer 1: (n_channels * n_patches * dim, n_patches * dim)
    - Layer 2: (n_patches * dim, dim)
    - Layer 3: (dim, n_class)

    Note: Requires shape info (n_patches, n_channels, dim) at initialization.
    """

    def __init__(
            self,
            n_patches: int,
            n_channels: int,
            dim: int,
            n_class: int,
            dropout: float = 0.3,
            activation=nn.ELU,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.n_channels = n_channels
        self.dim = dim
        self.flatten_in_dim = n_channels * n_patches * dim

        # Fixed 3-layer MLP structure following CSBrain design
        # Layer 1: (n_channels * n_patches * dim) -> (n_patches * dim)
        # Layer 2: (n_patches * dim) -> dim
        # Layer 3: dim -> n_class
        hidden1 = n_patches * dim
        hidden2 = dim

        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_in_dim, hidden1),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_class),
        )

    def forward(self, x: Tensor, capture_features: bool = False) -> Tensor:
        """
        Args:
            x: Features of shape [B, T, C, D]
            capture_features: Whether to return features for t-SNE
        Returns:
            logits: [B, n_class]
        """
        b = x.shape[0]
        # Flatten all dimensions: [B, T, C, D] -> [B, T*C*D]
        x = x.contiguous().view(b, -1)

        if capture_features:
            return x

        logits = self.mlp(x)
        return logits


###############################################################################
#                          Multi-Head Classifier                               #
###############################################################################

# Type alias for dataset shape info
# ds_shape_info: Dict[ds_name, (n_timepoints, n_channels, embed_dim)]
DatasetShapeInfo = Dict[str, tuple[int, int, int]]


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier that maintains separate classification heads
    for different datasets or montages.

    Supports multiple classification head types via configuration.
    All heads expect 4D input: [B, T, C, D]
    
    For FLATTEN_MLP head type, heads are indexed by montage_key (e.g., 'tuab/01_tcp_ar')
    because different montages have different shapes.
    For other head types, heads are indexed by ds_name (e.g., 'tuab').
    """

    def __init__(
            self,
            embed_dim: int,
            head_configs: Dict[str, int],  # {ds_name: n_class}
            head_cfg: ClassifierHeadConfig,
            ds_shape_info: Optional[DatasetShapeInfo] = None,  # Required for FLATTEN_MLP
            t_sne: bool = False,
    ):
        """
        Args:
            embed_dim: Feature dimension from encoder
            head_configs: Dict mapping ds_name -> n_classes
            head_cfg: Configuration for classification head type and parameters
            ds_shape_info: Dict mapping montage_key -> (n_timepoints, n_channels, embed_dim)
                           Required for FLATTEN_MLP head type.
                           Keys should be full montage keys like 'tuab/01_tcp_ar'.
            t_sne: Whether to capture features for t-SNE visualization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_configs = dict(head_configs)  # Make a copy: {ds_name: n_class}
        self.head_cfg = head_cfg
        self.ds_shape_info = ds_shape_info or {}  # {montage_key: (t, c, d)}
        self.t_sne = t_sne

        # Validate FLATTEN_MLP requirement
        if head_cfg.head_type == ClassifierHeadType.FLATTEN_MLP and not ds_shape_info:
            raise ValueError("ds_shape_info is required for FLATTEN_MLP head type")

        # Create separate classification heads
        # For FLATTEN_MLP: indexed by montage_key (ds_name/montage_name)
        # For other types: indexed by ds_name
        self.heads = nn.ModuleDict()
        if head_cfg.head_type == ClassifierHeadType.FLATTEN_MLP:
            # Create one head per montage_key
            for montage_key in ds_shape_info.keys():
                ds_name = montage_key.split('/')[0]
                n_class = head_configs[ds_name]
                self.add_head(montage_key, n_class)
        else:
            # Create one head per dataset
            for head_name, n_class in head_configs.items():
                self.add_head(head_name, n_class)

        self.cls_feature = None

    def _create_head(self, head_name: str, n_class: int) -> nn.Module:
        """Factory method to create classification head based on config."""
        head_type = self.head_cfg.head_type
        hidden_dims = self.head_cfg.hidden_dims
        dropout = self.head_cfg.dropout

        if head_type == ClassifierHeadType.AVG_POOL:
            return AvgPoolClassificationHead(
                embed_dim=self.embed_dim,
                hidden_dims=hidden_dims,
                n_class=n_class,
                dropout=dropout,
            )

        elif head_type == ClassifierHeadType.ATTENTION_POOL:
            return AttentionPoolHead(
                embed_dim=self.embed_dim,
                hidden_dims=hidden_dims,
                n_class=n_class,
                n_head=self.head_cfg.attn_n_head,
                head_dim=self.head_cfg.attn_head_dim,
                dropout=dropout,
            )

        elif head_type == ClassifierHeadType.DUAL_STREAM_FUSION:
            return DualStreamFusionHead(
                embed_dim=self.embed_dim,
                hidden_dims=hidden_dims,
                n_class=n_class,
                mode=self.head_cfg.fusion_mode,
                n_head=self.head_cfg.fusion_n_head,
                head_dim=self.head_cfg.fusion_head_dim,
                use_rope=self.head_cfg.fusion_use_rope,
                rope_theta=self.head_cfg.fusion_rope_theta,
                max_seq_len=self.head_cfg.fusion_max_seq_len,
                dropout=dropout,
            )

        elif head_type == ClassifierHeadType.FLATTEN_MLP:
            if head_name not in self.ds_shape_info:
                raise ValueError(f"Shape info for dataset '{head_name}' not found in ds_shape_info")
            n_patches, n_channels, embed_dim = self.ds_shape_info[head_name]
            return FlattenMLPHead(
                n_patches=n_patches,
                n_channels=n_channels,
                dim=embed_dim,
                n_class=n_class,
                dropout=dropout,
            )

        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, x: Tensor, montage: str) -> Tensor:
        """
        Forward pass using specified classification head.

        Args:
            x: Features of shape [B, T, C, D]
            montage: Full montage key (e.g., 'tuab/01_tcp_ar')

        Returns:
            logits: [B, n_class] for the specified head
        """
        # For FLATTEN_MLP, use full montage_key; for others, extract ds_name
        if self.head_cfg.head_type == ClassifierHeadType.FLATTEN_MLP:
            head_name = montage
        else:
            head_name = montage.split('/')[0]  # Extract ds_name from montage
            
        if head_name not in self.heads:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {list(self.heads.keys())}")

        if self.t_sne:
            # Capture shared features for t-SNE visualization
            shared_features = self.heads[head_name](x, capture_features=True)
            self.cls_feature = shared_features.clone()

        # Continue with normal forward pass for logits
        logits = self.heads[head_name](x, capture_features=False)
        return logits

    def add_head(self, head_name: str, n_class: int):
        """Add a new classification head.
        
        For FLATTEN_MLP: head_name should be montage_key (e.g., 'tuab/01_tcp_ar')
        For other types: head_name should be ds_name (e.g., 'tuab')
        """
        self.heads[head_name] = self._create_head(head_name, n_class)
        # Note: we don't update head_configs here since it uses ds_name as key

    def get_available_heads(self) -> list[str]:
        """Get a list of available classification heads."""
        return list(self.heads.keys())
