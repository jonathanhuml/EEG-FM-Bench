"""
Moment Model Implementation.

This module provides the MOMENT model adapted from the official implementation
for integration with our unified EEG framework.

Reference: AutonLab/MOMENT - momentfm/models/moment.py
"""

import logging
import math
import warnings
from argparse import Namespace

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import T5Config, T5EncoderModel, T5Model

from typing import Optional, NamedTuple


class TimeseriesOutputs(NamedTuple):
    """Output container for time series model results."""
    embeddings: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    reconstruction: Optional[torch.Tensor] = None
    forecast: Optional[torch.Tensor] = None
    anomaly_scores: Optional[torch.Tensor] = None
    pretrain_mask: Optional[torch.Tensor] = None
    input_mask: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None
    illegal_output: Optional[bool] = None


class TASKS:
    """Task type constants."""
    RECONSTRUCTION = "reconstruction"
    EMBED = "embed"
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"


class NamespaceWithDefaults(Namespace):
    """Namespace with default value support."""
    
    def getattr(self, key, default=None):
        return getattr(self, key, default)
    
    @classmethod
    def from_namespace(cls, ns: Namespace) -> 'NamespaceWithDefaults':
        return cls(**vars(ns))


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        # Set masked positions to NaN, and unmasked positions are taken from x
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# Patching
class Patching(nn.Module):
    """Patchify time series into non-overlapping or overlapping patches."""
    
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
    
    def forward(self, x):
        # x: (batch, n_channels, seq_len)
        batch, n_channels, seq_len = x.shape
        
        # Unfold to create patches
        # Output: (batch, n_channels, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        return x


# Masking utilities
class Masking:
    def __init__(
        self, mask_ratio: float = 0.3, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(
        mask: torch.Tensor, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len]
        Output
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        stride = patch_len if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
        # mask : [batch_size x n_patches x patch_len]
        return (mask.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(
        mask: torch.Tensor,
        patch_len: int = 8,
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len] or
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len] or
            [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(
            input_mask, self.patch_len, self.stride
        )
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # batch_size x 1

        batch_size, _, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(
            batch_size, n_patches, device=x.device
        )  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(
            input_mask == 1, noise, torch.ones_like(noise)
        )  # only keep the noise of observed patches

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(
            ids_shuffle, dim=1
        )  # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros(
            [batch_size, n_patches], device=x.device
        )  # mask: [batch_size x n_patches]
        for i in range(batch_size):
            mask[i, : len_keep[i]] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if (
            self.model_name == "MOMENT"
            or self.model_name == "TimesNet"
            or self.model_name == "GPT4TS"
        ):
            return self.pe[:, : x.size(2)]
        else:
            return self.pe[:, : x.size(1)]


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_len: int = 8,
        stride: int = 8,
        patch_dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding

        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(patch_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mask = Masking.convert_seq_to_patch_view(
            mask, patch_len=self.patch_len
        ).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = (
            mask.repeat_interleave(self.d_model, dim=-1)
            .unsqueeze(1)
            .repeat(1, n_channels, 1, 1)
        )
        # mask : [batch_size x n_channels x n_patches x d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


# Head modules
class PretrainHead(nn.Module):
    """Reconstruction head for pretraining."""
    
    def __init__(self, d_model: int, patch_len: int, head_dropout: float = 0.1, orth_gain: float = 1.41):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)
        
        if orth_gain is not None:
            nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()
    
    def forward(self, x):
        x = self.linear(self.dropout(x))
        x = x.flatten(start_dim=2, end_dim=3)
        return x


class ClassificationHead(nn.Module):
    """Classification head for time series classification."""
    
    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_classes: int,
        head_dropout: float = 0.1,
        reduction: str = "concat"
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.reduction = reduction
        
        if reduction == "mean":
            self.linear = nn.Linear(d_model, n_classes)
        elif reduction == "concat":
            self.linear = nn.Linear(n_channels * d_model, n_classes)
        else:
            raise ValueError(f"Reduction method {reduction} not implemented")
    
    def forward(self, x, input_mask=None):
        # x: (batch, n_patches, d_model) or (batch, n_patches, d_model * n_channels)
        x = torch.mean(x, dim=1)  # Mean across patches
        x = self.dropout(x)
        return self.linear(x)


class ForecastingHead(nn.Module):
    """Forecasting head."""
    
    def __init__(self, head_nf: int, forecast_horizon: int, head_dropout: float = 0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)
    
    def forward(self, x, input_mask=None):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


def freeze_parameters(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False
    return model


class MOMENT(nn.Module):
    """
    MOMENT: A Family of Open Time-series Foundation Models
    
    Multichannel time series model using T5 encoder backbone.
    Supports embedding, classification, forecasting, and reconstruction tasks.
    """
    
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name: str = config.task_name
        self.seq_len: int = config.seq_len
        self.patch_len: int = config.patch_len
        
        # Normalizer
        self.normalizer = RevIN(
            num_features=1, 
            affine=config.getattr("revin_affine", False)
        )
        
        # Tokenizer
        self.tokenizer = Patching(
            patch_len=config.patch_len,
            stride=config.patch_stride_len
        )
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        
        # Mask generator
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        
        # Transformer encoder
        self.encoder = self._get_transformer_backbone(config)
        
        # Task head
        self.head = self._get_head(self.task_name)
        
        # Freeze parameters if specified
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)
        
        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)
    
    def _update_inputs(self, config, **kwargs):
        if isinstance(config, dict):
            if "model_kwargs" in kwargs:
                return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
            return NamespaceWithDefaults(**config)
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config):
        if config.getattr("d_model") is None:
            if config.getattr("t5_config") and "d_model" in config.t5_config:
                config.d_model = config.t5_config["d_model"]
            else:
                raise ValueError("d_model must be specified")

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config
    
    def _get_head(self, task_name: str):
        if task_name == TASKS.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("head_dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction=self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.FORECASTING:
            num_patches = (max(self.config.seq_len, self.config.patch_len) - self.config.patch_len) // self.config.patch_stride_len + 1
            head_nf = self.config.d_model * num_patches
            return ForecastingHead(
                head_nf,
                self.config.forecast_horizon,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented")
    
    def _get_transformer_backbone(self, config):
        t5_config = T5Config.from_dict(config.t5_config)
        
        if config.getattr("randomly_initialize_backbone", False):
            transformer_backbone = T5Model(t5_config)
            logging.info(f"Initializing randomly initialized transformer from {config.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel(t5_config)
            logging.info(f"Initializing pre-trained transformer from {config.transformer_backbone}.")
        
        transformer_backbone = transformer_backbone.get_encoder()
        
        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
        
        return transformer_backbone
    
    def embed(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs
    ) -> TimeseriesOutputs:
        """
        Embed time series into latent representation.
        
        Args:
            x_enc: Input tensor of shape (batch, n_channels, seq_len)
            input_mask: Optional mask of shape (batch, seq_len)
            reduction: 'mean' to average over channels, 'none' to keep all
        
        Returns:
            TimeseriesOutputs with embeddings
        """
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)

        elif reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata= {"reduction": reduction}
        )

    
    def classify(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs
    ) -> TimeseriesOutputs:
        """
        Classify time series.
        
        Args:
            x_enc: Input tensor of shape (batch, n_channels, seq_len)
            input_mask: Optional mask of shape (batch, seq_len)
            reduction: 'mean' or 'concat' for handling channels
        
        Returns:
            TimeseriesOutputs with logits
        """
        batch_size, n_channels, seq_len = x_enc.shape
        
        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len), device=x_enc.device)
        
        # Normalize
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        
        # Patchify and embed
        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)
        
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(batch_size * n_channels, n_patches, self.config.d_model)
        
        # Create attention mask
        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        
        # Encode
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        
        # Reshape back
        enc_out = enc_out.reshape(batch_size, n_channels, n_patches, self.config.d_model)
        
        # Apply reduction
        if reduction == "mean":
            enc_out = enc_out.mean(dim=1)  # (batch, n_patches, d_model)
        elif reduction == "concat":
            enc_out = enc_out.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, self.config.d_model * n_channels
            )
        else:
            raise NotImplementedError(f"Reduction {reduction} not implemented")
        
        # Classify
        logits = self.head(enc_out, input_mask=input_mask)
        
        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata={"reduction": reduction})
    
    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs
    ) -> TimeseriesOutputs:
        """
        Forward pass based on task.
        
        Args:
            x_enc: Input tensor of shape (batch, n_channels, seq_len)
            input_mask: Optional mask of shape (batch, seq_len)
            mask: Optional mask for reconstruction
        
        Returns:
            Task-specific TimeseriesOutputs
        """
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])
        
        if self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not fully implemented in this adapter")


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    """MOMENT with HuggingFace Hub integration."""
    
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        self.new_task_name = kwargs.get("model_kwargs", {}).pop("task_name", TASKS.RECONSTRUCTION)
        super().__init__(config, **kwargs)
    
    def init(self):
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)
