#!/usr/bin/env python3
"""
BENDR Trainer using Abstract Base Class

A unified BENDR trainer that inherits from AbstractTrainer and supports multiple training patterns.
BENDR uses convolutional encoder + transformer contextualizer architecture.
"""

import logging
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import Conv1d
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classifier import MultiHeadClassifier, DynamicChannelConvRouter
from baseline.abstract.trainer import AbstractTrainer
from baseline.bendr.bendr_config import BendrConfig, BendrModelArgs
from baseline.bendr.model.trainable.layers import BENDRContextualizer, ConvEncoderBENDR


logger = logging.getLogger('baseline')


def zscore(x: Tensor) -> Tensor:
    mean = torch.mean(x, dim=(1, 2), keepdim=True)
    std = torch.std(x, dim=(1, 2), keepdim=True)
    return (x - mean) / std


class BendrDataLoaderFactory(AbstractDataLoaderFactory):
    """BENDR DataLoader factory that inherits from AbstractDataLoaderFactory."""

    def create_adapter(
            self,
            dataset: HFDataset,
            dataset_names: List[str],
            dataset_configs: List[str]
    ) -> HFDataset:
        return dataset


class BendrDynamicConvRouter(DynamicChannelConvRouter):
    def add_conv(self, mont_name: str, mont_len: int):
        # noinspection PyTypeChecker
        self.conv_router[mont_name] = Conv1d(
            mont_len, self.target_channel, 3, stride=1)


class BendrUnifiedModel(nn.Module):
    """Unified BENDR model that adapts BENDRClassifier for multi-dataset training."""
    
    def __init__(
            self,
            conv_router: BendrDynamicConvRouter,
            encoder: ConvEncoderBENDR,
            contextualizer: BENDRContextualizer,
            classifier: MultiHeadClassifier,
            grad_cam: bool,
    ):
        super().__init__()
        self.conv_router = conv_router
        self.encoder = encoder
        self.contextualizer = contextualizer
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None

    def forward(self, batch):
        x = batch['data'] * 0.001  # Shape: (batch_size, n_channels, n_timepoints)
        # x = zscore(x)
        montage = batch['montage'][0]  # Get montage from batch

        data = self.conv_router(x, montage)

        if self.grad_cam:
            self.grad_cam_activation = data.transpose(1, 2)

        # Extract features using BENDR (encoder + contextualizer)
        features = self.encoder(data)
        features = self.contextualizer(features)

        # features shape: (batch, encoder_h, seq_len)
        # Permute to: (batch, seq_len, encoder_h)
        features = torch.permute(features, (0, 2, 1))
        
        # Reshape to 4D: [B, T, C, D] where T=seq_len, C=1 (features already combined channels)
        features = features.unsqueeze(2)  # [B, seq_len, 1, encoder_h]

        # Pass montage; classifier will handle montage/ds_name split internally
        # Apply our multi-head classifier
        logits = self.classifier(features, montage)
        
        return logits


class BendrTrainer(AbstractTrainer):
    """BENDR trainer that inherits from AbstractTrainer."""
    
    def __init__(self, cfg: BendrConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = BendrDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )
        
        # Model components
        self.conv_router = None
        self.encoder = None
        self.contextualizer = None
        self.classifier = None
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Setup BENDR model architecture."""
        logger.info(f"Setting up BENDR model architecture...")
        model_cfg: BendrModelArgs = self.cfg.model

        self.encoder = ConvEncoderBENDR(
            in_features=model_cfg.max_channels,
            encoder_h=model_cfg.emb_dim,
            enc_width=tuple(model_cfg.conv_width),
            dropout=model_cfg.conv_drop_rate,
            projection_head=model_cfg.conv_proj_head,
            enc_downsample=tuple(model_cfg.conv_stride)
        )

        self.contextualizer = BENDRContextualizer(
            in_features=model_cfg.emb_dim,
            hidden_feedforward=model_cfg.ffn_dim,
            heads=model_cfg.heads,
            layers=model_cfg.context_layers,
            dropout=model_cfg.context_drop_rate,
            activation=model_cfg.activation,
            position_encoder=model_cfg.position_encoder,
            layer_drop=model_cfg.layer_drop,
            mask_p_t=model_cfg.mask_p_t,
            mask_p_c=model_cfg.mask_p_c,
            mask_t_span=model_cfg.mask_t_span,
            finetuning=True
        )

        conv_configs = {ds_name: info['config'] for ds_name, info in self.ds_info.items()}
        self.conv_router = BendrDynamicConvRouter(
            conv_configs,
            target_channel=model_cfg.max_channels,
        )
        logger.info(f"Created dynamic convolution router: {list(conv_configs.keys())}")

        # Create multi-head classifier
        embed_dim = model_cfg.emb_dim  # BENDR contextualizer outputs encoder_h features
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        
        # Get classifier head configuration
        head_cfg = model_cfg.classifier_head
        
        # Build ds_shape_info for FLATTEN_MLP head type
        # BENDR outputs features after conv downsampling: seq_len is reduced by encoder stride
        # Output shape: [B, seq_len_reduced, 1, encoder_h]
        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                # Calculate reduced sequence length after BENDR encoder
                # BENDR uses conv_stride (default: [5, 4]) for downsampling
                seq_len_reduced = n_timepoints
                for stride in model_cfg.conv_stride:
                    seq_len_reduced = seq_len_reduced // stride
                # C=1 since channel info is already combined by conv_router and encoder
                ds_shape_info[montage_key] = (seq_len_reduced, 1, embed_dim)
        
        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_info,
            t_sne=model_cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        # Load pretrained weights if available
        if self.cfg.model.pretrained_path and self.cfg.model.pretrained_conv_path:
            self.load_checkpoint([
                self.cfg.model.pretrained_path,
                self.cfg.model.pretrained_conv_path
            ])
        else:
            logger.info("No pretrained path specified, starting from scratch")

        # Create a unified model
        model = BendrUnifiedModel(
            conv_router=self.conv_router,
            encoder=self.encoder,
            contextualizer=self.contextualizer,
            classifier=self.classifier,
            grad_cam=self.cfg.model.grad_cam,
        )
        
        # Apply LoRA if enabled
        model = self.apply_lora(model)
        
        model = model.to(self.device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        self.model = model

        return model

    def load_checkpoint(self, checkpoint_path: list[str]):
        """Load separate checkpoints for encoder and contextualizer."""
        context_path = checkpoint_path[0]
        conv_path = checkpoint_path[1]

        logger.info(f"Loading separate checkpoints: {context_path} and {conv_path}")

        context_ckpt = torch.load(context_path, map_location=self.device, weights_only=False)
        conv_ckpt = torch.load(conv_path, map_location=self.device, weights_only=False)

        missing_keys, unexpected_keys = self.encoder.load_state_dict(conv_ckpt)

        if missing_keys:
            logger.warning(f"Missing keys for conv checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys for conv checkpoint: {unexpected_keys}")

        missing_keys, unexpected_keys = self.contextualizer.load_state_dict(context_ckpt)

        if missing_keys:
            logger.warning(f"Missing keys for contextualizer checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys for contextualizer checkpoint: {unexpected_keys}")

        logger.info("Successfully loaded separate checkpoints")


def main():
    """Main function to run BENDR training."""
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 2:
        print("Usage: python bendr_trainer.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    file_cfg = OmegaConf.load(config_path)
    code_cfg = OmegaConf.create(BendrConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = BendrConfig.model_validate(config_dict)
    
    # Create and run trainer
    trainer = BendrTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main() 