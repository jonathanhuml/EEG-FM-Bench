"""
Mantis Trainer that inherits from AbstractTrainer.

Mantis is a time series foundation model using Token Generation + ViT.
This trainer:
1. Supports multi-dataset joint training with dynamic channel/time routing
2. Handles multivariate time series (EEG with multiple channels)
3. Processes channels independently through encoder then concatenates
4. Integrates with the unified multi-head classifier framework
5. Supports LoRA fine-tuning

Reference: Mantis official implementation - trainer/trainer.py
"""

import logging
import os
from typing import Optional

import safetensors.torch
import torch
from torch import nn

from baseline.abstract.classifier import (
    MultiHeadClassifier,
    DynamicTemporalConvRouter,
)

from baseline.abstract.trainer import AbstractTrainer
from baseline.mantis.mantis_adapter import MantisDataLoaderFactory
from baseline.mantis.mantis_config import MantisConfig, MantisModelArgs
from baseline.mantis.model import Mantis8M


logger = logging.getLogger('baseline')


class MantisEncoder(nn.Module):
    """
    Wrapper for Mantis model that handles multi-channel EEG data.
    
    Following Mantis official implementation, each channel is processed
    independently through the encoder and the outputs are concatenated.
    """
    
    def __init__(self, cfg: MantisModelArgs):
        super().__init__()
        self.config = cfg
        self.hidden_dim = cfg.hidden_dim
        
        # Initialize Mantis model
        self.model = Mantis8M(
            seq_len=cfg.seq_len,
            hidden_dim=cfg.hidden_dim,
            num_patches=cfg.num_patches,
            scalar_scales=cfg.scalar_scales,
            hidden_dim_scalar_enc=cfg.hidden_dim_scalar_enc,
            epsilon_scalar_enc=cfg.epsilon_scalar_enc,
            transf_depth=cfg.transf_depth,
            transf_num_heads=cfg.transf_num_heads,
            transf_mlp_dim=cfg.transf_mlp_dim,
            transf_dim_head=cfg.transf_dim_head,
            transf_dropout=cfg.transf_dropout,
            pre_training=False,  # Classification mode
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that processes each channel independently.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
            
        Returns:
            features: Tensor of shape (batch, n_channels, hidden_dim)
        """
        batch_size, n_channels, seq_len = x.shape
        
        # Process each channel independently
        # x[:, i:i+1, :] maintains the channel dimension as required by Mantis
        channel_embeddings = []
        for i in range(n_channels):
            # Extract single channel: (batch, 1, seq_len)
            x_channel = x[:, i:i+1, :]
            # Get embedding: (batch, hidden_dim)
            embedding = self.model(x_channel)
            channel_embeddings.append(embedding)
        
        # Stack: (batch, n_channels, hidden_dim)
        features = torch.stack(channel_embeddings, dim=1)
        
        return features


class MantisUnifiedModel(nn.Module):
    """Unified Mantis model wrapper for multitask classification."""
    
    def __init__(
        self, 
        encoder: MantisEncoder, 
        classifier: MultiHeadClassifier,
        conv_router: DynamicTemporalConvRouter,
        grad_cam: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.conv_router = conv_router
        
        self.grad_cam = grad_cam
        self.grad_cam_activation = None
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary containing:
                - data: (batch, n_channels, seq_len)
                - montage: montage string
        """
        x = batch['data']  # Shape: (batch, n_channels, seq_len)
        montage = batch['montage'][0]

        # Route data to fixed (B, C, seq_len)
        x = self.conv_router(x, montage)
        
        # Forward through encoder
        # Output: (batch, n_channels, hidden_dim)
        features = self.encoder(x)

        # T=1 (CLS token), C=n_channels
        features = features.unsqueeze(1)  # [B, 1, n_channels, hidden_dim]

        if self.grad_cam:
            self.grad_cam_activation = features
        
        # Classify
        logits = self.classifier(features, montage)
        
        return logits


class MantisTrainer(AbstractTrainer):
    """
    Mantis trainer that inherits from AbstractTrainer.
    
    Supports both multitask training (single shared model) and
    separate training (one model per dataset).
    """
    
    def __init__(self, cfg: MantisConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = MantisDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_seq_len=self.cfg.data.target_seq_len,
            use_zscore=True,
        )
        
        # Model components
        self.conv_router = None
        self.encoder = None
        self.classifier = None
        
        # Loss function
        if self.cfg.training.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Setup Mantis model architecture."""
        logger.info(f"Setting up Mantis model architecture...")
        model_cfg: MantisModelArgs = self.cfg.model
        data_cfg = self.cfg.data
        
        # Initialize encoder
        self.encoder = MantisEncoder(model_cfg)

        # Create router
        embed_dim = model_cfg.hidden_dim
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg = model_cfg.classifier_head

        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                ds_shape_info[montage_key] = (n_timepoints, n_channels, embed_dim)

        self.conv_router = DynamicTemporalConvRouter(
                ds_shape_info,
                target_seq_len=data_cfg.target_seq_len,
        )

        ds_shape_out_info = {}
        for montage_key, (n_timepoints, n_channels, embed_dim) in ds_shape_info.items():
            ds_shape_out_info[montage_key] = (1, n_channels, embed_dim)

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_out_info,
            t_sne=model_cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")
        
        # Load pretrained weights
        self.load_checkpoint(model_cfg.pretrained_path)
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")
        
        # Create unified model
        model = MantisUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            conv_router=self.conv_router,
            grad_cam=self.cfg.model.grad_cam,
        )
        
        # Apply LoRA if enabled
        model = self.apply_lora(model)

        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        
        self.model = model
        return model

    def load_checkpoint(self, checkpoint_path: Optional[str]):
        """Load model checkpoint from local file."""
        # Try local checkpoint first
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return None

        logger.info(f"Loading pretrained weights from local: {checkpoint_path}")
        checkpoint = safetensors.torch.load_file(checkpoint_path)

        missing, unexpected = self.encoder.model.load_state_dict(checkpoint, strict=False)

        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        logger.info("Successfully loaded pretrained encoder weights from local file")
        return checkpoint


def main():
    """Main function for standalone execution."""
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) < 2:
        raise ValueError("Please provide a config file path")
    
    # Load configuration
    conf_file_path = sys.argv[1]
    file_cfg = OmegaConf.load(conf_file_path)
    code_cfg = OmegaConf.create(MantisConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = MantisConfig.model_validate(config_dict)
    
    # Create and run trainer
    trainer = MantisTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
