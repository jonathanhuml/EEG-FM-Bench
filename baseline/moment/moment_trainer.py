"""
MOMENT Trainer that inherits from AbstractTrainer.

MOMENT is a family of open time-series foundation models using T5 encoder backbone.
This trainer:
1. Supports multi-dataset joint training with dynamic channel/time routing
2. Handles multivariate time series (EEG with multiple channels)
3. Uses T5 encoder with patch embedding for time series processing
4. Integrates with the unified multi-head classifier framework
5. Supports LoRA fine-tuning

Reference: AutonLab/MOMENT - momentfm/models/moment.py
"""

import logging
import os
from typing import Optional

import torch
from torch import nn
import safetensors.torch

from baseline.abstract.classifier import (
    MultiHeadClassifier,
    DynamicTemporalConvRouter,
)

from baseline.abstract.trainer import AbstractTrainer
from baseline.moment.model import MOMENTPipeline
from baseline.moment.moment_adapter import MomentDataLoaderFactory
from baseline.moment.moment_config import MomentConfig, MomentModelArgs


logger = logging.getLogger('baseline')


class MomentEncoder(nn.Module):
    """
    Wrapper for MOMENT model that handles multi-channel EEG data.
    
    MOMENT processes all channels together through the T5 encoder.
    The output is a per-patch embedding that can be aggregated for classification.
    """
    
    def __init__(self, cfg: MomentConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.model.d_model
        self.patch_len = cfg.model.patch_len
        self.seq_len = cfg.model.seq_len
        
        # Calculate number of patches
        self.n_patches = (self.seq_len - self.patch_len) // cfg.model.patch_stride_len + 1
        
        # Build MOMENT config
        moment_config = self.cfg.build_moment_config()

        self.moment = MOMENTPipeline(config=moment_config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MOMENT encoder.
        
        Args:
            x: Input tensor of shape (batch, n_channels, seq_len)
            
        Returns:
            features: Tensor of shape (batch, n_channels, n_patches, d_model)
                      or (batch, n_patches, d_model) depending on reduction
        """
        batch_size, n_channels, seq_len = x.shape
        
        # Create input mask (all valid)
        input_mask = torch.ones((batch_size, seq_len), device=x.device)
        
        # Get embeddings from MOMENT
        outputs = self.moment(x_enc=x, input_mask=input_mask, reduction="none")
        
        # embeddings shape: (batch, n_channels, n_patches, d_model)
        embeddings = outputs.embeddings
        
        return embeddings


class MomentUnifiedModel(nn.Module):
    """Unified MOMENT model wrapper for multitask classification."""
    
    def __init__(
        self, 
        encoder: MomentEncoder, 
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
        # Output: (batch, n_channels, n_patches, d_model)
        features = self.encoder(x)
        
        # Permute to (batch, n_patches, n_channels, d_model) for classifier
        # T=n_patches, C=n_channels
        features = features.permute(0, 2, 1, 3)  # [B, n_patches, n_channels, d_model]

        if self.grad_cam:
            self.grad_cam_activation = features
        
        # Classify
        logits = self.classifier(features, montage)
        
        return logits


class MomentTrainer(AbstractTrainer):
    """
    MOMENT trainer that inherits from AbstractTrainer.
    
    Supports both multitask training (single shared model) and
    separate training (one model per dataset).
    """
    
    def __init__(self, cfg: MomentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = MomentDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_seq_len=self.cfg.data.target_seq_len,
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
        """Setup MOMENT model architecture."""
        logger.info(f"Setting up MOMENT model architecture...")
        model_cfg: MomentModelArgs = self.cfg.model
        data_cfg = self.cfg.data
        
        # Get dataset shape info
        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                ds_shape_info[montage_key] = (n_timepoints, n_channels, model_cfg.d_model)

        # Create temporal router
        self.conv_router = DynamicTemporalConvRouter(
            ds_shape_info,
            target_seq_len=data_cfg.target_seq_len,
        )
        
        # Initialize encoder with target channels
        self.encoder = MomentEncoder(self.cfg)
        
        # Calculate output shape for classifier
        # n_patches = (seq_len - patch_len) // patch_stride + 1
        n_patches = (data_cfg.target_seq_len - model_cfg.patch_len) // model_cfg.patch_stride_len + 1
        embed_dim = model_cfg.d_model

        # Create classifier head configs
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg = model_cfg.classifier_head

        # Output shape info for classifier: (n_patches, n_channels, embed_dim)
        ds_shape_out_info = {}
        for montage_key, (n_timepoints, n_channels, _) in ds_shape_info.items():
            ds_shape_out_info[montage_key] = (n_patches, n_channels, embed_dim)

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_out_info,
            t_sne=model_cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        self.load_checkpoint(model_cfg.pretrained_path)
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")
        
        # Create unified model
        model = MomentUnifiedModel(
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
        """Load model checkpoint - handled by MOMENTPipeline.from_pretrained."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return None

        logger.info(f"Loading pretrained weights from local: {checkpoint_path}")
        checkpoint = safetensors.torch.load_file(checkpoint_path)

        missing, unexpected = self.encoder.moment.load_state_dict(checkpoint, strict=False)

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
    code_cfg = OmegaConf.create(MomentConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = MomentConfig.model_validate(config_dict)
    
    # Create and run trainer
    trainer = MomentTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
