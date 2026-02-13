"""
CSBrain Trainer that inherits from AbstractTrainer.

CSBrain is a brain-region-aware EEG foundation model. This trainer:
1. Supports multi-dataset joint training with dynamic channel configurations
2. Handles brain region assignments for different electrode montages
3. Integrates with the unified multi-head classifier framework
"""

import logging
import os

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.csbrain.csbrain_adapter import CSBrainDataLoaderFactory
from baseline.csbrain.csbrain_config import CSBrainConfig, CSBrainModelArgs
from baseline.csbrain.model import CSBrain
from baseline.csbrain.utils import generate_area_config

logger = logging.getLogger('baseline')


class CSBrainUnifiedModel(nn.Module):
    """Unified CSBrain model wrapper for multitask training."""
    
    def __init__(
        self, 
        encoder: CSBrain, 
        classifier: MultiHeadClassifier,
        patch_size: int,
        grad_cam: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.patch_size = patch_size
        
        self.grad_cam = grad_cam
        self.grad_cam_activation = None
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary containing:
                - data: (batch, n_channels, n_patches * patch_size)
                - brain_regions: (n_channels,) brain region IDs
                - area_config: brain region configuration
                - montage: montage string
        """
        x = batch['data']  # Shape: (batch, n_channels, n_patches * patch_size)
        montage = batch['montage'][0]
        brain_regions = batch['brain_regions'][0]

        area_config = generate_area_config(brain_regions.cpu().tolist())
        batch_size, n_channels, n_timepoints = x.shape
        n_patches = n_timepoints // self.patch_size

        x = x.view(batch_size, n_channels, n_patches, self.patch_size)
        
        # Forward through encoder
        # Output shape: (batch, n_channels, n_patches, d_model)
        features = self.encoder(x, area_config)

        # Reshape to 4D: [B, T, C, D] where T=n_patches, C=n_channels
        # CSBrain output is (batch, n_channels, n_patches, d_model)
        # Permute to: (batch, n_patches, n_channels, d_model)
        features = features.permute(0, 2, 1, 3)  # [B, n_patches, n_channels, d_model]

        # Keep the exact tensor used for classification so gradients can be captured
        if self.grad_cam:
            self.grad_cam_activation = features

        # Pass montage; classifier will handle montage/ds_name split internally
        # Classify
        logits = self.classifier(features, montage)
        
        return logits


class CSBrainTrainer(AbstractTrainer):
    """
    CSBrain trainer that inherits from AbstractTrainer.
    
    Supports both multitask training (single shared model) and
    separate training (one model per dataset).
    """
    
    def __init__(self, cfg: CSBrainConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = CSBrainDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            patch_size=self.cfg.data.patch_size,
            max_seq_len=self.cfg.data.max_seq_len,  # None for dynamic length
        )
        
        # Model components
        self.encoder = None
        self.classifier = None
        
        # Loss function
        if self.cfg.training.label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def setup_model(self):
        """Setup CSBrain model architecture."""
        logger.info(f"Setting up CSBrain model architecture...")
        cfg: CSBrainModelArgs = self.cfg.model
        
        # Initialize encoder (seq_len is now dynamic, no need to pass fixed value)
        self.encoder = CSBrain(
            in_dim=cfg.in_dim,
            out_dim=cfg.out_dim,
            d_model=cfg.d_model,
            dim_feedforward=cfg.dim_feedforward,
            n_layer=cfg.n_layer,
            nhead=cfg.nhead,
            tem_embed_kernel_sizes=cfg.tem_embed_kernel_sizes,
        )
        
        # Create classifier - multi-head for multi-dataset support
        embed_dim = cfg.d_model
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        
        # Get classifier head configuration
        head_cfg = cfg.classifier_head
        
        # Build ds_shape_info for FLATTEN_MLP head type
        # CSBrain outputs: [B, n_patches, n_channels, d_model]
        ds_shape_info = {}
        patch_size = self.cfg.data.patch_size
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                # Calculate n_patches from n_timepoints and patch_size
                n_patches = n_timepoints // patch_size
                ds_shape_info[montage_key] = (n_patches, n_channels, embed_dim)
        
        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_info,
            t_sne=cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")
        
        if self.cfg.model.pretrained_path:
            self.load_checkpoint(self.cfg.model.pretrained_path)
        else:
            logger.info("No pretrained path specified, starting from scratch")

        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")
        
        # Create unified model
        model = CSBrainUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            patch_size=self.cfg.data.patch_size,
            grad_cam=self.cfg.model.grad_cam,
        )
        
        # Apply LoRA if enabled
        model = self.apply_lora(model)

        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        
        self.model = model
        return model

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return
        
        logger.info(f"Loading pretrained weights from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if any(key.startswith('module.') for key in checkpoint.keys()):
            checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
        missing, unexpected = self.encoder.load_state_dict(checkpoint, strict=False)
        
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")
        
        logger.info("Successfully loaded pretrained encoder weights")


def main():
    """Main function for standalone execution."""
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) < 2:
        raise ValueError("Please provide a config file path")
    
    # Load configuration
    conf_file_path = sys.argv[1]
    file_cfg = OmegaConf.load(conf_file_path)
    code_cfg = OmegaConf.create(CSBrainConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = CSBrainConfig.model_validate(config_dict)
    
    # Create and run trainer
    trainer = CSBrainTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
