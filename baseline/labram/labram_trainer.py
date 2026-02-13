import logging
import os
from functools import partial
from typing import Tuple

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.labram.labram_adapter import LabramDataLoaderFactory
from baseline.labram.labram_config import LabramConfig
from baseline.labram.model import NeuralTransformer


logger = logging.getLogger('baseline')


class LabramUnifiedModel(nn.Module):
    """Unified Labram model wrapper for multitask training."""
    
    def __init__(self, encoder, classifier, grad_cam: bool = False):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None
        
    def forward(self, batch):
        # Labram expects data in shape (batch, n_channels, n_patches, patch_size)
        # Data comes in as (batch, n_channels, n_timepoints)
        x = batch['data']
        chans_id = batch['chans_id'][0]
        montage = batch['montage'][0]
        ds_name = montage.split('/')[0]
        batch_size, n_channels, n_timepoints = x.shape
        
        # Calculate patch parameters
        patch_size = self.encoder.patch_size
        n_patches = n_timepoints // patch_size
        
        # Reshape to patches
        data = x[:, :, :n_patches * patch_size]  # Trim to fit patches
        data = data.view(batch_size, n_channels, n_patches, patch_size)

        chans_id = nn.functional.pad(chans_id+1, (1, 0), value=0)

        # Get features from encoder
        # features shape: [batch, n_ch * n_patches, embed_dim]
        features = self.encoder.forward_features(
            data,
            input_chans=chans_id,
            return_patch_tokens=True
        )

        if self.grad_cam:
            self.grad_cam_activation = features.transpose(1, 2)

        # Reshape features to 4D: [B, T, C, D]
        embed_dim = features.shape[-1]
        features = features.view(batch_size, n_channels, n_patches, embed_dim)
        features = features.permute(0, 2, 1, 3)  # [B, n_patches, n_channels, D] = [B, T, C, D]

        # Pass montage; classifier will handle montage/ds_name split internally
        logits = self.classifier(features, montage)

        return logits


class LabramTrainer(AbstractTrainer):
    """
    LABRAM trainer that inherits from AbstractTrainer.
    """
    
    def __init__(self, cfg: LabramConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Initialize dataloader factory
        self.dataloader_factory = LabramDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
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
        """Setup Labram model architecture."""
        logger.info(f"Setting up Labram model architecture...")
        cfg = self.cfg.model

        self.encoder = NeuralTransformer(
                EEG_size=cfg.eeg_size,
                patch_size=cfg.patch_size,
                in_chans=cfg.in_chans,
                out_chans=cfg.out_chans,
                num_classes=0,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=partial(nn.LayerNorm, eps=1e-6),
                qk_scale=None,
                drop_rate=cfg.dropout_rate,
                attn_drop_rate=cfg.attn_dropout_rate,
                drop_path_rate=cfg.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=cfg.init_values,
                use_abs_pos_emb=cfg.use_abs_pos_emb,
                use_rel_pos_bias=cfg.use_rel_pos_bias,
                use_shared_rel_pos_bias=cfg.use_shared_rel_pos_bias,
                use_mean_pooling=cfg.use_mean_pooling,
                init_scale=cfg.init_scale,
        )

        # Create a classifier - always use multi-head for compatibility
        embed_dim = cfg.embed_dim
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg = cfg.classifier_head

        # Build ds_shape_info for FLATTEN_MLP head type
        # Shape info: montage_key -> (n_patches, n_channels, embed_dim)
        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                n_patches = n_timepoints // cfg.patch_size
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

        model = LabramUnifiedModel(
            self.encoder,
            self.classifier,
            grad_cam=self.cfg.model.grad_cam
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
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        encoder_state_dict = {}
        for k, v in checkpoint['model'].items():
            if k.startswith('student.'):
                encoder_state_dict[k[len('student.'):]] = v

        # Load weights into encoder
        if self.encoder is not None:
            missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = [], []
        
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logger.info("Successfully loaded pretrained encoder weights")

    def pretrain_step_for_analysis(
        self,
        batch: dict,
        mask_ratio: float = 0.5,
        mask_strategy: str = "random_mixed",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pretrain step with MSE reconstruction objective.

        This performs a single pretraining step:
        1. Mask input patches
        2. Forward through encoder
        3. Reconstruct masked patches (using simple linear head)
        4. Compute MSE loss on masked positions only

        Args:
            batch: Input batch
            mask_ratio: Fraction of patches to mask
            mask_strategy: Masking strategy

        Returns:
            (loss, logits, mask): Reconstruction loss, predicted patches, mask
        """
        # Create masked batch
        masked_batch, mask, original_patches = self.create_masked_batch(
            batch, mask_ratio, mask_strategy
        )

        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp, dtype=torch.bfloat16):
            # Get encoder output
            encoder = getattr(self.model, 'encoder', self.model)

            # Forward through encoder
            # Most encoders expect [B, C, n_patches, patch_size]
            data = masked_batch['data']
            chans_id = masked_batch['chans_id'][0]
            batch_size, n_channels, n_timepoints = data.shape

            patch_size = getattr(encoder, 'patch_size', 200)
            n_patches = n_timepoints // patch_size

            # Reshape for encoder
            data_patches = data[:, :, :n_patches * patch_size].view(
                batch_size, n_channels, n_patches, patch_size
            )

            chans_id = nn.functional.pad(chans_id + 1, (1, 0), value=0)

            # Get features from encoder
            # Output shape varies by model, typically [B, C, n_patches, D] or [B, T, D]
            features = encoder(data_patches, input_chans=chans_id, return_patch_tokens=True)

            # Handle different output shapes
            if features.dim() == 3:
                # [B, T, D] - typical transformer output
                # Reshape to [B, C, n_patches, D] if possible
                if features.shape[1] == n_channels * n_patches:
                    features = features.view(batch_size, n_channels, n_patches, -1)
                else:
                    # Use as-is, project to reconstruction
                    embed_dim = features.shape[-1]
                    if self._pretrain_recon_head is None:
                        head = torch.nn.Linear(embed_dim, patch_size)
                        head = head.to(features.device).to(features.dtype)
                        target_model = getattr(self.model, "module", self.model)
                        target_model._pretrain_recon_head = head
                        self._pretrain_recon_head = head
                        if self.optimizer is not None:
                            self.optimizer.add_param_group({
                                "params": self._pretrain_recon_head.parameters(),
                                "lr": self.cfg.training.max_lr,
                            })
                    reconstructed = self._pretrain_recon_head(features)
                    # This path needs special handling - skip for now
                    raise NotImplementedError("3D output reconstruction not fully implemented")

            if features.dim() == 4:
                # [B, C, n_patches, D]
                embed_dim = features.shape[-1]

                # Create reconstruction head if not exists (register on model)
                if self._pretrain_recon_head is None:
                    head = torch.nn.Linear(embed_dim, patch_size)
                    head = head.to(features.device).to(features.dtype)
                    # Register on underlying model for checkpointing
                    target_model = getattr(self.model, "module", self.model)
                    target_model._pretrain_recon_head = head
                    self._pretrain_recon_head = head
                    # Ensure optimizer updates this head if optimizer already built
                    if self.optimizer is not None:
                        self.optimizer.add_param_group({
                            "params": self._pretrain_recon_head.parameters(),
                            "lr": self.cfg.training.max_lr,
                        })

                # Reconstruct: [B, C, n_patches, patch_size]
                reconstructed = self._pretrain_recon_head(features)
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

        # Compute MSE loss on masked positions only
        # mask: [B, C, n_patches], original_patches: [B, C, n_patches, patch_size]
        mask_expanded = mask.unsqueeze(-1).expand_as(original_patches)

        # Only compute loss on masked patches
        pred_masked = reconstructed[mask_expanded]
        target_masked = original_patches[mask_expanded]

        if pred_masked.numel() == 0:
            # No masked patches (edge case)
            loss = torch.tensor(0.0, device=reconstructed.device, requires_grad=True)
        else:
            loss = torch.nn.functional.mse_loss(pred_masked.float(), target_masked.float())

        return loss, reconstructed, mask

def main():
    """Main function for standalone execution."""
    import sys
    from omegaconf import OmegaConf
    from common.path import get_conf_file_path
    from common.utils import setup_yaml
    
    setup_yaml()
    
    if len(sys.argv) < 2:
        raise ValueError("Please provide a config file path")
    
    # Load configuration
    conf_file_path = get_conf_file_path(sys.argv[1])
    file_cfg = OmegaConf.load(conf_file_path)
    
    # Create config object
    cfg = LabramConfig.model_validate(OmegaConf.to_container(file_cfg, resolve=True))
    
    # Create and run trainer
    trainer = LabramTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main() 