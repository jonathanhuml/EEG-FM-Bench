import logging
import os
from typing import Tuple

import safetensors.torch
import torch
from optimi import StableAdamW
from torch import nn
from torch.utils.data import DataLoader

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer, format_console_log_dict
from baseline.reve.model import Reve
from baseline.reve.reve_adapter import ReveDataLoaderFactory
from baseline.reve.reve_config import ReveConfig, ReveModelArgs
from common.distributed.env import get_is_master
from common.distributed.loader import DistributedGroupBatchSampler


logger = logging.getLogger('baseline')


class ReveUnifiedModel(nn.Module):
    def __init__(
            self,
            encoder: Reve,
            classifier,
            grad_cam: bool,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None

    def forward(self, batch):
        # Reve expects data in shape (batch, n_channels, n_timepoints)
        x = batch['data']  # Shape: (batch_size, n_channels, n_timepoints)
        pos = batch['pos']  # Shape: (batch_size, n_channels, 3)
        montage = batch['montage'][0]  # Get montage from batch

        # features shape: [B, C, H, E] = [B, n_channels, n_patches, embed_dim]
        features = self.encoder(x, pos)

        # Reshape features to 4D: [B, T, C, D]
        # features is [B, C, H, E], permute to [B, H, C, E] = [B, T, C, D]
        features = features.permute(0, 2, 1, 3)

        # Keep the exact tensor used for classification so gradients can be captured
        if self.grad_cam:
            self.grad_cam_activation = features

        # Pass montage; classifier will handle montage/ds_name split internally
        logits = self.classifier(features, montage)

        return logits


class ReveTrainer(AbstractTrainer):
    def __init__(self, cfg: ReveConfig):
        super().__init__(cfg)
        self.cfg = cfg

        pos_bank_path = self.cfg.model.pos_bank_pretrained_path
        pos_bank_dict = self.load_safetensor(pos_bank_path, 'cpu')

        
        channel_restricted = True

        self.dataloader_factory = ReveDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            pos_bank_dict=pos_bank_dict,
            channel_restricted=channel_restricted,
        )

        # Model components
        self.encoder = None
        self.classifier = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.warmup_scheduler = None

        self.warmup_steps = 0
        self.warmup_freeze_state = False
        self.is_warmup_finished = False

    @staticmethod
    def compute_patches_num(n_timepoints: int, patch_size: int, patch_overlap: int) -> int:
        step = patch_size - patch_overlap
        if step <= 0:
            raise ValueError(
                f"Invalid REVE patch config: patch_size={patch_size}, patch_overlap={patch_overlap} (step={step})")
        if n_timepoints < patch_size:
            return 0
        return 1 + (n_timepoints - patch_size) // step

    def setup_model(self):
        logger.info(f"Setting up Reve model architecture...")
        cfg: ReveModelArgs = self.cfg.model

        self.encoder = Reve(cfg)

        embed_dim = cfg.embed_dim
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg = cfg.classifier_head

        # Build ds_shape_info for FLATTEN_MLP head type
        # Shape info: montage_key -> (n_patches, n_channels, embed_dim)
        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                n_patches = self.compute_patches_num(n_timepoints, cfg.patch_size, cfg.patch_overlap)
                if n_patches <= 0:
                    raise ValueError(
                        f"Dataset sample too short for REVE patching: montage={montage_key}, timepoints={n_timepoints}, "
                        f"patch_size={cfg.patch_size}, overlap={cfg.patch_overlap}"
                    )
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

        model = ReveUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            grad_cam=self.cfg.model.grad_cam,
        )
        
        # Apply LoRA if enabled
        model = self.apply_lora(model)

        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        self.model = model

        return model

    @staticmethod
    def load_safetensor(checkpoint_path: str, device: torch.device | str):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return None

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")

        checkpoint = safetensors.torch.load_file(checkpoint_path)
        checkpoint = {name: param.to(device) for name, param in checkpoint.items()}

        return checkpoint

    def load_checkpoint(self, checkpoint_path: str):
        missing_keys, unexpected_keys = [], []

        # Load encoder checkpoint
        ckpt = self.load_safetensor(checkpoint_path, self.device)
        if ckpt is not None:
            missing, unexpected = self.encoder.load_state_dict(ckpt, strict=False)
            missing_keys += missing
            unexpected_keys += unexpected

        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")

        logger.info("Successfully loaded pretrained encoder weights")

    def setup_optim_params(self, model):
        """Setup optimizer parameters with LoRA support."""
        lora_cfg = self.cfg.training.lora
        
        head_params = []
        encoder_params = []
        lora_params = []

        for name, param in model.named_parameters():
            # Check if this is a LoRA parameter
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            elif 'classifier' in name or 'conv_router' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        params = [{'params': head_params, 'lr': self.cfg.training.max_lr}]

        if lora_cfg.use_lora:
            # LoRA mode: train LoRA params + head, freeze encoder
            lora_lr = self.cfg.training.max_lr * lora_cfg.lora_lr_scale
            params.append({'params': lora_params, 'lr': lora_lr})
            
            # Freeze non-LoRA encoder parameters
            for param in encoder_params:
                param.requires_grad = False
            
            lora_param_count = sum(p.numel() for p in lora_params)
            head_param_count = sum(p.numel() for p in head_params)
            frozen_param_count = sum(p.numel() for p in encoder_params)
            
            logger.info(f"LoRA training mode:")
            logger.info(f"  - LoRA params: {lora_param_count:,} (lr={lora_lr:.2e})")
            logger.info(f"  - Head params: {head_param_count:,} (lr={self.cfg.training.max_lr:.2e})")
            logger.info(f"  - Frozen encoder params: {frozen_param_count:,}")
        else:
            # Original logic with warmup freeze support
            if self.cfg.training.freeze_encoder:
                logger.info("Encoder parameters frozen permanently, no adding to optimizer")
                for param in encoder_params:
                    param.requires_grad = False
            else:
                encoder_lr = self.cfg.training.max_lr * self.cfg.training.encoder_lr_scale
                params.append({'params': encoder_params, 'lr': encoder_lr})
                if self.cfg.training.warmup_freeze_encoder:
                    logger.info("Encoder parameters frozen for warmup")
                    self.warmup_freeze_state = True
                    # Freeze encoder parameters
                    for param in encoder_params:
                        param.requires_grad = False
                else:
                    logger.info("Encoder parameters trainable")
                    self.warmup_freeze_state = False

        return params

    def setup_optimizer_and_scheduler(self, model, train_loader: DataLoader):
        params = self.setup_optim_params(model)

        optimizer = StableAdamW(
            params,
            betas=(self.cfg.training.adam_beta_1, self.cfg.training.adam_beta_2),
            lr=self.cfg.training.max_lr,
            eps=self.cfg.training.eps,
        )

        scaler = torch.amp.GradScaler(enabled=self.cfg.training.use_amp)
        
        # Calculate scheduler parameters
        warmup_steps = len(train_loader) * self.cfg.training.warmup_epochs
        total_steps = len(train_loader) * self.cfg.training.max_epochs
        
        # Save warmup steps for tracking
        self.warmup_steps = warmup_steps
        self.is_warmup_finished = False
        
        # Warmup scheduler
        warm_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.cfg.training.min_lr / self.cfg.training.max_lr,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            threshold=1e-5,
            threshold_mode='rel',
            cooldown=0,
            min_lr=self.cfg.training.min_lr,
            eps=self.cfg.training.eps,
        )

        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = main_scheduler
        self.warmup_scheduler = warm_scheduler
        
        logger.info(f"Scheduler setup: warmup_steps={warmup_steps}, total_steps={total_steps}")

    def unfreeze_encoder(self):
        logger.info("Unfreezing encoder parameters...")
        
        # Set requires_grad=True for encoder parameters
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'conv_router' not in name:
                param.requires_grad = True

        self.warmup_freeze_state = False
        
        if get_is_master():
            logger.info(f"Encoder unfrozen, parameters now trainable")

    def on_train_step_end(self):
        """Hook called at the end of each training step for custom learning rate scheduling."""
        # Learning rate scheduling - only step warmup scheduler during warmup phase
        if not self.is_warmup_finished and self.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
        elif not self.is_warmup_finished and self.current_step >= self.warmup_steps:
            self.is_warmup_finished = True
            if get_is_master():
                logger.info(f"Warmup finished at step {self.current_step}")

    def train_epoch(self, train_loader: DataLoader, train_sampler: DistributedGroupBatchSampler):
        """Override to add warmup scheduler and custom learning rate logging."""
        self.model.train()
        if self.cfg.training.freeze_encoder:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.model.module.encoder.eval()
            else:
                self.model.encoder.eval()

        train_sampler.set_epoch(self.epoch)

        if (not self.cfg.training.freeze_encoder
            and self.cfg.training.warmup_freeze_encoder
            and self.warmup_freeze_state
            and self.epoch == self.cfg.training.warmup_freeze_encoder_epochs):
            self.unfreeze_encoder()

        batch: dict
        for step_in_epoch, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['label']
            ds_name = batch['montage'][0].split('/')[0]

            # Forward pass with mixed precision
            logits, loss = self.train_step(batch, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {self.current_step}")

            # Backward pass
            self.scaler.scale(loss).backward()
            grad_norm = self._clip_grad_norm_()

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Logging with distributed reduction
            if self.current_step % self.cfg.logging.log_step_interval == 0:
                # Calculate step accuracy
                preds = torch.argmax(logits, dim=-1)
                step_acc = (preds == labels).float().mean()

                # Create tensors for distributed reduction
                loss_tensor = loss.clone().detach()
                acc_tensor = step_acc.clone().detach()

                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.AVG)

                if get_is_master():
                    log_data = {
                        'train/epoch': self.epoch,
                        'train/step': self.current_step,
                        'train/loss_ce': loss_tensor.cpu().item(),
                        'train/acc': acc_tensor.cpu().item(),
                        'train/grad_norm': grad_norm,
                        'train/header_lr': self.get_current_lr()[0],
                    }

                    if not self.cfg.training.freeze_encoder:
                        log_data['train/encoder_lr'] = self.get_current_lr()[-1]

                    if not self.multitask:
                        log_data = {f"{ds_name}/{key}": value for key, value in log_data.items()}

                    # Log to cloud services
                    if self.cfg.logging.use_cloud:
                        self._log_to_cloud(log_data)

                    logger.info(format_console_log_dict(log_data, prefix='train'))

            self.current_step += 1
            self.on_train_step_end()


    def _compute_eval_loss(self, overall_metrics: dict) -> float:
        """Compute average evaluation loss across all datasets.

        Args:
            overall_metrics: Dictionary containing metrics for each dataset

        Returns:
            Average loss across all datasets
        """
        total_loss = torch.tensor(0.0, device=self.device)
        total_cnt = torch.tensor(0, device=self.device)

        for ds_name in self.ds_info.keys():
            total_loss += overall_metrics[ds_name]['loss_sum'].reshape([])
            total_cnt += overall_metrics[ds_name]['cnt'].reshape([])

        avg_loss = (total_loss / total_cnt.float()).cpu().item() if total_cnt > 0 else 0.0
        return avg_loss

    def on_eval_epoch_end(self, eval_loss: float):
        """Called after evaluation to step ReduceLROnPlateau scheduler.

        Args:
            eval_loss: Average evaluation loss for this epoch
        """
        if self.is_warmup_finished:
            old_lrs = self.get_current_lr()
            self.scheduler.step(eval_loss)
            new_lrs = self.get_current_lr()

            # Log if learning rate changed
            if old_lrs[0] != new_lrs[0] and get_is_master():
                logger.info(f"Learning rate reduced: {old_lrs} -> {new_lrs}")

    def eval_epoch(self, dataloaders: list[DataLoader], prefix: str):
        """Override to return average loss for scheduler."""
        # Call parent eval_epoch (which doesn't return anything)
        overall_metrics = super().eval_epoch(dataloaders, prefix)

        if prefix == 'eval':
            avg_val_loss = self._compute_eval_loss(overall_metrics)
            self.on_eval_epoch_end(avg_val_loss)

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
            pos = masked_batch['pos']  # Shape: (batch_size, n_channels, 3)
            batch_size, n_channels, n_timepoints = data.shape

            patch_size = getattr(encoder, 'patch_size', 200)
            n_patches = n_timepoints // patch_size

            # Reshape for encoder
            data_patches = data[:, :, :n_patches * patch_size]

            # Get features from encoder
            # Output shape varies by model, typically [B, C, n_patches, D] or [B, T, D]
            features = encoder(data_patches, pos)

            features = features[:, :, :n_patches, :]

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
    code_cfg = OmegaConf.create(ReveConfig().model_dump())
    merged_config = OmegaConf.merge(code_cfg, file_cfg)
    config_dict = OmegaConf.to_container(merged_config, resolve=True)
    cfg = ReveConfig.model_validate(config_dict)
    
    # Create and run trainer
    trainer = ReveTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()



