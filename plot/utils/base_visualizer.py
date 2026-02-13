import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Callable
from collections import defaultdict

import mne
import numpy as np
import seaborn
import torch
from torch import Tensor, autocast
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from captum.attr import IntegratedGradients, NoiseTunnel

from baseline.abstract.trainer import AbstractTrainer
from common.utils import ElectrodeSet
from data.processor.wrapper import get_dataset_n_class, get_dataset_category
from plot.utils.conf import TsneVisArgs, GradCamVisArgs, IntegratedGradientsVisArgs
from common.distributed.env import clean_torch_distributed


logger = logging.getLogger("plot_vis")


class BaseVisualizer(ABC):
    def __init__(self, model_config, vis_args):
        self.cfg = model_config
        self.vis_args = vis_args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_size: int = 1
        self.rank: int = 0
        self.local_rank: int = 0
        self.eps = 1e-8

        self.electrode_set = ElectrodeSet()

        # Dataset metadata dictionary
        self.ds_dict: dict[str, dict] = {
            ds_name: {
                'idx': idx,
                'config': ds_config,
                'n_class': get_dataset_n_class(ds_name, ds_config),
                'category': get_dataset_category(ds_name, ds_config)
            }
            for idx, (ds_name, ds_config) in enumerate(self.vis_args.datasets.items())
        }

        # Model-related
        self.model = None
        self.trainer: Optional[AbstractTrainer] = None

        # Feature collection for t-SNE
        self.feature_collection: dict[str, Tensor] = {}
        self.label_collection: dict[str, Tensor] = {}

        # Grad-CAM related
        self.target_layer = None
        self.gradients = None
        self.activations = None
        self.handles = []
        self.cam_val_collection: dict = {}

        # Integrated Gradients related
        self.attribution_collection: dict[str, dict[str, list]] = {}

        # Channel name mapping
        self.ch_name_map: dict[str, str] = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz', 'AFZ': 'AFz',
            'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz', 'PZ': 'Pz',
            'POZ': 'POz', 'OZ': 'Oz', 'IZ': 'Iz',
        }

    def get_model_from_ddp(self):
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    @abstractmethod
    def build_model(self):
        """Build the model (must be implemented by subclasses)."""
        pass

    @abstractmethod
    def create_dataloader(self, ds_name, ds_config) -> DataLoader:
        """Create the dataloader (must be implemented by subclasses)."""
        pass

    @abstractmethod
    def extract_model_t_sne_features(self, ds_name: str) -> Tensor:
        """Extract features from the model (must be implemented by subclasses)."""
        pass

    @abstractmethod
    def find_target_layer(self):
        """Find the target layer for Grad-CAM."""
        pass

    def forward_step(self, batch):
        """Forward step (subclasses may override)."""
        batch = self._normalize_csbrain_regions_for_vis(batch)
        device_type = self.device.type
        use_amp_cfg = getattr(getattr(self.cfg, "training", None), "use_amp", True)
        enable_amp = device_type == "cuda" and use_amp_cfg and self.vis_args.model_type != 'biot'

        with autocast(device_type=device_type, enabled=enable_amp, dtype=torch.bfloat16):
            logits = self.model(batch)
        return logits

    def _normalize_csbrain_regions_for_vis(self, batch: dict) -> dict:
        """Visualization-only: normalize CSBrain `brain_regions` into contiguous segments.

        This avoids slicing issues caused by non-contiguous region indices.
        """
        if getattr(self.vis_args, 'model_type', None) != 'csbrain':
            return batch
        if 'brain_regions' not in batch:
            return batch

        brain_regions = batch['brain_regions']
        if not isinstance(brain_regions, torch.Tensor):
            return batch

        if brain_regions.dim() == 2:
            regions = brain_regions[0].tolist()
        else:
            regions = brain_regions.tolist()

        new_regions = []
        last_region = None
        new_id = -1
        for region in regions:
            if region != last_region:
                new_id += 1
                last_region = region
            new_regions.append(new_id)

        new_tensor = torch.tensor(new_regions, device=brain_regions.device, dtype=brain_regions.dtype)
        if brain_regions.dim() == 2:
            new_tensor = new_tensor.unsqueeze(0)

        new_batch = dict(batch)
        new_batch['brain_regions'] = new_tensor
        return new_batch

    def load_checkpoint(self):
        """Load checkpoint (generic implementation)."""
        logger.info(f"Loading checkpoint from: {self.vis_args.ckpt_path}")

        try:
            if self.vis_args.ckpt_path.endswith('.gz'):
                import gzip
                with gzip.open(self.vis_args.ckpt_path, "rb") as f:
                    # noinspection PyTypeChecker
                    ckpt = torch.load(f, weights_only=False, map_location=self.device)
            else:
                ckpt = torch.load(self.vis_args.ckpt_path, weights_only=False, map_location=self.device)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return

        # Handle different checkpoint formats
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        # Remove DDP wrapper prefix
        state_dict = {
            k.replace("module.", ""): v.to(dtype=torch.float32)
            for k, v in state_dict.items()
        }

        try:
            if hasattr(self.model, 'module'):
                missing_keys, unexpected_keys = self.model.module.load_state_dict(state_dict, strict=False)
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                logger.warning(f"Missing keys for checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys for checkpoint: {unexpected_keys}")

        except Exception as e:
            logger.warning(f"Failed to load state dict: {e}")

    # ===== t-SNE methods =====
    def extract_features(self, dataloader: DataLoader, ds_name: str):
        """Extract features for t-SNE visualization."""
        self.model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            batch: dict
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.vis_args.num_batch:
                    break

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                labels = batch.get('label', batch.get('labels')).cpu()

                try:
                    with autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                        _ = self.forward_step(batch)

                    features = self.extract_model_t_sne_features(ds_name).float()
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(labels)

                except Exception as e:
                    logger.warning(f"Failed to extract features from batch {batch_idx}: {e}")
                    continue

        if features_list:
            self.feature_collection[ds_name] = torch.concat(features_list, dim=0)
            self.label_collection[ds_name] = torch.concat(labels_list, dim=0)
        else:
            logger.warning(f"No features extracted for dataset {ds_name}")

    def visualize_tsne(self, ds_name: str):
        """Create t-SNE visualization."""
        if ds_name not in self.feature_collection:
            logger.warning(f"No features collected for dataset {ds_name}")
            return

        categories = self.ds_dict[ds_name]['category']
        features = self.feature_collection[ds_name].numpy()
        labels = self.label_collection[ds_name].numpy()

        logger.info(f"Creating t-SNE for {ds_name}: {features.shape} features, {len(categories)} classes")

        # Apply PCA (if requested)
        if hasattr(self.vis_args, 'use_pca') and self.vis_args.use_pca:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            pca = PCA(n_components=min(features.shape[0], self.vis_args.pca_dims))
            features = pca.fit_transform(features)
            logger.info(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.2f}")

        # Choose perplexity based on dataset
        # TODO
        if ds_name not in ['workload', 'tusl']:
            perplexity = self.vis_args.perplexity
        else:
            perplexity = self.vis_args.small_perplexity

        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.vis_args.seed,
            n_jobs=4,
            max_iter=self.vis_args.max_iter,
        )
        embeddings = tsne.fit_transform(features)

        # Create visualization
        plt.figure(figsize=(6, 5))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        palette = seaborn.color_palette(n_colors=len(categories))

        for class_id, class_name in enumerate(categories):
            mask = (labels == int(class_id))
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                color=palette[int(class_id)],
                label=class_name,
                alpha=0.8,
                edgecolor='w',
                linewidth=0.3
            )

        plt.legend(
            bbox_to_anchor=(0.5, -0.1),
            loc='upper center',
            borderaxespad=0.,
            ncol=len(categories),
            handletextpad=0.2,
            borderpad=0.5,
            prop={'size': 12},
        )
        plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.15)

        plt.title(
            f"{ds_name} t-sne {self.vis_args.split}",
            fontname='Times New Roman',
            fontsize=14,
            pad=10
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"tsne_{self.vis_args.model_type}_{ds_name}_{self.vis_args.split}_{timestamp}.png"
        save_path = os.path.join(self.vis_args.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

        logger.info(f"Saved t-sne plot to {save_path}")

    # ===== Grad-CAM methods =====
    def register_hooks(self):
        """Register hooks for Grad-CAM."""
        self.target_layer = self.find_target_layer()
        if self.target_layer is None:
            raise ValueError(f"Could not find target layer with grad_cam_activation")

        def forward_hook(module, input_tensor, output_tensor):
            if hasattr(module, 'grad_cam_activation') and module.grad_cam_activation is not None:
                self.activations = module.grad_cam_activation
            else:
                self.activations = output_tensor
            
            if isinstance(self.activations, (list, tuple)):
                self.activations = self.activations[0]
            
            self.activations.retain_grad()

        handle = self.target_layer.register_forward_hook(forward_hook)
        self.handles.append(handle)

    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.gradients = None
        self.activations = None

    def _infer_time_channel_dims(self, input_data: torch.Tensor) -> tuple[int, int]:
        """Infer (n_timestep, n_channel) from input data.

        Supports shapes [B, C, T] or [B, T, C].
        """
        if input_data.dim() != 3:
            raise ValueError(f"Unsupported input_data shape: {input_data.shape}")

        _, dim1, dim2 = input_data.shape

        if dim1 == dim2:
            # Degenerate case: treat the smaller dimension as channels
            if dim1 <= 256:
                return dim2, dim1
            return dim1, dim2

        # Usually: n_channel < n_timestep
        if dim1 < dim2:
            return dim2, dim1  # [B, C, T]
        return dim1, dim2  # [B, T, C]

    def _normalize_gradient_activation_shape(self, gradient: torch.Tensor, activation: torch.Tensor, input_data: torch.Tensor):
        """Normalize gradient/activation shapes to [batch_size, n_timestep, n_channel, d_model]."""
        _, n_channel = self._infer_time_channel_dims(input_data)

        if gradient.shape != activation.shape:
            raise ValueError(f"Gradient shape {gradient.shape} doesn't match activation shape {activation.shape}")

        # 4D: [B, T, C, D] or [B, C, T, D]
        if gradient.dim() == 4:
            if gradient.shape[2] == n_channel:
                return gradient, activation
            if gradient.shape[1] == n_channel:
                return gradient.transpose(1, 2), activation.transpose(1, 2)
            raise ValueError(f"Unsupported gradient/activation 4D shape: {gradient.shape} with n_channel={n_channel}")

        # 3D: [B, T, C] or [B, C, T]
        if gradient.dim() == 3:
            if gradient.shape[2] == n_channel:
                return gradient.unsqueeze(-1), activation.unsqueeze(-1)
            if gradient.shape[1] == n_channel:
                grad_norm = gradient.transpose(1, 2).unsqueeze(-1)
                act_norm = activation.transpose(1, 2).unsqueeze(-1)
                return grad_norm, act_norm
            raise ValueError(f"Unsupported gradient/activation 3D shape: {gradient.shape} with n_channel={n_channel}")

        raise ValueError(f"Unsupported gradient/activation shape: {gradient.shape}")

    def generate_cam(self, batch, labels):
        """Generate Grad-CAM."""
        self.model.eval()
        self.register_hooks()
        self.model.zero_grad()

        logits = self.forward_step(batch)
        predictions = logits.argmax(dim=1)

        if self.vis_args.label_option == 'pred':
            sel = predictions
        elif self.vis_args.label_option == 'truth':
            sel = labels
        else:
            raise ValueError(f"Unsupported label: {self.vis_args.label_option}")

        # Backpropagation
        batch_size = logits.size(0)
        target_score = logits[torch.arange(batch_size), sel].sum()
        target_score.backward()

        self.gradients = self.activations.grad
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients/Activations not captured")

        # Compute CAM
        gradient = self.gradients.clone().detach().to(dtype=torch.float64)
        activation = self.activations.clone().detach().to(dtype=torch.float64)

        # Get input data for shape normalization
        input_data = batch['data']  # [batch_size, time, channel] or [batch_size, channel, time]
        
        # Normalize gradient/activation shapes
        gradient, activation = self._normalize_gradient_activation_shape(gradient, activation, input_data)
        # Now gradient/activation are shaped as [batch_size, n_timestep, n_channel, d_model]

        # CAM computation using normalized shapes
        if self.vis_args.grad_cam_target == 'channel':
            # Average over time and d_model dimensions to obtain weights
            weights = torch.mean(gradient, dim=[1, 3], keepdim=True)  # [batch_size, 1, n_channel, 1]
            cam = torch.sum(weights * activation, dim=[1, 3])  # [batch_size, n_channel]
        elif self.vis_args.grad_cam_target == 'temporal':
            # Average over channel and d_model dimensions to obtain weights
            weights = torch.mean(gradient, dim=[2, 3], keepdim=True)  # [batch_size, n_timestep, 1, 1]
            cam = torch.sum(weights * activation, dim=[2, 3])  # [batch_size, n_timestep]
        else:
            raise ValueError(f"Unsupported target layer: {self.vis_args.grad_cam_target}")

        self.remove_hooks()

        cam = torch.nn.functional.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + self.eps)

        return cam.detach().cpu().numpy(), predictions.detach().cpu().numpy()

    def visualize_cam(self, cam: np.ndarray, save_path: str, chs: list[str], show: bool = True):
        """Visualize CAM results."""
        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")

        if self.vis_args.grad_cam_target == 'channel':
            try:
                if len(cam.shape) == 2:
                    if cam.shape[0] > 1:
                        raise ValueError("CAM not supported for batch size > 1")
                    else:
                        cam = cam.squeeze(0)

                montage = mne.channels.make_standard_montage('easycap-M1')

                exclude = []
                ch_names = []
                for idx, ch in enumerate(chs):
                    if ch in self.ch_name_map:
                        ch = self.ch_name_map[ch]

                    if ch not in montage.ch_names:
                        exclude.append(idx)
                    else:
                        ch_names.append(ch)

                if len(ch_names) > 0:
                    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
                    info.set_montage(montage)

                    cam_filtered = np.delete(cam, exclude, axis=0)
                    mne.viz.plot_topomap(
                        cam_filtered, info, show=False, contours=0,
                        sensors=True, outlines='head', res=64, cmap='RdBu_r', axes=ax
                    )
            except Exception as e:
                logger.warning(f"Failed to create topographic map: {e}")
                ax.bar(range(len(cam)), cam)
                ax.set_title("Channel Grad-CAM")
        else:
            heatmap_data = cam.reshape(1, -1)
            seaborn.heatmap(heatmap_data, cmap='plasma', yticklabels=False, ax=ax)
            ax.set_title("Temporal Grad-CAM")

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()

        plt.close(fig)

    # ===== Integrated Gradients methods =====
    def _create_forward_func(self, batch_template: dict, target_key: str = 'data') -> Callable:
        """Create a forward function used for attribution."""
        # Keep all original values except for `target_key`
        fixed_batch = {k: v for k, v in batch_template.items() if k != target_key}
        
        def forward_func(input_tensor: Tensor) -> Tensor:
            # Create a new batch: keep everything fixed, replace only `target_key`
            batch_dict = fixed_batch.copy()
            batch_dict[target_key] = input_tensor
            
            batch_dict = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_dict.items()
            }
            
            return self.forward_step(batch_dict)
        
        return forward_func

    def _get_baseline(self, input_tensor: Tensor) -> Tensor:
        """Generate the baseline input."""
        if self.vis_args.baseline_type == 'zero':
            return torch.zeros_like(input_tensor)
        elif self.vis_args.baseline_type == 'random':
            return torch.rand_like(input_tensor) * 300 - 150
        elif self.vis_args.baseline_type == 'gaussian':
            return torch.randn_like(input_tensor) * input_tensor.std() + input_tensor.mean()
        else:
            raise ValueError(f"Unsupported baseline type: {self.vis_args.baseline_type}")

    def generate_integrated_gradients_attribution(self, batch: dict, labels: Tensor) -> np.ndarray:
        """Generate attributions using IntegratedGradients + NoiseTunnel."""
        model = self.get_model_from_ddp()
        model.eval()

        target_key = 'data'
        input_tensor = batch[target_key]
        forward_func = self._create_forward_func(batch, target_key)
        
        # Create IntegratedGradients
        ig = IntegratedGradients(forward_func)
        # Wrap with NoiseTunnel
        nt = NoiseTunnel(ig)
        
        baseline = self._get_baseline(input_tensor)
        # Captum supports a `target` list aligned with the batch dimension
        target_class = labels.tolist()
        
        try:
            # Use NoiseTunnel-wrapped IntegratedGradients
            attribution = nt.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class,
                n_steps=self.vis_args.n_steps,
                nt_type=self.vis_args.noise_tunnel_type,
                nt_samples=self.vis_args.noise_tunnel_samples,
                stdevs=self.vis_args.noise_tunnel_stdevs
            )
            
            # Return attributions for the entire batch; downstream code can visualize per-sample
            return attribution.detach().cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Failed to compute IntegratedGradients: {e}")
            raise e

    def visualize_attribution(self, attribution: np.ndarray, save_path: str, chs: list[str] = None, show: bool = True):
        """Visualize attribution results."""
        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
        if len(attribution.shape) > 2:
            attribution = attribution.squeeze(0)
        # Normalize attribution values
        attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + self.eps)

        if hasattr(self.vis_args, 'ig_target'):
            target = self.vis_args.ig_target
        else:
            target = 'channel'  # default

        if target == 'channel' and chs is not None:
            # Channel-level: average over time, then draw topomap
            channel_avg_attr: np.ndarray = np.mean(attr_norm, axis=1)
            channel_avg_attr = (channel_avg_attr - channel_avg_attr.mean()) / channel_avg_attr.std()
            try:
                montage = mne.channels.make_standard_montage('easycap-M1')

                exclude = []
                ch_names = []
                for idx, ch in enumerate(chs):
                    if ch in self.ch_name_map:
                        ch = self.ch_name_map[ch]

                    if ch not in montage.ch_names:
                        exclude.append(idx)
                    else:
                        ch_names.append(ch)

                if len(ch_names) > 0:
                    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
                    info.set_montage(montage)

                    attr_filtered = np.delete(channel_avg_attr, exclude, axis=0)
                    mne.viz.plot_topomap(
                        attr_filtered, info, show=False, contours=0,
                        sensors=True, outlines='head', res=64, cmap='RdBu_r', axes=ax
                    )
            except Exception as e:
                logger.warning(f"Failed to create topographic map: {e}")
                ax.bar(range(len(channel_avg_attr)), channel_avg_attr)
                ax.set_title("Channel Attribution")
        else:
            # Temporal-level: average over channels, then draw time heatmap
            time_avg_attr = np.mean(attr_norm, axis=0)
            heatmap_data = time_avg_attr.reshape(1, -1)
            seaborn.heatmap(heatmap_data, cmap='plasma', yticklabels=False, ax=ax)
            ax.set_title("Temporal Attribution")
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        if show:
            plt.show()

        plt.close(fig)

    # ===== Data collection =====
    def collect_class_average_data(self, ds_name: str, data: np.ndarray, 
                                 labels: np.ndarray, predictions: np.ndarray, 
                                 chs: list[str], vis_type: str):
        """Collect data for computing class-average visualizations."""
        if not self.vis_args.generate_class_average:
            return

        ds_info = self.ds_dict[ds_name]
        
        # batch_size=1: handle types carefully
        if isinstance(predictions, np.ndarray):
            pred = predictions.item() if predictions.size == 1 else predictions[0]
        else:
            pred = predictions
            
        if isinstance(labels, np.ndarray):
            truth = labels.item() if labels.size == 1 else labels[0]
        else:
            truth = labels[0].item() if hasattr(labels[0], 'item') else labels[0]
        
        # Only collect correctly predicted samples
        if pred == truth:
            class_name = ds_info['category'][truth]
            
            # Append directly into the corresponding collection
            if vis_type == 'grad_cam':
                self.cam_val_collection[ds_name][class_name].append({
                    'val': data.copy(),  # ensure we copy the data
                    'chs': chs.copy() if chs else [],
                })
            else:  # integrated_gradients
                self.attribution_collection[ds_name][class_name].append({
                    'val': data.copy(),  # ensure we copy the data
                    'chs': chs.copy() if chs else [],
                })

    def visualize_class_average(self, ds_name: str, vis_type: str = 'grad_cam'):
        """Generate class-average visualizations for each class."""
        if vis_type == 'grad_cam':
            collection = self.cam_val_collection.get(ds_name, {})
            method_name = 'grad_cam'
        else:
            collection = self.attribution_collection.get(ds_name, {})
            method_name = 'integrated_gradients'

        if not collection:
            return
        
        class_avg_dir = os.path.join(self.vis_args.output_dir, ds_name, f'{vis_type}_class_average')
        os.makedirs(class_avg_dir, exist_ok=True)

        for class_name, data_list in collection.items():
            if not data_list:
                continue

            # Collect all data
            all_data = []
            all_channels = set()
            
            for item in data_list:
                all_data.append(item['val'].squeeze(0))
                if item['chs']:
                    all_channels.update(item['chs'])
            
            if not all_data:
                continue
                
            # Remove reference electrodes
            if 'A1' in all_channels:
                all_channels.remove('A1')
            if 'A2' in all_channels:
                all_channels.remove('A2')

            # Create a canonical channel mapping
            channel_index_map = {ch: idx for idx, ch in enumerate(all_channels)}
            channel_count = np.zeros(len(all_channels))
            if vis_type == 'grad_cam':
                avg_data = np.zeros(len(all_channels))
            else:
                avg_data = np.zeros((len(all_channels),all_data[0].shape[-1]))

            # Accumulate data
            for item in data_list:
                sample_data = item['val'].squeeze(0)
                sample_chs = item['chs']

                for ch_idx, ch_name in enumerate(sample_chs):
                    if ch_name in ['A1', 'A2']:
                        continue
                    if ch_name in channel_index_map:
                        std_idx = channel_index_map[ch_name]
                        avg_data[std_idx] += sample_data[ch_idx]
                        channel_count[std_idx] += 1

            # Compute averages
            mask = channel_count > 0
            if vis_type == 'grad_cam':
                avg_data[mask] /= channel_count[mask]
            else:
                avg_data[mask] /= channel_count[mask][:, np.newaxis]

            filename = f'avg_{self.vis_args.split}_{method_name.lower()}_{class_name}_{len(data_list)}_samples.png'
            save_path = os.path.join(class_avg_dir, filename)

            if vis_type == 'grad_cam':
                self.visualize_cam(avg_data, save_path, list(all_channels), show=False)
            else:
                self.visualize_attribution(avg_data, save_path, list(all_channels), show=False)

            logger.info(f"Generated average {method_name} for class {class_name} with {len(data_list)} samples")

    # ===== Per-sample visualization =====
    def visualize_samples(self, ds_name: str, data: np.ndarray, batch_idx: int,
                         chs: list[str], labels: np.ndarray, predictions: np.ndarray,
                         vis_type: str = 'grad_cam'):
        """Visualize a single sample (batch_size=1)."""
        ds_info = self.ds_dict[ds_name]
        
        # batch_size=1: handle types carefully
        if isinstance(labels, np.ndarray):
            truth = labels.item() if labels.size == 1 else labels[0]
        else:
            truth = labels[0].item() if hasattr(labels[0], 'item') else labels[0]
            
        if isinstance(predictions, np.ndarray):
            pred = predictions.item() if predictions.size == 1 else predictions[0]
        else:
            pred = predictions

        truth_label = ds_info['category'][truth]
        pred_label = ds_info['category'][pred]

        if hasattr(self.vis_args, 'label_option') and self.vis_args.label_option == 'truth':
            output_dir = os.path.join(
                self.vis_args.output_dir, ds_name, vis_type,
                f'{truth_label}_{self.vis_args.label_option}'
            )
            filename = f'{self.vis_args.split}_b{batch_idx}_tru_{truth_label}_pre_{pred_label}.png'
        else:
            output_dir = os.path.join(
                self.vis_args.output_dir, ds_name, vis_type,
                f'{pred_label}_pred'
            )
            filename = f'{self.vis_args.split}_b{batch_idx}_pre_{pred_label}_tru_{truth_label}.png'

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)

        if vis_type == 'grad_cam':
            self.visualize_cam(data, save_path, chs, show=False)
        elif vis_type == 'integrated_gradients':
            self.visualize_attribution(data, save_path, chs, show=False)

    # ===== Main entrypoints =====
    def run(self):
        if isinstance(self.vis_args, TsneVisArgs):
            self.run_tsne()
        elif isinstance(self.vis_args, GradCamVisArgs):
            self.run_grad_cam()
        elif isinstance(self.vis_args, IntegratedGradientsVisArgs):
            self.run_integrated_gradients()
        else:
            raise NotImplementedError(f"Unsupported visualization args type: {type(self.vis_args)}")

        if self.trainer is not None:
            clean_torch_distributed(self.trainer.local_rank)

    def run_tsne(self):
        """Run t-SNE visualization."""
        self.vis_args.output_dir = os.path.join(self.vis_args.output_dir, datetime.now().strftime('%y%m%d%H%M%S'))
        os.makedirs(self.vis_args.output_dir, exist_ok=True)
        self.vis_args.dump_to_yaml(path=os.path.join(self.vis_args.output_dir, 't_sne_vis_conf.yaml'))

        self.build_model()

        for ds_name, ds_info in self.ds_dict.items():
            logger.info(f"Processing dataset: {ds_name}")

            try:
                dataloader = self.create_dataloader(ds_name, ds_info['config'])
                self.extract_features(dataloader, ds_name)
                self.visualize_tsne(ds_name)

            except Exception as e:
                logger.error(f"Failed to process dataset {ds_name}: {e}")
                continue

        logger.info("t-SNE visualization completed")

    def run_grad_cam(self):
        """Run Grad-CAM visualization."""
        self.vis_args.output_dir = os.path.join(self.vis_args.output_dir, datetime.now().strftime('%y%m%d%H%M%S'))
        os.makedirs(self.vis_args.output_dir, exist_ok=True)
        self.vis_args.dump_to_yaml(path=os.path.join(self.vis_args.output_dir, 'grad_cam_vis_conf.yaml'))

        self.build_model()

        for ds_name, ds_info in self.ds_dict.items():
            logger.info(f"Processing dataset: {ds_name}")
            
            # Initialize collectors
            if self.vis_args.generate_class_average:
                self.cam_val_collection[ds_name] = defaultdict(list)

            try:
                dataloader = self.create_dataloader(ds_name, ds_info['config'])

                batch: dict
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= self.vis_args.num_batch:
                        break

                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    labels = batch.get('label', batch.get('labels')).cpu().numpy()
                    chs = self.electrode_set.get_electrodes_name(
                        batch["chs"][0].tolist()
                    ) if "chs" in batch else []

                    try:
                        cam, predictions = self.generate_cam(batch, labels)
                        if self.vis_args.generate_per_sample:
                            self.visualize_samples(ds_name, cam, batch_idx, chs, labels, predictions, 'grad_cam')

                        self.collect_class_average_data(ds_name, cam, labels, predictions, chs, 'grad_cam')

                    except Exception as e:
                        logger.warning(f"Failed to generate Grad-CAM for batch {batch_idx}: {e}")
                        continue

                # Generate class-average visualizations
                if self.vis_args.generate_class_average:
                    self.visualize_class_average(ds_name, 'grad_cam')

            except Exception as e:
                logger.error(f"Failed to process dataset {ds_name}: {e}")
                continue

        logger.info("Grad-CAM visualization completed")

    def run_integrated_gradients(self):
        """Run Integrated Gradients visualization."""
        self.vis_args.output_dir = os.path.join(self.vis_args.output_dir, datetime.now().strftime('%y%m%d%H%M%S'))
        os.makedirs(self.vis_args.output_dir, exist_ok=True)
        self.vis_args.dump_to_yaml(path=os.path.join(self.vis_args.output_dir, 'integrated_gradients_vis_conf.yaml'))
        
        self.build_model()
        
        for ds_name, ds_info in self.ds_dict.items():
            logger.info(f"Processing dataset: {ds_name}")
            
            # Initialize collectors
            if self.vis_args.generate_class_average:
                self.attribution_collection[ds_name] = defaultdict(list)
            
            try:
                dataloader = self.create_dataloader(ds_name, ds_info['config'])

                batch: dict
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= self.vis_args.num_batch:
                        break
                    
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    labels = batch.get('label', batch.get('labels')).cpu().numpy()
                    chs = self.electrode_set.get_electrodes_name(
                        batch["chs"][0].tolist()
                    ) if "chs" in batch else []
                    
                    try:
                        # Get predictions
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.forward_step(batch)
                        predictions = logits.argmax(dim=1).cpu().numpy()
                        
                        # Generate attributions
                        attribution = self.generate_integrated_gradients_attribution(batch, labels)
                        
                        if attribution is not None:
                            if self.vis_args.generate_per_sample:
                                self.visualize_samples(ds_name, attribution, batch_idx, chs, labels, predictions, 'integrated_gradients')
                            self.collect_class_average_data(ds_name, attribution, labels, predictions, chs, 'integrated_gradients')
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate attribution for batch {batch_idx}: {e}")
                        continue
                
                # Generate class-average visualizations
                if self.vis_args.generate_class_average:
                    self.visualize_class_average(ds_name, 'integrated_gradients')
                    
            except Exception as e:
                logger.error(f"Failed to process dataset {ds_name}: {e}")
                continue
        
        logger.info("IntegratedGradients visualization completed")
