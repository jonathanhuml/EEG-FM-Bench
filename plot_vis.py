#!/usr/bin/env python3
"""Traditional visualization entry for baseline models only.

Usage:
    python visualize.py t_sne assets/conf/baseline/eegpt/eegpt_unified.yaml plot/configs/eegpt/tsne_config_eegpt.yaml
    python visualize.py grad_cam assets/conf/baseline/csbrain/csbrain_unified.yaml plot/configs/csbrain/gradcam_config_csbrain.yaml
    python visualize.py integrated_gradients assets/conf/baseline/reve/reve_unified.yaml plot/configs/reve/integrated_gradients_config_reve.yaml
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from baseline.abstract.config import AbstractConfig
from baseline.abstract.factory import ModelRegistry
from baseline.utils.common import seed_torch
from common.log import setup_log
from common.path import get_conf_file_path
from common.utils import setup_yaml
from plot.baseline_visualizer import BaselineVisualizer
from plot.utils.conf import load_vis_conf_dict, TsneVisArgs, GradCamVisArgs, IntegratedGradientsVisArgs

logger = logging.getLogger()


def load_model_config(config_path: str) -> AbstractConfig:
    """Load baseline model config from YAML."""
    config_path = get_conf_file_path(config_path)
    file_cfg = OmegaConf.load(config_path)
    specific_model_type = str(file_cfg.get('model_type', ''))

    if specific_model_type not in ModelRegistry.list_models():
        raise ValueError(
            f"Unsupported model_type '{specific_model_type}'. "
            f"Supported baseline models: {', '.join(ModelRegistry.list_models())}"
        )

    config_class = ModelRegistry.get_config_class(specific_model_type)
    code_cfg = OmegaConf.create(config_class().model_dump())

    cfg = OmegaConf.merge(code_cfg, file_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = config_class.model_validate(cfg)

    logger.info('change batch_size forcefully to 1')
    cfg.data.batch_size = 1

    logger.info('change pretrained model path to none')
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'pretrained_path'):
        cfg.model.pretrained_path = None

    return cfg


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Traditional visualization for baseline models")
    parser.add_argument("vis_type", choices=["t_sne", "grad_cam", "integrated_gradients"])
    parser.add_argument("model_config", help="Path to baseline model config yaml")
    parser.add_argument("vis_config", help="Path to visualization config yaml")
    args = parser.parse_args()

    setup_log()
    setup_yaml()

    logger.info(f"Starting {args.vis_type} visualization")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Visualization config: {args.vis_config}")

    if not Path(args.vis_config).exists():
        raise FileNotFoundError(f"Visualization config file not found: {args.vis_config}")

    model_config = load_model_config(args.model_config)
    logger.info(f"Loaded {type(model_config).__name__} configuration")

    if args.vis_type == 't_sne':
        vis_config: TsneVisArgs = load_vis_conf_dict(args.vis_config, args.vis_type)
        model_config.model.t_sne = True
    elif args.vis_type == 'grad_cam':
        vis_config: GradCamVisArgs = load_vis_conf_dict(args.vis_config, args.vis_type)
        model_config.model.grad_cam = True
        model_config.model.grad_cam_target = vis_config.grad_cam_target
    else:
        vis_config: IntegratedGradientsVisArgs = load_vis_conf_dict(args.vis_config, args.vis_type)

    logger.info(f'visualization config {vis_config}')
    logger.info(f'target model config {model_config}')

    model_config.data.datasets = vis_config.datasets

    seed_torch(vis_config.seed)
    visualizer = BaselineVisualizer(model_config, vis_config)
    visualizer.run()

    logger.info(f"{args.vis_type} visualization completed successfully")



if __name__ == "__main__":
    main()