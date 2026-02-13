#!/bin/bash

cd "$(dirname "$0")/.." || exit

# Example: traditional visualization (t-SNE / Grad-CAM / IG)
PYTHONPATH=$PWD python plot_vis.py \
  t_sne \
  assets/conf/baseline/csbrain/csbrain_unified.yaml \
  plot/configs/example/tsne_config_csbrain.yaml



