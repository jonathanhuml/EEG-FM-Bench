#!/bin/bash

cd "$(dirname "$0")/.." || exit

# Example: analysis stage-2 (single-seed visualization)
PYTHONPATH=$PWD python analysis_vis.py \
  --data-dir analysis_results/scratch_vs_pretrained_YYYYMMDD_HHMMSS









