#!/bin/bash

cd "$(dirname "$0")/.." || exit

# Example: analysis stage-1 (collect gradients/features)
PYTHONPATH=$PWD python analysis_run.py \
  --config assets/conf/analysis/analysis_example.yaml \
  --trainer-config assets/conf/baseline/csbrain/csbrain_unified.yaml


