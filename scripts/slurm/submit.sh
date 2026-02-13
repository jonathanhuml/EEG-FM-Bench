#!/bin/bash

cd "$(dirname "$0")/../.." || exit


# Example: submit to do dataset preprocessing with one config
sbatch scripts/slurm/preproc_submit.slurm conf_file=preproc/preproc_example.yaml

# Example: submit baseline training with one config
sbatch scripts/slurm/baseline_submit.slurm conf_file=baseline/csbrain/csbrain_unified.yaml

