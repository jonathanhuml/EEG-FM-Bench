---
name: TUEV benchmark experiment setup
description: Details of run_tuev_benchmark.py and associated configs for TUEV 6-class evaluation
type: project
---

Created run_tuev_benchmark.py to reproduce Table 4 of arXiv 2508.17742 on TUEV.

Four methods: PSD+LogReg, ZUNA, BENDR, BIOT — single-task frozen-backbone, avg_pool head.

**Why:** User wants to benchmark foundation models on TUEV epileptiform event classification (6 classes: spsw, gped, pled, eyem, artf, bckg), reproducing Table 4 of the EEG-FM-Bench paper.

**How to apply:** Run with `python run_tuev_benchmark.py --help`. The script handles environment setup (symlinks), preprocessing, training via torchrun, PSD baseline, t-SNE visualization, and results table.

Key files created (2026-03-18):
- run_tuev_benchmark.py — main orchestrator
- assets/conf/preproc/tuev_preproc_256.yaml — 256 Hz preprocessing (ZUNA+BENDR)
- assets/conf/preproc/tuev_preproc_200.yaml — 200 Hz preprocessing (BIOT)
- assets/conf/baseline/zuna/zuna_tuev.yaml — ZUNA single-task TUEV config
- assets/conf/baseline/bendr/bendr_tuev.yaml — BENDR single-task TUEV config
- assets/conf/baseline/biot/biot_tuev.yaml — BIOT single-task TUEV config
- plot/configs/tuev/tsne_{zuna,bendr,biot}.yaml — t-SNE vis configs (ckpt_path patched at runtime)

TUEV raw data: /data/datasets/bci/tuh_eeg_evals/tuh_eeg_events/v2.0.1/edf/{train,eval}/
Script auto-creates symlink: assets/data/raw/TUE/tuev → above path
Training uses: torchrun --nproc_per_node=1 (single GPU, no SLURM needed)
