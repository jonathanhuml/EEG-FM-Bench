---
name: EEG dataset paths on this machine
description: Actual filesystem locations of EEG datasets
type: reference
---

TUEV (TUH EEG Events v2.0.1): /data/datasets/bci/tuh_eeg_evals/tuh_eeg_events/v2.0.1/edf/{train,eval}/
Things-EEG-2: /data/datasets/bci/eeg2image/things-eeg2/
Workload: /data/datasets/bci/workload/
SEED: /data/datasets/bci/seed/
HBN: /data/datasets/bci/HBN_EEG/
Mimul-11: /data/datasets/bci/Mimul-11/

Framework path convention: EEGFM_DATABASE_RAW_ROOT/{suffix_path}/
TUEV suffix_path = TUE/tuev → symlink must point to the v2.0.1 directory.
run_tuev_benchmark.py creates this symlink automatically.
