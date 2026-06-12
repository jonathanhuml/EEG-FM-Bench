"""
DREAMER EEG Emotion Dataset
============================
A low-cost wireless EEG dataset for emotion recognition collected using an Emotiv
Epoc+ headset (14 channels, 128 Hz). 23 participants watched 18 emotionally
evocative film clips; each trial is labelled with a valence and an arousal rating
on a 1–5 Likert scale.

Dataset structure (expected on disk)::

    $EEGFM_DATABASE_RAW_ROOT/DREAMER/
    └── DREAMER.mat           (single .mat file, all subjects and trials)

Source / download: https://zenodo.org/records/546113

Citation::

    @article{katsigiannis2017dreamer,
      title   = {DREAMER: A database for emotion recognition through EEG and ECG
                 signals from wireless low-cost off-the-shelf devices},
      author  = {Katsigiannis, Stamos and Ramzan, Naeem},
      journal = {IEEE Journal of Biomedical and Health Informatics},
      volume  = {22}, number = {1}, pages = {98--107},
      year    = {2017}, publisher = {IEEE},
      doi     = {10.1109/JBHI.2016.2612843}
    }

Preprocessing (identical to dreamer/preprocess.py for reproducibility):
  1.  Baseline subtraction — subtract per-channel mean of baseline recording.
  2.  Resample 128 → 256 Hz — polyphase resampling (scipy.signal.resample_poly).
  3.  High-pass filter — 4th-order Butterworth at 0.5 Hz (full trial before windowing).
  4.  Notch filter — IIR notch at 50 Hz, Q = 30.
  5.  Z-score normalization — per channel over the full trial.
  6.  Non-overlapping 5 s windows — 1280 samples @ 256 Hz; trailing samples dropped.
  Labels: valence / arousal integer scores 1–5 mapped to 0-indexed class labels 0–4.

Split protocol (matches compare_models.py exactly):
  All windows from all subjects are pooled, then split **at the window level**
  using a stratified shuffle (by label) — 80% train / 10% valid / 10% test.
  Subject identity is not used to assign splits; the same subject can appear
  in train and test, mirroring the Stratified 5-Fold CV used in the original
  dreamer experiments.

Builder configs:
  pretrain          — no labels (whole-trial windows)
  finetune_valence  — 5-class valence classification
  finetune_arousal  — 5-class arousal classification

Usage::

    builder = DreamerBuilder('finetune_valence')
    builder.preproc()
    builder.download_and_prepare(num_proc=4)
    dataset = builder.as_dataset()
"""

import json
import logging
import os
from dataclasses import dataclass, field
from math import gcd
from typing import Optional, Union, Any

import datasets
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from numpy import ndarray
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger('preproc')


# ── Preprocessing constants (must match dreamer/preprocess.py exactly) ────────
_ORIG_FS = 128
_TARGET_FS = 256
_WINDOW_SEC = 5
_WINDOW_SAMPLES = _WINDOW_SEC * _TARGET_FS   # 1280
_HP_FREQ = 0.5         # Hz — high-pass cut-off
_NOTCH_FREQ = 50.0     # Hz — UK power-line
_NOTCH_Q = 30.0

# Emotiv Epoc+ 14-channel electrode names in standard 10-20 notation
# Order matches the DREAMER.mat channel ordering
_CH_NAMES = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
]
_MONTAGE_KEY = 'emotiv_14'


# ── Signal processing helpers (mirrors dreamer/preprocess.py exactly) ─────────

def _resample_dreamer(data: ndarray) -> ndarray:
    """Polyphase resample (n_samples, n_ch) from _ORIG_FS to _TARGET_FS."""
    up = _TARGET_FS // gcd(_ORIG_FS, _TARGET_FS)
    down = _ORIG_FS // gcd(_ORIG_FS, _TARGET_FS)
    return scipy.signal.resample_poly(data, up, down, axis=0)


def _highpass_dreamer(data: ndarray) -> ndarray:
    """4th-order Butterworth high-pass at _HP_FREQ applied to full trial."""
    nyq = _TARGET_FS / 2
    b, a = scipy.signal.butter(4, _HP_FREQ / nyq, btype='high')
    return scipy.signal.filtfilt(b, a, data, axis=0)


def _notch_dreamer(data: ndarray) -> ndarray:
    """IIR notch at _NOTCH_FREQ with Q = _NOTCH_Q."""
    nyq = _TARGET_FS / 2
    b, a = scipy.signal.iirnotch(_NOTCH_FREQ / nyq, Q=_NOTCH_Q)
    return scipy.signal.filtfilt(b, a, data, axis=0)


def _zscore_dreamer(data: ndarray) -> ndarray:
    """Z-score per channel over the full trial."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return (data - mean) / std


def process_dreamer_trial(stimulus: ndarray, baseline: ndarray) -> ndarray:
    """
    Full preprocessing pipeline for one DREAMER trial.

    Arguments
    ---------
    stimulus : ndarray, shape (n_samples, 14)
        Raw EEG recorded during the video clip.
    baseline : ndarray, shape (n_samples_base, 14)
        Pre-trial resting-state baseline recording.

    Returns
    -------
    windows : ndarray, shape (n_windows, 14, 1280), float32
        Non-overlapping 5 s windows; trailing samples dropped.
    """
    # 1. Baseline subtraction (per channel)
    data = stimulus - baseline.mean(axis=0, keepdims=True)

    # 2. Resample 128 → 256 Hz (polyphase)
    data = _resample_dreamer(data)

    # 3. High-pass filter (applied to full trial to avoid edge artefacts)
    data = _highpass_dreamer(data)

    # 4. Notch filter at 50 Hz
    data = _notch_dreamer(data)

    # 5. Z-score per channel
    data = _zscore_dreamer(data)

    # 6. Slice into non-overlapping 5 s windows
    n_windows = len(data) // _WINDOW_SAMPLES
    if n_windows == 0:
        return np.empty((0, len(_CH_NAMES), _WINDOW_SAMPLES), dtype=np.float32)
    data = data[: n_windows * _WINDOW_SAMPLES]
    # (n_windows, WINDOW_SAMPLES, n_ch) → (n_windows, n_ch, WINDOW_SAMPLES)
    windows = data.reshape(n_windows, _WINDOW_SAMPLES, data.shape[1]).transpose(0, 2, 1)
    return windows.astype(np.float32)


# ── Configuration dataclasses ─────────────────────────────────────────────────

@dataclass
class DreamerConfig(EEGConfig):
    name: str = 'pretrain'
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "DREAMER: EEG emotion recognition dataset. 23 subjects, 18 film-clip trials each. "
        "Emotiv Epoc+ headset, 14 channels, 128 Hz original. Preprocessed: baseline "
        "subtraction, 128→256 Hz polyphase resample, 0.5 Hz high-pass (Butterworth 4th "
        "order), 50 Hz IIR notch (Q=30), per-channel z-score, 5 s non-overlapping windows."
    )
    citation: Optional[str] = """\
    @article{katsigiannis2017dreamer,
      title   = {DREAMER: A database for emotion recognition through EEG and ECG
                 signals from wireless low-cost off-the-shelf devices},
      author  = {Katsigiannis, Stamos and Ramzan, Naeem},
      journal = {IEEE Journal of Biomedical and Health Informatics},
      volume  = {22}, number  = {1}, pages   = {98--107},
      year    = {2017}, publisher = {IEEE},
      doi     = {10.1109/JBHI.2016.2612843}
    }
    """

    # Override filter params to match dreamer/preprocess.py:
    #   high-pass at 0.5 Hz only — no explicit low-pass; notch at 50 Hz.
    # NOTE: These are informational; the actual filtering is done inside
    # process_dreamer_trial() which bypasses the MNE pipeline.
    filter_low: float = 0.5
    filter_notch: float = 50.0
    is_notched: bool = True   # mark as already notched so builder doesn't re-notch

    dataset_name: Optional[str] = 'dreamer'
    task_type: DatasetTaskType = DatasetTaskType.EMOTION
    file_ext: str = 'mat'     # single DREAMER.mat file

    montage: dict[str, list[str]] = field(default_factory=lambda: {
        _MONTAGE_KEY: list(_CH_NAMES),
    })

    # Window-level stratified split (matches compare_models.py StratifiedKFold on windows).
    # Subject identity is ignored; same subject can appear in train and test.
    valid_ratio: float = 0.10   # 10% of windows → valid
    test_ratio: float = 0.10    # 10% of windows → test  (80% → train)

    # 5 s windows, matching dreamer/preprocess.py
    wnd_div_sec: int = _WINDOW_SEC

    # Path to the folder containing DREAMER.mat
    suffix_path: str = 'DREAMER'
    scan_sub_dir: str = ''

    # Which emotion dimension to label (set per config variant below)
    emotion_target: str = 'valence'

    # 5-class ratings (score 1–5 mapped to labels 0–4)
    category: list[str] = field(
        default_factory=lambda: ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']
    )


# ── Builder ───────────────────────────────────────────────────────────────────

class DreamerBuilder(EEGDatasetBuilder):
    """
    EEG-FM-Bench dataset builder for DREAMER.

    The builder overrides ``preproc()`` entirely because DREAMER is stored as a
    single .mat file (all 23 subjects × 18 trials) rather than one file per
    recording.  All signal processing is performed by ``process_dreamer_trial()``,
    which is a direct transcription of dreamer/preprocess.py for reproducibility.
    """

    BUILDER_CONFIG_CLASS = DreamerConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name='pretrain'),
        DreamerConfig(
            name='finetune_valence',
            is_finetune=True,
            emotion_target='valence',
        ),
        DreamerConfig(
            name='finetune_arousal',
            is_finetune=True,
            emotion_target='arousal',
        ),
    ]

    def __init__(self, config_name: str = 'pretrain', **kwargs):
        super().__init__(config_name, **kwargs)

        # The base class writes summary metadata into raw_path/summary/, which
        # is read-only for DREAMER (the .mat lives in a shared dataset directory).
        # Redirect to database_proc_root so preproc can write freely.
        conf = self.config
        self.summary_path = os.path.join(
            conf.database_proc_root, 'summary', conf.dataset_name, conf.name
        )
        self.info_csv_path = os.path.join(
            self.summary_path,
            f'{self.dataset_name}_{conf.name}_info.csv',
        )
        self.mid_file_csv_path = os.path.join(
            self.summary_path,
            f'{self.dataset_name}_{conf.name}_{conf.get_fs_id()}_cache_files.csv',
        )

    # ── Abstract method stubs ─────────────────────────────────────────────────
    # These are not reached during normal operation (preproc() is fully
    # overridden), but must be present to satisfy the ABC contract.

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        # Not used — preproc() handles DREAMER.mat directly
        raise NotImplementedError("DreamerBuilder does not use per-file resolution.")

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        raise NotImplementedError("DreamerBuilder does not use per-file resolution.")

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        raise NotImplementedError("DreamerBuilder does not use per-file resolution.")

    def _divide_split(self, df: DataFrame) -> DataFrame:
        return self._divide_all_split_by_sub(df)

    def standardize_chs_names(self, montage: str) -> list[str]:
        return self.config.montage[montage]

    # ── Core preprocessing override ───────────────────────────────────────────

    def preproc(self, n_proc: Optional[int] = None):
        """
        Load DREAMER.mat, apply the exact same preprocessing pipeline as
        dreamer/preprocess.py, and write parquet files for the EEG-FM-Bench
        intermediate cache.

        Split protocol matches compare_models.py:
          All windows from all subjects are pooled, then split at the window
          level using a stratified shuffle by label (80% train / 10% valid /
          10% test).  Subject identity is not used to assign splits, so the
          same subject can appear in train and test — identical to the
          Stratified 5-Fold CV on windows used in the original dreamer repo.
        """
        if self._is_preproc_cached():
            logger.info(f'Using cached DREAMER summary at {self.mid_file_csv_path}')
            return

        np.random.seed(self.config.seed)
        self.clean_disk_cache()
        self.create_dir_structure()

        mat_path = os.path.join(self.config.raw_path, 'DREAMER.mat')
        logger.info(f'Loading {mat_path} …')
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        dreamer = mat['DREAMER']
        subjects = dreamer['Data']
        n_subjects = int(dreamer['noOfSubjects'])
        n_videos = int(dreamer['noOfVideoSequences'])

        # ── Channel index mapping ─────────────────────────────────────────────
        chs_idx = self._fetch_chs_index(_MONTAGE_KEY)
        montage_key = f'{self.config.dataset_name}/{_MONTAGE_KEY}'

        # ── Pass 1: collect every window from every subject/trial ─────────────
        all_examples: list[dict] = []
        all_labels: list[int] = []   # parallel list for stratified split

        for s_idx, subj in enumerate(subjects):
            subj_id = s_idx + 1
            eeg_stim   = subj['EEG']['stimuli']
            eeg_base   = subj['EEG']['baseline']
            scores_val = subj['ScoreValence']
            scores_aro = subj['ScoreArousal']

            for t_idx in range(n_videos):
                stimulus = np.array(eeg_stim[t_idx], dtype=np.float64)
                baseline = np.array(eeg_base[t_idx], dtype=np.float64)

                windows = process_dreamer_trial(stimulus, baseline)  # (n_wnd, 14, 1280)
                if len(windows) == 0:
                    continue

                # Shift 1-indexed scores to 0-indexed labels
                y_valence = int(scores_val[t_idx]) - 1
                y_arousal = int(scores_aro[t_idx]) - 1
                label = y_valence if self.config.emotion_target == 'valence' else y_arousal

                for wnd in windows:
                    example: dict = {
                        'data': wnd.flatten().astype(np.float32),
                        'chs': chs_idx,
                        'montage': montage_key,
                        'task': self.config.task_type.value,
                        'subject': str(subj_id),
                    }
                    if self.config.is_finetune:
                        example['label'] = label
                    all_examples.append(example)
                    all_labels.append(label if self.config.is_finetune else 0)

            logger.info(f'  Subject {subj_id:02d}/{n_subjects} processed')

        n_total = len(all_examples)
        logger.info(f'Total windows collected: {n_total}')

        # ── Pass 2: stratified window-level split (by label) ──────────────────
        # Mirrors StratifiedKFold(n_splits=5) from compare_models.py:
        # each window is assigned independently of its subject.
        labels_arr = np.array(all_labels)
        train_idx, valid_idx, test_idx = self._stratified_window_split(
            labels_arr,
            valid_ratio=self.config.valid_ratio,
            test_ratio=self.config.test_ratio if self.config.is_finetune else 0.0,
            seed=self.config.seed,
        )

        split_map = {}
        for i in train_idx:
            split_map[i] = 'train'
        for i in valid_idx:
            split_map[i] = 'valid'
        for i in test_idx:
            split_map[i] = 'test'

        logger.info(
            f'Window split — train: {len(train_idx)}, '
            f'valid: {len(valid_idx)}, test: {len(test_idx)}'
        )

        # ── Pass 3: write one parquet per split ───────────────────────────────
        split_examples: dict[str, list[dict]] = {'train': [], 'valid': [], 'test': []}
        for i, example in enumerate(all_examples):
            split_examples[split_map[i]].append(example)

        mid_records: list[dict] = []
        for split_name, examples in split_examples.items():
            if not examples or (split_name == 'test' and not self.config.is_finetune):
                continue
            df = pd.DataFrame(examples)
            filename = f'{split_name}.parquet'
            output_path = self._build_output_dir(split_name, filename)
            df.to_parquet(
                output_path,
                compression=self.config.mid_compress_algo,
                engine='pyarrow',
                index=False,
            )
            mid_records.append({'key': filename, 'split': split_name, 'cnt': len(examples)})
            logger.info(f'  Wrote {split_name}.parquet ({len(examples)} windows)')

        mid_df = pd.DataFrame(mid_records)
        mid_df.to_csv(self.mid_file_csv_path, index=False)
        self._mark_preproc_done()
        logger.info('DREAMER preprocessing complete.')

    @staticmethod
    def _stratified_window_split(
        labels: ndarray,
        valid_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Stratified shuffle split at the window level, by label.

        For each class, windows are shuffled then partitioned into test /
        valid / train in that order.  Mirrors sklearn StratifiedKFold on
        the full window pool (no subject-level grouping).
        """
        rng = np.random.default_rng(seed)
        train_idx, valid_idx, test_idx = [], [], []

        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0].copy()
            rng.shuffle(cls_idx)
            n = len(cls_idx)
            n_test  = max(1, int(n * test_ratio))  if test_ratio  > 0 else 0
            n_valid = max(1, int(n * valid_ratio)) if valid_ratio > 0 else 0
            test_idx.extend(cls_idx[:n_test])
            valid_idx.extend(cls_idx[n_test:n_test + n_valid])
            train_idx.extend(cls_idx[n_test + n_valid:])

        return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


if __name__ == '__main__':
    for config in ('finetune_valence', 'finetune_arousal'):
        builder = DreamerBuilder(config)
        builder.preproc()
        builder.download_and_prepare(num_proc=4)
        dataset = builder.as_dataset()
        print(f'\n{config}:')
        print(dataset)
