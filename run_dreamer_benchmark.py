#!/usr/bin/env python3
"""
run_dreamer_benchmark.py — DREAMER emotion EEG benchmark
=========================================================

Replicates the compare_models.py setup from the dreamer repository, replacing
the sklearn LogisticRegression with the FM-Bench PyTorch MLP head.

Two emotion classification tasks (5-class each):
  • Valence  — arousal score 1–5 (low→high valence)
  • Arousal  — arousal score 1–5 (low→high arousal)

Two methods, frozen-backbone:
  1. PSD   — Welch log-power spectra (parameter-free encoder)
  2. ZUNA  — Zyphra/ZUNA frozen encoder (auto-downloads from HuggingFace)

Preprocessing (identical to dreamer/preprocess.py):
  baseline subtraction → polyphase resample 128→256 Hz → 0.5 Hz high-pass
  (Butterworth 4th-order) → 50 Hz IIR notch (Q=30) → per-channel z-score
  → 5 s non-overlapping windows (1280 samples @ 256 Hz)

Prerequisites
-------------
  1. Raw DREAMER data at:
       /data/datasets/bci/dataset_downloads_cw/DREAMER/DREAMER.mat

  2. Python environment with all repo dependencies installed:
       source env.sh

Usage
-----
  # Full run (preprocess → train valence → train arousal → table)
  python run_dreamer_benchmark.py

  # Use GPU 4
  python run_dreamer_benchmark.py --gpu 4

  # Skip preprocessing if already done
  python run_dreamer_benchmark.py --skip-preproc

  # Run only specific models
  python run_dreamer_benchmark.py --models psd zuna

  # Run only one emotion target
  python run_dreamer_benchmark.py --targets valence

  # Override classifier head type
  python run_dreamer_benchmark.py --head-type avg_pool

Design decisions
----------------
  - DREAMER is a single .mat file; DreamerBuilder.preproc() handles it directly.
  - Valence and arousal share the same preprocessing but are trained separately.
  - ZUNA features are cached under assets/data/cache/zuna_dreamer_{valence,arousal}.
  - PSD has a parameter-free encoder; no pretrained weights exist or are needed.
  - Cloud logging (wandb/comet) is disabled; all logs go to assets/run/.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()

# DREAMER raw data location on this machine (single .mat file)
DREAMER_RAW_SRC = Path("/data/datasets/bci/dataset_downloads_cw/DREAMER")

LOCAL_RAW_ROOT   = REPO_ROOT / "assets" / "data" / "raw"
LOCAL_PROC_ROOT  = REPO_ROOT / "assets" / "data" / "processed"
LOCAL_CACHE_ROOT = REPO_ROOT / "assets" / "data" / "cache"

PREPROC_CFG = "assets/conf/preproc/dreamer_preproc_256.yaml"

# Model configs keyed by (model, target)
MODEL_CONFIGS: dict[tuple[str, str], str] = {
    ("psd",  "valence"): "assets/conf/baseline/psd/psd_dreamer_valence.yaml",
    ("psd",  "arousal"): "assets/conf/baseline/psd/psd_dreamer_arousal.yaml",
    ("zuna", "valence"): "assets/conf/baseline/zuna/zuna_dreamer_valence.yaml",
    ("zuna", "arousal"): "assets/conf/baseline/zuna/zuna_dreamer_arousal.yaml",
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ── Head-type CLI overrides ───────────────────────────────────────────────────

HEAD_TYPE_OVERRIDES: dict[str, list[str]] = {
    "avg_pool":               ["model.classifier_head.head_type=avg_pool"],
    "attention_pool":         ["model.classifier_head.head_type=attention_pool"],
    "flatten_mlp":            ["model.classifier_head.head_type=flatten_mlp"],
    "time_first":             ["model.classifier_head.head_type=dual_stream_fusion",
                               "model.classifier_head.fusion_mode=time_first"],
    "channel_first":          ["model.classifier_head.head_type=dual_stream_fusion",
                               "model.classifier_head.fusion_mode=channel_first"],
    "dual":                   ["model.classifier_head.head_type=dual_stream_fusion",
                               "model.classifier_head.fusion_mode=dual"],
    "linear_pool":            ["model.classifier_head.head_type=avg_pool",
                               "model.classifier_head.hidden_dims=[]"],
    "linear_pool_time_first": ["model.classifier_head.head_type=dual_stream_fusion",
                               "model.classifier_head.fusion_mode=time_first",
                               "model.classifier_head.hidden_dims=[]"],
}

# ── Environment setup ─────────────────────────────────────────────────────────

def setup_environment() -> None:
    """
    Ensure environment variables and symlink structure are in place.

    Expected layout after this function:
        assets/data/raw/DREAMER/  →  symlink to DREAMER_RAW_SRC
    """
    LOCAL_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_PROC_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    dreamer_link = LOCAL_RAW_ROOT / "DREAMER"
    if not dreamer_link.exists():
        if not DREAMER_RAW_SRC.exists():
            raise FileNotFoundError(
                f"DREAMER raw data not found at {DREAMER_RAW_SRC}.\n"
                "Please update DREAMER_RAW_SRC in this script to point to the "
                "directory containing DREAMER.mat."
            )
        dreamer_link.symlink_to(DREAMER_RAW_SRC)
        logger.info(f"Created symlink: {dreamer_link} → {DREAMER_RAW_SRC}")
    else:
        logger.info(f"Symlink already exists: {dreamer_link}")

    env_defaults = {
        "EEGFM_DATABASE_RAW_ROOT":   str(LOCAL_RAW_ROOT),
        "EEGFM_DATABASE_PROC_ROOT":  str(LOCAL_PROC_ROOT),
        "EEGFM_DATABASE_CACHE_ROOT": str(LOCAL_CACHE_ROOT),
    }
    for key, val in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = val
            logger.info(f"Set {key}={val}")
        else:
            logger.info(f"Using existing {key}={os.environ[key]}")


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run(cmd: list[str], *, capture: bool = False) -> str:
    """Run a command, streaming output live; optionally also capture it."""
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    if not capture:
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): "
                f"{' '.join(str(c) for c in cmd)}"
            )
        return ""
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=os.environ,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): "
            f"{' '.join(str(c) for c in cmd)}"
        )
    return "".join(lines)


def _torchrun(conf_file: str, model_type: str, extra_args: list[str] | None = None,
              port: int = 29500) -> str:
    """Launch baseline_main.py via torchrun on a single GPU."""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        f"--master_port={port}",
        "baseline_main.py",
        f"conf_file={conf_file}",
        f"model_type={model_type}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return _run(cmd, capture=True)


def _extract_test_metrics(output: str, label: str) -> dict[str, float]:
    """Parse the last test-split metrics line from torchrun stdout."""
    metrics: dict[str, float] = {}

    pattern = re.compile(r"test\s+(.*)")
    matches = pattern.findall(output)
    if not matches:
        logger.warning(f"[{label}] No test metrics found in output.")
    else:
        last_line = matches[-1]
        for kv in last_line.split(","):
            kv = kv.strip()
            if ":" not in kv:
                continue
            k, _, v = kv.partition(":")
            leaf = k.strip().split("/")[-1]
            try:
                metrics[leaf] = float(v.strip())
            except ValueError:
                pass

    param_pattern = re.compile(
        r"PARAM_COUNT\s+encoder=(\d+)\s+classifier=(\d+)\s+total=(\d+)\s+trainable=(\d+)"
    )
    param_match = param_pattern.search(output)
    if param_match:
        metrics["encoder_params"]    = int(param_match.group(1))
        metrics["classifier_params"] = int(param_match.group(2))
        metrics["trainable_params"]  = int(param_match.group(4))

    return metrics


# ── Preprocessing ─────────────────────────────────────────────────────────────

def run_preprocessing(targets: list[str]) -> None:
    """
    Preprocess DREAMER at 256 Hz.

    Both valence and arousal share the same signal preprocessing; only the
    label column differs.  We run preproc once per target so both parquet
    caches exist.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Preprocessing DREAMER")
    logger.info("=" * 60)

    for target in targets:
        # Patch the finetune_datasets line in the preproc config via CLI override
        logger.info(f"  Preprocessing DREAMER ({target}) at 256 Hz …")
        _run([
            sys.executable, "preproc.py",
            f"conf_file={PREPROC_CFG}",
            f"finetune_datasets={{dreamer: finetune_{target}}}",
        ])


# ── Model training ────────────────────────────────────────────────────────────

def run_model(
    model_type: str,
    target: str,
    port: int,
    run_tag: Optional[str] = None,
    head_type: Optional[str] = None,
) -> dict[str, float]:
    """Train one (model, target) combination and return its test metrics."""
    label = f"{model_type.upper()}/{target}"
    logger.info("-" * 60)
    logger.info(f"Training {label} …")
    logger.info("-" * 60)

    conf = MODEL_CONFIGS[(model_type, target)]
    extra: list[str] = []

    if run_tag:
        extra.append(f"logging.run_dir=assets/run/{run_tag}")

    if head_type:
        if head_type not in HEAD_TYPE_OVERRIDES:
            raise ValueError(
                f"Unknown head type '{head_type}'. "
                f"Choose from: {list(HEAD_TYPE_OVERRIDES)}"
            )
        extra.extend(HEAD_TYPE_OVERRIDES[head_type])

    output = _torchrun(conf, model_type, extra_args=extra or None, port=port)
    metrics = _extract_test_metrics(output, label)
    if metrics:
        logger.info(f"[{label}] test metrics: {metrics}")
    else:
        logger.warning(f"[{label}] Could not parse test metrics; check log files.")
    return metrics


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(
    all_results: dict[tuple[str, str], dict[str, float]],
    head_type: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> None:
    """Print a comparison table of balanced accuracy and F1 for all runs."""
    head_label = head_type or "yaml default"
    tag_label  = run_tag   or "assets/run"
    W = 90
    sep = "═" * W
    print(f"\n{sep}")
    print("  DREAMER — 5-class emotion classification (valence & arousal)")
    print(f"  Single-task · Frozen backbone · Head: {head_label} · Run: {tag_label}")
    print(sep)
    print(f"  {'Method':<12}  {'Target':<10}  {'Bal. Acc':>10}  {'F1':>8}  "
          f"{'Clf params':>12}  {'Enc params':>12}  {'Trainable':>12}")
    print(f"  {'-'*(W-2)}")

    order = [("psd", "valence"), ("psd", "arousal"),
             ("zuna", "valence"), ("zuna", "arousal")]
    for model, target in order:
        m = all_results.get((model, target))
        if m is None:
            continue
        bal  = m.get("balanced_acc",      float("nan"))
        f1   = m.get("f1",                float("nan"))
        clf  = m.get("classifier_params", float("nan"))
        enc  = m.get("encoder_params",    float("nan"))
        trai = m.get("trainable_params",  float("nan"))

        def fmt_p(v):
            return f"{int(v):,}" if not (isinstance(v, float) and v != v) else "n/a"

        print(f"  {model.upper():<12}  {target:<10}  {bal:>10.4f}  {f1:>8.4f}  "
              f"{fmt_p(clf):>12}  {fmt_p(enc):>12}  {fmt_p(trai):>12}")

    print(sep)
    print("  Notes:")
    print("    PSD = Welch log-power encoder; no pretrained weights.")
    print("    ZUNA = Zyphra/ZUNA frozen encoder (auto-downloaded from HuggingFace).")
    print("    Preprocessing mirrors dreamer/preprocess.py exactly.")
    print(f"  {sep}\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DREAMER benchmark: PSD + ZUNA on valence and arousal"
    )
    p.add_argument(
        "--models", nargs="+", default=["psd", "zuna"],
        choices=["psd", "zuna"],
        help="Which models to train (default: both).",
    )
    p.add_argument(
        "--targets", nargs="+", default=["valence", "arousal"],
        choices=["valence", "arousal"],
        help="Which emotion targets to train (default: both).",
    )
    p.add_argument("--skip-preproc",  action="store_true",
                   help="Skip preprocessing (data already preprocessed).")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip model training (just print table from existing results).")
    p.add_argument("--gpu", default=None, metavar="ID",
                   help="GPU device index, e.g. --gpu 4 (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--port-base", type=int, default=29500, metavar="PORT",
                   help="Base rendezvous port for torchrun. Each run uses "
                        "base, base+1, base+2, base+3 (default: 29500).")
    p.add_argument("--run-tag", default=None, metavar="TAG",
                   help="Tag for output organisation (logs go to assets/run/<TAG>/).")
    p.add_argument("--head-type", default=None,
                   choices=list(HEAD_TYPE_OVERRIDES),
                   help="Classifier head type to use for all models, overriding the yaml.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete existing ZUNA feature caches before training so they are "
                        "recomputed with the current config (required after changing "
                        "skip_input_norm or other encoder settings).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.info(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    setup_environment()

    # ── Optional cache clearing ────────────────────────────────────────────────
    if args.clear_cache:
        zuna_caches = [
            LOCAL_CACHE_ROOT / "zuna_dreamer_valence",
            LOCAL_CACHE_ROOT / "zuna_dreamer_arousal",
        ]
        for cache_dir in zuna_caches:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared ZUNA cache: {cache_dir}")
            else:
                logger.info(f"Cache dir not found (nothing to clear): {cache_dir}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    if not args.skip_preproc:
        run_preprocessing(args.targets)
    else:
        logger.info("Skipping preprocessing (--skip-preproc).")

    # ── Model training ────────────────────────────────────────────────────────
    all_results: dict[tuple[str, str], dict[str, float]] = {}

    if not args.skip_training:
        logger.info("=" * 60)
        logger.info("STEP 2: Training models")
        logger.info("=" * 60)

        # Assign distinct ports: (psd/val, psd/aro, zuna/val, zuna/aro)
        combos = [(m, t) for m in args.models for t in args.targets]
        ports = {combo: args.port_base + i for i, combo in enumerate(combos)}

        for model, target in combos:
            try:
                metrics = run_model(
                    model_type=model,
                    target=target,
                    port=ports[(model, target)],
                    run_tag=args.run_tag,
                    head_type=args.head_type,
                )
                all_results[(model, target)] = metrics
            except RuntimeError as exc:
                logger.error(f"Run {model}/{target} failed: {exc}")
                all_results[(model, target)] = {}
    else:
        logger.info("Skipping training (--skip-training).")

    # ── Results table ─────────────────────────────────────────────────────────
    print_results_table(all_results, head_type=args.head_type, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
