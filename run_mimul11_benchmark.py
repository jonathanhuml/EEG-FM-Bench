#!/usr/bin/env python3
"""
run_mimul11_benchmark.py — Mimul-11 motor-imagery benchmark
============================================================

3-class motor imagery: reach (6 directions) vs grasp (3 types) vs twist (2 types).
60-channel 10-20 montage; native 2500 Hz resampled to 256/200 Hz.
Data format: EEG_ConvertedData .mat files (Mimul11ConvertedBuilder).

Four methods (single-task, frozen-backbone):
  1. PSD   — log-power spectra (Welch) → classifier head
  2. ZUNA  — Zyphra/ZUNA frozen encoder (auto-downloads from HuggingFace)
  3. BENDR — BENDR frozen encoder (256 Hz; auto-downloads from GitHub releases)
  4. BIOT  — BIOT frozen encoder (200 Hz; auto-downloads from GitHub)

Prerequisites
-------------
  1. Raw Mimul-11 data at:
       /data/datasets/bci/Mimul-11/
     Expected layout:
       EEG_ConvertedData/EEG_session{N}_sub{M}_{task}_MI.mat

  2. Python environment with all repo dependencies installed.

Usage
-----
  # Full run (preprocess → train → table)
  python run_mimul11_benchmark.py

  # Skip preprocessing if already done
  python run_mimul11_benchmark.py --skip-preproc

  # Run only specific models
  python run_mimul11_benchmark.py --models psd zuna

  # Run attention_pool head
  python run_mimul11_benchmark.py --head-type attention_pool

  # Use a specific GPU
  python run_mimul11_benchmark.py --gpu 0

Design decisions
----------------
  - 3-class task (reach / grasp / twist) using the combined finetune config.
    Subsets (finetune-reach, finetune-grasp, finetune-twist) can be run by
    editing the PREPROC and MODEL_CONFIGS dicts.
  - BENDR max_channels=20: pretrained conv encoder constraint. conv_router
    maps 60 channels → 20.
  - BIOT max_channels=64: upper bound; use_channel_conv handles 60 channels.
  - ZUNA features are cached under assets/data/cache/zuna_mimul11.
  - Master ports 51400-51403 differ from TUEV (51200s) and THINGS-EEG2 (51300s).
  - Two preprocessing steps: 256 Hz (PSD/ZUNA/BENDR) and 200 Hz (BIOT).
  - num_preproc_mid_workers=4 (reduced from 6) due to large .mat files (1+ GB each).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()

# Mimul-11 raw data location on this machine
MIMUL11_RAW_SRC = Path("/data/datasets/bci/Mimul-11")

# Local symlink tree that the framework reads via $EEGFM_DATABASE_RAW_ROOT
LOCAL_RAW_ROOT   = REPO_ROOT / "assets" / "data" / "raw"
LOCAL_PROC_ROOT  = REPO_ROOT / "assets" / "data" / "processed"
LOCAL_CACHE_ROOT = REPO_ROOT / "assets" / "data" / "cache"
WEIGHTS_DIR      = REPO_ROOT / "assets" / "weights"

# Config files
PREPROC_256_CFG = "assets/conf/preproc/mimul11_preproc_256.yaml"
PREPROC_200_CFG = "assets/conf/preproc/mimul11_preproc_200.yaml"

MODEL_CONFIGS = {
    "psd":   "assets/conf/baseline/psd/psd_mimul11.yaml",
    "zuna":  "assets/conf/baseline/zuna/zuna_mimul11.yaml",
    "bendr": "assets/conf/baseline/bendr/bendr_mimul11.yaml",
    "biot":  "assets/conf/baseline/biot/biot_mimul11.yaml",
}

# Pretrained weight download URLs (same as TUEV/THINGS-EEG2 benchmarks)
BENDR_URLS = {
    "contextualizer": "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt",
    "encoder":        "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt",
}
BIOT_URL = (
    "https://github.com/ycq91044/BIOT/raw/main/pretrained-models/"
    "EEG-six-datasets-18-channels.ckpt"
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ── Environment setup ─────────────────────────────────────────────────────────

def setup_environment() -> None:
    """
    Ensure environment variables and symlink structure are in place.

    Expected directory layout after this function:
        assets/data/raw/Mimul-11  →  symlink to MIMUL11_RAW_SRC
    The framework reads raw data from $EEGFM_DATABASE_RAW_ROOT/Mimul-11/.
    """
    LOCAL_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_PROC_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    mimul_link = LOCAL_RAW_ROOT / "Mimul-11"
    if not mimul_link.exists():
        if not MIMUL11_RAW_SRC.exists():
            raise FileNotFoundError(
                f"Mimul-11 raw data not found at {MIMUL11_RAW_SRC}.\n"
                "Please update MIMUL11_RAW_SRC in this script to point to the "
                "directory containing EEG_ConvertedData/."
            )
        mimul_link.symlink_to(MIMUL11_RAW_SRC)
        logger.info(f"Created symlink: {mimul_link} → {MIMUL11_RAW_SRC}")
    else:
        logger.info(f"Symlink already exists: {mimul_link}")

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


# ── Weight downloading ────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    """Download url → dest with a simple progress indicator."""
    if dest.exists():
        logger.info(f"  Already cached: {dest}")
        return
    logger.info(f"  Downloading {url}")
    logger.info(f"    → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"  Downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_bendr_weights() -> tuple[str, str]:
    """Return (contextualizer_path, encoder_path), downloading if needed."""
    ctx_path  = WEIGHTS_DIR / "bendr" / "contextualizer.pt"
    conv_path = WEIGHTS_DIR / "bendr" / "encoder.pt"
    _download_file(BENDR_URLS["contextualizer"], ctx_path)
    _download_file(BENDR_URLS["encoder"],        conv_path)
    return str(ctx_path), str(conv_path)


def download_biot_weights() -> str:
    """Return biot_ckpt_path, downloading if needed."""
    biot_path = WEIGHTS_DIR / "biot" / "EEG-six-datasets-18-channels.ckpt"
    _download_file(BIOT_URL, biot_path)
    return str(biot_path)


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run(cmd: list[str], *, capture: bool = False, cwd: Optional[Path] = None) -> str:
    """Run a command, always streaming output live; optionally also capture it."""
    cwd = cwd or REPO_ROOT
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    if not capture:
        result = subprocess.run(cmd, cwd=cwd, env=os.environ)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")
        return ""
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=os.environ,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}")
    return "".join(lines)


def _torchrun(conf_file: str, model_type: str, extra_args: list[str] = None,
              port: int = 51400) -> str:
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


def _extract_test_metrics(output: str, model_type: str) -> dict[str, float]:
    """
    Parse the last occurrence of test-split metrics from torchrun stdout.

    The AbstractTrainer logs lines such as:
      INFO  test mimul_11_conv/balanced_acc: 0.720, mimul_11_conv/f1_weighted: 0.715
    Also parses PARAM_COUNT lines:
      INFO  PARAM_COUNT encoder=... classifier=... total=... trainable=...
    """
    metrics: dict[str, float] = {}

    pattern = re.compile(r"test\s+(.*)")
    matches = pattern.findall(output)
    if not matches:
        logger.warning(f"[{model_type}] No test metrics found in output.")
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

def run_preprocessing(skip_biot: bool = False) -> None:
    """Preprocess Mimul-11 at 256 Hz (and optionally 200 Hz for BIOT)."""
    logger.info("=" * 60)
    logger.info("STEP 1: Preprocessing Mimul-11")
    logger.info("=" * 60)

    logger.info("Preprocessing at 256 Hz (PSD + ZUNA + BENDR) …")
    _run([sys.executable, "preproc.py", f"conf_file={PREPROC_256_CFG}"])

    if not skip_biot:
        logger.info("Preprocessing at 200 Hz (BIOT) …")
        _run([sys.executable, "preproc.py", f"conf_file={PREPROC_200_CFG}"])


# ── Model training ────────────────────────────────────────────────────────────

HEAD_TYPE_OVERRIDES: dict[str, list[str]] = {
    "avg_pool":              ["model.classifier_head.head_type=avg_pool"],
    "attention_pool":        ["model.classifier_head.head_type=attention_pool"],
    "flatten_mlp":           ["model.classifier_head.head_type=flatten_mlp"],
    "time_first":            ["model.classifier_head.head_type=dual_stream_fusion",
                              "model.classifier_head.fusion_mode=time_first"],
    "channel_first":         ["model.classifier_head.head_type=dual_stream_fusion",
                              "model.classifier_head.fusion_mode=channel_first"],
    "dual":                  ["model.classifier_head.head_type=dual_stream_fusion",
                              "model.classifier_head.fusion_mode=dual"],
    "linear_pool":           ["model.classifier_head.head_type=avg_pool",
                              "model.classifier_head.hidden_dims=[]"],
    "linear_pool_time_first":["model.classifier_head.head_type=dual_stream_fusion",
                               "model.classifier_head.fusion_mode=time_first",
                               "model.classifier_head.hidden_dims=[]"],
}


def run_model(
    model_type: str,
    port: int,
    bendr_ctx_ckpt: Optional[str] = None,
    bendr_conv_ckpt: Optional[str] = None,
    biot_ckpt: Optional[str] = None,
    auto_download: bool = True,
    run_tag: Optional[str] = None,
    head_type: Optional[str] = None,
) -> dict[str, float]:
    """Train one model and return its final test metrics."""
    logger.info("-" * 60)
    logger.info(f"Training {model_type.upper()} …")
    logger.info("-" * 60)

    conf = MODEL_CONFIGS[model_type]
    extra: list[str] = []

    if run_tag:
        extra.append(f"logging.run_dir=assets/run/{run_tag}")

    if head_type:
        if head_type not in HEAD_TYPE_OVERRIDES:
            raise ValueError(f"Unknown head type '{head_type}'. "
                             f"Choose from: {list(HEAD_TYPE_OVERRIDES)}")
        extra.extend(HEAD_TYPE_OVERRIDES[head_type])

    if model_type == "bendr":
        if not bendr_ctx_ckpt and not bendr_conv_ckpt and auto_download:
            logger.info("  Auto-downloading BENDR pretrained weights …")
            try:
                bendr_ctx_ckpt, bendr_conv_ckpt = download_bendr_weights()
            except RuntimeError as exc:
                logger.warning(f"  Weight download failed ({exc}); using random init.")
        if bendr_ctx_ckpt:
            extra.append(f"model.pretrained_path={bendr_ctx_ckpt}")
        if bendr_conv_ckpt:
            extra.append(f"model.pretrained_conv_path={bendr_conv_ckpt}")

    elif model_type == "biot":
        if not biot_ckpt and auto_download:
            logger.info("  Auto-downloading BIOT pretrained weights …")
            try:
                biot_ckpt = download_biot_weights()
            except RuntimeError as exc:
                logger.warning(f"  Weight download failed ({exc}); using random init.")
        if biot_ckpt:
            extra.append(f"model.pretrained_path={biot_ckpt}")

    output = _torchrun(conf, model_type, extra_args=extra or None, port=port)
    metrics = _extract_test_metrics(output, model_type)
    if metrics:
        logger.info(f"[{model_type}] test metrics: {metrics}")
    else:
        logger.warning(f"[{model_type}] Could not parse test metrics; check log files.")
    return metrics


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(
    all_results: dict[str, dict[str, float]],
    head_type: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> None:
    """Print a comparison table for Mimul-11 3-class motor imagery."""
    head_label = head_type or "yaml default"
    tag_label  = run_tag   or "assets/run"
    W = 80
    header = f"\n{'═'*W}"
    print(header)
    print("  Mimul-11 — 3-class motor imagery (reach vs grasp vs twist)")
    print(f"  Single-task · Frozen backbone · Head: {head_label} · Run: {tag_label}")
    print(f"{'═'*W}")
    print(f"  {'Method':<10}  {'Bal. Acc':>10}  {'F1':>8}  {'Clf params':>12}  {'Enc params':>12}  {'Trainable':>12}")
    print(f"  {'-'*(W-2)}")
    order = ["PSD", "BENDR", "BIOT", "ZUNA"]
    for name in order:
        m = all_results.get(name)
        if m is None:
            continue
        bal  = m.get("balanced_acc",      float("nan"))
        f1   = m.get("f1_weighted",        float("nan"))
        clf  = m.get("classifier_params", float("nan"))
        enc  = m.get("encoder_params",    float("nan"))
        trai = m.get("trainable_params",  float("nan"))

        def fmt_p(v):
            return f"{int(v):,}" if not (isinstance(v, float) and v != v) else "n/a"

        print(f"  {name:<10}  {bal:>10.4f}  {f1:>8.4f}  {fmt_p(clf):>12}  {fmt_p(enc):>12}  {fmt_p(trai):>12}")
    print(header)
    print("  Notes:")
    print("    PSD = Welch log-power encoder, only head trained.")
    print("    BENDR: conv_router projects 60 → 20 channels before pretrained encoder.")
    print("    BIOT: use_channel_conv handles 60 channels via per-channel projection.")
    print("    Chance level = 1/3 ≈ 0.333 balanced accuracy.")
    print("    Supply --bendr-ctx-ckpt / --biot-ckpt (or use --no-auto-download)")
    print("    to control whether pretrained weights are used.")
    print(header + "\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mimul-11 benchmark: PSD + ZUNA + BENDR + BIOT"
    )
    p.add_argument(
        "--models", nargs="+", default=["psd", "zuna", "bendr", "biot"],
        choices=["psd", "zuna", "bendr", "biot"],
        help="Which models to train (default: all four).",
    )
    p.add_argument("--skip-preproc",  action="store_true",
                   help="Skip preprocessing (data already exists).")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip model training (use existing checkpoints).")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Disable automatic download of BENDR/BIOT pretrained weights.")
    p.add_argument("--bendr-ctx-ckpt",  default=None, metavar="PATH",
                   help="Path to pretrained BENDR contextualiser checkpoint (.pt).")
    p.add_argument("--bendr-conv-ckpt", default=None, metavar="PATH",
                   help="Path to pretrained BENDR conv-encoder checkpoint (.pt).")
    p.add_argument("--biot-ckpt",       default=None, metavar="PATH",
                   help="Path to pretrained BIOT encoder checkpoint (.ckpt).")
    p.add_argument("--gpu", default=None, metavar="ID",
                   help="GPU device index to use, e.g. --gpu 0 (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--port-base", type=int, default=51399, metavar="PORT",
                   help="Base port for torchrun rendezvous. Models use base+1..base+4. "
                        "Default 51399 gives ports 51400-51403.")
    p.add_argument("--run-tag", default=None, metavar="TAG",
                   help="Experiment tag for output organisation. Logs go to "
                        "assets/run/<TAG>/log/... (default: assets/run/).")
    p.add_argument("--head-type", default=None,
                   choices=list(HEAD_TYPE_OVERRIDES),
                   help="Classifier head type to use for all models, overriding the yaml. "
                        "Options: avg_pool, attention_pool, flatten_mlp, time_first, "
                        "channel_first, dual, linear_pool, linear_pool_time_first.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    auto_download = not args.no_auto_download

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.info(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    setup_environment()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    if not args.skip_preproc:
        skip_biot = "biot" not in args.models
        run_preprocessing(skip_biot=skip_biot)
    else:
        logger.info("Skipping preprocessing (--skip-preproc).")

    # ── Model training ────────────────────────────────────────────────────────
    all_results: dict[str, dict[str, float]] = {}

    if not args.skip_training:
        logger.info("=" * 60)
        logger.info("STEP 2: Training models")
        logger.info("=" * 60)

        base = args.port_base
        ports = {"psd": base + 1, "zuna": base + 2, "bendr": base + 3, "biot": base + 4}

        for model in args.models:
            try:
                m = run_model(
                    model,
                    port=ports[model],
                    bendr_ctx_ckpt=args.bendr_ctx_ckpt,
                    bendr_conv_ckpt=args.bendr_conv_ckpt,
                    biot_ckpt=args.biot_ckpt,
                    auto_download=auto_download,
                    run_tag=args.run_tag,
                    head_type=args.head_type,
                )
                all_results[model.upper()] = m
            except RuntimeError as exc:
                logger.error(f"Model {model} failed: {exc}")
                all_results[model.upper()] = {}
    else:
        logger.info("Skipping training (--skip-training).")

    # ── Results table ─────────────────────────────────────────────────────────
    print_results_table(all_results, head_type=args.head_type, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
