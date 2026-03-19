#!/usr/bin/env python3
"""
run_tuev_benchmark.py — TUEV 6-class epileptiform event benchmark
==================================================================

Reproduces results comparable to Table 4 of:
  "EEG Foundation Model Benchmark" (arXiv 2508.17742)

Four methods on TUEV (single-task, frozen-backbone, avg_pool head):
  1. PSD    — log-power spectra (rfft) → avg_pool classifier head
               Parameter-free encoder; only the head is trained.
  2. ZUNA   — Zyphra/ZUNA frozen encoder (auto-downloads from HuggingFace)
  3. BENDR  — BENDR frozen encoder (256 Hz; auto-downloads from GitHub
               releases if no local weights are given via CLI flags)
  4. BIOT   — BIOT frozen encoder (200 Hz; auto-downloads from GitHub
               if no local weights are given via CLI flags)

Metrics reported: balanced accuracy, weighted F1.

Visualization
-------------
After training, t-SNE plots of each model's learned features are produced
via the repository's own plot_vis.py pipeline.

Prerequisites
-------------
  1. Raw TUEV data at:
       /data/datasets/bci/tuh_eeg_evals/tuh_eeg_events/v2.0.1/
     (train/ and eval/ sub-directories of EDF files + matching .rec files)

  2. Environment variables (set automatically by this script if not present):
       EEGFM_DATABASE_RAW_ROOT  — points to the raw-data symlink tree
       EEGFM_DATABASE_PROC_ROOT — writable directory for Arrow datasets
       EEGFM_DATABASE_CACHE_ROOT

  3. Python environment with all repo dependencies installed.

Usage
-----
  # Full run (preprocess → train → visualise → table)
  python run_tuev_benchmark.py

  # Skip preprocessing if already done
  python run_tuev_benchmark.py --skip-preproc

  # Skip all training (re-run vis + table from existing checkpoints)
  python run_tuev_benchmark.py --skip-training

  # Run only specific models
  python run_tuev_benchmark.py --models psd zuna bendr

  # Provide pretrained BENDR weights (contextualiser, conv-encoder)
  python run_tuev_benchmark.py \\
      --bendr-ctx-ckpt /path/to/contextualizer.pt \\
      --bendr-conv-ckpt /path/to/conv_encoder.pt

  # Provide pretrained BIOT weights
  python run_tuev_benchmark.py --biot-ckpt /path/to/biot_encoder.ckpt

  # Disable auto-download of pretrained weights (use random init for BENDR/BIOT)
  python run_tuev_benchmark.py --no-auto-download

  # Skip visualisation (faster)
  python run_tuev_benchmark.py --skip-vis

Design decisions
----------------
  - ZUNA auto-downloads from HuggingFace; no local weights needed.
  - BENDR runs at 256 Hz (shares preprocessing with ZUNA and PSD).
  - BIOT  runs at 200 Hz (requires a separate 200 Hz preprocessing step).
  - PSD has a parameter-free encoder; no pretrained weights exist or are needed.
  - By default, BENDR/BIOT pretrained weights are auto-downloaded from GitHub
    into assets/weights/ if no local path is supplied.  Pass --no-auto-download
    to skip this and use a randomly initialised (frozen) encoder instead.
  - Cloud logging (wandb/comet) is disabled; all logs go to assets/run/.
  - Training uses torchrun --nproc_per_node=1 (single GPU, distributed init).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()

# TUEV raw data location on this machine
TUEV_RAW_SRC = Path("/data/datasets/bci/tuh_eeg_evals/tuh_eeg_events/v2.0.1")

# Local symlink tree that the framework reads via $EEGFM_DATABASE_RAW_ROOT
LOCAL_RAW_ROOT   = REPO_ROOT / "assets" / "data" / "raw"
LOCAL_PROC_ROOT  = REPO_ROOT / "assets" / "data" / "processed"
LOCAL_CACHE_ROOT = REPO_ROOT / "assets" / "data" / "cache"
WEIGHTS_DIR      = REPO_ROOT / "assets" / "weights"

# Config files
PREPROC_256_CFG = "assets/conf/preproc/tuev_preproc_256.yaml"
PREPROC_200_CFG = "assets/conf/preproc/tuev_preproc_200.yaml"

MODEL_CONFIGS = {
    "psd":   "assets/conf/baseline/psd/psd_tuev.yaml",
    "zuna":  "assets/conf/baseline/zuna/zuna_tuev.yaml",
    "bendr": "assets/conf/baseline/bendr/bendr_tuev.yaml",
    "biot":  "assets/conf/baseline/biot/biot_tuev.yaml",
}

TSNE_VIS_CONFIGS = {
    "psd":   "plot/configs/tuev/tsne_psd.yaml",
    "zuna":  "plot/configs/tuev/tsne_zuna.yaml",
    "bendr": "plot/configs/tuev/tsne_bendr.yaml",
    "biot":  "plot/configs/tuev/tsne_biot.yaml",
}

# Pretrained weight download URLs
BENDR_URLS = {
    "contextualizer": "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt",
    "encoder":        "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt",
}
# 18-channel BIOT checkpoint trained on six EEG datasets — best match for TUEV
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
        assets/data/raw/TUE/tuev  →  symlink to TUEV_RAW_SRC
    The framework reads raw data from $EEGFM_DATABASE_RAW_ROOT/TUE/tuev/.
    """
    LOCAL_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_PROC_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    tue_dir = LOCAL_RAW_ROOT / "TUE"
    tue_dir.mkdir(exist_ok=True)
    tuev_link = tue_dir / "tuev"
    if not tuev_link.exists():
        if not TUEV_RAW_SRC.exists():
            raise FileNotFoundError(
                f"TUEV raw data not found at {TUEV_RAW_SRC}.\n"
                "Please update TUEV_RAW_SRC in this script to point to the "
                "directory containing edf/train/ and edf/eval/."
            )
        tuev_link.symlink_to(TUEV_RAW_SRC)
        logger.info(f"Created symlink: {tuev_link} → {TUEV_RAW_SRC}")
    else:
        logger.info(f"Symlink already exists: {tuev_link}")

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
    # Stream stdout+stderr live while also capturing for metric parsing.
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


def _find_latest_ckpt(model_type: str) -> Optional[Path]:
    """Find the most recently created checkpoint file for a given model type."""
    ckpt_base = REPO_ROOT / "assets" / "run" / "ckpt" / "baseline" / model_type
    if not ckpt_base.exists():
        return None
    ckpts = sorted(ckpt_base.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def _extract_test_metrics(output: str, model_type: str) -> dict[str, float]:
    """
    Parse the last occurrence of test-split metrics from torchrun stdout.

    The AbstractTrainer logs lines such as:
      INFO  test tuev/01_tcp_ar/balanced_acc: 0.850, tuev/01_tcp_ar/f1_weighted: 0.847
    """
    metrics: dict[str, float] = {}
    pattern = re.compile(r"test\s+(.*)")
    matches = pattern.findall(output)
    if not matches:
        logger.warning(f"[{model_type}] No test metrics found in output.")
        return metrics
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
    return metrics


# ── Preprocessing ─────────────────────────────────────────────────────────────

def run_preprocessing(skip_biot: bool = False) -> None:
    """Preprocess TUEV at 256 Hz (and optionally 200 Hz for BIOT)."""
    logger.info("=" * 60)
    logger.info("STEP 1: Preprocessing TUEV")
    logger.info("=" * 60)

    logger.info("Preprocessing at 256 Hz (PSD + ZUNA + BENDR) …")
    _run([sys.executable, "preproc.py", f"conf_file={PREPROC_256_CFG}"])

    if not skip_biot:
        logger.info("Preprocessing at 200 Hz (BIOT) …")
        _run([sys.executable, "preproc.py", f"conf_file={PREPROC_200_CFG}"])


# ── Model training ────────────────────────────────────────────────────────────

def patch_yaml_inplace(yaml_path: str, key_path: str, value: str) -> None:
    """Minimal YAML line-patch: replace `key: <anything>` with `key: value`."""
    path = REPO_ROOT / yaml_path
    lines = path.read_text().splitlines()
    leaf_key = key_path.split(".")[-1]
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"{leaf_key}:"):
            indent = " " * (len(line) - len(stripped))
            line = f"{indent}{leaf_key}: {value}"
        new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n")


def run_model(
    model_type: str,
    port: int,
    bendr_ctx_ckpt: Optional[str] = None,
    bendr_conv_ckpt: Optional[str] = None,
    biot_ckpt: Optional[str] = None,
    auto_download: bool = True,
) -> dict[str, float]:
    """Train one model and return its final test metrics."""
    logger.info("-" * 60)
    logger.info(f"Training {model_type.upper()} …")
    logger.info("-" * 60)

    conf = MODEL_CONFIGS[model_type]
    extra: list[str] = []

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


# ── Visualization ──────────────────────────────────────────────────────────────

def run_tsne_visualization(model_type: str) -> None:
    """
    Run t-SNE latent-space visualisation for one trained model.

    Finds the most recent checkpoint for the model, patches the vis config,
    then calls:
        python plot_vis.py t_sne <model_cfg> <vis_cfg>
    """
    logger.info(f"  [t-SNE] {model_type.upper()} …")

    ckpt = _find_latest_ckpt(model_type)
    if ckpt is None:
        logger.warning(f"  No checkpoint found for {model_type}; skipping t-SNE.")
        return

    logger.info(f"  Using checkpoint: {ckpt}")

    vis_cfg_path = REPO_ROOT / TSNE_VIS_CONFIGS[model_type]
    vis_text = vis_cfg_path.read_text()
    vis_text = re.sub(
        r"^ckpt_path:.*$",
        f"ckpt_path: '{ckpt}'",
        vis_text,
        flags=re.MULTILINE,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=REPO_ROOT / "plot" / "configs" / "tuev"
    ) as tf:
        tf.write(vis_text)
        tmp_vis_cfg = tf.name

    try:
        _run([
            sys.executable, "plot_vis.py",
            "t_sne",
            MODEL_CONFIGS[model_type],
            tmp_vis_cfg,
        ])
        logger.info(f"  t-SNE saved to assets/vis/tuev/{model_type}/tsne/")
    except RuntimeError as exc:
        logger.warning(f"  t-SNE visualisation failed for {model_type}: {exc}")
    finally:
        Path(tmp_vis_cfg).unlink(missing_ok=True)


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(all_results: dict[str, dict[str, float]]) -> None:
    """Print a comparison table similar to Table 4 of the paper."""
    header = f"\n{'═'*65}"
    print(header)
    print("  TUEV — 6-class epileptiform event classification")
    print("  Single-task · Frozen backbone · Avg-pool head")
    print(f"{'═'*65}")
    print(f"  {'Method':<22}  {'Bal. Acc':>10}  {'F1 weighted':>12}")
    print(f"  {'-'*59}")
    order = ["PSD", "BENDR", "BIOT", "ZUNA"]
    for name in order:
        m = all_results.get(name)
        if m is None:
            continue
        bal = m.get("balanced_acc", float("nan"))
        f1w = m.get("f1_weighted",  float("nan"))
        print(f"  {name:<22}  {bal:>10.4f}  {f1w:>12.4f}")
    print(header)
    print("  Notes:")
    print("    PSD = log-power rfft encoder, only head trained.")
    print("    BENDR/BIOT without pretrained weights = frozen random encoder.")
    print("    Supply --bendr-ctx-ckpt / --biot-ckpt (or use --no-auto-download)")
    print("    to control whether pretrained weights are used.")
    print(header + "\n")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TUEV benchmark: PSD + ZUNA + BENDR + BIOT"
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
    p.add_argument("--skip-vis",      action="store_true",
                   help="Skip t-SNE visualisation.")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Disable automatic download of BENDR/BIOT pretrained weights.")
    p.add_argument("--bendr-ctx-ckpt",  default=None, metavar="PATH",
                   help="Path to pretrained BENDR contextualiser checkpoint (.pt).")
    p.add_argument("--bendr-conv-ckpt", default=None, metavar="PATH",
                   help="Path to pretrained BENDR conv-encoder checkpoint (.pt).")
    p.add_argument("--biot-ckpt",       default=None, metavar="PATH",
                   help="Path to pretrained BIOT encoder checkpoint (.ckpt).")
    p.add_argument("--gpu", default=None, metavar="ID",
                   help="GPU device index to use, e.g. --gpu 2 (sets CUDA_VISIBLE_DEVICES).")
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

        ports = {"psd": 29499, "zuna": 29500, "bendr": 29501, "biot": 29502}

        for model in args.models:
            try:
                m = run_model(
                    model,
                    port=ports[model],
                    bendr_ctx_ckpt=args.bendr_ctx_ckpt,
                    bendr_conv_ckpt=args.bendr_conv_ckpt,
                    biot_ckpt=args.biot_ckpt,
                    auto_download=auto_download,
                )
                all_results[model.upper()] = m
            except RuntimeError as exc:
                logger.error(f"Model {model} failed: {exc}")
                all_results[model.upper()] = {}
    else:
        logger.info("Skipping training (--skip-training).")

    # ── Visualisation ─────────────────────────────────────────────────────────
    if not args.skip_vis:
        logger.info("=" * 60)
        logger.info("STEP 3: t-SNE visualisation")
        logger.info("=" * 60)
        for model in args.models:
            try:
                run_tsne_visualization(model)
            except Exception as exc:
                logger.warning(f"Visualisation failed for {model}: {exc}")

    # ── Results table ─────────────────────────────────────────────────────────
    print_results_table(all_results)


if __name__ == "__main__":
    main()
