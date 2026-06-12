#!/usr/bin/env python3
"""Run frozen-feature benchmarks over the six local evaluation datasets.

Usage:
    source env.sh
    python run_eval_datasets_benchmark.py

The script maps the downloaded dataset layouts into FM-Bench's raw-data
layout, preprocesses at 256 Hz and/or 200 Hz, downloads pretrained weights,
and trains an attention-pooling readout for each selected model and dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/groups/bci/jonhuml/fm-bench-data/eval_datasets")
AY2LATENT_ROOT = Path("/data/groups/bci/jonhuml/workspace/AY2latent/lingua")
ZUNA_CHECKPOINT = Path(
    "/data/groups/bci/checkpoints/bci/ZUNA2_5e-4/checkpoints/0000052500"
)

RAW_ROOT = REPO_ROOT / "assets" / "data" / "raw"
PROC_ROOT = REPO_ROOT / "assets" / "data" / "processed"
CACHE_ROOT = REPO_ROOT / "assets" / "data" / "cache"
WEIGHTS_ROOT = REPO_ROOT / "assets" / "weights"

MODELS = ("psd", "zuna", "bendr", "biot", "labram", "cbramod", "reve")
MODEL_FS = {
    "psd": 256,
    "zuna": 256,
    "bendr": 256,
    "biot": 200,
    "labram": 200,
    "cbramod": 200,
    "reve": 200,
}
MODEL_BATCH_SIZE = {
    "psd": 64,
    "zuna": 4,
    "bendr": 64,
    "biot": 32,
    "labram": 32,
    "cbramod": 16,
    "reve": 16,
}

BENDR_URLS = {
    "contextualizer": (
        "https://github.com/SPOClab-ca/BENDR/releases/download/"
        "v0.1-alpha/contextualizer.pt"
    ),
    "encoder": (
        "https://github.com/SPOClab-ca/BENDR/releases/download/"
        "v0.1-alpha/encoder.pt"
    ),
}
BIOT_URL = (
    "https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/"
    "EEG-six-datasets-18-channels.ckpt"
)
HF_WEIGHTS = {
    "labram": ("eeg-telecom-paris/labram-base-official", "weights.safetensors"),
    "cbramod": ("weighting666/CBraMod", "pretrained_weights.pth"),
    "reve": ("brain-bzh/reve-base", "model.safetensors"),
    "reve_positions": ("brain-bzh/reve-positions", "model.safetensors"),
}

METRIC_RE_TEMPLATE = (
    r"{dataset}/(?P<split>eval|test)\s+epoch:\s*(?P<epoch>\d+)"
    r".*?balanced_acc:\s*(?P<bacc>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class DatasetSpec:
    folder: str
    fm_name: str
    duration_sec: int
    required_paths: tuple[str, ...]

    @property
    def source(self) -> Path:
        return DATA_ROOT / self.folder


DATASETS = {
    "adtfd": DatasetSpec(
        folder="adtfd",
        fm_name="adftd",
        duration_sec=10,
        required_paths=("participants.tsv", "sub-001/eeg"),
    ),
    "seed": DatasetSpec(
        folder="seed",
        fm_name="seed",
        duration_sec=4,
        required_paths=("SEED/SEED_EEG/SEED_RAW_EEG",),
    ),
    "seed_v": DatasetSpec(
        folder="seed_v",
        fm_name="seed_v",
        duration_sec=10,
        required_paths=("SEED-V/EEG_raw",),
    ),
    "seed_vii": DatasetSpec(
        folder="seed_vii",
        fm_name="seed_vii",
        duration_sec=15,
        required_paths=("SEED-VII/EEG_raw", "SEED-VII/save_info"),
    ),
    "siena": DatasetSpec(
        folder="siena",
        fm_name="siena_scalp",
        duration_sec=10,
        required_paths=("subject_info.csv", "PN00"),
    ),
    "workload": DatasetSpec(
        folder="workload",
        fm_name="workload",
        duration_sec=4,
        required_paths=("subject-info.csv", "Subject00_1.edf"),
    ),
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval-benchmark")


class ModelUnavailableError(RuntimeError):
    """A requested model cannot run because an external artifact is unavailable."""


def _run(
    cmd: list[str],
    *,
    capture: bool = False,
    tee_path: Optional[Path] = None,
) -> str:
    logger.info("Running: %s", " ".join(cmd))
    if not capture:
        subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ, check=True)
        return ""

    output = []
    if tee_path is not None:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        tee_path.open("w") if tee_path is not None else contextlib.nullcontext()
    ) as tee:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                if tee is not None:
                    tee.write(line)
                    tee.flush()
                output.append(line)
            return_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    return "".join(output)


def _ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == target:
            return
        raise FileExistsError(
            f"Cannot create dataset compatibility link; path already exists: {link}"
        )
    link.symlink_to(target, target_is_directory=target.is_dir())
    logger.info("Mapped %s -> %s", link, target)


def _validate_source(spec: DatasetSpec) -> None:
    if not spec.source.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {spec.source}")
    missing = [
        relative
        for relative in spec.required_paths
        if not (spec.source / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{spec.folder} is missing expected paths: {', '.join(missing)}"
        )


def _setup_dataset_layout(spec: DatasetSpec) -> None:
    _validate_source(spec)
    source = spec.source

    if spec.folder == "adtfd":
        data_root = RAW_ROOT / "ADFTD" / "data"
        # Older runs linked the whole checkout and therefore scanned the
        # duplicate BIDS dataset under derivatives/. Replace only that known
        # generated link with a filtered view of root metadata and sub-* data.
        if (
            data_root.is_symlink()
            and data_root.resolve() == source.resolve()
        ):
            data_root.unlink()
        data_root.mkdir(parents=True, exist_ok=True)
        for metadata_file in source.iterdir():
            if metadata_file.is_file():
                _ensure_link(data_root / metadata_file.name, metadata_file)
        for subject_dir in sorted(source.glob("sub-*")):
            if subject_dir.is_dir():
                mirrored_subject = data_root / subject_dir.name
                if (
                    mirrored_subject.is_symlink()
                    and mirrored_subject.resolve() == subject_dir.resolve()
                ):
                    mirrored_subject.unlink()
                mirrored_subject.mkdir(parents=True, exist_ok=True)
                for source_path in subject_dir.rglob("*"):
                    relative = source_path.relative_to(subject_dir)
                    destination = mirrored_subject / relative
                    if source_path.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif source_path.is_file():
                        _ensure_link(destination, source_path)
    elif spec.folder == "seed":
        _ensure_link(
            RAW_ROOT / "SEED" / "SEED" / "SEED_EEG" / "SEED_RAW_EEG" / "resampled",
            source / "SEED" / "SEED_EEG" / "SEED_RAW_EEG",
        )
    elif spec.folder == "seed_v":
        _ensure_link(
            RAW_ROOT / "SEED" / "SEED-V" / "EEG_raw" / "resampled",
            source / "SEED-V" / "EEG_raw",
        )
    elif spec.folder == "seed_vii":
        _ensure_link(
            RAW_ROOT / "SEED" / "SEED-VII" / "SEED-VII",
            source / "SEED-VII",
        )
    elif spec.folder == "siena":
        _ensure_link(
            RAW_ROOT / "Siena Scalp EEG Dataset" / "siena-scalp-eeg-database-1.0.0",
            source,
        )
    elif spec.folder == "workload":
        workload_root = RAW_ROOT / "Workload EEGMAT"
        _ensure_link(workload_root / "data", source)
        _ensure_link(workload_root / "subject-info.csv", source / "subject-info.csv")
    else:
        raise KeyError(f"Unhandled dataset layout: {spec.folder}")


def _setup_environment(gpu: str) -> None:
    for path in (RAW_ROOT, PROC_ROOT, CACHE_ROOT, WEIGHTS_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["EEGFM_DATABASE_RAW_ROOT"] = str(RAW_ROOT)
    os.environ["EEGFM_DATABASE_PROC_ROOT"] = str(PROC_ROOT)
    os.environ["EEGFM_DATABASE_CACHE_ROOT"] = str(CACHE_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _download_file(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, destination)
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _download_hf(repo_id: str, filename: str, destination_dir: Path) -> Path:
    destination = destination_dir / filename
    if destination.is_file():
        return destination

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download LaBraM, CBraMod, and REVE"
        ) from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=str(destination_dir),
        )
    except GatedRepoError as exc:
        raise ModelUnavailableError(
            "REVE access was denied. Accept the terms at "
            "https://huggingface.co/brain-bzh/reve-base, then run either "
            "`hf auth login` or export HF_TOKEN."
        ) from exc
    return Path(downloaded)


def _resolve_weights(args: argparse.Namespace) -> dict[str, str]:
    weights: dict[str, str] = {}
    requested = set(args.models)

    if "bendr" in requested:
        if args.bendr_ctx_ckpt and args.bendr_conv_ckpt:
            weights["bendr_ctx"] = str(Path(args.bendr_ctx_ckpt).resolve())
            weights["bendr_conv"] = str(Path(args.bendr_conv_ckpt).resolve())
        elif not args.no_auto_download:
            weights["bendr_ctx"] = str(
                _download_file(
                    BENDR_URLS["contextualizer"],
                    WEIGHTS_ROOT / "bendr" / "contextualizer.pt",
                )
            )
            weights["bendr_conv"] = str(
                _download_file(
                    BENDR_URLS["encoder"],
                    WEIGHTS_ROOT / "bendr" / "encoder.pt",
                )
            )
        else:
            raise ValueError("BENDR requires both checkpoint paths")

    if "biot" in requested:
        if args.biot_ckpt:
            weights["biot"] = str(Path(args.biot_ckpt).resolve())
        elif not args.no_auto_download:
            weights["biot"] = str(
                _download_file(
                    BIOT_URL,
                    WEIGHTS_ROOT / "biot" / "EEG-six-datasets-18-channels.ckpt",
                )
            )
        else:
            raise ValueError("BIOT requires --biot-ckpt")

    for model, arg_name in (
        ("labram", "labram_ckpt"),
        ("cbramod", "cbramod_ckpt"),
        ("reve", "reve_ckpt"),
    ):
        if model not in requested:
            continue
        supplied = getattr(args, arg_name)
        if supplied:
            weights[model] = str(Path(supplied).resolve())
        elif not args.no_auto_download:
            repo_id, filename = HF_WEIGHTS[model]
            weights[model] = str(
                _download_hf(repo_id, filename, WEIGHTS_ROOT / model)
            )
        else:
            raise ValueError(f"{model} requires --{arg_name.replace('_', '-')}")

    if "reve" in requested:
        if args.reve_positions_ckpt:
            weights["reve_positions"] = str(
                Path(args.reve_positions_ckpt).resolve()
            )
        elif not args.no_auto_download:
            repo_id, filename = HF_WEIGHTS["reve_positions"]
            weights["reve_positions"] = str(
                _download_hf(repo_id, filename, WEIGHTS_ROOT / "reve_positions")
            )
        else:
            raise ValueError("REVE requires --reve-positions-ckpt")

    return weights


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _processed_dataset_ready(spec: DatasetSpec, fs: int) -> bool:
    config_root = PROC_ROOT / f"fs_{fs}" / spec.fm_name / "finetune"
    for version_dir in config_root.glob("*"):
        if not (version_dir / "dataset_info.json").is_file():
            continue
        split_files = {
            split: list(version_dir.glob(f"*-{split}*.arrow"))
            for split in ("train", "validation", "test")
        }
        if all(split_files.values()):
            return True
    return False


def _preprocess(
    spec: DatasetSpec,
    fs: int,
    config_dir: Path,
    workers: int,
) -> None:
    config_path = config_dir / f"preproc_{fs}.json"
    _write_json(
        config_path,
        {
            "fs": fs,
            "clean_middle_cache": False,
            "clean_shared_info": False,
            "num_preproc_arrow_writers": workers,
            "num_preproc_mid_workers": workers,
            "pretrain_datasets": [],
            "finetune_datasets": {spec.fm_name: "finetune"},
        },
    )
    output = _run(
        [sys.executable, "preproc.py", f"conf_file={config_path}"],
        capture=True,
    )
    success_marker = (
        f"Dataset {spec.fm_name} finetune at fs={fs}Hz is prepared."
    )
    if success_marker not in output:
        raise RuntimeError(
            f"Preprocessing did not complete for {spec.fm_name} at {fs} Hz. "
            "Inspect the preproc log above; preproc.py logs exceptions without "
            "returning a non-zero exit status."
        )


def _head_config() -> dict[str, Any]:
    return {
        "head_type": "attention_pool",
        "hidden_dims": [],
        "dropout": 0.0,
        "attn_n_head": 4,
        "attn_head_dim": 64,
    }


def _model_overrides(
    model: str,
    spec: DatasetSpec,
    weights: dict[str, str],
    run_root: Path,
) -> dict[str, Any]:
    common = {
        "classifier_head": _head_config(),
        "t_sne": False,
    }
    if model == "psd":
        return common | {
            "nperseg": 256,
            "noverlap": None,
            "fmin": 1.0,
            "fmax": 45.0,
        }
    if model == "zuna":
        return common | {
            "ay2latent_root": str(AY2LATENT_ROOT),
            "checkpoint_path": str(ZUNA_CHECKPOINT),
        }
    if model == "bendr":
        return common | {
            "pretrained_path": weights["bendr_ctx"],
            "pretrained_conv_path": weights["bendr_conv"],
        }
    if model == "biot":
        return common | {"pretrained_path": weights["biot"]}
    if model == "labram":
        return common | {
            "pretrained_path": weights["labram"],
            "eeg_size": spec.duration_sec * MODEL_FS[model],
        }
    if model == "cbramod":
        return common | {"pretrained_path": weights["cbramod"]}
    if model == "reve":
        return common | {
            "pretrained_path": weights["reve"],
            "pos_bank_pretrained_path": weights["reve_positions"],
        }
    raise KeyError(model)


def _training_config(
    model: str,
    spec: DatasetSpec,
    epochs: int,
    run_root: Path,
) -> dict[str, Any]:
    training: dict[str, Any] = {
        "max_epochs": epochs,
        "freeze_encoder": True,
        "encoder_lr_scale": 0.0,
        "warmup_epochs": min(5, max(1, epochs // 10)),
        "lora": {"use_lora": False},
    }
    if model == "zuna":
        training.update(
            {
                "cache_features": True,
                "features_cache_dir": str(
                    run_root / "feature_cache" / f"zuna_{spec.folder}"
                ),
                "precompute_batch_size": 1,
            }
        )
    return training


def _build_model_config(
    model: str,
    spec: DatasetSpec,
    weights: dict[str, str],
    run_root: Path,
    epochs: int,
    port: int,
    workers: int,
) -> dict[str, Any]:
    return {
        "seed": 42,
        "master_port": port,
        "multitask": True,
        "model_type": model,
        "fs": MODEL_FS[model],
        "data": {
            "batch_size": MODEL_BATCH_SIZE[model],
            "num_workers": workers,
            "datasets": {spec.fm_name: "finetune"},
        },
        "model": _model_overrides(model, spec, weights, run_root),
        "training": _training_config(model, spec, epochs, run_root),
        "logging": {
            "experiment_name": f"{model}_{spec.fm_name}",
            "run_dir": str(run_root),
            "use_cloud": False,
            "offline": True,
            "tags": [
                model,
                spec.fm_name,
                "frozen",
                "attention_pool",
                "linear_readout",
            ],
            "log_step_interval": 10,
            "ckpt_interval": epochs,
        },
    }


def _latest_log(run_root: Path, model: str) -> Optional[Path]:
    log_root = run_root / "log" / "baseline" / model
    if not log_root.exists():
        return None
    logs = list(log_root.glob(f"*/{model}_trainer.log"))
    return max(logs, key=lambda path: path.stat().st_mtime) if logs else None


def _parse_history(
    text: str,
    dataset: str,
) -> dict[str, list[tuple[int, float]]]:
    pattern = re.compile(
        METRIC_RE_TEMPLATE.format(dataset=re.escape(dataset))
    )
    history: dict[str, list[tuple[int, float]]] = {"eval": [], "test": []}
    for match in pattern.finditer(text):
        history[match.group("split")].append(
            (int(match.group("epoch")), float(match.group("bacc")))
        )
    return history


def _write_results(
    path: Path,
    spec: DatasetSpec,
    results: dict[str, dict[str, Any]],
) -> None:
    lines = [
        f"dataset_folder: {spec.folder}",
        f"fm_bench_dataset: {spec.fm_name}",
        "encoder_features: frozen",
        "readout: learned attention pooling + linear classifier",
        "classifier_hidden_dims: []",
        "",
    ]
    for model in MODELS:
        if model not in results:
            continue
        result = results[model]
        lines.append(f"[{model}]")
        lines.append(f"status: {result['status']}")
        if result.get("error"):
            lines.append(f"error: {result['error']}")
        for split in ("eval", "test"):
            points = result.get(split, [])
            for epoch, balanced_acc in points:
                lines.append(
                    f"{split} epoch={epoch} balanced_acc={balanced_acc:.6f}"
                )
            if points:
                lines.append(
                    f"final_{split}_balanced_acc: {points[-1][1]:.6f}"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _plot_results(
    path: Path,
    spec: DatasetSpec,
    results: dict[str, dict[str, Any]],
    split: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "psd": "#2196F3",
        "zuna": "#F44336",
        "bendr": "#4CAF50",
        "biot": "#FF9800",
        "labram": "#9C27B0",
        "cbramod": "#00ACC1",
        "reve": "#795548",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for model in MODELS:
        points = results.get(model, {}).get(split, [])
        if not points:
            continue
        epochs, values = zip(*points)
        ax.plot(
            epochs,
            values,
            color=colors[model],
            linewidth=2,
            label=model.upper(),
        )
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"{spec.fm_name} {split.capitalize()} Balanced Accuracy")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def _train_model(
    model: str,
    spec: DatasetSpec,
    weights: dict[str, str],
    dataset_run_root: Path,
    config_dir: Path,
    epochs: int,
    port: int,
    workers: int,
) -> Path:
    config_path = config_dir / f"{model}.json"
    config = _build_model_config(
        model,
        spec,
        weights,
        dataset_run_root,
        epochs,
        port,
        workers,
    )
    _write_json(config_path, config)
    _run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            f"--master_port={port}",
            "baseline_main.py",
            f"conf_file={config_path}",
            f"model_type={model}",
        ],
        capture=True,
        tee_path=dataset_run_root / "subprocess" / f"{model}_torchrun.log",
    )
    log_path = _latest_log(dataset_run_root, model)
    if log_path is None:
        raise FileNotFoundError(f"No trainer log produced for {model}/{spec.fm_name}")
    return log_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all frozen-feature models over six evaluation datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--port-base", type=int, default=53100)
    parser.add_argument("--run-tag", default="eval_datasets_attention_linear")
    parser.add_argument("--skip-preproc", action="store_true")
    parser.add_argument(
        "--force-preproc",
        action="store_true",
        help="Rebuild preprocessing even when a complete Arrow dataset exists",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Retrain models whose existing log already reached --epochs",
    )
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--bendr-ctx-ckpt")
    parser.add_argument("--bendr-conv-ckpt")
    parser.add_argument("--biot-ckpt")
    parser.add_argument("--labram-ckpt")
    parser.add_argument("--cbramod-ckpt")
    parser.add_argument("--reve-ckpt")
    parser.add_argument("--reve-positions-ckpt")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_environment(args.gpu)
    logger.info("Using CUDA_VISIBLE_DEVICES=%s", args.gpu)

    if "zuna" in args.models:
        if not (AY2LATENT_ROOT / "apps" / "AY2latent_bci").is_dir():
            raise FileNotFoundError(f"AY2latent checkout not found: {AY2LATENT_ROOT}")
        if not (ZUNA_CHECKPOINT / ".metadata").is_file():
            raise FileNotFoundError(f"ZUNA checkpoint not found: {ZUNA_CHECKPOINT}")

    weights: dict[str, str] = {}
    unavailable_models: dict[str, str] = {}
    weight_failures: dict[str, str] = {}
    if not args.skip_training:
        for model in args.models:
            model_args = argparse.Namespace(**vars(args))
            model_args.models = [model]
            try:
                weights.update(_resolve_weights(model_args))
            except ModelUnavailableError as exc:
                logger.warning("Skipping %s: %s", model, exc)
                unavailable_models[model] = str(exc)
            except Exception as exc:
                logger.exception("Weight setup failed for %s", model)
                weight_failures[model] = str(exc)

    benchmark_root = REPO_ROOT / "assets" / "run" / args.run_tag
    config_root = benchmark_root / "generated_configs"
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for dataset_index, dataset_name in enumerate(args.datasets):
        spec = DATASETS[dataset_name]
        dataset_run_root = benchmark_root / spec.folder
        result_path = benchmark_root / "results" / f"{spec.folder}_accuracies.txt"
        plot_path = benchmark_root / "plots" / f"{spec.folder}_balanced_accuracy.png"
        eval_plot_path = benchmark_root / "plots" / f"{spec.folder}_eval_balanced_accuracy.png"
        test_plot_path = benchmark_root / "plots" / f"{spec.folder}_test_balanced_accuracy.png"
        results: dict[str, dict[str, Any]] = {}
        summary[spec.folder] = results

        try:
            _setup_dataset_layout(spec)
            if not args.skip_preproc:
                runnable_models = [
                    model for model in args.models
                    if args.skip_training
                    or (
                        model not in unavailable_models
                        and model not in weight_failures
                    )
                ]
                required_fs = sorted({MODEL_FS[model] for model in runnable_models})
                for fs in required_fs:
                    if not args.force_preproc and _processed_dataset_ready(spec, fs):
                        logger.info(
                            "Using completed preprocessing for %s at %s Hz",
                            spec.fm_name,
                            fs,
                        )
                        continue
                    _preprocess(
                        spec,
                        fs,
                        config_root / spec.folder,
                        args.workers,
                    )
        except Exception as exc:
            logger.exception("Dataset setup/preprocessing failed for %s", spec.folder)
            for model in args.models:
                results[model] = {"status": "failed", "error": str(exc)}
            _write_results(result_path, spec, results)
            if args.fail_fast:
                raise
            continue

        for model_index, model in enumerate(args.models):
            try:
                if model in unavailable_models:
                    results[model] = {
                        "status": "skipped",
                        "error": unavailable_models[model],
                    }
                    logger.warning(
                        "Skipping %s/%s: %s",
                        spec.folder,
                        model,
                        unavailable_models[model],
                    )
                    continue
                if model in weight_failures:
                    raise RuntimeError(weight_failures[model])
                if args.skip_training:
                    log_path = _latest_log(dataset_run_root, model)
                    if log_path is None:
                        raise FileNotFoundError(
                            f"No existing log for {model}/{spec.fm_name}"
                        )
                elif not args.rerun_completed:
                    log_path = _latest_log(dataset_run_root, model)
                    existing_history = (
                        _parse_history(
                            log_path.read_text(errors="replace"),
                            spec.fm_name,
                        )
                        if log_path is not None
                        else {"eval": [], "test": []}
                    )
                    final_epoch = max(
                        (epoch for epoch, _value in existing_history["eval"]),
                        default=-1,
                    )
                    if final_epoch >= args.epochs - 1:
                        logger.info(
                            "Using completed training for %s/%s through epoch %s",
                            spec.folder,
                            model,
                            final_epoch,
                        )
                    else:
                        log_path = None
                    if log_path is None:
                        port = args.port_base + dataset_index * 20 + model_index
                        log_path = _train_model(
                            model,
                            spec,
                            weights,
                            dataset_run_root,
                            config_root / spec.folder,
                            args.epochs,
                            port,
                            args.workers,
                        )
                else:
                    port = args.port_base + dataset_index * 20 + model_index
                    log_path = _train_model(
                        model,
                        spec,
                        weights,
                        dataset_run_root,
                        config_root / spec.folder,
                        args.epochs,
                        port,
                        args.workers,
                    )
                history = _parse_history(
                    log_path.read_text(errors="replace"),
                    spec.fm_name,
                )
                if not history["eval"]:
                    raise RuntimeError(
                        f"No balanced-accuracy history found in {log_path}"
                    )
                results[model] = {
                    "status": "complete",
                    "log": str(log_path),
                    **history,
                }
            except Exception as exc:
                logger.exception("Benchmark failed for %s/%s", spec.folder, model)
                results[model] = {"status": "failed", "error": str(exc)}
                if args.fail_fast:
                    _write_results(result_path, spec, results)
                    raise
            finally:
                _write_results(result_path, spec, results)

        try:
            _plot_results(eval_plot_path, spec, results, split="eval")
            _plot_results(test_plot_path, spec, results, split="test")
            _plot_results(plot_path, spec, results, split="eval")
        except Exception:
            logger.exception("Plotting failed for %s", spec.folder)

    failed = [
        f"{dataset}/{model}"
        for dataset, model_results in summary.items()
        for model, result in model_results.items()
        if result["status"] == "failed"
    ]
    skipped = [
        f"{dataset}/{model}"
        for dataset, model_results in summary.items()
        for model, result in model_results.items()
        if result["status"] == "skipped"
    ]
    if failed:
        logger.error("Incomplete runs: %s", ", ".join(failed))
        return 1
    if skipped:
        logger.warning("Skipped unavailable runs: %s", ", ".join(skipped))
    logger.info("All available benchmark runs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
