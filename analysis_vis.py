#!/usr/bin/env python3
"""Root entrypoint for analysis stage-2 single-seed visualization.

This script delegates to `baseline.analysis.visualize_single_seed`.
"""
import argparse
from pathlib import Path

from baseline.analysis.vis import load_analysis_config, setup_matplotlib_style, VisConfig, discover_seed_dirs, \
    visualize_multi_condition_paradigm, visualize_two_condition_paradigm


def detect_paradigm(data_dir: str) -> str:
    config = load_analysis_config(Path(data_dir))
    return config.paradigm


def main():
    parser = argparse.ArgumentParser(description="EEG-FM Visualization (Single Seed)")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to analysis data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--figure-format", type=str, default="pdf", help="Figure format")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--ema-alpha", type=float, default=0.3, help="EMA smoothing factor")
    parser.add_argument("--step-ema-alpha", type=float, default=0.6, help="Light EMA for step dynamics")
    parser.add_argument("--delta-window", type=int, default=5, help="Early/late window (steps) for Δ plots")
    parser.add_argument("--topk", type=int, default=10, help="Top-K condition pairs for trajectory plots")
    parser.add_argument("--diag-max-samples", type=int, default=256, help="Max samples per step for SVD diagnostics")
    parser.add_argument("--diag-num-steps", type=int, default=6, help="How many steps to sample for SVD diagnostics")
    parser.add_argument("--svcca-sample-rate", type=float, default=None, help="SVCCA subsample rate (0-1, <1 enables trials)")
    parser.add_argument("--subspace-sample-rate", type=float, default=None, help="Subspace subsample rate (0-1, <1 enables trials)")
    parser.add_argument("--sample-trials", type=int, default=None, help="Number of subsampling trials (e.g., 3/5)")
    parser.add_argument("--max-subsample", type=int, default=None, help="Max samples per subsampling trial (optional)")

    args = parser.parse_args()

    setup_matplotlib_style()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (data_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VisConfig(
        figure_format=args.figure_format,
        ema_alpha=args.ema_alpha,
        step_dynamics_ema_alpha=args.step_ema_alpha,
        delta_window=args.delta_window,
        topk_pairs=args.topk,
        diagnostics_max_samples_per_step=args.diag_max_samples,
        diagnostics_num_steps=args.diag_num_steps,
        svcca_sample_rate=args.svcca_sample_rate,
        subspace_sample_rate=args.subspace_sample_rate,
        sample_trials=args.sample_trials,
        max_subsample=args.max_subsample,
    )

    seed_dirs = discover_seed_dirs(str(data_dir))
    if seed_dirs:
        for seed, seed_dir in sorted(seed_dirs.items()):
            seed_out = output_dir / f"seed_{seed}"
            seed_out.mkdir(parents=True, exist_ok=True)
            paradigm = detect_paradigm(str(seed_dir))
            if paradigm == "multi_dataset_joint":
                visualize_multi_condition_paradigm(str(seed_dir), str(seed_out), config)
            else:
                visualize_two_condition_paradigm(str(seed_dir), str(seed_out), config)
        return

    paradigm = detect_paradigm(str(data_dir))
    if paradigm == "multi_dataset_joint":
        visualize_multi_condition_paradigm(str(data_dir), str(output_dir), config)
    else:
        visualize_two_condition_paradigm(str(data_dir), str(output_dir), config)


if __name__ == "__main__":
    main()
