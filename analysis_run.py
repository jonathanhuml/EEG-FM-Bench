#!/usr/bin/env python3
"""Root entrypoint for analysis stage-1 data collection.

This script delegates to `baseline.run_analysis`.
"""
import argparse
import logging

from baseline.analysis.run import load_analysis_config, load_trainer_config, run_analysis


logger = logging.getLogger("analysis_run")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EEG-FM Gradient Analysis")

    parser.add_argument("--config", type=str, default=None, help="Path to analysis config YAML")
    parser.add_argument("--trainer-config", type=str, default=None, help="Path to trainer config YAML")

    args = parser.parse_args()

    # Load configs
    analysis_config = load_analysis_config(args.config)

    trainer_config = load_trainer_config(
        analysis_config.model_type,
        args.trainer_config or analysis_config.trainer_config_path,
    )

    # Run analysis (handles multi-seed internally)
    results = run_analysis(analysis_config, trainer_config)

    logger.info("=" * 60)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {analysis_config.output.output_dir}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()

