"""
Plot balanced accuracy training curves from trainer log files.

Parses tuev/eval and tuev/test balanced_acc from the standard trainer log format:
  0:INFO ... - tuev/eval epoch: 5, loss: 0.91, acc: 0.65, balanced_acc: 0.31, ...

Usage
-----
  # Auto-discover latest run per method, plot eval curves:
  python plot/plot_training_curves.py

  # Specify log files explicitly:
  python plot/plot_training_curves.py \
    --logs psd:assets/run/log/baseline/psd/torchrun_260318184545/psd_trainer.log \
           zuna:assets/run/log/baseline/zuna/torchrun_260319125424/zuna_trainer.log

  # Include test curves as dashed lines:
  python plot/plot_training_curves.py --test

  # Save to custom path:
  python plot/plot_training_curves.py --out assets/vis/training_curves/balanced_acc.png
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Log parsing ───────────────────────────────────────────────────────────────

_EVAL_RE = re.compile(
    r'tuev/(?P<split>eval|test)\s+epoch:\s*(?P<epoch>\d+).*?balanced_acc:\s*(?P<bacc>[0-9.]+)'
)


def parse_log(log_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Parse a trainer log file.

    Returns
    -------
    {'eval': [(epoch, balanced_acc), ...], 'test': [...]}
    """
    result: Dict[str, List[Tuple[int, float]]] = {'eval': [], 'test': []}
    with open(log_path) as f:
        for line in f:
            m = _EVAL_RE.search(line)
            if m:
                split = m.group('split')
                epoch = int(m.group('epoch'))
                bacc  = float(m.group('bacc'))
                result[split].append((epoch, bacc))
    return result


def _latest_log(method: str, run_dir: str = 'assets/run/log/baseline') -> Optional[str]:
    """Return the log file from the most recent run directory for a method."""
    method_dir = Path(run_dir) / method
    if not method_dir.exists():
        return None
    runs = sorted(method_dir.iterdir())
    for run in reversed(runs):
        logs = list(run.glob(f'{method}_trainer.log'))
        if logs:
            return str(logs[0])
    return None


# ── Plot ──────────────────────────────────────────────────────────────────────

# Consistent colours across methods
_COLORS = {
    'psd':   '#2196F3',   # blue
    'zuna':  '#F44336',   # red
    'bendr': '#4CAF50',   # green
    'biot':  '#FF9800',   # orange
}
_DEFAULT_COLOR = '#9E9E9E'

_LABELS = {
    'psd':   'PSD',
    'zuna':  'ZUNA',
    'bendr': 'BENDR',
    'biot':  'BIOT',
}


def plot_curves(
    method_logs: Dict[str, str],
    include_test: bool = False,
    out_path: str = 'assets/vis/training_curves/balanced_acc.png',
    dataset: str = 'TUEV',
):
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for method, log_path in sorted(method_logs.items()):
        if not os.path.exists(log_path):
            print(f"[skip] {method}: log not found at {log_path}")
            continue

        curves = parse_log(log_path)

        eval_pts = curves['eval']
        test_pts = curves['test']

        if not eval_pts:
            print(f"[skip] {method}: no eval entries in log")
            continue

        color = _COLORS.get(method, _DEFAULT_COLOR)
        label = _LABELS.get(method, method.upper())

        epochs_e, baccs_e = zip(*eval_pts)
        ax.plot(epochs_e, baccs_e, color=color, linewidth=2, label=f'{label} (eval)')

        if include_test and test_pts:
            epochs_t, baccs_t = zip(*test_pts)
            ax.plot(epochs_t, baccs_t, color=color, linewidth=1.5,
                    linestyle='--', alpha=0.7, label=f'{label} (test)')

        plotted += 1

    if plotted == 0:
        print("No data to plot.")
        return

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_title(f'{dataset} — Balanced Accuracy During Training', fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot balanced accuracy training curves')
    parser.add_argument(
        '--logs', nargs='*', metavar='METHOD:PATH',
        help='Log files as method:path pairs. If omitted, auto-discovers latest run per method.'
    )
    parser.add_argument('--methods', nargs='*', default=['psd', 'zuna', 'bendr', 'biot'],
                        help='Methods to include when auto-discovering (default: all four)')
    parser.add_argument('--run-dir', default=None,
                        help='Base log directory, e.g. assets/run/avg_pool/log/baseline. '
                             'If omitted, defaults to assets/run/log/baseline.')
    parser.add_argument('--run-tag', default=None,
                        help='Experiment tag used with --run-tag in the benchmark script '
                             '(sets run-dir to assets/run/<TAG>/log/baseline).')
    parser.add_argument('--test', action='store_true',
                        help='Also plot test curves as dashed lines')
    parser.add_argument('--out', default='assets/vis/training_curves/balanced_acc.png',
                        help='Output image path')
    parser.add_argument('--dataset', default='TUEV')
    args = parser.parse_args()

    # Resolve run directory
    run_dir = args.run_dir
    if run_dir is None:
        if args.run_tag:
            run_dir = f'assets/run/{args.run_tag}/log/baseline'
        else:
            run_dir = 'assets/run/log/baseline'

    if args.logs:
        method_logs = {}
        for entry in args.logs:
            method, path = entry.split(':', 1)
            method_logs[method] = path
    else:
        method_logs = {}
        for method in args.methods:
            log = _latest_log(method, run_dir)
            if log:
                method_logs[method] = log
                print(f"[auto] {method}: {log}")
            else:
                print(f"[auto] {method}: no log found, skipping")

    plot_curves(method_logs, include_test=args.test, out_path=args.out, dataset=args.dataset)


if __name__ == '__main__':
    main()
