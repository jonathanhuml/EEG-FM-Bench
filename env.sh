#!/usr/bin/env bash
# env.sh — activate the EEG-FM-Bench Python environment
#
# Usage:
#   source env.sh
#
# To create the environment for the first time:
#   python3 -m venv /data/home/jonhuml/venvs/eegfm
#   source env.sh
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
#   pip install -r requirements.txt

VENV=/data/home/jonhuml/venvs/eegfm
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Create it with:"
    echo "  python3 -m venv $VENV"
    echo "  source $REPO/env.sh"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu128"
    echo "  pip install -r $REPO/requirements.txt"
    return 1
fi

source "$VENV/bin/activate"
cd "$REPO"
echo "EEG-FM-Bench env active ($(python --version), $(pwd))"
