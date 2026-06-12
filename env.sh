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

VENV=/data/groups/bci/jonhuml/venvs/eegfm
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
# The environment was relocated from /data/home; its generated activate script
# still contains the old absolute prefix.
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
hash -r
export _MNE_FAKE_HOME_DIR="${_MNE_FAKE_HOME_DIR:-$REPO/assets/data/cache/mne-home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO/assets/data/cache/matplotlib}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$_MNE_FAKE_HOME_DIR/.mne" "$MPLCONFIGDIR"

# Console-script shebangs inside the relocated venv still point to /data/home.
hf() {
    "$VENV/bin/python" -c 'from huggingface_hub.cli.hf import main; main()' "$@"
}

cd "$REPO"
echo "EEG-FM-Bench env active ($(python --version), $(pwd))"
