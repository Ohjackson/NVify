#!/usr/bin/env bash

# NVify end-to-end training helper
# - Ensures Python 3.11 virtualenv (.venv311) exists
# - Installs project dependencies & libomp
# - Forces CF (and optional ranking) training via main.py
# - Accepts optional USER_ID argument; defaults to 2nd row of User Listening History

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Detect / install Python 3.11 (required for scikit-surprise)
PY311="/opt/homebrew/bin/python3.11"
if [ ! -x "$PY311" ]; then
  echo "[INFO] Installing python@3.11 via Homebrew..."
  brew install python@3.11
fi

# Create/activate virtualenv if needed
if [ ! -f ".venv311/bin/activate" ]; then
  echo "[INFO] Creating .venv311 using $PY311"
  "$PY311" -m venv .venv311
fi
. .venv311/bin/activate
python -V

# Base tooling + project deps
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt

# LightGBM runtime (macOS)
brew list libomp >/dev/null 2>&1 || brew install libomp

# Optional config tweak: ensure ranking is enabled (toggle true->false once)
if grep -q "skip_ranking:\s*true" config.yaml; then
  sed -i '' -e 's/^\(\s*skip_ranking:\s*\)true/\1false/' config.yaml
fi

# Training knobs (override via env vars)
: "${NVIFY_FORCE_TRAIN:=1}"
: "${NVIFY_TRAIN_MAX_ROWS:=50000}"
export NVIFY_FORCE_TRAIN NVIFY_TRAIN_MAX_ROWS

# Resolve user id (arg > CSV > fallback)
if [ $# -gt 0 ]; then
  USER_ID="$1"
else
  USER_ID=$(awk -F, 'NR==2{print $2}' "data/raw/User Listening History.csv" 2>/dev/null || echo "u0")
fi

echo "[INFO] Training with user_id=$USER_ID, MAX_ROWS=$NVIFY_TRAIN_MAX_ROWS"
python main.py --user_id "$USER_ID" --eval

echo "[INFO] Artifacts after training:"
ls -lh data/artifacts

