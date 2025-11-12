#!/usr/bin/env bash
set -euo pipefail

# Create and prepare a local virtual environment and install deps

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
# Prefer Homebrew's Python 3.11 if available
PY311_BIN="${PY311_BIN:-/opt/homebrew/bin/python3.11}"
if [[ -x "$PY311_BIN" ]]; then
  PY_CMD="$PY311_BIN"
else
  # Fallback to python3
  PY_CMD="python3"
fi

echo "[NVify] Bootstrapping virtualenv at: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[NVify] Creating venv with: $PY_CMD"
  "$PY_CMD" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip || true

if [[ -f "requirements.txt" ]]; then
  echo "[NVify] Installing requirements (numpy from source to avoid macOS wheel issues)"
  python -m pip install --no-binary numpy -r requirements.txt
fi

echo "[NVify] Bootstrap complete. Active python: $(command -v python)"
