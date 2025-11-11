#!/usr/bin/env bash
set -euo pipefail

# NVify launcher: ensures artifacts exist and starts the web app

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONUNBUFFERED=1

# Use config default artifacts path
ARTIFACTS_DIR="$REPO_DIR/data/artifacts"

echo "[NVify] Using artifacts at: $ARTIFACTS_DIR"
mkdir -p "$ARTIFACTS_DIR"

# Optional: allow override via env
export NVIFY_ARTIFACTS_DIR="${NVIFY_ARTIFACTS_DIR:-$ARTIFACTS_DIR}"

# Ensure venv and deps via bootstrap.sh
echo "[NVify] Bootstrapping environment (venv + deps; prefer Python 3.11)"
bash "$REPO_DIR/bootstrap.sh" || {
  echo "[NVify] bootstrap failed; continuing without venv (may error)";
}

# Ensure venv uses Python 3.11; if not, recreate with 3.11 when available
if [[ -d "$REPO_DIR/.venv" ]]; then
  VENV_PY_VER="$($REPO_DIR/.venv/bin/python -V 2>/dev/null || echo unknown)"
  if ! echo "$VENV_PY_VER" | grep -q "Python 3.11"; then
    if [[ -x "/opt/homebrew/bin/python3.11" ]]; then
      echo "[NVify] Recreating venv with Python 3.11 (current: $VENV_PY_VER)"
      rm -rf "$REPO_DIR/.venv"
      PY311_BIN="/opt/homebrew/bin/python3.11" VENV_DIR="$REPO_DIR/.venv" bash "$REPO_DIR/bootstrap.sh"
    else
      echo "[NVify] Python 3.11 not found at /opt/homebrew/bin/python3.11. Consider: brew install python@3.11"
    fi
  fi
fi

cd "$REPO_DIR"

# Ensure processed data exists for the app
if [[ ! -f "data/processed/tracks.csv" || ! -f "data/processed/interactions.csv" ]]; then
  echo "[NVify] Processed data missing. Running make_processed.sh"
  bash "$REPO_DIR/make_processed.sh"
fi

# Choose interpreter: prefer project venv
EXEC_PY="python3"
if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  EXEC_PY="$REPO_DIR/.venv/bin/python"
fi

# Start Flask app
echo "[NVify] Starting app (Flask) with: $EXEC_PY"
exec "$EXEC_PY" app.py
