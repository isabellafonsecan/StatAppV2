#!/usr/bin/env bash
set -euo pipefail

# Install ffmpeg if missing
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg
fi

# Minimize disk usage from pip caches (safe even if folders don't exist)
pip cache purge >/dev/null 2>&1 || true
rm -rf ~/.cache/pip /tmp/pip-* || true

# Install python deps without cache, then run
pip install --no-cache-dir -r requirements.txt
python transcribe.py