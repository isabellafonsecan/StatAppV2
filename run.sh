#!/usr/bin/env bash
set -euo pipefail

# Install ffmpeg if missing (needed to read mp3)
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg
fi

# Reduce disk usage from pip caches
pip cache purge >/dev/null 2>&1 || true
rm -rf ~/.cache/pip /tmp/pip-* || true

# Install deps without cache
pip install --no-cache-dir -r requirements.txt

# Run transcription + diarization
python transcribe.py
python diarize.py