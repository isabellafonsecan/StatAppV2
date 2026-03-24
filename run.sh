#!/usr/bin/env bash
set -euo pipefail

# Install ffmpeg if missing (needed to read mp3)
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg
fi

# (Optional) activate venv if it exists (common in SSP Cloud)
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

# Upgrade packaging tooling (avoids some install issues)
python -m pip install -U pip setuptools wheel

# Install deps
pip install -r requirements.txt

# Run transcription + diarization
python transcribe.py
python diarize.py