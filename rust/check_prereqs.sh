#!/usr/bin/env bash
set -Eeuo pipefail

echo "[VoiceStand] prerequisite verification"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $1"
    return 1
  fi
  echo "[OK] $1"
}

need_cmd rustc
need_cmd cargo
need_cmd pkg-config
need_cmd gcc

if pkg-config --exists alsa; then
  echo "[OK] alsa development package detected"
else
  echo "[WARN] alsa.pc not found (Linux audio build may fail)"
fi

MODEL_DIR="${HOME}/.config/voice-to-text/models"
if [ -d "$MODEL_DIR" ]; then
  count=$(find "$MODEL_DIR" -maxdepth 1 -name 'ggml-*.bin' | wc -l)
  if [ "$count" -gt 0 ]; then
    echo "[OK] model files detected in $MODEL_DIR ($count found)"
  else
    echo "[WARN] no model files in $MODEL_DIR"
  fi
else
  echo "[WARN] model directory missing: $MODEL_DIR"
fi

echo "[VoiceStand] prerequisite verification complete"
