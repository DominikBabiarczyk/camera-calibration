#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EPOCHS="${1:-30}"
BATCH_SIZE="${2:-64}"
LR="${3:-1e-3}"
PATIENT="${4:-10}"
MIN_DELTA="${5:-0.0}"
TRAIN_SEQUENCES="${6:-10000}"
CALIBRATION_RESULT="${7:-extern/XHOG-007_charuco/result.npz}"
VAL_SEQUENCES="${8:-2000}"
SEED="${9:-42}"

echo "Training fisheye corner GRU from $CALIBRATION_RESULT"
env PYTHONPATH="$ROOT_DIR" .venv/bin/python -m src.networks_tool.train \
  --fisheye-corner-sequence \
  --sequence-length 10 \
  --fisheye-calibration-result "$CALIBRATION_RESULT" \
  --model-name fisheye_corner_gru_sequence \
  --output-dir outputs/calibration_net_fisheye \
  --checkpoint-path outputs/calibration_net_fisheye/best_model.pth \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
  --max-source-items "$TRAIN_SEQUENCES" \
  --max-validation-items "$VAL_SEQUENCES" \
  --synthetic-seed "$SEED" \
  --early-stopping-patience "$PATIENT" \
  --early-stopping-min-delta "$MIN_DELTA"