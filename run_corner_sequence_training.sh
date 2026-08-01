#!/usr/bin/env bash
set -euo pipefail

# Run compact corner-based sequence training with synthetic sequences generated
# on-the-fly during training. No intermediate dataset is written to disk.
#
# Usage:
#   ./run_corner_sequence_training.sh [epochs] [batch_size] [lr] [patience] [min_delta] [train_sequences] [generator_config] [val_sequences] [seed]
# Example:
#   ./run_corner_sequence_training.sh 30 16 1e-3 8 0.0 20000 generate_dataset/creating_various_perspectives/camera_calibration_config.yaml 5000 42

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="corner_gru_sequence"
SEQUENCE_LENGTH="10"
EPOCHS="${1:-15}"
BATCH_SIZE="${2:-64}"
LR="${3:-1e-3}"
EARLY_STOPPING_PATIENCE="${4:-10}"
EARLY_STOPPING_MIN_DELTA="${5:-0.0}"
MAX_SOURCE_ITEMS="${6:-10000000}"
SYNTHETIC_CONFIG="${7:-generate_dataset/creating_various_perspectives/camera_calibration_config.yaml}"
MAX_VALIDATION_ITEMS="${8:-10000}"
SYNTHETIC_SEED="${9:-42}"

echo "Training corner-sequence model: $MODEL_NAME"
echo "Synthetic config: $SYNTHETIC_CONFIG"
echo "Frames per sequence: $SEQUENCE_LENGTH"
echo "Epochs: $EPOCHS, batch size: $BATCH_SIZE, lr: $LR"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE, min delta: $EARLY_STOPPING_MIN_DELTA"
echo "Train sequences generated per epoch: $MAX_SOURCE_ITEMS"
echo "Validation sequences generated on-the-fly: $MAX_VALIDATION_ITEMS"
echo "Synthetic seed: $SYNTHETIC_SEED"

env PYTHONPATH="$ROOT_DIR" .venv/bin/python -m src.networks_tool.train \
  --synthetic-corner-sequence \
  --sequence-length "$SEQUENCE_LENGTH" \
  --synthetic-corner-config "$SYNTHETIC_CONFIG" \
  --synthetic-seed "$SYNTHETIC_SEED" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-source-items "$MAX_SOURCE_ITEMS" \
  --max-validation-items "$MAX_VALIDATION_ITEMS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"