#!/usr/bin/env bash
set -euo pipefail

# Run LSTM sequence training on the generated augmented dataset.
# Usage:
#   ./run_sequence_training_lstm.sh [sequence_length] [model_name] [epochs] [batch_size] [lr] [patience] [min_delta]
# Example:
#   ./run_sequence_training_lstm.sh 5 cnn_lstm_sequence 50 8 1e-4 10 0.0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SEQUENCE_LENGTH="${1:-5}"
MODEL_NAME="${2:-cnn_lstm_sequence}"
EPOCHS="${3:-1}"
BATCH_SIZE="${4:-8}"
LR="${5:-1e-4}"
EARLY_STOPPING_PATIENCE="${6:-10}"
EARLY_STOPPING_MIN_DELTA="${7:-0.0}"

echo "Training sequence model: $MODEL_NAME"
echo "Source dir: data/aug_camera_test_seq"
echo "Sequence length: $SEQUENCE_LENGTH"
echo "Epochs: $EPOCHS, batch size: $BATCH_SIZE, lr: $LR"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE, min delta: $EARLY_STOPPING_MIN_DELTA"

env PYTHONPATH="$ROOT_DIR" python -m src.networks_tool.train \
  --source-dir data/aug_camera_test_seq \
  --sequence \
  --sequence-length "$SEQUENCE_LENGTH" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
