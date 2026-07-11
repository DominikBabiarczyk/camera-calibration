#!/usr/bin/env bash
set -euo pipefail

# Run sequence training on the generated augmented dataset.
# Usage:
#   ./run_sequence_training.sh [sequence_length] [model_name] [epochs] [batch_size] [lr]
# Example:
#   ./run_sequence_training.sh 5 cnn_lstm_sequence 50 8 1e-4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SEQUENCE_LENGTH="${1:-5}"
MODEL_NAME="${2:-cnn_lstm_sequence}"
EPOCHS="${3:-50}"
BATCH_SIZE="${4:-8}"
LR="${5:-1e-4}"

echo "Training sequence model: $MODEL_NAME"
echo "Source dir: data/aug_camera_test_seq"
echo "Sequence length: $SEQUENCE_LENGTH"
echo "Epochs: $EPOCHS, batch size: $BATCH_SIZE, lr: $LR"

env PYTHONPATH="$ROOT_DIR" python -m src.networks_tool.train \
  --source-dir data/aug_camera_test_seq \
  --sequence \
  --sequence-length "$SEQUENCE_LENGTH" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR"
