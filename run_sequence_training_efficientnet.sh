#!/usr/bin/env bash
set -euo pipefail

# Run single-frame training with the EfficientNet-B0 backbone on the generated augmented dataset.
# Usage:
#   ./run_sequence_training_efficientnet.sh [model_name] [epochs] [batch_size] [lr] [patience] [min_delta] [max_images] [samples_per_image] [source_dir] [max_val_images]
# Example:
#   ./run_sequence_training_efficientnet.sh efficientnet_b0_single 10 8 1e-4 5 0.0 50000 1 data/aug_camera_test_seq 5000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${1:-efficientnet_b0_single}"
EPOCHS="${2:-1}"
BATCH_SIZE="${3:-8}"
LR="${4:-1e-4}"
EARLY_STOPPING_PATIENCE="${5:-10}"
EARLY_STOPPING_MIN_DELTA="${6:-0.0}"
MAX_SOURCE_ITEMS="${7:-900000}"
SAMPLES_PER_IMAGE="${8:-1}"
SOURCE_DIR="${9:-data/aug_camera_test_seq}"
MAX_VALIDATION_ITEMS="${10:-5000}"

echo "Training single-frame model: $MODEL_NAME"
echo "Source dir: $SOURCE_DIR"
echo "Epochs: $EPOCHS, batch size: $BATCH_SIZE, lr: $LR"
echo "Early stopping patience: $EARLY_STOPPING_PATIENCE, min delta: $EARLY_STOPPING_MIN_DELTA"
echo "Max source images: $MAX_SOURCE_ITEMS, samples per image: $SAMPLES_PER_IMAGE"
echo "Max validation images: $MAX_VALIDATION_ITEMS"

env PYTHONPATH="$ROOT_DIR" python -m src.networks_tool.train \
  --source-dir "$SOURCE_DIR" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --max-source-items "$MAX_SOURCE_ITEMS" \
  --max-validation-items "$MAX_VALIDATION_ITEMS" \
  --samples-per-image "$SAMPLES_PER_IMAGE" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
