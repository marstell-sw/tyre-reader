#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/c/Lavoro/MS/TyreReader"
PYTHON_BIN="$REPO_ROOT/.venv-yolo/bin/python3"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train_roi_detector.py"
DATASET_YAML="$REPO_ROOT/ml_artifacts/sidewall_v2_roi_yolo/dataset.yaml"
PROJECT_DIR="$REPO_ROOT/ml_artifacts/yolo_runs"

exec "$PYTHON_BIN" -u "$TRAIN_SCRIPT" \
  --data "$DATASET_YAML" \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 12 \
  --workers 8 \
  --device cpu \
  --project "$PROJECT_DIR" \
  --name sidewall_v2_50ep_cpu
