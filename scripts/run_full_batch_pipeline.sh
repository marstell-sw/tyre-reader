#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Lavoro/MS/TyreReader"
RUN_ROOT="${ROOT}/batch_runs/20260330_full_eval"

mkdir -p "${RUN_ROOT}"

INPUT_A="${ROOT}/prep_output_5mp"
INPUT_B="${ROOT}/prep_output_5mp_sidewall_v2"
INPUT_C="${ROOT}/prep_output_5mp_tyre_to_text"

PROGRAM_EXTRACT="${RUN_ROOT}/program_extract"
PROGRAM_EVAL="${RUN_ROOT}/program_eval"
YOLO_DETECT="${RUN_ROOT}/yolo_detect"
ASSISTANT_FULL="${RUN_ROOT}/assistant_full"

echo "[1/4] Fast proposal extraction..."
python3 "${ROOT}/scripts/run_program_batch.py" \
  --binary "${ROOT}/build-wsl24/tyre_reader_v3" \
  --input-dir "${INPUT_A}" \
  --input-dir "${INPUT_B}" \
  --input-dir "${INPUT_C}" \
  --output-dir "${PROGRAM_EXTRACT}" \
  --skip-ocr \
  --pretty

echo "[2/4] Full C++ program evaluation (Tesseract)..."
python3 "${ROOT}/scripts/run_program_batch.py" \
  --binary "${ROOT}/build-wsl24/tyre_reader_v3" \
  --input-dir "${INPUT_A}" \
  --input-dir "${INPUT_B}" \
  --input-dir "${INPUT_C}" \
  --output-dir "${PROGRAM_EVAL}" \
  --pretty

echo "[3/4] YOLO raw detection crops..."
"${ROOT}/.venv-yolo/bin/python3" "${ROOT}/scripts/detect_yolo_rois.py" \
  --input-dir "${INPUT_A}" \
  --input-dir "${INPUT_B}" \
  --input-dir "${INPUT_C}" \
  --weights "${ROOT}/ml_artifacts/yolo_runs/sidewall_v2_50ep_cpu/weights/best.pt" \
  --output-dir "${YOLO_DETECT}"

echo "[4/4] Assistant OCR aggregation..."
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
"${ROOT}/.venv-ocr/bin/python3" "${ROOT}/scripts/assistant_full_batch.py" \
  --detections-json "${YOLO_DETECT}/detections_summary.json" \
  --program-json "${PROGRAM_EXTRACT}/program_results.json" \
  --output-dir "${ASSISTANT_FULL}"

echo "Done."
