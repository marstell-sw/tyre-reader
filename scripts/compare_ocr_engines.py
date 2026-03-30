#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OCR engines on a single ROI image.")
    parser.add_argument("--image", required=True, help="Input ROI image")
    parser.add_argument("--easyocr", action="store_true", help="Run EasyOCR")
    parser.add_argument("--paddleocr", action="store_true", help="Run PaddleOCR")
    return parser.parse_args()


def run_easyocr(image_path: str) -> dict:
    import easyocr

    start = time.perf_counter()
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    init_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    results = reader.readtext(image_path, detail=1, paragraph=False)
    infer_ms = (time.perf_counter() - start) * 1000.0

    normalized = []
    for item in results:
        bbox = [[int(point[0]), int(point[1])] for point in item[0]]
        normalized.append({
            "bbox": bbox,
            "text": str(item[1]),
            "confidence": float(item[2]),
        })

    joined = " ".join(item["text"] for item in normalized).strip()
    mean_conf = sum(item["confidence"] for item in normalized) / max(1, len(normalized))
    return {
        "engine": "easyocr",
        "text": joined,
        "confidence": mean_conf,
        "init_ms": init_ms,
        "infer_ms": infer_ms,
        "raw": normalized,
    }


def run_paddleocr(image_path: str) -> dict:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    start = time.perf_counter()
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    init_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    results = ocr.predict(image_path)
    infer_ms = (time.perf_counter() - start) * 1000.0

    flat = []
    if results:
        for item in results:
            rec_texts = item.get("rec_texts") or []
            rec_scores = item.get("rec_scores") or []
            rec_boxes = item.get("rec_polys") or []
            for idx, text in enumerate(rec_texts):
                box = rec_boxes[idx].tolist() if idx < len(rec_boxes) else []
                norm_box = [[int(point[0]), int(point[1])] for point in box] if box else []
                conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                flat.append({"bbox": norm_box, "text": str(text), "confidence": conf})

    joined = " ".join(item["text"] for item in flat).strip()
    mean_conf = sum(item["confidence"] for item in flat) / max(1, len(flat))
    return {
        "engine": "paddleocr",
        "text": joined,
        "confidence": mean_conf,
        "init_ms": init_ms,
        "infer_ms": infer_ms,
        "raw": flat,
    }


def main() -> int:
    args = parse_args()
    image_path = str(Path(args.image))
    outputs = []
    if args.easyocr:
        outputs.append(run_easyocr(image_path))
    if args.paddleocr:
        outputs.append(run_paddleocr(image_path))
    print(json.dumps(outputs, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
