#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import easyocr
from paddleocr import PaddleOCR


SIZE_BASE_RE = re.compile(r"\b([1-3]\d{2})\s*[/\\ ]\s*([2-9]\d)\s*([RZ]?)\s*([1-2]\d)\b", re.I)
SIZE_FULL_RE = re.compile(r"\b([1-3]\d{2})\s*[/\\ ]\s*([2-9]\d)\s*([RZ]?)\s*([1-2]\d)(?:\s*[- ]?\s*(\d{2,3}\s*[A-Z]))?\b", re.I)
DOT_RE = re.compile(r"\bDOT\b[\s:-]*([A-Z0-9 ]{3,24})", re.I)
WWYY_RE = re.compile(r"\b([0-5]\d)(\d{2})\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build assistant suggestions from YOLO raw crops + local unwrap crops.")
    parser.add_argument("--detections-json", required=True)
    parser.add_argument("--program-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def squeeze_spaces(text: str) -> str:
    return " ".join(text.split())


def normalize_size(match: re.Match[str]) -> str:
    width = match.group(1)
    aspect = match.group(2)
    construction = match.group(3).upper() if match.group(3) else "R"
    rim = match.group(4)
    return f"{width}/{aspect} {construction}{rim}"


def extract_size(text: str) -> tuple[str, str]:
    compact = squeeze_spaces(text.upper().replace("I5", "15").replace("RI5", "R15").replace("ZR", "R"))
    base_match = SIZE_BASE_RE.search(compact)
    full_match = SIZE_FULL_RE.search(compact)
    size_base = normalize_size(base_match) if base_match else ""
    size_full = size_base
    if full_match:
        suffix = squeeze_spaces((full_match.group(5) or "").replace("-", " "))
        if suffix:
            size_full = f"{normalize_size(full_match)} {suffix}"
        else:
            size_full = normalize_size(full_match)
    return size_base, size_full


def extract_dot(text: str) -> tuple[str, str]:
    compact = squeeze_spaces(text.upper())
    dot_match = DOT_RE.search(compact)
    dot_text = ""
    if dot_match:
        dot_text = "DOT " + squeeze_spaces(dot_match.group(1))
    week_year = ""
    wwyy_match = WWYY_RE.search(compact)
    if wwyy_match:
        week = int(wwyy_match.group(1))
        if 1 <= week <= 53:
            week_year = f"{wwyy_match.group(1)}{wwyy_match.group(2)}"
    return dot_text, week_year


def flatten_easyocr(result: list) -> tuple[str, float]:
    if not result:
        return "", 0.0
    text = squeeze_spaces(" ".join(str(item[1]) for item in result))
    conf = sum(float(item[2]) for item in result) / max(1, len(result))
    return text, conf


def flatten_paddle(result: list[dict]) -> tuple[str, float]:
    if not result:
        return "", 0.0
    texts = result[0].get("rec_texts") or []
    scores = result[0].get("rec_scores") or []
    text = squeeze_spaces(" ".join(str(t) for t in texts))
    conf = float(sum(float(v) for v in scores) / max(1, len(scores))) if scores else 0.0
    return text, conf


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = {row["image_path"]: row for row in json.loads(Path(args.detections_json).read_text(encoding="utf-8"))}
    program_rows = json.loads(Path(args.program_json).read_text(encoding="utf-8"))

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    paddle = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    def run_easy(path: str) -> tuple[str, float, float]:
        if not path or not Path(path).exists():
            return "", 0.0, 0.0
        t0 = time.perf_counter()
        result = reader.readtext(path, detail=1, paragraph=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        text, conf = flatten_easyocr(result)
        return text, conf, elapsed

    def run_paddle(path: str) -> tuple[str, float, float]:
        if not path or not Path(path).exists():
            return "", 0.0, 0.0
        t0 = time.perf_counter()
        result = paddle.predict(path)
        elapsed = (time.perf_counter() - t0) * 1000.0
        text, conf = flatten_paddle(result)
        return text, conf, elapsed

    for row in program_rows:
        image_path = row["image_path"]
        dataset = row["dataset"]
        detect = detections.get(image_path, {})

        summary = {
            "dataset": dataset,
            "image_path": image_path,
            "brand_text": "",
            "brand_box": ",".join(str(v) for v in detect.get("brand_box", [])),
            "model_text": "",
            "model_box": ",".join(str(v) for v in detect.get("model_box", [])),
            "size_base": "",
            "size_full": "",
            "size_box": row.get("size_crop_path", ""),
            "size_engine": "",
            "size_confidence": 0.0,
            "dot_text": "",
            "dot4": "",
            "dot_box": row.get("dot_crop_path", ""),
            "dot_engine": "",
            "dot_confidence": 0.0,
            "wheel_found": row.get("wheel_found", False),
            "yolo_inference_ms": row.get("yolo_inference_ms", 0.0),
            "wheel_total_ms": row.get("wheel_total_ms", 0.0),
            "wheel_unwrap_total_ms": row.get("wheel_unwrap_total_ms", 0.0),
        }

        for label in ("brand", "model"):
            crop_path = detect.get(f"{label}_crop_path", "")
            if not crop_path:
                continue
            easy_text, easy_conf, easy_ms = run_easy(crop_path)
            summary[f"{label}_text"] = easy_text
            detail_rows.append({
                "dataset": dataset,
                "image_path": image_path,
                "label": label,
                "engine": "easyocr",
                "crop_path": crop_path,
                "raw_text": easy_text,
                "confidence": easy_conf,
                "elapsed_ms": easy_ms,
                "parsed_base": "",
                "parsed_full": "",
                "parsed_dot": "",
                "parsed_dot4": "",
            })

        size_crop_path = row.get("size_crop_path", "")
        easy_text, easy_conf, easy_ms = run_easy(size_crop_path)
        paddle_text, paddle_conf, paddle_ms = run_paddle(size_crop_path)
        easy_base, easy_full = extract_size(easy_text)
        paddle_base, paddle_full = extract_size(paddle_text)
        if paddle_base:
            summary["size_base"] = paddle_base
            summary["size_full"] = paddle_full or paddle_base
            summary["size_engine"] = "paddleocr"
            summary["size_confidence"] = paddle_conf
        elif easy_base:
            summary["size_base"] = easy_base
            summary["size_full"] = easy_full or easy_base
            summary["size_engine"] = "easyocr"
            summary["size_confidence"] = easy_conf
        detail_rows.extend([
            {
                "dataset": dataset,
                "image_path": image_path,
                "label": "size",
                "engine": "easyocr",
                "crop_path": size_crop_path,
                "raw_text": easy_text,
                "confidence": easy_conf,
                "elapsed_ms": easy_ms,
                "parsed_base": easy_base,
                "parsed_full": easy_full,
                "parsed_dot": "",
                "parsed_dot4": "",
            },
            {
                "dataset": dataset,
                "image_path": image_path,
                "label": "size",
                "engine": "paddleocr",
                "crop_path": size_crop_path,
                "raw_text": paddle_text,
                "confidence": paddle_conf,
                "elapsed_ms": paddle_ms,
                "parsed_base": paddle_base,
                "parsed_full": paddle_full,
                "parsed_dot": "",
                "parsed_dot4": "",
            },
        ])

        dot_crop_path = row.get("dot_crop_path", "")
        easy_text, easy_conf, easy_ms = run_easy(dot_crop_path)
        paddle_text, paddle_conf, paddle_ms = run_paddle(dot_crop_path)
        easy_dot, easy_wwyy = extract_dot(easy_text)
        paddle_dot, paddle_wwyy = extract_dot(paddle_text)
        if paddle_dot or paddle_wwyy:
            summary["dot_text"] = paddle_dot
            summary["dot4"] = paddle_wwyy
            summary["dot_engine"] = "paddleocr"
            summary["dot_confidence"] = paddle_conf
        elif easy_dot or easy_wwyy:
            summary["dot_text"] = easy_dot
            summary["dot4"] = easy_wwyy
            summary["dot_engine"] = "easyocr"
            summary["dot_confidence"] = easy_conf
        detail_rows.extend([
            {
                "dataset": dataset,
                "image_path": image_path,
                "label": "dot",
                "engine": "easyocr",
                "crop_path": dot_crop_path,
                "raw_text": easy_text,
                "confidence": easy_conf,
                "elapsed_ms": easy_ms,
                "parsed_base": "",
                "parsed_full": "",
                "parsed_dot": easy_dot,
                "parsed_dot4": easy_wwyy,
            },
            {
                "dataset": dataset,
                "image_path": image_path,
                "label": "dot",
                "engine": "paddleocr",
                "crop_path": dot_crop_path,
                "raw_text": paddle_text,
                "confidence": paddle_conf,
                "elapsed_ms": paddle_ms,
                "parsed_base": "",
                "parsed_full": "",
                "parsed_dot": paddle_dot,
                "parsed_dot4": paddle_wwyy,
            },
        ])

        summary_rows.append(summary)

    (output_dir / "assistant_suggestions.json").write_text(
        json.dumps(summary_rows, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "assistant_suggestions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "image_path",
            "brand_text", "brand_box",
            "model_text", "model_box",
            "size_base", "size_full", "size_box", "size_engine", "size_confidence",
            "dot_text", "dot4", "dot_box", "dot_engine", "dot_confidence",
            "wheel_found", "wheel_total_ms", "wheel_unwrap_total_ms", "yolo_inference_ms",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "ocr_details.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "image_path", "label", "engine", "crop_path", "raw_text",
            "confidence", "elapsed_ms", "parsed_base", "parsed_full", "parsed_dot", "parsed_dot4",
        ])
        writer.writeheader()
        writer.writerows(detail_rows)

    print(json.dumps({"images": len(summary_rows), "detail_rows": len(detail_rows), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
