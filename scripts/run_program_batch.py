#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tyre_reader_v3 on one or more directories and aggregate results.")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    return parser.parse_args()


def iter_images(input_dirs: list[str]) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for input_dir in input_dirs:
        root = Path(input_dir)
        dataset = root.name
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                items.append((dataset, path))
    return items


def flatten_timing(step_timings: list[dict], name: str) -> float:
    for item in step_timings:
        if item.get("name") == name:
            return float(item.get("ms", 0.0))
    return 0.0


def main() -> int:
    args = parse_args()
    binary = Path(args.binary)
    output_dir = Path(args.output_dir)
    per_image_dir = output_dir / "per_image"
    runs_dir = output_dir / "runs"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    images = iter_images(args.input_dir)

    for dataset, image_path in images:
        safe_stem = image_path.stem
        per_image_json = per_image_dir / f"{dataset}__{safe_stem}.json"
        run_output_dir = runs_dir / dataset / safe_stem
        run_output_dir.mkdir(parents=True, exist_ok=True)

        if per_image_json.exists():
            summary_rows.append(json.loads(per_image_json.read_text(encoding="utf-8")))
            continue

        command = [
            str(binary),
            "--image", str(image_path),
            "--output", str(run_output_dir),
        ]
        if args.skip_ocr:
            command.append("--skip-ocr")
        if args.pretty:
            command.append("--pretty")

        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            row = {
                "dataset": dataset,
                "image_path": str(image_path),
                "status": "error",
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
            per_image_json.write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")
            summary_rows.append(row)
            continue

        payload = json.loads(completed.stdout)
        row = {
            "dataset": dataset,
            "image_path": str(image_path),
            "status": "ok",
            "wheel_found": bool(payload.get("wheelFound", False)),
            "size_found": bool(payload.get("tyreSizeFound", False)),
            "size_raw": payload.get("tyreSize", {}).get("raw", ""),
            "size_normalized": payload.get("tyreSize", {}).get("normalized", ""),
            "size_confidence": payload.get("tyreSize", {}).get("confidence", 0.0),
            "dot_found": bool(payload.get("dotFound", False)),
            "dot_week_year_found": bool(payload.get("dotWeekYearFound", False)),
            "dot_raw": payload.get("dot", {}).get("raw", ""),
            "dot_normalized": payload.get("dot", {}).get("normalized", ""),
            "dot_week_year": payload.get("dotWeekYear", ""),
            "dot_confidence": payload.get("dot", {}).get("confidence", 0.0),
            "total_ms": float(payload.get("timings", {}).get("totalMs", 0.0)),
            "wheel_total_ms": flatten_timing(payload.get("stepTimings", []), "wheel_total_ms"),
            "wheel_unwrap_total_ms": flatten_timing(payload.get("stepTimings", []), "wheel_unwrap_total_ms"),
            "yolo_inference_ms": flatten_timing(payload.get("stepTimings", []), "yolo_inference_ms"),
            "size_branch_total_ms": flatten_timing(payload.get("stepTimings", []), "size_branch_total_ms"),
            "dot_branch_total_ms": flatten_timing(payload.get("stepTimings", []), "dot_branch_total_ms"),
            "size_crop_path": payload.get("tyreSize", {}).get("cropPath", ""),
            "dot_crop_path": payload.get("dot", {}).get("cropPath", ""),
            "notes": " | ".join(payload.get("notes", [])),
            "run_output_dir": str(run_output_dir),
        }
        per_image_json.write_text(json.dumps(row, ensure_ascii=True, indent=2), encoding="utf-8")
        summary_rows.append(row)

    with (output_dir / "program_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=True, indent=2)

    with (output_dir / "program_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "image_path", "status",
            "wheel_found", "size_found", "size_raw", "size_normalized", "size_confidence",
            "dot_found", "dot_week_year_found", "dot_raw", "dot_normalized", "dot_week_year", "dot_confidence",
            "total_ms", "wheel_total_ms", "wheel_unwrap_total_ms", "yolo_inference_ms",
            "size_branch_total_ms", "dot_branch_total_ms", "size_crop_path", "dot_crop_path", "notes", "run_output_dir", "error",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({
        "images": len(summary_rows),
        "output_dir": str(output_dir),
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
