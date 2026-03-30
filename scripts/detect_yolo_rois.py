#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO ROI detection on one or more image directories.")
    parser.add_argument("--input-dir", action="append", required=True, help="Input image directory. Repeat for multiple datasets.")
    parser.add_argument("--weights", required=True, help="YOLO weights (.pt)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.04, help="Relative crop margin")
    return parser.parse_args()


def iter_images(input_dirs: Iterable[str]) -> list[tuple[str, Path]]:
    images: list[tuple[str, Path]] = []
    for input_dir in input_dirs:
        root = Path(input_dir)
        dataset_name = root.name
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append((dataset_name, path))
    return images


def clamp_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def expand_box(box: tuple[int, int, int, int], image_shape: tuple[int, int, int], margin: float) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    dx = int(round(bw * margin))
    dy = int(round(bh * margin))
    return clamp_box(x0 - dx, y0 - dy, x1 + dx, y1 + dy, w, h)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    per_image_dir = output_dir / "per_image"
    crops_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    images = iter_images(args.input_dir)

    summary_rows: list[dict] = []
    class_summary_rows: list[dict] = []

    for dataset_name, image_path in images:
        safe_stem = image_path.stem
        per_image_json = per_image_dir / f"{dataset_name}__{safe_stem}.json"
        if per_image_json.exists():
            data = json.loads(per_image_json.read_text(encoding="utf-8"))
            summary_rows.append(data["summary"])
            class_summary_rows.extend(data["detections"])
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            summary = {
                "dataset": dataset_name,
                "image_path": str(image_path),
                "status": "read_error",
                "image_width": 0,
                "image_height": 0,
            }
            record = {"summary": summary, "detections": []}
            per_image_json.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
            summary_rows.append(summary)
            continue

        result = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            device="cpu",
            verbose=False,
            max_det=16,
        )[0]

        names = result.names
        best_by_label: dict[str, dict] = {}
        detections: list[dict] = []
        if result.boxes is not None:
            for box, cls_id, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                         result.boxes.cls.cpu().numpy(),
                                         result.boxes.conf.cpu().numpy()):
                label = str(names[int(cls_id)])
                x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
                x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, image.shape[1], image.shape[0])
                crop_box = expand_box((x0, y0, x1, y1), image.shape, args.margin)
                cx0, cy0, cx1, cy1 = crop_box
                crop = image[cy0:cy1, cx0:cx1]
                crop_rel = Path("crops") / dataset_name / label
                crop_abs = output_dir / crop_rel
                crop_abs.mkdir(parents=True, exist_ok=True)
                crop_path = crop_abs / f"{safe_stem}.png"
                cv2.imwrite(str(crop_path), crop)

                detection = {
                    "dataset": dataset_name,
                    "image_path": str(image_path),
                    "label": label,
                    "confidence": float(conf),
                    "box": [x0, y0, x1 - x0, y1 - y0],
                    "crop_box": [cx0, cy0, cx1 - cx0, cy1 - cy0],
                    "crop_path": str(crop_path),
                }
                detections.append(detection)
                current_best = best_by_label.get(label)
                if current_best is None or float(conf) > float(current_best["confidence"]):
                    best_by_label[label] = detection

        summary = {
            "dataset": dataset_name,
            "image_path": str(image_path),
            "status": "ok",
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "brand_crop_path": best_by_label.get("Brand", {}).get("crop_path", ""),
            "brand_box": best_by_label.get("Brand", {}).get("box", []),
            "model_crop_path": best_by_label.get("Model", {}).get("crop_path", ""),
            "model_box": best_by_label.get("Model", {}).get("box", []),
            "size_crop_path": best_by_label.get("Size", {}).get("crop_path", ""),
            "size_box": best_by_label.get("Size", {}).get("box", []),
            "dot_crop_path": best_by_label.get("DOT", {}).get("crop_path", ""),
            "dot_box": best_by_label.get("DOT", {}).get("box", []),
            "num_detections": len(detections),
        }

        record = {"summary": summary, "detections": detections}
        per_image_json.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        summary_rows.append(summary)
        class_summary_rows.extend(detections)

    with (output_dir / "detections_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=True, indent=2)

    with (output_dir / "detections_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "image_path", "status", "image_width", "image_height",
            "brand_crop_path", "brand_box", "model_crop_path", "model_box",
            "size_crop_path", "size_box", "dot_crop_path", "dot_box", "num_detections",
        ])
        writer.writeheader()
        for row in summary_rows:
            row = dict(row)
            for key in ("brand_box", "model_box", "size_box", "dot_box"):
                row[key] = ",".join(str(v) for v in row.get(key, []))
            writer.writerow(row)

    with (output_dir / "detections_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "image_path", "label", "confidence", "box", "crop_box", "crop_path"
        ])
        writer.writeheader()
        for row in class_summary_rows:
            row = dict(row)
            row["box"] = ",".join(str(v) for v in row.get("box", []))
            row["crop_box"] = ",".join(str(v) for v in row.get("crop_box", []))
            writer.writerow(row)

    print(json.dumps({
        "images": len(summary_rows),
        "detections": len(class_summary_rows),
        "output_dir": str(output_dir),
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
