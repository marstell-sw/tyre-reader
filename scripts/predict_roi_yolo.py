#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO ROI detection on a tyre image.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--weights", required=True, help="YOLO weights path")
    parser.add_argument("--output-dir", required=True, help="Directory for debug overlay")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = YOLO(str(Path(args.weights)))
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
        device="cpu",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    overlay = image.copy() if image is not None else None

    detections = []
    names = results[0].names if results else {}
    if results:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            label = names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "w": max(1, x2 - x1),
                    "h": max(1, y2 - y1),
                }
            )
            if overlay is not None:
                color = (0, 255, 255)
                if label.lower() == "size":
                    color = (0, 255, 0)
                elif label.lower() == "dot":
                    color = (0, 140, 255)
                elif label.lower() == "brand":
                    color = (255, 0, 255)
                elif label.lower() == "model":
                    color = (255, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    f"{label} {conf:.2f}",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    overlay_path = output_dir / f"{image_path.stem}_yolo_overlay.png"
    if overlay is not None:
        cv2.imwrite(str(overlay_path), overlay)

    payload = {
        "overlayPath": str(overlay_path),
        "elapsedMs": elapsed_ms,
        "detections": detections,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
