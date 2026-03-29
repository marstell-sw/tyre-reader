#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO ROI detector with Ultralytics.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--project", default="runs/roi_detector", help="Training project dir")
    parser.add_argument("--name", default="sidewall_roi", help="Run name")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - helper script
        raise SystemExit(
            "Ultralytics is not installed. Install it first, for example in WSL: "
            "python3 -m pip install ultralytics torch torchvision"
        ) from exc

    data_arg = Path(args.data)
    effective_data = data_arg
    if data_arg.suffix.lower() in {".yaml", ".yml"} and data_arg.exists():
        import yaml

        data_obj = yaml.safe_load(data_arg.read_text(encoding="utf-8"))
        if isinstance(data_obj, dict):
            data_obj["path"] = str(data_arg.resolve().parent)
            tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
            with tmp:
                yaml.safe_dump(data_obj, tmp, sort_keys=False)
            effective_data = Path(tmp.name)

    model = YOLO(args.model)
    model.train(
        data=str(effective_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
