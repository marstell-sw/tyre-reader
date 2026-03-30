#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained YOLO model to ONNX.")
    parser.add_argument("--weights", required=True, help="Path to the .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    model = YOLO(str(weights))
    output = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
