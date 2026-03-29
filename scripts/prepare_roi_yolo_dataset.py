#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a COCO ROI dataset to YOLO detection format.")
    parser.add_argument("--coco", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--images-dir", required=True, help="Directory containing source images")
    parser.add_argument("--output-dir", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of hardlinking")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(output_path: Path, class_names: list[str]) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        f"names: {class_names}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    coco_path = Path(args.coco)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    obj = json.loads(coco_path.read_text(encoding="utf-8"))
    used_category_ids = {ann["category_id"] for ann in obj["annotations"]}
    categories = sorted(
        [cat for cat in obj["categories"] if cat["id"] in used_category_ids],
        key=lambda item: item["id"],
    )
    class_names = [cat["name"] for cat in categories]
    cat_id_to_yolo = {cat["id"]: index for index, cat in enumerate(categories)}

    images = obj["images"]
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in obj["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    rng = random.Random(args.seed)
    image_ids = [img["id"] for img in images]
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * args.val_fraction)))
    val_ids = set(image_ids[:val_count])

    image_by_id = {img["id"]: img for img in images}

    for split in ("train", "val"):
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)

    summary = {
        "images_total": len(images),
        "annotations_total": len(obj["annotations"]),
        "class_names": class_names,
        "splits": {"train": 0, "val": 0},
    }

    for image_id in image_ids:
        image = image_by_id[image_id]
        split = "val" if image_id in val_ids else "train"
        src_image_path = images_dir / image["file_name"]
        dst_image_path = output_dir / "images" / split / image["file_name"]
        safe_link_or_copy(src_image_path, dst_image_path, args.copy_images)

        image_width = float(image["width"])
        image_height = float(image["height"])
        label_lines: list[str] = []

        for ann in annotations_by_image.get(image_id, []):
            x, y, w, h = map(float, ann["bbox"])
            cx = (x + w / 2.0) / image_width
            cy = (y + h / 2.0) / image_height
            nw = w / image_width
            nh = h / image_height
            yolo_class = cat_id_to_yolo[ann["category_id"]]
            label_lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (output_dir / "labels" / split / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        summary["splits"][split] += 1

    write_yaml(output_dir / "dataset.yaml", class_names)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
