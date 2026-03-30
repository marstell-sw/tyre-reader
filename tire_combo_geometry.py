from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TireGeometry:
    cx: int
    cy: int
    inner_r: int
    outer_r: int
    inner_points: list[tuple[int, int]]

    @property
    def sidewall(self) -> int:
        return self.outer_r - self.inner_r


def fit_circle(points: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    if len(points) < 3:
        return None
    pts = np.array(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    d, e, f = np.linalg.lstsq(a, b, rcond=None)[0]
    cx, cy = d / 2, e / 2
    r = np.sqrt(max(0.0, f + cx**2 + cy**2))
    return int(cx), int(cy), int(r)


def find_inner_from_borders(gray: np.ndarray) -> tuple[tuple[int, int, int] | None, list[tuple[int, int]]]:
    h, w = gray.shape[:2]
    scans = [
        (w // 2, 0, 0, 1),
        (w // 2, h - 1, 0, -1),
        (0, h // 2, 1, 0),
        (w - 1, h // 2, -1, 0),
    ]

    points: list[tuple[int, int]] = []
    for sx, sy, dx, dy in scans:
        values: list[float] = []
        coords: list[tuple[int, int]] = []
        x, y = sx, sy
        while 0 <= x < w and 0 <= y < h:
            values.append(float(gray[y, x]))
            coords.append((x, y))
            x += dx
            y += dy

        if len(values) < 50:
            continue

        smooth = np.convolve(values, np.ones(15) / 15, mode="same")
        in_tire = False
        for i, value in enumerate(smooth):
            if value > 80:
                in_tire = True
            if in_tire and value < 50:
                points.append(coords[i])
                break

    return fit_circle(points), points


def find_outer_radial(gray: np.ndarray, cx: int, cy: int) -> int | None:
    h, w = gray.shape[:2]
    max_r = min(cx, cy, w - cx, h - cy) - 10
    outer_radii: list[int] = []

    for i in range(360):
        angle = 2 * np.pi * i / 360
        rr = np.arange(10, max_r, 2)
        xs = (cx + rr * np.cos(angle)).astype(int)
        ys = (cy + rr * np.sin(angle)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys, rr = xs[valid], ys[valid], rr[valid]
        if len(xs) < 20:
            continue

        values = np.convolve(gray[ys, xs].astype(float), np.ones(7) / 7, mode="same")
        grad = np.diff(values)
        for j in range(len(grad) - 1, 0, -1):
            if grad[j] < -10 and rr[j] > 100:
                outer_radii.append(int(rr[j]))
                break

    if not outer_radii:
        return None
    return int(np.median(outer_radii))


def detect_tire_geometry(gray: np.ndarray) -> TireGeometry | None:
    inner_circle, inner_points = find_inner_from_borders(gray)
    if not inner_circle:
        return None

    cx, cy, inner_r = inner_circle
    outer_r = find_outer_radial(gray, cx, cy)
    if outer_r is None or outer_r <= inner_r:
        return None

    return TireGeometry(
        cx=cx,
        cy=cy,
        inner_r=inner_r,
        outer_r=outer_r,
        inner_points=inner_points,
    )


def unwrap_sidewall(
    image: np.ndarray,
    geometry: TireGeometry,
    out_h: int | None = None,
    outer_margin: float = 0.02,
    inner_margin: float = 0.02,
) -> np.ndarray:
    sidewall = max(geometry.sidewall, 1)
    if out_h is None:
        out_h = max(220, min(420, sidewall * 2))

    outer_r = int(geometry.outer_r - sidewall * outer_margin)
    inner_r = int(geometry.inner_r + sidewall * inner_margin)
    width = int(2 * np.pi * outer_r)

    angles = np.linspace(0, 2 * np.pi, width, endpoint=False)
    radii = np.linspace(outer_r, inner_r, out_h)

    map_x = np.zeros((out_h, width), dtype=np.float32)
    map_y = np.zeros((out_h, width), dtype=np.float32)
    for i, radius in enumerate(radii):
        map_x[i, :] = geometry.cx + radius * np.cos(angles)
        map_y[i, :] = geometry.cy + radius * np.sin(angles)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def draw_geometry(image: np.ndarray, geometry: TireGeometry) -> np.ndarray:
    vis = image.copy()
    for px, py in geometry.inner_points:
        cv2.circle(vis, (px, py), 12, (0, 255, 0), -1)

    overlay = vis.copy()
    cv2.circle(overlay, (geometry.cx, geometry.cy), geometry.outer_r, (0, 180, 255), -1)
    cv2.circle(overlay, (geometry.cx, geometry.cy), geometry.inner_r, (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)

    cv2.circle(vis, (geometry.cx, geometry.cy), geometry.inner_r, (0, 255, 0), 3)
    cv2.circle(vis, (geometry.cx, geometry.cy), geometry.outer_r, (0, 100, 255), 3)
    cv2.circle(vis, (geometry.cx, geometry.cy), 8, (0, 0, 255), -1)
    return vis


def process_image(path: Path, out_dir: Path) -> None:
    image = cv2.imread(str(path))
    if image is None:
        print(f"{path.name}: read failed")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    geometry = detect_tire_geometry(gray)
    if geometry is None:
        print(f"{path.name}: geometry failed")
        return

    vis = draw_geometry(image, geometry)
    unwrap = unwrap_sidewall(image, geometry)

    stem = path.stem
    cv2.imwrite(str(out_dir / f"{stem}_combo.jpg"), vis)
    cv2.imwrite(str(out_dir / f"{stem}_unwrap.jpg"), unwrap)

    print(
        f"{path.name}: center=({geometry.cx},{geometry.cy}) "
        f"inner=r{geometry.inner_r} outer=r{geometry.outer_r} sidewall={geometry.sidewall}px "
        f"unwrap={unwrap.shape[1]}x{unwrap.shape[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rilevamento combo corona circolare pneumatico")
    parser.add_argument("paths", nargs="*", help="immagini input; default: vere_foto/*.jpeg")
    parser.add_argument("--out-dir", default="debug_combo_geometry", help="cartella output debug")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths] if args.paths else sorted(Path("vere_foto").glob("*.jpeg"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for path in paths:
        if "(1)" in path.name:
            continue
        process_image(path, out_dir)


if __name__ == "__main__":
    main()
