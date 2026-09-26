#!/usr/bin/env python3
"""Derives a YOLO detection dataset from the segmentation dataset by reducing each
polygon to its enclosing box. The split is carried over unchanged.
"""
import shutil
from pathlib import Path

SEG_DIR = Path("dataset_seg")
DET_DIR = Path("dataset_det")


def polygon_to_box(line):
    """YOLO polygon line -> YOLO box line, both normalised.

    Polygon: cls x1 y1 x2 y2 ... xn yn
    Box:     cls xc yc w h
    """
    parts = line.split()
    cls, coords = parts[0], [float(v) for v in parts[1:]]
    xs, ys = coords[0::2], coords[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def main():
    if DET_DIR.exists():
        shutil.rmtree(DET_DIR)

    n_img = n_box = n_empty = 0
    for split in ("train", "val"):
        (DET_DIR / "images" / split).mkdir(parents=True)
        (DET_DIR / "labels" / split).mkdir(parents=True)

        for img in sorted((SEG_DIR / "images" / split).iterdir()):
            if img.suffix.lower() != ".png":
                continue
            (DET_DIR / "images" / split / img.name).symlink_to(img.resolve())

            src = SEG_DIR / "labels" / split / f"{img.stem}.txt"
            lines = []
            if src.exists():
                for line in src.read_text().splitlines():
                    if not line.strip():
                        continue
                    box = polygon_to_box(line)
                    if box:
                        lines.append(box)
            (DET_DIR / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            n_img += 1
            n_box += len(lines)
            if not lines:
                n_empty += 1

    yaml_path = DET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DET_DIR.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: Plaque\n"
    )
    print(f"Detection dataset written to {DET_DIR.resolve()}")
    print(f"  images: {n_img}  boxes: {n_box}  background images: {n_empty}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
