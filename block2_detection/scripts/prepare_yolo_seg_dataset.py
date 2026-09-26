#!/usr/bin/env python3
"""Builds a YOLO segmentation dataset from the ground-truth masks by converting
each connected component into a polygon.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

INPUT_DIR = paths.TRAIN
RESULTS_CSV = "labeling_results.csv"
DATASET_DIR = Path("dataset_seg")
CLASS_ID = 0
CLASS_NAME = "Plaque"
MIN_BLOB_AREA = 15      # px^2 - filters noise specks from mask thresholding
POLY_EPSILON_FRAC = 0.002  # contour simplification, fraction of arc length
MIN_POLY_POINTS = 3
VAL_FRACTION = 0.15
SEED = 42


def mask_to_polygon_lines(mask_path, img_w, img_h):
    """One line per mask blob: class + flattened normalized polygon."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in contours:
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            continue
        # Simplify: raw contours carry hundreds of near-collinear points,
        # which bloats the label files without adding shape information.
        eps = POLY_EPSILON_FRAC * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < MIN_POLY_POINTS:
            continue
        coords = []
        for x, y in approx:
            coords.append(f"{np.clip(x / img_w, 0, 1):.6f}")
            coords.append(f"{np.clip(y / img_h, 0, 1):.6f}")
        lines.append(f"{CLASS_ID} " + " ".join(coords))
    return lines


def main():
    df = pd.read_csv(RESULTS_CSV, dtype={"image": str})

    rng = np.random.default_rng(SEED)
    val_names = set()
    for _, group in df.groupby("has_label"):
        names = group["image"].tolist()
        rng.shuffle(names)
        n_val = max(1, int(len(names) * VAL_FRACTION))
        val_names.update(names[:n_val])

    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = n_polys = n_empty = 0
    point_counts = []
    for _, row in df.iterrows():
        name = row["image"]
        src_img = INPUT_DIR / name
        if not src_img.exists():
            continue

        img = cv2.imread(str(src_img))
        h, w = img.shape[:2]

        lines = []
        if row["has_label"] == "yes":
            lines = mask_to_polygon_lines(INPUT_DIR / f"{Path(name).stem}_labeled.png", w, h)

        split = "val" if name in val_names else "train"
        shutil.copy(src_img, DATASET_DIR / "images" / split / name)
        (DATASET_DIR / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        n_written += 1
        n_polys += len(lines)
        point_counts.extend((len(l.split()) - 1) // 2 for l in lines)
        if not lines:
            n_empty += 1

    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  {CLASS_ID}: {CLASS_NAME}\n"
    )

    n_train = len(list((DATASET_DIR / "images" / "train").glob("*.png")))
    n_val = len(list((DATASET_DIR / "images" / "val").glob("*.png")))
    pc = np.array(point_counts) if point_counts else np.array([0])
    print(f"Segmentation dataset written to {DATASET_DIR.resolve()}")
    print(f"  images: {n_written}  (train: {n_train}, val: {n_val})")
    print(f"  polygons: {n_polys}, background (no-polygon) images: {n_empty}")
    print(f"  polygon points: median {int(np.median(pc))}, min {pc.min()}, max {pc.max()}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
