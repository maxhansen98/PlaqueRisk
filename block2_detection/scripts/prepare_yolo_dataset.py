#!/usr/bin/env python3
"""Builds a YOLO detection dataset from the ground-truth masks by converting each
connected component into a bounding box.
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
DATASET_DIR = Path("dataset")
CLASS_ID = 0
CLASS_NAME = "Plaque"
MIN_BLOB_AREA = 15  # px^2 - filters noise specks from mask thresholding
VAL_FRACTION = 0.15
SEED = 42


def mask_to_yolo_lines(mask_path, img_w, img_h):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in contours:
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        xc, yc = (x + w / 2) / img_w, (y + h / 2) / img_h
        wn, hn = w / img_w, h / img_h
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
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

    n_boxes_total = 0
    n_empty = 0
    n_written = 0
    for _, row in df.iterrows():
        name = row["image"]
        src_img = INPUT_DIR / name
        if not src_img.exists():
            continue

        img = cv2.imread(str(src_img))
        h, w = img.shape[:2]

        lines = []
        if row["has_label"] == "yes":
            mask_path = INPUT_DIR / f"{Path(name).stem}_labeled.png"
            lines = mask_to_yolo_lines(mask_path, w, h)

        split = "val" if name in val_names else "train"
        shutil.copy(src_img, DATASET_DIR / "images" / split / name)
        label_path = DATASET_DIR / "labels" / split / f"{Path(name).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        n_written += 1
        n_boxes_total += len(lines)
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
    print(f"Dataset written to {DATASET_DIR.resolve()}")
    print(f"  images written: {n_written}  (train: {n_train}, val: {n_val})")
    print(f"  total boxes: {n_boxes_total}, background (no-box) images: {n_empty}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
