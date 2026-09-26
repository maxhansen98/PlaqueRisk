#!/usr/bin/env python3
"""Evaluates inference-time variants of the segmentation models -- single models,
test-time augmentation and pooled predictions -- on the combined validation and
test images. No retraining is involved.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_seg_testset import evaluate, SWEEP

TRAIN_DIR = paths.TRAIN
TEST_DIR = paths.TEST
VAL_DIR = paths.DATASET_SEG / "images" / "val"
MASK_SUFFIX = "_labeled.png"


def build_pool():
    """(image, mask) pairs for val + test, with their origin recorded."""
    pairs = []
    for img in sorted(VAL_DIR.glob("*.png")):
        pairs.append((img, TRAIN_DIR / f"{img.stem}{MASK_SUFFIX}", "val"))
    masks = {p.name[: -len(MASK_SUFFIX)].strip(): p
             for p in TEST_DIR.glob(f"*{MASK_SUFFIX}")}
    for img in sorted(TEST_DIR.glob("*.png")):
        if img.name.endswith(MASK_SUFFIX):
            continue
        m = masks.get(img.stem.strip())   # 'test/544 .png' -> '544_labeled.png'
        if m:
            pairs.append((img, m, "test"))
    return [(i, m, o) for i, m, o in pairs if m.exists()]


def run_models(models, pairs, device, augment):
    """One record per image holding every instance every model produced.

    Instances are concatenated across models rather than NMS-merged: for a
    single class the downstream union-of-masks is what matters, and NMS
    across models would silently drop the very detections that only one
    model found - which is the whole point of the ensemble.
    """
    records = []
    for img_path, mask_path, origin in pairs:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = gt > 127

        confs, insts = [], []
        for model, imgsz in models:
            res = model.predict(img, conf=0.001, imgsz=imgsz, device=device,
                                retina_masks=True, augment=augment,
                                verbose=False)[0]
            if res.masks is None or len(res.masks) == 0:
                continue
            m = res.masks.data.cpu().numpy() > 0.5
            if m.shape[1:] != (h, w):
                m = np.stack([cv2.resize(x.astype(np.uint8), (w, h),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                              for x in m])
            confs.append(res.boxes.conf.cpu().numpy())
            insts.append(m)

        records.append({
            "name": img_path.name, "gt": gt,
            "confs": np.concatenate(confs) if confs else np.array([]),
            "inst": (np.concatenate(insts) if insts
                     else np.zeros((0, h, w), dtype=bool)),
            "origin": origin,
        })
    return records


def summarise(label, records):
    sweep = pd.DataFrame([evaluate(records, t)[0] for t in SWEEP])
    bj = sweep.loc[sweep["youden_J"].idxmax()]
    bd = sweep.loc[sweep["dice_mean_pos"].idxmax()]
    return {
        "variant": label,
        "conf*": bj["conf"], "sens": bj["sensitivity"],
        "spec": bj["specificity"], "J": bj["youden_J"],
        "dice@J": bj["dice_mean_pos"],
        "dice_best": bd["dice_mean_pos"], "conf_dice": bd["conf"],
        "TP": int(bj["TP"]), "FP": int(bj["FP"]),
        "TN": int(bj["TN"]), "FN": int(bj["FN"]),
    }, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=str(paths.seg_weights("seg_v1")))
    ap.add_argument("--v2", default=str(paths.seg_weights("seg_v2")))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="variant_comparison.csv")
    args = ap.parse_args()

    pairs = build_pool()
    n_pos = sum(1 for _, m, _ in pairs
                if (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 127).any())
    print(f"Pool: {len(pairs)} images ({n_pos} positive, "
          f"{len(pairs) - n_pos} negativ) aus val + test\n")

    from ultralytics import YOLO
    v1 = (YOLO(args.v1), 480)
    v2 = (YOLO(args.v2), 640)

    variants = [
        ("seg_v1 (m@480)",        [v1],     False),
        ("seg_v2 (s@640)",        [v2],     False),
        ("seg_v1 + TTA",          [v1],     True),
        ("seg_v2 + TTA",          [v2],     True),
        ("Ensemble v1+v2",        [v1, v2], False),
        ("Ensemble v1+v2 + TTA",  [v1, v2], True),
    ]

    rows = []
    for label, models, aug in variants:
        print(f"-> {label} ...", flush=True)
        recs = run_models(models, pairs, args.device, aug)
        row, sweep = summarise(label, recs)
        sweep.to_csv(f"variant_{label.split()[0].lower()}"
                     f"{'_tta' if aug else ''}"
                     f"{'_ens' if len(models) > 1 else ''}_sweep.csv",
                     index=False)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    res = pd.DataFrame(rows)
    print(f"\n=== Variant comparison (pool: {len(pairs)} images, "
          f"{n_pos} positive), each at its own best-J point ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    base = res.iloc[0]
    print("\nDelta gegen seg_v1:")
    for _, r in res.iloc[1:].iterrows():
        print(f"  {r['variant']:<24} J {r['J'] - base['J']:+.3f}   "
              f"sens {r['sens'] - base['sens']:+.3f}   "
              f"spec {r['spec'] - base['spec']:+.3f}   "
              f"Dice {r['dice@J'] - base['dice@J']:+.3f}")
    print(f"\nGeschrieben: {args.out}, variant_*_sweep.csv")


if __name__ == "__main__":
    main()
