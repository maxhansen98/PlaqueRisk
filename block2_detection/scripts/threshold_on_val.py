#!/usr/bin/env python3
"""Selects the confidence threshold on the validation split and reports test-set
performance at that threshold, alongside the value the test set would have
chosen for itself.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from compare_variants import build_pool, run_models
from evaluate_seg_testset import evaluate

GRID = np.round(np.arange(0.01, 0.96, 0.01), 2)


def preprocess(gray, pp):
    c = gray[pp["roi_y"]:pp["roi_y"] + pp["roi_h"],
             pp["roi_x"]:pp["roi_x"] + pp["roi_w"]]
    b = cv2.medianBlur(c, pp["median_blur_ksize"])
    clahe = cv2.createCLAHE(clipLimit=pp["clip_limit"],
                            tileGridSize=tuple(pp["tile_grid_size"]))
    return clahe.apply(b)


def sweep_seg(records):
    return pd.DataFrame([evaluate(records, t)[0] for t in GRID])


def sweep_imagelevel(scores):
    """scores: list of (max_confidence, gt_positive). Detection models give
    no masks, so only the image-level decision is scored here."""
    rows = []
    for t in GRID:
        tp = sum(1 for s, y in scores if y and s >= t)
        fn = sum(1 for s, y in scores if y and s < t)
        fp = sum(1 for s, y in scores if not y and s >= t)
        tn = sum(1 for s, y in scores if not y and s < t)
        se = tp / (tp + fn) if tp + fn else 0.0
        sp = tn / (tn + fp) if tn + fp else 0.0
        rows.append({"conf": t, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "sensitivity": se, "specificity": sp,
                     "youden_J": se + sp - 1, "dice_mean_pos": np.nan})
    return pd.DataFrame(rows)


def report(name, val_sw, test_sw, out):
    t_star = float(val_sw.loc[val_sw.youden_J.idxmax(), "conf"])
    at = test_sw.loc[(test_sw.conf - t_star).abs().idxmin()]
    opt = test_sw.loc[test_sw.youden_J.idxmax()]
    out.append({
        "model": name, "t_star_from_val": t_star,
        "sens": at["sensitivity"], "spec": at["specificity"],
        "J": at["youden_J"], "Dice": at["dice_mean_pos"],
        "J_test_optimiert": opt["youden_J"],
        "Optimismus": opt["youden_J"] - at["youden_J"],
    })


def main():
    pairs = build_pool()
    val = [p for p in pairs if p[2] == "val"]
    test = [p for p in pairs if p[2] == "test"]
    print(f"val {len(val)} images / test {len(test)} images\n")

    from ultralytics import YOLO
    rows = []

    # --- Georgakis detection baseline, both preprocessing variants ---
    cfg = json.load(open(paths.BLOCK1_CONFIG / "params.json"))
    pp, inf = cfg["preprocess"], cfg["inference"]
    base = YOLO(str(paths.WEIGHTS / "best_YOLO27Jan2024.pt"))

    def base_scores(subset, use_pp):
        out = []
        for img_p, mask_p, _ in subset:
            g = cv2.imread(str(img_p), 0)
            gt = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            im = preprocess(g, pp) if use_pp else g
            # The checkpoint is a 3-channel model; run_plaque_inference.py
            # feeds it a 2-D array and would crash here.
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            r = base.predict(im, conf=0.001, imgsz=inf["img_size"],
                             iou=inf["iou"], max_det=inf["max_det"],
                             device="mps", verbose=False)[0]
            c = (r.boxes.conf.cpu().numpy()
                 if r.boxes is not None and len(r.boxes) else np.array([]))
            out.append((float(c.max()) if len(c) else 0.0, bool((gt > 127).any())))
        return out

    variants = {}
    for tag, use_pp in [("roh", False), ("ROI+CLAHE", True)]:
        v = sweep_imagelevel(base_scores(val, use_pp))
        variants[tag] = (v, float(v.youden_J.max()), use_pp)
        print(f"Baseline {tag:<10} val-J {v.youden_J.max():.3f}")
    best_tag = max(variants, key=lambda k: variants[k][1])
    v_sw, _, use_pp = variants[best_tag]
    print(f"-> Variante auf val gewaehlt: {best_tag}\n")
    report(f"Baseline Georgakis ({best_tag})", v_sw,
           sweep_imagelevel(base_scores(test, use_pp)), rows)

    # --- segmentation models and ensembles ---
    M = {"v1": (YOLO(str(paths.seg_weights("seg_v1"))), 480),
         "v2": (YOLO(str(paths.seg_weights("seg_v2"))), 640),
         "v3": (YOLO(str(paths.seg_weights("seg_v3"))), 640),
         "v4": (YOLO(str(paths.seg_weights("seg_v4"))), 640)}
    for combo in [("v1",), ("v2",), ("v3",), ("v4",),
                  ("v1", "v2"), ("v1", "v4"), ("v2", "v4"),
                  ("v1", "v2", "v4"), ("v1", "v2", "v3", "v4")]:
        ms = [M[c] for c in combo]
        print(f"-> {'+'.join(combo)} ...", flush=True)
        report("+".join(combo),
               sweep_seg(run_models(ms, val, "mps", False)),
               sweep_seg(run_models(ms, test, "mps", False)), rows)

    res = pd.DataFrame(rows)
    res.to_csv("threshold_on_val_results.csv", index=False)
    print("\n=== Schwelle auf val gewaehlt, Ergebnis auf test ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
