#!/usr/bin/env python3
"""Renders the five-panel qualitative figure at the operating point derived from
the validation split.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

import figlabel as fl  # from common/, on sys.path via paths
from compare_variants import build_pool, run_models
from evaluate_seg_testset import evaluate

OUT = Path(str(paths.FIGURES / "qualitative_examples.png"))
GT_COL = (0, 255, 0)       # green (BGR)
PR_COL = (0, 0, 255)       # red   (BGR)
CROP_X = (130, 766)
PANEL_H = 430
SCALE = 0.98
PANELS = ["526", "530", "541", "524", "538"]


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def panel(rec, t):
    vis = cv2.cvtColor(cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE),
                       cv2.COLOR_GRAY2BGR)
    gt = rec["gt"]
    keep = rec["confs"] >= t
    pred = (rec["inst"][keep].any(0) if keep.any()
            else np.zeros_like(gt, dtype=bool))
    for mask, col in ((pred, PR_COL), (gt, GT_COL)):
        if mask.any():
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 4)
    sub = f"Dice {dice(gt, pred):.2f}" if pred.any() else "not reported"
    return fl.label(vis[:, CROP_X[0]:CROP_X[1]], PANEL_H, rec["label"], sub,
                    scale=SCALE)


def main():
    models = [(YOLO(str(paths.seg_weights("seg_v1"))), 480),
              (YOLO(str(paths.seg_weights("seg_v2"))), 640)]
    pairs = build_pool()
    recs = run_models(models, pairs, "mps", augment=False)
    for r, (p, _, _) in zip(recs, pairs):
        r["path"], r["label"] = p, p.stem.strip()

    grid = np.arange(0.01, 0.96, 0.01)
    val = [r for r in recs if r["origin"] == "val"]
    sw = pd.DataFrame([evaluate(val, float(x))[0] for x in grid])
    t = float(sw.loc[sw.youden_J.idxmax(), "conf"])
    print(f"t* aus val: {t:.2f}")

    by_name = {r["label"]: r for r in recs if r["origin"] == "test"}
    cv2.imwrite(str(OUT), fl.row([panel(by_name[n], t) for n in PANELS]))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
