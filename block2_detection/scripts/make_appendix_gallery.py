#!/usr/bin/env python3
"""Renders the predictions of the pooled v1+v2 model on every held-out test image.
The confidence threshold is derived from the validation split. Ground truth is
outlined in cyan, the prediction in orange with a light fill.
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

OUT = paths.FIGURES
GT_COL = (255, 255, 0)     # cyan (BGR)
PR_COL = (0, 150, 255)     # orange (BGR)
COLS, ROWS = 4, 4
PANEL_H = 315
CROP_X = (130, 766)   # common non-black column range across all 48 test images


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def panel(rec, t):
    img = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt = rec["gt"]
    keep = rec["confs"] >= t
    pred = (rec["inst"][keep].any(0) if keep.any()
            else np.zeros_like(gt, dtype=bool))

    if pred.any():                                    # fill, then outline
        overlay = vis.copy()
        overlay[pred] = PR_COL
        vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)
        cs, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, PR_COL, 3)
    if gt.any():
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, GT_COL, 3)

    top = rec["confs"].max() if len(rec["confs"]) else 0.0
    if gt.any():
        sub = (f"Dice {dice(gt, pred):.2f}   conf {top:.3f}" if pred.any()
               else f"missed   conf {top:.3f}")
    else:
        sub = f"false positive   conf {top:.3f}" if pred.any() else "correct reject"
    vis = vis[:, CROP_X[0]:CROP_X[1]]
    return fl.label(vis, PANEL_H, rec["label"], sub, scale=0.62)


def main():
    device = "mps"
    models = [(YOLO(str(paths.seg_weights("seg_v1"))), 480),
              (YOLO(str(paths.seg_weights("seg_v2"))), 640)]

    pairs = build_pool()
    recs = run_models(models, pairs, device, augment=False)
    for r, (p, _, _) in zip(recs, pairs):
        r["path"] = p
        r["label"] = p.stem.strip()

    grid = np.arange(0.01, 0.96, 0.01)
    val = [r for r in recs if r["origin"] == "val"]
    sw = pd.DataFrame([evaluate(val, float(t))[0] for t in grid])
    t = float(sw.loc[sw.youden_J.idxmax(), "conf"])
    print(f"t* aus val (n={len(val)}): {t:.2f}  J={sw.youden_J.max():.3f}")

    test = [r for r in recs if r["origin"] == "test"]
    row = evaluate(test, t)[0]
    print(f"test @ t*: sens {row['sensitivity']:.3f}  spec {row['specificity']:.3f}  "
          f"J {row['youden_J']:.3f}  ({len(test)} images)")

    test.sort(key=lambda r: (not r["gt"].any(), int(''.join(
        c for c in r["label"] if c.isdigit()) or 0)))

    per = COLS * ROWS
    for s in range((len(test) + per - 1) // per):
        chunk = test[s * per:(s + 1) * per]
        panels = [panel(r, t) for r in chunk]
        blank = np.full_like(panels[0], 255)
        rows = [fl.row(panels[i:i + COLS] + [blank] * (COLS - len(panels[i:i + COLS])))
                for i in range(0, len(panels), COLS)]
        out = OUT / f"appendix_gallery_{s + 1}.png"
        cv2.imwrite(str(out), fl.stack(rows))
        print(f"{out}  {len(chunk)} images")


if __name__ == "__main__":
    main()
