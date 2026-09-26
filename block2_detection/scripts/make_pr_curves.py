#!/usr/bin/env python3
"""Computes and plots instance-level precision-recall curves on the validation
split. A prediction counts as a true positive at a mask intersection over union
of 0.5; instances are matched in descending order of confidence, so each
annotation is claimed at most once.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

VAL_DIR = paths.DATASET_SEG / "images" / "val"
MASK_DIR = paths.TRAIN
OUT = str(paths.FIGURES / "pr_curves.png")
IOU_T = 0.5
MIN_PX = 15

# Okabe-Ito: distinguishable under the common forms of colour vision deficiency.
MODELS = [
    ("seg_v1", str(paths.seg_weights("seg_v1")), 480, "#0072B2", "-"),
    ("seg_v2", str(paths.seg_weights("seg_v2")), 640, "#E69F00", "-"),
    ("seg_v3", str(paths.seg_weights("seg_v3")), 640, "#009E73", "-"),
    ("seg_v4", str(paths.seg_weights("seg_v4")), 640, "#CC79A7", "-"),
]


def gt_instances(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    return [(lab == i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_PX]


def average_precision(conf, tp, n_gt):
    """All-point interpolated AP, as used for mAP@50."""
    o = np.argsort(-conf)
    tp = tp[o]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]      # precision envelope
    ap = np.sum(np.diff(mrec) * mpre[1:])
    return rec, prec, float(ap)


def evaluate(model, imgsz):
    conf, tp, n_gt = [], [], 0
    for img_p in sorted(VAL_DIR.glob("*.png")):
        m = cv2.imread(str(MASK_DIR / f"{img_p.stem}_labeled.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        gts = gt_instances(m)
        n_gt += len(gts)

        r = model.predict(img, conf=0.001, imgsz=imgsz, device="mps",
                          retina_masks=True, verbose=False)[0]
        if r.masks is None or len(r.masks) == 0:
            continue
        pm = r.masks.data.cpu().numpy() > 0.5
        if pm.shape[1:] != (h, w):
            pm = np.stack([cv2.resize(x.astype(np.uint8), (w, h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                           for x in pm])
        cs = r.boxes.conf.cpu().numpy()
        taken = set()
        for k in np.argsort(-cs):                       # highest score claims first
            best, bi = 0.0, -1
            for j, g in enumerate(gts):
                if j in taken:
                    continue
                u = (pm[k] | g).sum()
                iou = (pm[k] & g).sum() / u if u else 0.0
                if iou > best:
                    best, bi = iou, j
            hit = best >= IOU_T
            if hit:
                taken.add(bi)
            conf.append(cs[k])
            tp.append(1.0 if hit else 0.0)
    return np.array(conf), np.array(tp), n_gt


def main():
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 14, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for name, wts, imgsz, colour, ls in MODELS:
        c, t, n = evaluate(YOLO(wts), imgsz)
        rec, prec, ap = average_precision(c, t, n)
        ax.plot(rec, prec, ls, color=colour, lw=2.0,
                label=f"{name}   AP {ap:.3f}")
        print(f"{name}: {n} annotated instances, {len(c)} predictions, AP@50 = {ap:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=220)
    print(f"{OUT} written")


if __name__ == "__main__":
    main()
