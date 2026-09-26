#!/usr/bin/env python3
"""Fits and evaluates a post-hoc re-scorer for predicted plaque instances, and
compares the resulting image-level decisions against the original confidences.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from compare_variants import build_pool, MASK_SUFFIX
from evaluate_seg_testset import evaluate, SWEEP

DICE_POSITIVE = 0.5
MIN_CONF = 0.001
FEATURES = ["conf_log", "area_frac", "aspect", "fill", "y_center",
            "contrast", "inside_std", "rank", "n_inst"]


def instance_features(img_gray, mask, conf, rank, n_inst):
    """Features for one predicted instance. All scale-free or normalised,
    so a model fitted at one image size still applies at another."""
    h, w = img_gray.shape
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    area = mask.sum()

    ring = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) - mask
    inside = img_gray[mask].astype(float)
    outside = img_gray[ring > 0].astype(float)
    contrast = inside.mean() - outside.mean() if outside.size >= 50 else 0.0

    return {
        # log, because raw confidences span three orders of magnitude and a
        # linear model would otherwise see 0.0019 and 0.0010 as identical.
        "conf_log": float(np.log10(max(conf, 1e-6))),
        "area_frac": float(area / (h * w)),
        "aspect": float(bw / max(bh, 1)),
        "fill": float(area / max(bw * bh, 1)),
        "y_center": float(ys.mean() / h),
        "contrast": float(contrast),
        "inside_std": float(inside.std()),
        "rank": float(rank),
        "n_inst": float(n_inst),
    }


def harvest(models, pairs, device):
    """One row per predicted instance, with its Dice against GT."""
    rows, store = [], []
    for img_path, mask_path, origin in pairs:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = gt > 127

        confs, insts = [], []
        for model, imgsz in models:
            res = model.predict(img, conf=MIN_CONF, imgsz=imgsz, device=device,
                                retina_masks=True, verbose=False)[0]
            if res.masks is None or len(res.masks) == 0:
                continue
            m = res.masks.data.cpu().numpy() > 0.5
            if m.shape[1:] != (h, w):
                m = np.stack([cv2.resize(x.astype(np.uint8), (w, h),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                              for x in m])
            confs.append(res.boxes.conf.cpu().numpy())
            insts.append(m)

        confs = np.concatenate(confs) if confs else np.array([])
        insts = (np.concatenate(insts) if insts
                 else np.zeros((0, h, w), dtype=bool))
        order = np.argsort(-confs)
        kept_conf, kept_mask = [], []
        for rank, i in enumerate(order):
            f = instance_features(gray, insts[i], confs[i], rank, len(order))
            if f is None:
                continue
            inter = np.logical_and(insts[i], gt).sum()
            denom = insts[i].sum() + gt.sum()
            f["dice"] = 2 * inter / denom if denom else 0.0
            f["y"] = int(f["dice"] >= DICE_POSITIVE)
            f["image"] = img_path.name
            f["origin"] = origin
            rows.append(f)
            kept_conf.append(confs[i])
            kept_mask.append(insts[i])

        store.append({"name": img_path.name, "gt": gt, "origin": origin,
                      "confs": np.array(kept_conf),
                      "inst": (np.stack(kept_mask) if kept_mask
                               else np.zeros((0, h, w), dtype=bool))})
    return pd.DataFrame(rows), store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=str(paths.seg_weights("seg_v1")))
    ap.add_argument("--v2", default=str(paths.seg_weights("seg_v2")))
    ap.add_argument("--extra", default="", help="optional third checkpoint")
    ap.add_argument("--extra-imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from ultralytics import YOLO
    models = [(YOLO(args.v1), 480), (YOLO(args.v2), 640)]
    if args.extra:
        models.append((YOLO(args.extra), args.extra_imgsz))

    pairs = build_pool()
    print(f"Collecting instances from {len(pairs)} images "
          f"({len(models)} models, conf >= {MIN_CONF}) ...", flush=True)
    df, store = harvest(models, pairs, args.device)

    tr = df[df.origin == "val"]
    te = df[df.origin == "test"]
    print(f"Instances: val {len(tr)} ({tr.y.sum()} positive), "
          f"test {len(te)} ({te.y.sum()} positive)\n")

    # class_weight="balanced" is not used: it inflates the predicted
    # probabilities and so distorts comparisons at a fixed threshold.
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=2000))
    X, y, g = tr[FEATURES].values, tr.y.values, tr.image.values
    cv = GroupKFold(n_splits=5)
    oof = cross_val_predict(clf, X, y, groups=g, cv=cv,
                            method="predict_proba")[:, 1]
    print(f"val, gruppierte 5-fold CV:  AUC {roc_auc_score(y, oof):.3f}")
    print(f"val, rohe Konfidenz allein: AUC "
          f"{roc_auc_score(y, tr.conf_log.values):.3f}")

    clf.fit(X, y)
    coefs = clf.named_steps["logisticregression"].coef_[0]
    print("\nGewichte (standardisiert, Betrag absteigend):")
    for n, c in sorted(zip(FEATURES, coefs), key=lambda t: -abs(t[1])):
        print(f"  {n:<12} {c:+.3f}")

    te_scores = clf.predict_proba(te[FEATURES].values)[:, 1]
    print(f"\ntest, rohe Konfidenz: AUC {roc_auc_score(te.y, te.conf_log):.3f}")
    print(f"test, neu bewertet:   AUC {roc_auc_score(te.y, te_scores):.3f}")

    # Replace confidences with re-scored values, then run the standard sweep.
    by_image = {}
    for name, s in zip(te.image.values, te_scores):
        by_image.setdefault(name, []).append(s)
    test_recs = [r for r in store if r["origin"] == "test"]

    rescored = []
    for r in test_recs:
        s = np.array(by_image.get(r["name"], []))
        n = min(len(s), len(r["confs"]))
        rescored.append({**r, "confs": s[:n], "inst": r["inst"][:n]})

    def sweep_quantile(records):
        """Threshold ladder taken from the score distribution itself.

        A fixed 0.05..0.90 grid is meaningless across two different scoring
        scales: at 0.40 it retains 11% of raw-confidence instances but 25%
        of re-scored ones, so the two methods get compared at different
        operating points and the more permissive one looks worse purely
        because its mask union collects more false positives. Sweeping by
        quantile fixes the retained fraction instead, which is the axis
        both methods actually share.
        """
        pool = np.concatenate([r["confs"] for r in records
                               if len(r["confs"])])
        qs = np.round(np.arange(0.50, 0.996, 0.01), 3)
        out = []
        for q in qs:
            t = float(np.quantile(pool, q))
            row, _ = evaluate(records, t)
            row["quantile"], row["kept_frac"] = q, float((pool >= t).mean())
            out.append(row)
        return pd.DataFrame(out)

    base = sweep_quantile(test_recs)
    new = sweep_quantile(rescored)

    for label, sw in [("Ensemble, rohe Konfidenz", base),
                      ("Ensemble, neu bewertet", new)]:
        bj = sw.loc[sw.youden_J.idxmax()]
        bd = sw.loc[sw.dice_mean_pos.idxmax()]
        print(f"\n{label}")
        print(f"  bestes J    : q={bj['quantile']:.2f} (conf {bj['conf']:.4f}) "
              f"sens={bj['sensitivity']:.3f} spec={bj['specificity']:.3f} "
              f"J={bj['youden_J']:.3f} Dice={bj['dice_mean_pos']:.3f}")
        print(f"  bester Dice : q={bd['quantile']:.2f} "
              f"Dice={bd['dice_mean_pos']:.3f} "
              f"(sens={bd['sensitivity']:.3f} spec={bd['specificity']:.3f})")
    base.to_csv("testset_eval_rawconf_qsweep.csv", index=False)
    new.to_csv("testset_eval_rescored_sweep.csv", index=False)
    df.to_csv("rescore_instances.csv", index=False)
    print("\nGeschrieben: testset_eval_rescored_sweep.csv, rescore_instances.csv")


if __name__ == "__main__":
    main()
