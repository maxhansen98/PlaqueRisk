#!/usr/bin/env python3
"""Evaluates a YOLO segmentation checkpoint on the held-out test set. Sweeps the
confidence threshold and reports image-level sensitivity and specificity
alongside pixel-level Dice.
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

TEST_DIR = paths.TEST
DEFAULT_WEIGHTS = str(paths.seg_weights("seg_v1"))
MASK_SUFFIX = "_labeled.png"
SWEEP = np.round(np.arange(0.05, 0.91, 0.05), 2)


def pair_test_images(test_dir):
    """(image, mask) pairs. Filenames are matched on the stripped stem -
    one image is called '544 .png' while its mask is '544_labeled.png';
    without the strip that pair silently drops out of the evaluation."""
    masks = {p.name[: -len(MASK_SUFFIX)].strip(): p
             for p in test_dir.glob(f"*{MASK_SUFFIX}")}
    pairs, orphans = [], []
    for img in sorted(test_dir.glob("*.png")):
        if img.name.endswith(MASK_SUFFIX):
            continue
        mask = masks.get(img.stem.strip())
        (pairs if mask else orphans).append((img, mask) if mask else img)
    return pairs, orphans


def predict_all(model, pairs, device, imgsz):
    """One inference pass at conf=0.001, keeping every instance with its
    score. Thresholding afterwards makes the sweep free - re-running
    predict() per threshold would be ~20x the work for identical masks."""
    records = []
    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        gt = gt > 127

        res = model.predict(img, conf=0.001, imgsz=imgsz, device=device,
                            retina_masks=True, verbose=False)[0]

        if res.masks is None or len(res.masks) == 0:
            confs, inst = np.array([]), np.zeros((0, h, w), dtype=bool)
        else:
            confs = res.boxes.conf.cpu().numpy()
            inst = res.masks.data.cpu().numpy() > 0.5
            if inst.shape[1:] != (h, w):
                inst = np.stack([
                    cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in inst
                ]) if len(inst) else np.zeros((0, h, w), dtype=bool)

        records.append({"name": img_path.name, "gt": gt,
                        "confs": confs, "inst": inst})
    return records


def evaluate(records, thr):
    """Image-level confusion + pixel overlap at one confidence threshold."""
    tp = fp = tn = fn = 0
    inter_sum = pred_sum = gt_sum = 0
    dice_pos, iou_pos, rows = [], [], []

    for r in records:
        keep = r["confs"] >= thr
        pred = (r["inst"][keep].any(axis=0) if keep.any()
                else np.zeros_like(r["gt"], dtype=bool))

        gt_pos, pred_pos = bool(r["gt"].any()), bool(keep.any())
        if gt_pos and pred_pos:
            tp += 1
        elif gt_pos:
            fn += 1
        elif pred_pos:
            fp += 1
        else:
            tn += 1

        i = int(np.logical_and(pred, r["gt"]).sum())
        p, g = int(pred.sum()), int(r["gt"].sum())
        inter_sum, pred_sum, gt_sum = inter_sum + i, pred_sum + p, gt_sum + g

        dice = (2 * i / (p + g)) if (p + g) else 1.0
        union = p + g - i
        iou = (i / union) if union else 1.0
        if gt_pos:
            # A missed plaque enters as 0.0 rather than being dropped -
            # averaging only over detections would reward misses.
            dice_pos.append(dice)
            iou_pos.append(iou)

        rows.append({"image": r["name"], "gt_positive": gt_pos,
                     "pred_positive": pred_pos, "n_pred": int(keep.sum()),
                     "max_conf": float(r["confs"].max()) if len(r["confs"]) else 0.0,
                     "dice": dice, "iou": iou,
                     "gt_px": g, "pred_px": p})

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "conf": thr, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "youden_J": sens + spec - 1,
        "f1_image": (2 * prec * sens / (prec + sens))
                    if prec and sens and not np.isnan(prec) else float("nan"),
        "dice_mean_pos": float(np.mean(dice_pos)) if dice_pos else float("nan"),
        "iou_mean_pos": float(np.mean(iou_pos)) if iou_pos else float("nan"),
        "dice_global": (2 * inter_sum / (pred_sum + gt_sum))
                       if (pred_sum + gt_sum) else float("nan"),
    }, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-prefix", default="testset_eval")
    args = ap.parse_args()

    pairs, orphans = pair_test_images(TEST_DIR)
    n_pos = sum(1 for _, m in pairs
                if (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 127).any())
    print(f"Test set: {len(pairs)} images ({n_pos} positive, "
          f"{len(pairs) - n_pos} negative)")
    if orphans:
        print(f"  WARNING: {len(orphans)} image(s) without a mask, skipped: "
              f"{[p.name for p in orphans]}")

    from ultralytics import YOLO
    print(f"Weights: {args.weights}  imgsz={args.imgsz}  device={args.device}")
    records = predict_all(YOLO(args.weights), pairs, args.device, args.imgsz)

    sweep = pd.DataFrame([evaluate(records, t)[0] for t in SWEEP])
    sweep.to_csv(f"{args.out_prefix}_sweep.csv", index=False)

    show = ["conf", "TP", "FP", "TN", "FN", "sensitivity", "specificity",
            "precision", "youden_J", "dice_mean_pos", "dice_global"]
    print("\n=== Confidence sweep ===")
    print(sweep[show].to_string(index=False,
                                float_format=lambda v: f"{v:.3f}"))

    best_j = sweep.loc[sweep["youden_J"].idxmax()]
    best_d = sweep.loc[sweep["dice_mean_pos"].idxmax()]
    print(f"\nBest image-level operating point (Youden's J): conf="
          f"{best_j['conf']:.2f}  sens={best_j['sensitivity']:.3f}  "
          f"spec={best_j['specificity']:.3f}  J={best_j['youden_J']:.3f}")
    print(f"Best mask overlap:                              conf="
          f"{best_d['conf']:.2f}  Dice={best_d['dice_mean_pos']:.3f}  "
          f"IoU={best_d['iou_mean_pos']:.3f}")

    _, per_image = evaluate(records, float(best_j["conf"]))
    per_image.to_csv(f"{args.out_prefix}_per_image.csv", index=False)
    print(f"\nWritten: {args.out_prefix}_sweep.csv, "
          f"{args.out_prefix}_per_image.csv (at conf={best_j['conf']:.2f})")

    worst = per_image[per_image.gt_positive].nsmallest(5, "dice")
    print("\nWorst 5 positives (candidates for a look at the annotation):")
    print(worst[["image", "dice", "n_pred", "max_conf", "gt_px", "pred_px"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
