#!/usr/bin/env python3
"""Computes detection metrics from labeling_results.csv using the result codes
written by the labelling tool.
"""
import json

import pandas as pd

RESULTS_CSV = "labeling_results.csv"
DETECTIONS_CACHE = "yolo_detections_cache.json"


def load_detections(path=DETECTIONS_CACHE):
    with open(path) as f:
        return {c["image"]: c for c in json.load(f)}


def evaluate(results_csv=RESULTS_CSV, detections_cache=DETECTIONS_CACHE):
    df = pd.read_csv(results_csv, dtype={"image": str})
    dets = load_detections(detections_cache)

    def has_boxes(name):
        return bool(dets.get(name, {}).get("boxes"))

    df["det_present"] = df["image"].map(has_boxes)

    # Structural, not result-based: labelers may leave these unlabeled or
    # click "Korrekt" [1] through them - either way they belong in TN, not TP.
    is_tn = (df["has_label"] == "no") & (~df["det_present"])
    is_tp = (df["has_label"] == "yes") & (df["result"] == 1)
    pending = df["result"].isna() & ~is_tn

    tp = int(is_tp.sum())
    fn_direct = int((df["result"] == 0).sum())
    fp_direct = int((df["result"] == 2).sum())
    both = int((df["result"] == 7).sum())
    tn = int(is_tn.sum())

    fn_total = fn_direct + both
    fp_total = fp_direct + both
    n_pending = int(pending.sum())
    total_resolved = tp + fn_total + fp_total + tn

    precision = tp / (tp + fp_total) if (tp + fp_total) else float("nan")
    recall = tp / (tp + fn_total) if (tp + fn_total) else float("nan")
    specificity = tn / (tn + fp_total) if (tn + fp_total) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))
    accuracy = (tp + tn) / total_resolved if total_resolved else float("nan")

    return {
        "TP": tp,
        "FN (direct, result==0)": fn_direct,
        "FP (direct, result==2)": fp_direct,
        "Beides falsch [7] (+1 FN, +1 FP each)": both,
        "FN (total)": fn_total,
        "FP (total)": fp_total,
        "TN": tn,
        "Pending (needs manual review)": n_pending,
        "Flagged for re-check": int((df["flag"] == "yes").sum()),
        "Total images": len(df),
        "Total resolved (TP+FN+FP+TN)": total_resolved,
        "Precision": precision,
        "Recall / Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
        "Accuracy": accuracy,
    }, df


def print_report(report):
    print("=" * 55)
    print("Plaque detection evaluation")
    print("=" * 55)
    print(f"{'TP':30s} {report['TP']}")
    print(f"{'FN':30s} {report['FN (total)']} "
          f"(direct: {report['FN (direct, result==0)']}, "
          f"from [7]: {report['Beides falsch [7] (+1 FN, +1 FP each)']})")
    print(f"{'FP':30s} {report['FP (total)']} "
          f"(direct: {report['FP (direct, result==2)']}, "
          f"from [7]: {report['Beides falsch [7] (+1 FN, +1 FP each)']})")
    print(f"{'TN':30s} {report['TN']}")
    print("-" * 55)
    print(f"{'Total resolved (TP+FN+FP+TN)':30s} {report['Total resolved (TP+FN+FP+TN)']}")
    both = report['Beides falsch [7] (+1 FN, +1 FP each)']
    if both:
        print(f"  (exceeds total images by {both} - each [7] image counts as"
              f" both an FN and an FP)")
    print(f"{'Pending (manual review needed)':30s} {report['Pending (needs manual review)']}")
    print(f"{'Flagged for re-check':30s} {report['Flagged for re-check']}")
    print(f"{'Total images':30s} {report['Total images']}")
    print("-" * 55)
    for k in ("Precision", "Recall / Sensitivity", "Specificity", "F1", "Accuracy"):
        v = report[k]
        print(f"{k:30s} {v:.3f}" if v == v else f"{k:30s} n/a")
    print("=" * 55)


if __name__ == "__main__":
    report, _ = evaluate()
    print_report(report)
