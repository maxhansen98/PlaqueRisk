#!/usr/bin/env python3
"""Five-fold cross-validation of the segmentation model over the development set.
Folds are cut over the training images only; the test set is not touched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from evaluate_seg_testset import predict_all, evaluate, SWEEP
from prepare_yolo_seg_dataset import mask_to_polygon_lines

INPUT_DIR = paths.TRAIN
RESULTS_CSV = "labeling_results.csv"
CV_DIR = Path("dataset_cv")
SEED = 42
N_FOLDS = 5
HEADLINE_CONF = 0.30


def make_folds(df):
    """Stratified fold assignment, one list of image names per fold."""
    rng = np.random.default_rng(SEED)
    folds = [[] for _ in range(N_FOLDS)]
    for _, group in df.groupby("has_label"):
        names = group["image"].tolist()
        rng.shuffle(names)
        # Round-robin after shuffling keeps fold sizes within one image of
        # each other for both strata, which even splitting by slice does not.
        for i, n in enumerate(names):
            folds[i % N_FOLDS].append(n)
    return folds


def build_fold(fold_idx, folds, label_cache):
    """Materialise dataset_cv/fold{k} as a YOLO dataset. Returns data.yaml."""
    root = CV_DIR / f"fold{fold_idx}"
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    val_names = set(folds[fold_idx])
    for name in label_cache:
        split = "val" if name in val_names else "train"
        src = (INPUT_DIR / name).resolve()
        (root / "images" / split / name).symlink_to(src)
        lines = label_cache[name]
        (root / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: Plaque\n"
    )
    return yaml_path, sorted(val_names)


def cache_labels(df):
    """Polygon lines per image, computed once and reused for every fold."""
    cache = {}
    for _, row in df.iterrows():
        name = row["image"]
        src = INPUT_DIR / name
        if not src.exists():
            continue
        if row["has_label"] == "yes":
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            cache[name] = mask_to_polygon_lines(
                INPUT_DIR / f"{Path(name).stem}_labeled.png", w, h)
        else:
            cache[name] = []
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--folds", default="", help="e.g. '0,1' to run a subset")
    ap.add_argument("--out", default="cv_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(RESULTS_CSV, dtype={"image": str})
    folds = make_folds(df)
    label_cache = cache_labels(df)
    print(f"{len(label_cache)} images, {N_FOLDS} folds "
          f"(sizes: {[len(f) for f in folds]})")

    wanted = ([int(x) for x in args.folds.split(",")] if args.folds
              else list(range(N_FOLDS)))

    from ultralytics import YOLO
    rows = []
    for k in wanted:
        yaml_path, val_names = build_fold(k, folds, label_cache)
        n_pos = sum(1 for n in val_names if label_cache[n])
        print(f"\n=== Fold {k}: {len(val_names)} val images "
              f"({n_pos} positive) ===", flush=True)

        model = YOLO(args.weights)
        model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, name=f"cv_fold{k}",
            patience=args.patience, pretrained=True, optimizer="AdamW",
            lr0=0.001, lrf=0.01, warmup_epochs=3,
            box=7.5, cls=0.3, dfl=1.5,
            fliplr=0.5, flipud=0.0, degrees=5.0, translate=0.1, scale=0.3,
            shear=0.0, perspective=0.0, mosaic=0.5, mixup=0.0,
            copy_paste=0.3, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, erasing=0.0,
            plots=False, val=True, exist_ok=True,
        )

        best = paths.RUNS / "segment" / f"cv_fold{k}" / "weights" / "best.pt"
        # Masks live next to the originals in train/, not in the fold dir -
        # the fold only holds symlinks and YOLO polygon labels.
        pairs = [(INPUT_DIR / n, INPUT_DIR / f"{Path(n).stem}_labeled.png")
                 for n in val_names]
        records = predict_all(YOLO(str(best)), pairs, args.device, args.imgsz)

        sweep = pd.DataFrame([evaluate(records, t)[0] for t in SWEEP])
        sweep.to_csv(f"cv_fold{k}_sweep.csv", index=False)

        fixed = sweep.loc[(sweep["conf"] - HEADLINE_CONF).abs().idxmin()]
        bestj = sweep.loc[sweep["youden_J"].idxmax()]
        rows.append({
            "fold": k, "n_val": len(val_names), "n_pos": n_pos,
            "sensitivity": fixed["sensitivity"],
            "specificity": fixed["specificity"],
            "youden_J": fixed["youden_J"],
            "dice_mean_pos": fixed["dice_mean_pos"],
            "dice_global": fixed["dice_global"],
            "bestJ_conf": bestj["conf"], "bestJ": bestj["youden_J"],
        })
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Fold {k} @conf {HEADLINE_CONF}: "
              f"sens={fixed['sensitivity']:.3f} spec={fixed['specificity']:.3f} "
              f"Dice={fixed['dice_mean_pos']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    print(f"\n=== {len(res)}-fold CV @ conf {HEADLINE_CONF} ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMittelwert ± Standardabweichung:")
    for c in ["sensitivity", "specificity", "youden_J",
              "dice_mean_pos", "dice_global"]:
        print(f"  {c:<16} {res[c].mean():.3f} ± {res[c].std():.3f}")
    print(f"\nGeschrieben: {args.out}, cv_fold*_sweep.csv")


if __name__ == "__main__":
    main()
