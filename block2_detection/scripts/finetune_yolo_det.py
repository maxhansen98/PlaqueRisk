#!/usr/bin/env python3
"""Fine-tunes a YOLO detector on Ar-PlaqSegm1 under the same split and schedule as
the segmentation runs, with each annotation reduced to its enclosing box.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import argparse
from pathlib import Path

DATA_YAML = Path(str(paths.DATASET_DET / "data.yaml"))
SEG_YAML = Path(str(paths.DATASET_SEG / "data.yaml"))

# Held constant across both controls and seg_v2. Only `model` and `data`
# differ between the detection and segmentation runs.
COMMON = dict(
    epochs=300, imgsz=640, batch=16, device="mps", patience=50,
    pretrained=True, optimizer="AdamW", lr0=0.001, lrf=0.01,
    warmup_epochs=3, box=7.5, cls=0.3, dfl=1.5,
    fliplr=0.5, flipud=0.0, degrees=5.0, translate=0.1, scale=0.3,
    shear=0.0, perspective=0.0, mosaic=0.5, mixup=0.0, copy_paste=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, erasing=0.0, plots=True, val=True,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.task == "detect":
        weights, data = "yolov8s.pt", DATA_YAML
        name = args.name or "det_ctrl"
    else:
        weights, data = "yolov8s-seg.pt", SEG_YAML
        name = args.name or "seg_ctrl"

    if not data.exists():
        raise SystemExit(f"{data} not found.")

    from ultralytics import YOLO
    YOLO(weights).train(data=str(data), name=name, **COMMON)

    out = Path("runs") / ("detect" if args.task == "detect" else "segment") / name
    print(f"\nDone. Weights: {out/'weights'/'best.pt'}")


if __name__ == "__main__":
    main()
