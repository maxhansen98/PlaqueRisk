#!/usr/bin/env python3
"""Fine-tunes the base detection model on the dataset built by
prepare_yolo_dataset.py, warm-starting from the existing weights.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import argparse
from pathlib import Path

DATA_YAML = Path(str(paths.DATASET / "data.yaml"))
BASE_WEIGHTS = Path(str(paths.WEIGHTS / "best_YOLO27Jan2024.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps", help="mps | cpu | 0 (cuda)")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--weights", default=str(BASE_WEIGHTS))
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found - run prepare_yolo_dataset.py first.")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,        # modest LR: warm start, not from scratch
        lrf=0.01,
        warmup_epochs=3,
        box=7.5,
        cls=0.3,
        dfl=1.5,
        # augmentation tuned for B-mode ultrasound
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        erasing=0.0,
        plots=True,
        val=True,
    )

    print("\nTraining done. Best weights:")
    print(f"  runs/detect/{args.name}/weights/best.pt")
    print("Validate with:")
    print(f"  yolo detect val model=runs/detect/{args.name}/weights/best.pt data={DATA_YAML} imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
