#!/usr/bin/env python3
"""Fine-tunes a YOLO segmentation model on the dataset built by
prepare_yolo_seg_dataset.py, starting from a COCO-pretrained checkpoint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import argparse
from pathlib import Path

DATA_YAML = Path(str(paths.DATASET_SEG / "data.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps", help="mps | cpu | 0 (cuda)")
    ap.add_argument("--name", default="seg_finetune")
    ap.add_argument("--weights", default="yolov8s-seg.pt",
                    help="yolov8n/s/m/l-seg.pt - s fits ~400 images better "
                         "than m and trains ~35%% faster on MPS")
    # 50, not 30: the seg_v1 metric curve was noisy epoch-to-epoch and
    # still trending up at 107; a 30-epoch window would stop it early.
    ap.add_argument("--patience", type=int, default=50)
    # Ultralytics defaults to 0.5; seg_v1 and seg_v2 ran at 0.3.
    ap.add_argument("--cls", type=float, default=1.0,
                    help="classification loss gain - with one class this "
                         "IS the confidence score (ultralytics default 0.5)")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found - run prepare_yolo_seg_dataset.py first.")

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
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        box=7.5,
        cls=args.cls,
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
        copy_paste=0.3,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        erasing=0.0,
        plots=True,
        val=True,
    )

    out = paths.RUNS / "segment" / args.name / "weights" / "best.pt"
    print(f"\nTraining done. Best weights: {out}")
    print("Validate with:")
    print(f"  yolo segment val model={out} data={DATA_YAML} imgsz={args.imgsz}")
    # The val split selected these weights, so its mAP is a selection score,
    # not a performance estimate. test/ is the untouched set.
    print("Then evaluate on the held-out test set (and re-derive the "
          "confidence threshold for config/params_seg.json):")
    print(f"  python3 evaluate_seg_testset.py --weights {out} "
          f"--imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
