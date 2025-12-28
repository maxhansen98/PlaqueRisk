import argparse
import os

import pandas as pd
from ultralytics import YOLO
import cv2
from tqdm import tqdm

PATHS = {
    "weights": "best_YOLO27Jan2024.pt",
    "out_counts": ""
}

# =========================
# Inference parameters (source: paper)
# =========================
CONFIDENCE_THRESHOLD = 0.13
IMAGE_SIZE = 480
MAX_DETECTIONS = 3

# =========================
# Preprocessing parameters (source: image_ext_36k.ipynb)
# =========================
ROI_X = 296
ROI_Y = 110
ROI_W = 448
ROI_H = 480
MEDIAN_BLUR_KSIZE = 5
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)

def arg_parser():
    p = argparse.ArgumentParser(
        description="YOLOv8 plaque detection inference"
    )

    # Required
    p.add_argument("--manifest", required=True,
                   help="CSV with columns: participant_id, side, image_path")

    # Outputs
    p.add_argument("--out-image-level", default="outputs/image_level.csv")
    p.add_argument("--out-summary", default="outputs/image_count_summary.csv")

    # Optional overrides (defaults = paper / notebook)
    p.add_argument("--weights", default=PATHS["weights"])
    p.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    p.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    p.add_argument("--max-det", type=int, default=MAX_DETECTIONS)

    # Optional technical
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--device", default=None)

    return p.parse_args()


def preprocess(img):
    # Preprocessing as defined in the paper and notebook
    if img.ndim != 2:
        raise TypeError(f"Expected 2D grayscale image, got shape={img.shape}")
    h, w = img.shape
    if ROI_X + ROI_W > w or ROI_Y + ROI_H > h:
        raise ValueError("ROI out of bounds")

    crop = img[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    blur = cv2.medianBlur(crop, MEDIAN_BLUR_KSIZE)

    clahe = cv2.createCLAHE(
        clipLimit=CLIP_LIMIT,
        tileGridSize=TILE_GRID_SIZE
    )

    return clahe.apply(blur)

def main():
    args = arg_parser()

    os.makedirs(os.path.dirname(args.out_image_level) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    df = pd.read_csv(args.manifest)

    model = YOLO(args.weights)
    rows = []

    # counts for images with 0, 1, 2, or 3 detected plaques
    plaque_counts = [0, 0, 0, 0]

    predict_kwargs = dict(
        conf=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        verbose=False
    )

    if args.iou is not None:
        predict_kwargs["iou"] = args.iou
    if args.device is not None:
        predict_kwargs["device"] = args.device

    errors = []
    done_paths = set()

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        image_path = str(r["image_path"])
        if done_paths and image_path in done_paths:
            continue

        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(image_path)

            img_pp = preprocess(img)
            res = model.predict(img_pp, **predict_kwargs)[0]

            if res.boxes is None:
                n_boxes = 0
            else:
                n_boxes = len(res.boxes)

            n_boxes_capped = min(n_boxes, MAX_DETECTIONS)
            plaque_counts[n_boxes_capped] += 1

            rows.append({
                "participant_id": r["participant_id"],
                "side": r["side"],
                "image_path": image_path,
                "n_boxes": n_boxes
            })

        except Exception as e:
            errors.append({
                "participant_id": r.get("participant_id", ""),
                "side": r.get("side", ""),
                "image_path": image_path,
                "error": repr(e)
            })


    # image-level output
    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out_image_level, index=False)

    if errors:
        err_path = os.path.splitext(args.out_image_level)[0] + "_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print("Errors written to:", err_path)

    # plaque counter
    total = sum(plaque_counts)
    summary = []
    for k in (0, 1, 2, 3):
        n = plaque_counts[k]
        summary.append({
            "n_detected_plaques": k,
            "n_images": n,
            "fraction": n / total if total > 0 else 0.0
        })

    pd.DataFrame(summary).to_csv(args.out_summary, index=False)

    print("Image-level output:", args.out_image_level)
    print("Summary (0/1/2/3 plaques):", args.out_summary)


if __name__ == "__main__":
    main()