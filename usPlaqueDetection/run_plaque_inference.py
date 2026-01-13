import argparse
import json
import os

import pandas as pd
from ultralytics import YOLO
import cv2
from tqdm import tqdm


cfg = json.load(open("config/params.json"))
params_preprocess = cfg["preprocess"]
params_inference = cfg["inference"]

def arg_parser():
    p = argparse.ArgumentParser(
        description="YOLOv8 plaque detection and plaque counter"
    )

    #  Inputs:
    p.add_argument("-m", "--manifest", required=True,
                   help="CSV with columns: participant_id, side, image_path")
    p.add_argument("-w", "--weights", default="weights/best_YOLO27Jan2024.pt")
    p.add_argument("-s", "--save_predictions", action="store_true")

    #  Outputs:
    p.add_argument("-oi", "--out-image-level", default="outputs/image_level.csv")
    p.add_argument("-oc", "--out-counts", default="outputs/plaque_counts.csv")

    #  Optional:
    p.add_argument("--device", default="cpu")

    return p.parse_args()


def preprocess(
        img,
        roi_x: int,
        roi_y: int,
        roi_w: int,
        roi_h: int,
        median_blur_ksize: int,
        clip_limit: float,
        tile_grid_size: tuple[int, int]
):
    # Preprocessing as defined in the paper and notebook
    h, w = img.shape
    if roi_x + roi_w > w or roi_y + roi_h > h:
        raise ValueError("ROI out of bounds")

    crop = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    blur = cv2.medianBlur(crop, median_blur_ksize)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(blur)

def main():
    args = arg_parser()

    os.makedirs(os.path.dirname(args.out_image_level) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_counts) or ".", exist_ok=True)

    df = pd.read_csv(args.manifest)

    model = YOLO(args.weights)
    rows = []

    pp  = (
        params_preprocess["roi_x"],
        params_preprocess["roi_y"],
        params_preprocess["roi_w"],
        params_preprocess["roi_h"],
        params_preprocess["median_blur_ksize"],
        params_preprocess["clip_limit"],
        tuple(params_preprocess["tile_grid_size"])
    )

    plaque_counts = [0] * (params_inference["max_det"] + 1)

    predict_kwargs = dict(
        conf=params_inference["confidence_threshold"],
        imgsz=params_inference["img_size"],
        max_det=params_inference["max_det"],
        iou=params_inference["iou"],
        device=args.device,
        verbose=False
    )

    if args.save_predictions:
        predict_kwargs["save_txt"]=True
        predict_kwargs["save_conf"]=True
        predict_kwargs["save"] = True

    errors = []
    done_paths = set()

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        image_path = str(r["image_path"])
        if done_paths and image_path in done_paths:
            continue

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(image_path)

            img_pp = preprocess(img, *pp)
            res = model.predict(img_pp, **predict_kwargs)[0]

            if res.boxes is None:
                n_boxes = 0
            else:
                n_boxes = len(res.boxes)

            plaque_counts[n_boxes] += 1

            rows.append({
                "participant_id": r["participant_id"],
                "side": r["side"],
                "image_path": image_path,
                "n_boxes": n_boxes
            })

            done_paths.add(image_path)

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

    # plaque counter output
    total = sum(plaque_counts)
    summary = []
    for k, n in enumerate(plaque_counts):
        summary.append({
            "n_detected_plaques": k,
            "n_images": n,
            "fraction": n / total if total > 0 else 0.0
        })

    pd.DataFrame(summary).to_csv(args.out_counts, index=False)

    print("Image-level output:", args.out_image_level)
    print(f"Summary (0..{params_inference['max_det']} plaques): {args.out_counts}")


if __name__ == "__main__":
    main()