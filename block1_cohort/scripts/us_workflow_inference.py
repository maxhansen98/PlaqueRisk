#!/usr/bin/env python3
"""Cohort-scale inference workflow: extracts archives, selects the main-axis
frame, preprocesses and runs the detection model.
"""
import argparse
import json
import os
import re
import zipfile
import tempfile
import time
import numpy as np

from pathlib import Path
from multiprocessing import get_context
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

import cv2
from ultralytics import YOLO

try:
    import pydicom
except Exception:
    pydicom = None

ID_RE = re.compile(r"^(\d+)_")

def _extract_zip_with_retries(zip_path: Path, out_dir: Path, retries: int = 3, base_sleep: float = 1.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            return
        except (OSError, IOError, zipfile.BadZipFile) as e:
            last_exc = e
            # BadZipFile is usually permanent; don't spin forever
            if isinstance(e, zipfile.BadZipFile):
                break
            time.sleep(base_sleep * attempt)
    raise last_exc

# ---------- Utilities ----------
def participant_id_from_zipname(zip_name: str) -> str:
    m = ID_RE.match(zip_name)
    return m.group(1) if m else Path(zip_name).stem

def side_from_path(p: Path) -> str:
    s = str(p).lower()
    if "carotid artery (left)" in s or "left" in s:
        return "left"
    if "carotid artery (right)" in s or "right" in s:
        return "right"
    return "unknown"

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def is_dicom(p: Path) -> bool:
    return p.suffix.lower() == ".dcm"

def to_uint8(arr):
    import numpy as np
    arr = arr.astype("float32")
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax <= vmin:
        return (arr * 0).astype("uint8")
    arr = (arr - vmin) / (vmax - vmin)
    return (arr * 255.0).clip(0, 255).astype("uint8")

def load_grayscale(path: Path):
    """
    Returns a 2D uint8 numpy array.
    Supports common image formats via cv2, and DICOM via pydicom (if available).
    """
    if is_dicom(path):
        if pydicom is None:
            raise RuntimeError("pydicom not available but DICOM encountered")
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
        if getattr(arr, "ndim", 0) == 3 and arr.shape[-1] == 3:
            gray = arr[..., 0]

        elif getattr(arr, "ndim", 0) == 3:
            gray = arr[0]
        else:
            gray = arr

        if gray.dtype != np.uint8:
            gray = to_uint8(gray)
        else:
            gray = np.ascontiguousarray(gray)

        return gray



    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img

# ---------- Preprocess (copied from your run_plaque_inference.py) ----------
def preprocess(
    img,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    median_blur_ksize: int,
    clip_limit: float,
    tile_grid_size: Tuple[int, int]
):

    crop = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    if crop.size == 0:
        raise ValueError("Empty ROI crop (notebook-style slicing)")

    blur = cv2.medianBlur(crop, median_blur_ksize)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(blur)

# ---------- Worker state for multiprocessing ----------
_MODEL = None
_PP = None
_PREDICT_KWARGS = None
_MAX_DET = None

def _init_worker(weights: str, params_json: str, device: str, save_predictions: bool):
    global _MODEL, _PP, _PREDICT_KWARGS, _MAX_DET

    cfg = json.load(open(params_json))
    pp_cfg = cfg["preprocess"]
    inf_cfg = cfg["inference"]

    _PP = (
        pp_cfg["roi_x"],
        pp_cfg["roi_y"],
        pp_cfg["roi_w"],
        pp_cfg["roi_h"],
        pp_cfg["median_blur_ksize"],
        pp_cfg["clip_limit"],
        tuple(pp_cfg["tile_grid_size"]),
    )
    _MAX_DET = inf_cfg["max_det"]

    _PREDICT_KWARGS = dict(
        conf=inf_cfg["confidence_threshold"],
        imgsz=inf_cfg["img_size"],
        max_det=inf_cfg["max_det"],
        iou=inf_cfg["iou"],
        device=device,
        verbose=False
    )
    if save_predictions:
        _PREDICT_KWARGS.update(save_txt=True, save_conf=True, save=True)

    _MODEL = YOLO(weights)

def process_one_zip(zip_path: str) -> Tuple[List[Dict], List[Dict], List[int]]:
    """
    Returns (rows, errors, plaque_counts)
    plaque_counts has length (_MAX_DET + 1): counts of images with k boxes.
    """
    global _MODEL, _PP, _PREDICT_KWARGS, _MAX_DET

    z = Path(zip_path)
    pid = participant_id_from_zipname(z.name)
    side = side_from_path(z)

    rows = []
    errors = []
    plaque_counts = [0] * (_MAX_DET + 1)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        try:
            _extract_zip_with_retries(z, td, retries=3, base_sleep=1.0)
        except Exception as e:
            errors.append({
                "participant_id": pid,
                "side": side,
                "source_zip": z.name,
                "image_path_in_zip": "",
                "error": f"ZIP_OPEN_OR_EXTRACT_FAILED: {repr(e)}"
            })
            return [], errors, plaque_counts

        # collect candidates
        files = [p for p in td.rglob("*") if p.is_file() and (is_dicom(p) or is_image_file(p))]

        done_paths = set()
        for p in files:
            image_path = str(p)
            if image_path in done_paths:
                continue
            try:
                img = load_grayscale(p)
                img_pp = preprocess(img, *_PP)

                if getattr(img_pp, "ndim", 0) == 2:
                    img_pp = cv2.cvtColor(img_pp, cv2.COLOR_GRAY2BGR)

                if img_pp.dtype != np.uint8:
                    img_pp = img_pp.astype(np.uint8, copy=False)
                img_pp = np.ascontiguousarray(img_pp)

                if not (img_pp.ndim == 3 and img_pp.shape[2] == 3):
                    raise RuntimeError(f"BAD_INPUT_SHAPE_FOR_YOLO: {img_pp.shape}, dtype={img_pp.dtype}")

                res = _MODEL.predict(img_pp, **_PREDICT_KWARGS)[0]

                n_boxes = 0 if (res.boxes is None) else len(res.boxes)
                if n_boxes > _MAX_DET:
                    n_boxes = _MAX_DET  # clamp (shouldn't happen due to max_det)

                plaque_counts[n_boxes] += 1

                rows.append({
                    "participant_id": pid,
                    "side": side,
                    "source_zip": z.name,
                    "image_path_in_zip": image_path,
                    "n_boxes": n_boxes
                })
                done_paths.add(image_path)
            except Exception as e:
                errors.append({
                    "participant_id": pid,
                    "side": side,
                    "source_zip": z.name,
                    "image_path_in_zip": image_path,
                    "error": repr(e)
                })

    return rows, errors, plaque_counts

def main():
    ap = argparse.ArgumentParser("Unzip -> Predict -> Save (single script)")
    ap.add_argument("--left-root", default="/mnt/project/Bulk/Carotid Ultrasound/Carotid artery (left)")
    ap.add_argument("--right-root", default="/mnt/project/Bulk/Carotid Ultrasound/Carotid artery (right)")
    ap.add_argument("--weights", default="/mnt/project/PlaqueRisk/usPlaqueDetection/weights/best_YOLO27Jan2024.pt")
    ap.add_argument("--params-json", default="/mnt/project/PlaqueRisk/usPlaqueDetection/config/params.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--save-predictions", action="store_true")

    ap.add_argument("--out-image-level", default="/tmp/image_level.csv")
    ap.add_argument("--out-counts", default="/tmp/plaque_counts.csv")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_image_level) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_counts) or ".", exist_ok=True)

    left_zips = list(Path(args.left_root).rglob("*.zip"))
    right_zips = list(Path(args.right_root).rglob("*.zip"))
    zip_files = [str(p) for p in (left_zips + right_zips)]

    if not zip_files:
        raise RuntimeError("No .zip files found under left/right roots")

    # Initialize worker (also for single-worker mode)
    cfg = json.load(open(args.params_json))
    max_det = cfg["inference"]["max_det"]

    all_rows = []
    all_errors = []
    total_counts = [0] * (max_det + 1)

    FLUSH_EVERY = 500
    processed = 0

    out_img = args.out_image_level
    out_err = os.path.splitext(args.out_image_level)[0] + "_errors.csv"

    # If files already exist (e.g., resume), avoid writing headers twice
    img_header_written = os.path.exists(out_img)
    err_header_written = os.path.exists(out_err)

    def _flush_buffers():
        """Append buffered rows/errors to disk and clear buffers."""
        nonlocal img_header_written, err_header_written
        if all_rows:
            pd.DataFrame(all_rows).to_csv(
                out_img,
                mode="a",
                header=not img_header_written,
                index=False,
            )
            img_header_written = True
            all_rows.clear()
        if all_errors:
            pd.DataFrame(all_errors).to_csv(
                out_err,
                mode="a",
                header=not err_header_written,
                index=False,
            )
            err_header_written = True
            all_errors.clear()


    if args.workers <= 1:
        _init_worker(args.weights, args.params_json, args.device, args.save_predictions)
        for zp in tqdm(zip_files, desc="ZIPs"):
            rows, errs, counts = process_one_zip(zp)
            all_rows.extend(rows)
            all_errors.extend(errs)
            for k in range(len(total_counts)):
                total_counts[k] += counts[k]

            processed += 1
            if processed % FLUSH_EVERY == 0:
                _flush_buffers()
                print(f"[flush] wrote results after {processed} ZIPs")
    else:
        # Use spawn for better compatibility with ML libs
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.weights, args.params_json, args.device, args.save_predictions),
        ) as pool:
            for rows, errs, counts in tqdm(pool.imap_unordered(process_one_zip, zip_files, chunksize=20),
                                           total=len(zip_files), desc="ZIPs"):
                all_rows.extend(rows)
                all_errors.extend(errs)
                for k in range(len(total_counts)):
                    total_counts[k] += counts[k]

                processed += 1
                if processed % FLUSH_EVERY == 0:
                    _flush_buffers()
                    print(f"[flush] wrote results after {processed} ZIPs")

    _flush_buffers()

    if os.path.exists(out_err):
        print("Errors written to:", out_err)

    # summary output
    total = sum(total_counts)
    summary = [{
        "n_detected_plaques": k,
        "n_images": n,
        "fraction": (n / total) if total > 0 else 0.0
    } for k, n in enumerate(total_counts)]
    pd.DataFrame(summary).to_csv(args.out_counts, index=False)

    print("Image-level output:", out_img)
    print("Summary output:", args.out_counts)

if __name__ == "__main__":
    main()