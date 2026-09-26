#!/usr/bin/env python3
"""Runs batch inference for any images not yet covered by the detection cache and
opens the interactive labelling tool in its own window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib
# TkAgg first: it grabs real keyboard focus, which the native macOS
# backend does not do reliably outside an application bundle.
for _backend in ("TkAgg", "MacOSX", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

WEIGHTS_PATH = "best_YOLO27Jan2024.pt"
INPUT_DIR = "train"
DETECTIONS_CACHE_PATH = Path("yolo_detections_cache.json")
RESULTS_CSV = "labeling_results.csv"
OVERLAY_ALPHA = 0.4

LOGO_PATH = Path("tum_logo.png")
COPYRIGHT_TEXT = "© 2026 Max-Malte Hansen (03686413)   ·   PlaqueLabeling"

OVERLAY_CACHE_SIZE = 30  # bounds memory while still speeding up back/forth navigation

NUMBERED_PNG = re.compile(r"^\d{1,3}\.png$", re.IGNORECASE)


def find_xxx_pngs(folder):
    folder = Path(folder)
    return sorted(
        (p for p in folder.iterdir()
         if p.is_file() and NUMBERED_PNG.match(p.name)),
        key=lambda p: int(p.stem),
    )


def load_detections_cache(cache_path):
    if not cache_path.exists():
        return {}
    with open(cache_path) as f:
        return {c["image"]: c for c in json.load(f)}


def save_detections_cache(cache_path, by_name):
    with open(cache_path, "w") as f:
        json.dump(list(by_name.values()), f)


def yolo_txt_to_boxes(txt_path, img_w, img_h):
    """Converts a saved YOLO label .txt (normalized xc,yc,w,h[,conf]) back
    into absolute-pixel (x1,y1,x2,y2) boxes, mirroring what
    box.xyxy[0].tolist() would have produced at inference time."""
    boxes, confs, clss = [], [], []
    if not txt_path.exists():
        return boxes, confs, clss
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:5])
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        x1, y1 = (xc - w / 2) * img_w, (yc - h / 2) * img_h
        x2, y2 = (xc + w / 2) * img_w, (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss


def recover_from_previous_runs(missing_paths):
    """Tries to reconstruct detections for `missing_paths` from earlier
    runs/detect/predict*/ folders (most recent first), without re-running
    the model. Returns (recovered_by_name, still_missing_paths)."""
    predict_dirs = sorted(
        (d for d in Path(str(paths.RUNS / "detect")).glob("predict*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    remaining = {p.name: p for p in missing_paths}
    recovered = {}

    for pdir in predict_dirs:
        if not remaining:
            break
        labels_dir = pdir / "labels"
        for name, path in list(remaining.items()):
            stem = path.stem
            matches = [m for m in sorted(pdir.glob(f"{stem}.*")) if m.suffix.lower() != ".txt"]
            if not matches:
                continue  # this image wasn't part of this run
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            boxes, confs, clss = yolo_txt_to_boxes(labels_dir / f"{stem}.txt", w, h)
            recovered[name] = {"image": name, "boxes": boxes, "confs": confs, "clss": clss}
            del remaining[name]

    return recovered, list(remaining.values())


def run_inference(image_paths):
    from ultralytics import YOLO
    model = YOLO(WEIGHTS_PATH)
    results = model.predict(
        [str(p) for p in image_paths],
        max_det=3,
        iou=0.1,
        imgsz=480,
        conf=0.13,
        save_txt=True,
        save_conf=True,
        save=True,
    )
    return {
        p.name: {
            "image": p.name,
            "boxes": [box.xyxy[0].tolist() for box in r.boxes],
            "confs": [float(box.conf[0]) for box in r.boxes],
            "clss": [int(box.cls[0]) for box in r.boxes],
        }
        for p, r in zip(image_paths, results)
    }


def get_detections(image_paths):
    by_name = load_detections_cache(DETECTIONS_CACHE_PATH)
    missing = [p for p in image_paths if p.name not in by_name]

    if not missing:
        print("Cache already covers all images - nothing to catch up on.")
        return [by_name[p.name] for p in image_paths]

    print(f"{len(missing)} image(s) missing from cache - trying to recover from earlier runs...")
    recovered, still_missing = recover_from_previous_runs(missing)
    by_name.update(recovered)
    if recovered:
        print(f"Recovered {len(recovered)} image(s) from runs/detect/predict*/ without re-running inference.")
    if still_missing:
        print(f"Running fresh inference for {len(still_missing)} image(s) not found in any previous run...")
        by_name.update(run_inference(still_missing))

    save_detections_cache(DETECTIONS_CACHE_PATH, by_name)
    return [by_name[p.name] for p in image_paths]


def has_any_label(mask_path):
    if not Path(mask_path).exists():
        return "no"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return "no"
    return "yes" if (mask > 127).any() else "no"


def build_label_df(image_paths, input_dir, results_csv=RESULTS_CSV):
    if Path(results_csv).exists():
        df = pd.read_csv(results_csv, dtype={"image": str})
    else:
        df = pd.DataFrame(columns=["image", "has_label", "result", "flag"])

    existing = set(df["image"])
    new_rows = []
    for p in image_paths:
        if p.name in existing:
            continue
        labeled_path = Path(input_dir) / f"{p.stem}_labeled.png"
        new_rows.append({
            "image": p.name,
            "has_label": has_any_label(labeled_path),
            "result": pd.NA,
            "flag": "no",
        })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"Added {len(new_rows)} new image(s) to {results_csv}.")

    df.to_csv(results_csv, index=False)
    return df


def show_splash(fig, on_done, logo_path=LOGO_PATH):
    """Shows a start screen (TUM logo on white background + copyright +
    a Start button) in its own axes on `fig`. Calls on_done() once the user
    clicks the Start button."""
    fig.patch.set_facecolor("white")

    logo_ax = fig.add_axes((0.1, 0.35, 0.8, 0.55))
    logo_ax.set_facecolor("white")
    logo_ax.axis("off")
    if logo_path.exists():
        logo_ax.imshow(plt.imread(str(logo_path)))
    else:
        logo_ax.text(0.5, 0.5, "TUM", ha="center", va="center",
                      fontsize=48, color="#3070b3", weight="bold")

    copyright_text = fig.text(0.5, 0.22, COPYRIGHT_TEXT,
                               ha="center", va="center", fontsize=10, color="black")

    button_ax = fig.add_axes((0.4, 0.08, 0.2, 0.07))
    button = Button(button_ax, "Start", color="#3070b3", hovercolor="#254f80")
    button.label.set_color("white")
    button.label.set_fontsize(11)
    fig._splash_button = button  # keep a strong reference so the callback stays alive

    def on_click(_event):
        button.disconnect_events()  # drop the widget's hover/click callbacks
        # before removing its axes, or later mouse-move events crash on the
        # now-orphaned axes
        logo_ax.remove()
        copyright_text.remove()
        button_ax.remove()
        fig.canvas.draw_idle()
        on_done()

    button.on_clicked(on_click)
    fig.canvas.draw_idle()


class LabelingTool:
    def __init__(self, image_paths, input_dir, df, detections, fig, ax,
                 results_csv=RESULTS_CSV, alpha=OVERLAY_ALPHA):
        self.paths_by_name = {p.name: p for p in image_paths}
        self.detections_by_name = {p.name: d for p, d in zip(image_paths, detections)}
        self.input_dir = Path(input_dir)
        self.order = [p.name for p in image_paths]
        self.df = df.set_index("image")
        self.results_csv = results_csv
        self.alpha = alpha

        self.fig, self.ax = fig, ax
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._overlay_cache = OrderedDict()
        self._im_artist = None

        self._build_buttons()

        self.pos = self._first_unlabeled_index()
        self.render()

    def _build_buttons(self):
        """Clickable buttons mirroring the keyboard shortcuts. Mouse clicks
        have proven reliable in this environment (the splash Start button
        always worked) while key_press_event never reached the app despite
        the window being active, so buttons are the primary interaction."""
        self.ax.set_position((0.05, 0.24, 0.90, 0.68))

        specs = [
            ((0.05, 0.14, 0.27, 0.07), "↑ Korrekt [1]", lambda _e: self.set_result(1)),
            ((0.36, 0.14, 0.27, 0.07), "↓ Nicht det. [0]", lambda _e: self.set_result(0)),
            ((0.67, 0.14, 0.27, 0.07), "→ Falsch det. [2]", lambda _e: self.set_result(2)),
            ((0.05, 0.05, 0.27, 0.07), "# Beides falsch [7]", lambda _e: self.set_result(7)),
            ((0.36, 0.05, 0.27, 0.07), "← Zurück", lambda _e: self.go_back()),
            ((0.67, 0.05, 0.27, 0.07), "z Flag", lambda _e: self.toggle_flag()),
        ]
        self._buttons = []
        for rect, label, callback in specs:
            bax = self.fig.add_axes(rect)
            btn = Button(bax, label, color="#eef2f7", hovercolor="#c9d8ea")
            btn.label.set_fontsize(8)
            btn.on_clicked(callback)
            self._buttons.append(btn)  # keep a strong reference

    # --- Hilfsfunktionen -------------------------------------------------

    def _first_unlabeled_index(self):
        for i, name in enumerate(self.order):
            if pd.isna(self.df.loc[name, "result"]):
                return i
        return len(self.order) - 1  # alles bereits gelabelt

    def remaining_count(self):
        return int(self.df["result"].isna().sum())

    def current_name(self):
        return self.order[self.pos]

    def load_overlay(self, name):
        """Cached form of _build_overlay. The composite image does not change
        within a session, so revisiting an image costs nothing."""
        cached = self._overlay_cache.get(name)
        if cached is not None:
            self._overlay_cache.move_to_end(name)
            return cached
        overlay = self._build_overlay(name)
        self._overlay_cache[name] = overlay
        if len(self._overlay_cache) > OVERLAY_CACHE_SIZE:
            self._overlay_cache.popitem(last=False)
        return overlay

    def _build_overlay(self, name):
        """Composite of the image, the ground-truth mask in red and the
        predicted boxes in green. Boxes are drawn after the mask so that they
        remain visible on top of it."""
        base = cv2.imread(str(self.paths_by_name[name]))
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB).astype(np.float32)

        labeled_path = self.input_dir / f"{Path(name).stem}_labeled.png"
        if labeled_path.exists():
            mask = cv2.imread(str(labeled_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != base.shape[:2]:
                    mask = cv2.resize(mask, (base.shape[1], base.shape[0]))
                alpha_mask = (mask > 127).astype(np.float32) * self.alpha
                red = np.zeros_like(base)
                red[..., 0] = 255  # reiner Rotkanal
                for c in range(3):
                    base[..., c] = base[..., c] * (1 - alpha_mask) + red[..., c] * alpha_mask

        overlay = base.astype(np.uint8).copy()

        det = self.detections_by_name[name]
        for (x1, y1, x2, y2), conf in zip(det["boxes"], det["confs"]):
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(overlay, f"{conf:.2f}", (int(x1), max(int(y1) - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return overlay

    # --- Rendering ---------------------------------------------------------

    def render(self):
        name = self.current_name()
        img = self.load_overlay(name)

        if self._im_artist is None:
            self.ax.axis("off")
            self._im_artist = self.ax.imshow(img)
        else:
            self._im_artist.set_data(img)

        row = self.df.loc[name]
        result_str = "-" if pd.isna(row["result"]) else str(int(row["result"]))
        remaining = self.remaining_count()
        flag_val = row["flag"]

        title = (f"{name}   |   Ergebnis: {result_str}   |   Flag: {flag_val}\n"
                 f"Verbleibend: {remaining} von {len(self.order)}")
        self.ax.set_title(title, fontsize=10)

        self.fig.canvas.draw_idle()

    def save(self):
        self.df.reset_index().to_csv(self.results_csv, index=False)

    # --- Aktionen ------------------------------------------------------

    def set_result(self, value):
        name = self.current_name()
        self.df.loc[name, "result"] = value
        self.save()
        print(f"{name}: labeled {value}  ({self.remaining_count()} remaining)")
        self.advance()

    def advance(self):
        for i in range(self.pos + 1, len(self.order)):
            if pd.isna(self.df.loc[self.order[i], "result"]):
                self.pos = i
                self.render()
                return
        for i in range(len(self.order)):
            if pd.isna(self.df.loc[self.order[i], "result"]):
                self.pos = i
                self.render()
                return
        self.render()  # all images labelled

    def go_back(self):
        if self.pos > 0:
            self.pos -= 1
            self.render()

    def toggle_flag(self):
        name = self.current_name()
        current = self.df.loc[name, "flag"]
        self.df.loc[name, "flag"] = "no" if current == "yes" else "yes"
        self.save()
        self.render()

    def on_key(self, event):
        key = event.key
        if key == "up":
            self.set_result(1)
        elif key == "down":
            self.set_result(0)
        elif key == "right":
            self.set_result(2)
        elif key in ("#", "numbersign"):
            self.set_result(7)
        elif key == "left":
            self.go_back()
        elif key in ("z", "Z"):
            self.toggle_flag()


def force_focus(fig):
    """Best-effort: grab real OS-level keyboard focus for the figure window.
    Focusing the canvas widget alone isn't enough if the top-level window
    was never raised/activated at the OS level - it can look active and
    accept clicks while still not being the 'key window' that keyboard
    events get routed to. Lift + briefly force topmost + focus both the
    toplevel and the canvas widget to cover that case."""
    try:
        widget = fig.canvas.get_tk_widget()
        root = widget.winfo_toplevel()
        root.lift()
        root.attributes("-topmost", True)
        root.after(50, lambda: root.attributes("-topmost", False))
        root.focus_force()
        widget.focus_force()
    except Exception as e:
        print(f"force_focus failed: {e}")


def main():
    image_paths = find_xxx_pngs(INPUT_DIR)
    print(f"Found {len(image_paths)} image(s) in {INPUT_DIR}/")

    detections = get_detections(image_paths)
    label_df = build_label_df(image_paths, INPUT_DIR)

    n_done = int(label_df["result"].notna().sum())
    print(f"{n_done} of {len(label_df)} images already labeled.")

    print(f"Using matplotlib backend: {matplotlib.get_backend()}")

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_visible(False)
    try:
        fig.canvas.manager.set_window_title("YOLO Labeling Tool")
    except Exception:
        pass

    # The native macOS backend overwrites the dock icon on figure creation.
    if matplotlib.get_backend().lower() == "macosx":
        try:
            from matplotlib.backends import _macosx
            dock_icon = Path("tum_logo_square.png")
            if dock_icon.exists():
                _macosx.FigureManager.set_icon(str(dock_icon.resolve()))
                print(f"Dock icon set to {dock_icon.resolve()}")
        except Exception as e:
            print(f"Could not set dock icon: {e}")

    fig.canvas.mpl_connect("button_press_event", lambda _e: force_focus(fig))

    def schedule_focus_retries(delays_ms=(50, 300, 800, 1500)):
        """The window may not be mapped/raised yet on the very first call
        (e.g. right after launch, or right after the splash Start click),
        so retry a few times over the next ~1.5s rather than relying on a
        single call."""
        try:
            root = fig.canvas.get_tk_widget().winfo_toplevel()
            for ms in delays_ms:
                root.after(ms, lambda: force_focus(fig))
        except Exception:
            pass

    def start_tool():
        ax.set_visible(True)
        LabelingTool(image_paths, INPUT_DIR, label_df, detections, fig, ax)
        force_focus(fig)
        schedule_focus_retries()
        print("Labeling tool ready - use the buttons below the image to label "
              "(arrow keys / # / z also work if the window has keyboard focus).")

    show_splash(fig, on_done=start_tool)
    force_focus(fig)
    schedule_focus_retries()
    print("Start screen opened - waiting for the Start button to be pressed.")
    plt.show()


if __name__ == "__main__":
    main()
