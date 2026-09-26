#!/usr/bin/env python3
"""Renders the echogenicity figure from two annotated images of near-equal aspect
ratio, so that the comparison varies echogenicity rather than lesion shape.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import cv2
import numpy as np
from pathlib import Path

import figlabel as fl  # from common/, on sys.path via paths

GREEN = (0, 255, 0)
PANEL_H = 430
SCALE = 0.98
RING = 25

PAIRS = [(str(paths.TEST / "539.png"),                   str(paths.TEST / "539_labeled.png"),  "hypoechoic"),
         (str(paths.DATASET_SEG / "images" / "val" / "155.png"), str(paths.TRAIN / "155_labeled.png"), "hyperechoic")]


def stats(img, mask):
    """Mean intensity inside the lesion minus that of a ring around it."""
    k = np.ones((2 * RING + 1, 2 * RING + 1), np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    inst = (lab == i).astype(np.uint8)
    ring = cv2.dilate(inst, k) - inst
    ring[mask > 0] = 0
    ar = st[i, cv2.CC_STAT_WIDTH] / max(st[i, cv2.CC_STAT_HEIGHT], 1)
    return img[inst > 0].mean() - img[ring > 0].mean(), ar


def render(img_path, mask_path, title):
    """Crop, outline and resize; returns the panel and its two caption lines."""
    g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m.shape != g.shape:
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)
    contrast, ar = stats(g, m)

    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, GREEN, 4)
    xs = np.where(g.max(0) > 12)[0]
    vis = vis[:, xs.min():xs.max() + 1]
    print(f"{Path(img_path).stem.strip():>4}  {title:<12} contrast {contrast:+6.1f}  "
          f"aspect ratio {ar:.2f}")
    return vis, title, f"contrast {contrast:+.0f} grey levels"


if __name__ == "__main__":
    panels = [render(*p) for p in PAIRS]
    # Fit the font to the narrowest panel so no caption is clipped.
    widths = [int(round(v.shape[1] * PANEL_H / v.shape[0])) for v, _, _ in panels]
    scale = min(SCALE, fl.fit_scale([t for _, t, _ in panels], min(widths)),
                fl.fit_scale([s for _, _, s in panels], min(widths)) / 0.78)
    out = str(paths.FIGURES / "echogenicity.png")
    cv2.imwrite(out, fl.row([fl.label(v, PANEL_H, t, s, scale=scale)
                             for v, t, s in panels]))
    print(f"{out} written (font scale {scale:.2f})")
