#!/usr/bin/env python3
"""Shared panel labelling for the multi-panel image figures. The font scale is
fitted to the narrowest panel in a row so that long captions shrink rather than
overflow.
"""
import cv2
import numpy as np

F = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (195, 195, 195)


def fit_scale(texts, width, thickness=2, hi=0.95, lo=0.40, margin=18):
    """Largest font scale at which every text fits into `width`."""
    s = hi
    while s > lo:
        if all(cv2.getTextSize(t, F, s, thickness)[0][0] <= width - margin
               for t in texts if t):
            return s
        s -= 0.01
    return lo


def label(img, height, title, sub="", scale=0.7, sub_scale=None):
    """Resize to `height` and draw a centred title (and optional subtitle)."""
    w = int(round(img.shape[1] * height / img.shape[0]))
    s = cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)
    if s.ndim == 2:
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    sub_scale = sub_scale or scale * 0.78
    bar = int(round(34 * scale / 0.7)) + (int(round(24 * scale / 0.7)) if sub else 0)
    cv2.rectangle(s, (0, 0), (w, bar), (0, 0, 0), -1)

    (tw, th), _ = cv2.getTextSize(title, F, scale, 2)
    y = int(bar * 0.42) + th // 2 if sub else (bar + th) // 2
    cv2.putText(s, title, ((w - tw) // 2, y), F, scale, WHITE, 2, cv2.LINE_AA)
    if sub:
        (sw, sh), _ = cv2.getTextSize(sub, F, sub_scale, 1)
        cv2.putText(s, sub, ((w - sw) // 2, int(bar * 0.85)), F, sub_scale,
                    GREY, 1, cv2.LINE_AA)
    return s


def row(panels, gap=7):
    """Horizontally concatenate equal-height panels with white gutters."""
    out = [panels[0]]
    h = panels[0].shape[0]
    for p in panels[1:]:
        out += [np.full((h, gap, 3), 255, np.uint8), p]
    return np.hstack(out)


def stack(rows, gap=7):
    w = max(r.shape[1] for r in rows)
    padded = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)])
              if r.shape[1] < w else r for r in rows]
    out = [padded[0]]
    for p in padded[1:]:
        out += [np.full((gap, w, 3), 255, np.uint8), p]
    return np.vstack(out)
