#!/usr/bin/env python3
"""Plots training progress from a run's results.csv, which the training framework
updates continuously, so that a run in progress can be inspected.

Usage: python3 plot_training_progress.py [run_name]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import paths
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "seg_v1"
CSV = paths.RUNS / "segment" / RUN / "results.csv"
OUT = paths.RUNS / "segment" / RUN / "progress.png"

# validated palette values
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=SECOND, length=0, labelsize=9)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main():
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found")
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    # 1) mask mAP - the headline metric for segmentation
    ax = axes[0]
    ax.plot(df["epoch"], df["metrics/mAP50(M)"], color=BLUE, linewidth=2, label="mAP50 (Mask)")
    ax.plot(df["epoch"], df["metrics/mAP50-95(M)"], color=AQUA, linewidth=2, label="mAP50-95 (Mask)")
    best_i = df["metrics/mAP50(M)"].idxmax()
    ax.scatter([df.loc[best_i, "epoch"]], [df.loc[best_i, "metrics/mAP50(M)"]],
               color=BLUE, s=45, zorder=5)
    ax.annotate(f"best {df.loc[best_i, 'metrics/mAP50(M)']:.3f}\n(Ep. {int(df.loc[best_i, 'epoch'])})",
                (df.loc[best_i, "epoch"], df.loc[best_i, "metrics/mAP50(M)"]),
                textcoords="offset points", xytext=(6, -22), fontsize=9, color=INK)
    ax.set_title("Mask mAP", color=INK, fontsize=12)
    ax.set_xlabel("Epoche", color=SECOND, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECOND)
    style(ax)

    # 2) precision / recall
    ax = axes[1]
    ax.plot(df["epoch"], df["metrics/precision(M)"], color=BLUE, linewidth=2, label="Precision")
    ax.plot(df["epoch"], df["metrics/recall(M)"], color=ORANGE, linewidth=2, label="Recall")
    ax.set_title("Precision / Recall (Mask)", color=INK, fontsize=12)
    ax.set_xlabel("Epoche", color=SECOND, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECOND)
    style(ax)

    # 3) losses - train vs val, to spot overfitting
    ax = axes[2]
    ax.plot(df["epoch"], df[str(paths.TRAIN / "seg_loss")], color=BLUE, linewidth=2, label="train seg_loss")
    if "val/seg_loss" in df:
        ax.plot(df["epoch"], df["val/seg_loss"], color=ORANGE, linewidth=2, label="val seg_loss")
    ax.set_title("Segmentation Loss", color=INK, fontsize=12)
    ax.set_xlabel("Epoche", color=SECOND, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECOND)
    style(ax)

    fig.suptitle(f"Training progress - {RUN}  (Epoche {int(df['epoch'].max())})",
                 color=INK, fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"saved {OUT}  (epochs: {len(df)}, best mAP50(M): {df['metrics/mAP50(M)'].max():.3f})")


if __name__ == "__main__":
    main()
