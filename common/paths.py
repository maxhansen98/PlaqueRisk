#!/usr/bin/env python3
"""Repository-relative locations of the data and output directories.

Resolving paths from the location of this file rather than from the working
directory lets the scripts be run from anywhere.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCK1 = ROOT / "block1_cohort"
BLOCK2 = ROOT / "block2_detection"
THESIS = ROOT / "thesis"

# Ar-PlaqSegm1 source material and derived datasets
DATA = BLOCK2 / "data"
TRAIN = DATA / "train"
TEST = DATA / "test"
DATASET = DATA / "dataset"
DATASET_SEG = DATA / "dataset_seg"
DATASET_DET = DATA / "dataset_det"
WEIGHTS = DATA / "weights"
RUNS = DATA / "runs"

# Outputs
RESULTS = BLOCK2 / "results"
TABLES = RESULTS / "tables"
LOGS = RESULTS / "logs"
RESULT_FIGURES = RESULTS / "figures"
ASSETS = BLOCK2 / "assets"
BLOCK1_CONFIG = BLOCK1 / "config"
FIGURES = THESIS / "figures"


def seg_weights(run: str, which: str = "best") -> Path:
    """Checkpoint of a segmentation run, e.g. seg_weights("seg_v1")."""
    return RUNS / "segment" / run / "weights" / f"{which}.pt"
