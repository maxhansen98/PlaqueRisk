#!/usr/bin/env python3
"""Aggregates the per-side main-axis selection into per-participant image counts.
"""
from pathlib import Path
import re
import pandas as pd

BASE = Path("/Users/Maxi/Desktop/PlaqueRisk_data")
LEFT_FILE = BASE / "left_pxlsum.txt"
RIGHT_FILE = BASE / "right_pxlsum.txt"

# Example token: ".../10/1000547_20222_2_0.zip"
PID_RE = re.compile(r"/(\d+)_\d+_\d+_\d+\.zip\b|\\(\d+)_\d+_\d+_\d+\.zip\b")


def extract_pid(line: str):
    m = PID_RE.search(line)
    if not m:
        return None
    # one of the groups will be set depending on / vs \
    return m.group(1) or m.group(2)


def count_pids(path: Path) -> pd.Series:
    counts = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pid = extract_pid(line)
            if pid is None:
                continue
            counts[pid] = counts.get(pid, 0) + 1
    return pd.Series(counts, dtype="int64")


left_counts = count_pids(LEFT_FILE).rename("Img left")
right_counts = count_pids(RIGHT_FILE).rename("Img right")

df = pd.concat([left_counts, right_counts], axis=1).fillna(0).astype(int)
df.insert(0, "PID", df.index)
df = df.reset_index(drop=True)

# Replace 0 with False as requested
df["Img left"] = df["Img left"].where(df["Img left"] != 0, False)
df["Img right"] = df["Img right"].where(df["Img right"] != 0, False)

# Optional: sort by PID numerically if possible
df["PID_num"] = pd.to_numeric(df["PID"], errors="coerce")
df = df.sort_values(["PID_num", "PID"]).drop(columns=["PID_num"]).reset_index(drop=True)

print(df.head(20))

# Save
out_csv = BASE / "main_axis_count.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")