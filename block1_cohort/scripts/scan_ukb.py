from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Set, List
from datetime import date
from tqdm import tqdm
import pandas as pd


def is_numeric_folder(name: str) -> bool:
    return name.isdigit()


def extract_id_from_filename(filename: str) -> Optional[str]:
    first = Path(filename).name.split("_", 1)[0]
    return first if first.isdigit() and len(first) >= 6 else None


def compute_path(rel_path: Path) -> Optional[str]:
    parts = rel_path.parts
    for i, part in enumerate(parts):
        if is_numeric_folder(part):
            return "/".join(parts[:i]) if i > 0 else None
    return None


# ai generated
def anonymize_ids(ids: pd.Index, salt: str) -> pd.Index:
    def anon_one(x: str) -> str:
        h = hashlib.sha256((salt + str(x)).encode("utf-8")).hexdigest()[:16]
        return f"P_{h}"

    return pd.Index([anon_one(x) for x in ids], name="anon_id")


def build_id_path_presence_df(
    bulk_root: str | Path,
    require_id_in_filename: bool = True,
    allowed_ext: Optional[Set[str]] = None,
    tqdm_total: Optional[int] = 4_192_137,
) -> pd.DataFrame:
    bulk_root = Path(bulk_root).resolve()
    if not bulk_root.exists():
        raise FileNotFoundError(f"Not found: {bulk_root}")

    rows: List[Tuple[str, str]] = []

    if allowed_ext is None:
        allowed_ext_norm: Optional[Set[str]] = None
    else:
        allowed_ext_norm = {e.lower() for e in allowed_ext}

    pbar = tqdm(total=tqdm_total, desc="Scanning .zip/.xml", unit="files")

    try:
        for root, dirnames, filenames in os.walk(bulk_root):
            rel_root = Path(root).relative_to(bulk_root)

            # Skip/prune the entire "Previous WGS releases" subtree
            if "Previous WGS releases" in rel_root.parts:
                dirnames[:] = []
                continue

            # Prune chr* directories anywhere (case-insensitive) and also the "Previous WGS releases" dir
            dirnames[:] = [
                d for d in dirnames
                if not d.lower().startswith("chr") and d != "Previous WGS releases"
            ]

            for fn in filenames:
                fpath = Path(root) / fn

                if allowed_ext_norm is not None:
                    ext = ("".join(fpath.suffixes) if len(fpath.suffixes) > 1 else fpath.suffix).lower()
                    if ext not in allowed_ext_norm:
                        continue

                # Only count relevant files in the progress bar
                pbar.update(1)

                rel = fpath.relative_to(bulk_root)

                path_key = compute_path(rel)
                if not path_key:
                    continue

                if require_id_in_filename:
                    pid = extract_id_from_filename(fn)
                    if not pid:
                        continue
                else:
                    pid = ""

                rows.append((pid, path_key))

    finally:
        pbar.close()

    if not rows:
        return pd.DataFrame()

    long = pd.DataFrame(rows, columns=["id", "path"]).drop_duplicates()
    out = pd.crosstab(long["id"], long["path"]).astype(bool)
    out.index.name = "id"
    return out


if __name__ == "__main__":
    BULK_ROOT = "/mnt/project/Bulk"
    ALLOWED_EXT = {".zip", ".xml"}

    # SET SALT IN TERMINAL
    SALT = os.environ["UKB_ANON_SALT"]

    df_presence = build_id_path_presence_df(
        BULK_ROOT,
        allowed_ext=ALLOWED_EXT,
        require_id_in_filename=True,
        tqdm_total=4_192_137,
    )

    # Debugging
    if df_presence.empty:
        print("Keine passenden Files gefunden (prüfe BULK_ROOT, allowed_ext, ID-Pattern).")
        raise SystemExit(0)

    df_presence = df_presence.copy()
    df_presence.insert(0, "anon_id", anonymize_ids(df_presence.index, SALT))

    # Delete original IDs:
    df_presence = df_presence.set_index("anon_id", drop=True)

    # Just in case: Wenn wir die IDs doch brauchen
    # df_presence.drop(columns=[], inplace=True)

    out_dir = Path("/home/dnanexus/ukb_bulk")
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().strftime("%Y%m%d")
    out_file = out_dir / f"{today}_ukb_bulk_scan.csv"

    df_presence.to_csv(out_file, index=True)
    print(f"Wrote: {out_file}")