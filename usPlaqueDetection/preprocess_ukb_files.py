import csv
import zipfile
from pathlib import Path
import regex as re

EXTRACT_ROOT = Path("extracted")
EXTRACT_ROOT.mkdir(exist_ok=True)

def unzip_and_rename(zip_path: Path) -> Path:
    prefix = zip_path.stem
    out_dir = EXTRACT_ROOT / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)

    for f in out_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".dcm":
            parts = f.name.split(".")
            num = parts[-2]
            new_name = f"{prefix}_{num}.dcm"
            f.rename(out_dir / new_name)

    return out_dir


def is_long_axis(path: Path) -> bool:
    # NOTE: adapt this to what the notebook uses / your UKB export naming.
    # If you have a metadata/manifest file, use that instead.
    name = path.name.lower()
    return ("long" in name) or ("long_axis" in name) or ("lax" in name)


def infer_side(path: Path) -> str:
    # Best-effort: adapt to your naming. Otherwise put empty or "unknown".
    name = path.name.lower()
    if "left" in name or "_l_" in name or "lhs" in name:
        return "left"
    if "right" in name or "_r_" in name or "rhs" in name:
        return "right"
    return "unknown"


def infer_participant_id(zip_stem: str) -> str:
    m = re.match(r"^(\d+)_", zip_stem)
    return m.group(1) if m else zip_stem


def main(zip_dir: str, out_manifest: str = "manifest.csv"):
    zip_dir = Path(zip_dir)
    zips = sorted(zip_dir.glob("*.zip"))

    rows = []
    for zp in zips:
        out_dir = unzip_and_rename(zp)
        pid = infer_participant_id(zp.stem)

        # choose image files (DCM or PNG depending on your export)
        for img in sorted(out_dir.glob("*.dcm")):
            if not is_long_axis(img):
                continue
            side = infer_side(img)

            rows.append({
                "participant_id": pid,
                "side": side,
                "image_path": str(img.resolve()),
            })

    with open(out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "side", "image_path"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_manifest}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--zip-dir", required=True)
    p.add_argument("--out", default="manifest.csv")
    args = p.parse_args()
    main(args.zip_dir, args.out)