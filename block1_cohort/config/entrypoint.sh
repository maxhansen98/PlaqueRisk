#!/usr/bin/env bash
set -euo pipefail

CODEDIR="$HOME/usPlaqueDetection"

echo "[worker] start"
mkdir -p /tmp/work
cd /tmp/work

#python3 -m pip install --no-cache-dir -r "$CODEDIR/config/requirements.txt"

mkdir -p zips

zip_name=$(dx describe "$sample_zip" --name)
zip_stem=${zip_name%.zip}

dx download "$sample_zip" -o "zips/$zip_name"

dx download "$weights" -o weights.pt
dx download "$params_json" -o params.json

mkdir -p "$CODEDIR/config"
cp -f params.json "$CODEDIR/config/params.json"

python3 "$CODEDIR/preprocess_ukb_files.py" \
  --zip-dir zips \
  --out manifest.csv

mkdir -p outputs
python3 "$CODEDIR/run_plaque_inference.py" \
  -m manifest.csv \
  -w weights.pt \
  -oi outputs/image_level.csv \
  -oc outputs/plaque_counts.csv

out_dir="$out_root/$zip_stem"

dx mkdir -p "$out_dir" || true

[ -f outputs/image_level.csv ] || { echo "Missing outputs/image_level.csv"; exit 2; }
[ -f outputs/plaque_counts.csv ] || { echo "Missing outputs/plaque_counts.csv"; exit 2; }

dx upload outputs/image_level.csv   --path "$out_dir/image_level.csv"
dx upload outputs/plaque_counts.csv --path "$out_dir/plaque_counts.csv"
dx upload manifest.csv             --path "$out_dir/manifest.csv"

if [ -f outputs/image_level_errors.csv ]; then
  dx upload outputs/image_level_errors.csv --path "$out_dir/image_level_errors.csv"
fi

echo "[worker] done"