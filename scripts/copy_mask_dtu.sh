#!/usr/bin/env bash
# Copy DTU evaluation masks into the workspace mask/ folder with 5-digit names
# (00000.png, ...) as expected by metrics_dtu.py.
# Usage: bash scripts/copy_mask_dtu.sh <workspace> <scan_id>
base=$1
scan_id=$2

if [ ! -d "$base" ]; then
    echo "Workspace $base does not exist, skip mask copy."
    exit 0
fi

mkdir -p "$base/mask"

# Candidate mask sources in priority order.
candidates=(
    "data/DTU/submission_data/idrmasks/${scan_id}/mask"
    "data/DTU/submission_data/${scan_id}/mask"
)

src_dir=""
for c in "${candidates[@]}"; do
    if [ -d "$c" ]; then
        src_dir="$c"
        break
    fi
done

if [ -z "$src_dir" ]; then
    echo "No mask directory found for scan ${scan_id}, skip mask copy."
    exit 0
fi

echo "Copying masks from $src_dir -> $base/mask"
id=0
# Sort masks numerically so they align with the rendered test views.
for file in $(ls "$src_dir"/*.png 2>/dev/null | sort -V); do
    file_name=$(printf "%05d" "$id").png
    cp "$file" "$base/mask/$file_name"
    ((id = id + 1))
done
echo "Copied $id mask(s)."
