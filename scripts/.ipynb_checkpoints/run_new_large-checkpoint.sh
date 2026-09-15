#!/usr/bin/env bash
# =============================================================================
# Floater-only DTU experiment.
#
# ConfSurf innovations A1-A4 are disabled; only the 6-stage Floater pipeline
# from SparseSurf is active.
#
# For each scan: train (7000 iters) -> extract mesh -> DTU Chamfer eval.
# Iterates over every scan* directory found under the submission_data path.
# Finally prints a summary of the Floater-only Chamfer metrics.
#
# Usage:
#   bash scripts/run_compare_confsurf_vs_sparsesurf.sh [gpu_id] [scan|all]
#
# Default: gpu_id=0, scan=all (auto-discover all scan* directories)
# =============================================================================
set -e

gpu_id=${1:-0}
scan_arg=${2:-all}
iters=7000

# ---- paths ----------------------------------------------------------------
CONFSURF_ROOT="/root/autodl-tmp/ConfSurf"
DATA_ROOT="${CONFSURF_ROOT}/data/DTU/submission_data"
DTU_GT="${CONFSURF_ROOT}/data/SampleSet/MVS Data"

OUT_ROOT="${CONFSURF_ROOT}/outputs/large"

# ---- environment ----------------------------------------------------------
source activate confsurf

if [ -z "${FOUNDATION_STEREO_CKPT:-}" ]; then
    export FOUNDATION_STEREO_CKPT="${CONFSURF_ROOT}/utils/FoundationStereo/pretrained_models/23-51-11/model_best_bp2-001.pth"
fi
if [ ! -f "$FOUNDATION_STEREO_CKPT" ]; then
    echo "ERROR: FoundationStereo checkpoint not found at $FOUNDATION_STEREO_CKPT"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$gpu_id

# ---- discover scans -------------------------------------------------------
# Collect every valid scan<number> directory under DATA_ROOT.
# The glob scan[0-9]* naturally excludes stray entries like the literal "scan*".
if [ "$scan_arg" = "all" ]; then
    scans=()
    for d in "$DATA_ROOT"/scan[0-9]*; do
        [ -d "$d" ] && scans+=("$(basename "$d")")
    done
    if [ ${#scans[@]} -eq 0 ]; then
        echo "ERROR: no scan directories found under $DATA_ROOT"
        exit 1
    fi
    # sort numerically by scan id for deterministic ordering
    IFS=$'\n' scans=($(printf '%s\n' "${scans[@]}" | sort -V)); unset IFS
else
    scans=("$scan_arg")
fi

echo "============================================================"
echo " Floater-only DTU run (A1-A4 OFF, Floater ON)"
echo " iters=$iters  gpu=$gpu_id  n_scans=${#scans[@]}"
echo " scans: ${scans[*]}"
echo "============================================================"

mkdir -p "$OUT_ROOT"
SUMMARY_FILE="$OUT_ROOT/summary_$(date +%Y%m%d_%H%M%S).txt"

# Helper: read the Chamfer "overall" value from a results.json (empty if missing)
read_cd() {
    local json_file="$1"
    if [ -f "$json_file" ]; then
        python -c "import json,sys; print(json.load(open('$json_file')).get('overall',''))" 2>/dev/null
    fi
}

for scan in "${scans[@]}"; do
    scan_id="${scan#scan}"
    DATA_PATH="${DATA_ROOT}/${scan}"
    CONFSURF_OUT="${OUT_ROOT}/floater_only/${scan}"

    if [ ! -d "$DATA_PATH" ]; then
        echo "WARNING: $DATA_PATH not found, skipping $scan"
        continue
    fi

    echo ""
    echo "############################################################"
    echo " Processing $scan (scan_id=$scan_id)"
    echo "############################################################"

    # ===========================================================================
    # PHASE 1: Floater-only (A1-A4 confidence OFF, 6-stage Floater pipeline ON)
    # ===========================================================================
    echo ""
    echo "############ Floater-only (A1-A4 OFF, Floater ON) [$scan] ############"
    cd "$CONFSURF_ROOT"
    export PYTHONPATH="$CONFSURF_ROOT:${PYTHONPATH:-}"

    rm -rf "$CONFSURF_OUT"
    mkdir -p "$CONFSURF_OUT"

    echo "--- [Floater-only][$scan] training ($iters iters) ---"
    python train.py \
      --source_path "$DATA_PATH" \
      -m "$CONFSURF_OUT" \
      --eval -r 2 --n_views 3 \
      --iterations "$iters" \
      --total_virtual_num 240 \
      --foundation_stereo_ckpt "$FOUNDATION_STEREO_CKPT" \
      --use_confidence_weighting 0 \
      --adaptive_baseline 0 \
      --confidence_aware_schedule 0 \
      --enable_depth_propagation 0 \
      --enable_floater 1 \
      --detect_from_iter 500 \
      --soft_from_iter 500 \
      --hard_from_iter 500 \
      --prune_depth_diff_thresh 0.01 \
      --prune_rel_floor 0.01 \
      --prune_mad_k 2.5 \
      --max_prune_ratio 0.1

    echo "--- [Floater-only][$scan] extracting mesh ---"
    python extract_mesh.py \
      -s "$DATA_PATH" \
      -m "$CONFSURF_OUT" \
      -r 2 \
      --iteration "$iters" \
      --sdf_trunc_mul 4.0 \
      --skip_test

    echo "--- [Floater-only][$scan] evaluating (DTU Chamfer) ---"
    python scripts/eval_dtu/evaluate_single_scene.py \
      --input_mesh "$CONFSURF_OUT/mesh/tsdf_fusion_post.ply" \
      --scan_id "$scan_id" \
      --output_dir "$CONFSURF_OUT/result" \
      --scene_dir "$DATA_PATH" \
      --mask_dir "$DATA_PATH/mask" \
      --DTU "$DTU_GT"

done

# ===========================================================================
# SUMMARY: Floater-only Chamfer Distance (lower is better)
# ===========================================================================
echo ""
echo "============================================================"
echo " SUMMARY: Floater-only Chamfer Distance"
echo " (overall = (mean_d2s + mean_s2d)/2, lower is better)"
echo "============================================================"
{
    printf "%-10s %14s\n" "scan" "Floater-only"
    printf "%-10s %14s\n" "----" "--------"
    for scan in "${scans[@]}"; do
        CONFSURF_OUT="${OUT_ROOT}/floater_only/${scan}"
        confsurf_cd=$(read_cd "$CONFSURF_OUT/result/results.json")
        [ -z "$confsurf_cd" ] && confsurf_cd="N/A"
        printf "%-10s %14s\n" "$scan" "$confsurf_cd"
    done
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
