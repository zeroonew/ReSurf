#!/usr/bin/env bash
# =============================================================================
# Baseline DTU experiment (pure SparseSurf behaviour).
#
# All ConfSurf additions are disabled:
#   - A1 continuous confidence weighting  (--use_confidence_weighting 0)
#   - A2 adaptive multi-baseline fusion   (--adaptive_baseline 0)
#   - A3 confidence-aware schedule        (--confidence_aware_schedule 0)
#   - A4 normal-guided depth propagation  (--enable_depth_propagation 0)
#   - Floater 6-stage pipeline            (--enable_floater 0)
#
# This reproduces the original SparseSurf training: binary LR-check mask,
# single baseline, fixed stereo schedule, no floater suppression.
#
# For each scan: train (7000 iters) -> extract mesh -> DTU Chamfer eval.
# Iterates over every scan* directory found under the submission_data path.
# Finally prints a summary of the baseline Chamfer metrics.
#
# Usage:
#   bash scripts/run_baseline.sh [gpu_id] [scan|all]
#
# Default: gpu_id=0, scan=all (auto-discover all scan* directories)
# =============================================================================
set -e

gpu_id=${1:-0}
scan_arg=${2:-all}
iters=7000

# ---- paths ----------------------------------------------------------------
CONFSURF_ROOT="/root/autodl-tmp/ConfSurf"
DATA_ROOT="${CONFSURF_ROOT}/data/DTU/submission_data_little"
DTU_GT="${CONFSURF_ROOT}/data/SampleSet/MVS Data"

OUT_ROOT="${CONFSURF_ROOT}/outputs/baseline"

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
if [ "$scan_arg" = "all" ]; then
    scans=()
    for d in "$DATA_ROOT"/scan[0-9]*; do
        [ -d "$d" ] && scans+=("$(basename "$d")")
    done
    if [ ${#scans[@]} -eq 0 ]; then
        echo "ERROR: no scan directories found under $DATA_ROOT"
        exit 1
    fi
    IFS=$'\n' scans=($(printf '%s\n' "${scans[@]}" | sort -V)); unset IFS
else
    scans=("$scan_arg")
fi

echo "============================================================"
echo " Baseline DTU run (SparseSurf: A1-A4 OFF, Floater OFF)"
echo " iters=$iters  gpu=$gpu_id  n_scans=${#scans[@]}"
echo " scans: ${scans[*]}"
echo "============================================================"

mkdir -p "$OUT_ROOT"
SUMMARY_FILE="$OUT_ROOT/summary_$(date +%Y%m%d_%H%M%S).txt"

# Helper: read a metric from results.json
read_metric() {
    local json_file="$1"
    local key="$2"
    if [ -f "$json_file" ]; then
        python -c "import json,sys; print(json.load(open('$json_file')).get('$key',''))" 2>/dev/null
    fi
}

for scan in "${scans[@]}"; do
    scan_id="${scan#scan}"
    DATA_PATH="${DATA_ROOT}/${scan}"
    BASELINE_OUT="${OUT_ROOT}/${scan}"

    if [ ! -d "$DATA_PATH" ]; then
        echo "WARNING: $DATA_PATH not found, skipping $scan"
        continue
    fi

    echo ""
    echo "############################################################"
    echo " Baseline [$scan] (scan_id=$scan_id)"
    echo "############################################################"
    cd "$CONFSURF_ROOT"
    export PYTHONPATH="$CONFSURF_ROOT:${PYTHONPATH:-}"

    rm -rf "$BASELINE_OUT"
    mkdir -p "$BASELINE_OUT"

    # ---- training (pure SparseSurf: no confidence, no floater) ----
    echo "--- [Baseline][$scan] training ($iters iters) ---"
    python train.py \
      --source_path "$DATA_PATH" \
      -m "$BASELINE_OUT" \
      --eval -r 2 --n_views 3 \
      --iterations "$iters" \
      --total_virtual_num 240 \
      --foundation_stereo_ckpt "$FOUNDATION_STEREO_CKPT" \
      --use_confidence_weighting 0 \
      --adaptive_baseline 0 \
      --confidence_aware_schedule 0 \
      --enable_depth_propagation 0 \
      --enable_floater 0

    # ---- extract mesh ----
    echo "--- [Baseline][$scan] extracting mesh ---"
    python extract_mesh.py \
      -s "$DATA_PATH" \
      -m "$BASELINE_OUT" \
      -r 2 \
      --iteration "$iters" \
      --sdf_trunc_mul 4.0 \
      --skip_test

    # ---- evaluate (DTU Chamfer) ----
    echo "--- [Baseline][$scan] evaluating (DTU Chamfer) ---"
    python scripts/eval_dtu/evaluate_single_scene.py \
      --input_mesh "$BASELINE_OUT/mesh/tsdf_fusion_post.ply" \
      --scan_id "$scan_id" \
      --output_dir "$BASELINE_OUT/result" \
      --scene_dir "$DATA_PATH" \
      --mask_dir "$DATA_PATH/mask" \
      --DTU "$DTU_GT"
done

# ===========================================================================
# SUMMARY: Chamfer Distance (lower is better)
#   mean_d2s = Accuracy (surface->GT)
#   mean_s2d = Completeness (GT->surface)
#   overall  = (mean_d2s + mean_s2d) / 2
# ===========================================================================
echo ""
echo "============================================================"
echo " SUMMARY: Baseline Chamfer Distance (lower is better)"
echo "============================================================"
{
    printf "%-10s %12s %12s %12s\n" "scan" "mean_d2s" "mean_s2d" "overall"
    printf "%-10s %12s %12s %12s\n" "----" "--------" "--------" "-------"
    for scan in "${scans[@]}"; do
        BASELINE_OUT="${OUT_ROOT}/${scan}"
        d2s=$(read_metric "$BASELINE_OUT/result/results.json" "mean_d2s")
        s2d=$(read_metric "$BASELINE_OUT/result/results.json" "mean_s2d")
        overall=$(read_metric "$BASELINE_OUT/result/results.json" "overall")
        [ -z "$d2s" ] && d2s="N/A"
        [ -z "$s2d" ] && s2d="N/A"
        [ -z "$overall" ] && overall="N/A"
        printf "%-10s %12s %12s %12s\n" "$scan" "$d2s" "$s2d" "$overall"
    done
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
