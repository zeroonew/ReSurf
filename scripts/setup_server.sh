#!/usr/bin/env bash
# ConfSurf server-side setup (autodl / generic Ubuntu + conda).
# Usage: bash scripts/setup_server.sh
# Prereqs: conda installed, this repo synced to the server (e.g. via git/scp).
set -e

echo "=== [1/4] Creating conda environment 'confsurf' ==="
if conda env list | grep -q "^confsurf "; then
    echo "env 'confsurf' already exists, skipping creation."
else
    conda env create -f environment.yml
fi

echo "=== [2/4] Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate confsurf

echo "=== [3/4] Building CUDA submodules ==="
pip install -e submodules/simple-knn
pip install -e submodules/diff-plane-rasterization

echo "=== [4/4] Checking FoundationStereo checkpoint ==="
FS_CKPT="utils/FoundationStereo/pretrained_models/23-51-11/model_best_bp2-001.pth"
if [ -f "$FS_CKPT" ]; then
    echo "FoundationStereo checkpoint found: $FS_CKPT"
else
    echo "WARNING: FoundationStereo checkpoint NOT found at $FS_CKPT"
    echo "Download it from https://github.com/NVlabs/FoundationStereo (model_best_bp2-001.pth + cfg.yaml)"
    echo "and place both files under utils/FoundationStereo/pretrained_models/23-51-11/"
fi

echo ""
echo "Setup finished. Next steps:"
echo "  1) Put DTU data under data/DTU/ (see README 'Required Data')"
echo "  2) export FOUNDATION_STEREO_CKPT=\$(pwd)/$FS_CKPT"
echo "  3) Quick single-scene NVS test:"
echo "     bash scripts/run_dtu_nvs.sh 0 scan24 ./outputs/dtu_nvs/scan24"
echo "  4) Ablation study:"
echo "     bash scripts/run_ablation.sh 0 scan24 ./outputs/ablation full"
