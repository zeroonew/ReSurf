<div align="center">

# ReSurf: Floater-Guided Sparse-View Surface Reconstruction

Built on top of [SparseSurf](https://arxiv.org/abs/2511.14633) (AAAI 2026).

</div>

ReSurf extends SparseSurf with a **Floater pipeline** — a 6-stage outlier suppression mechanism that detects and removes floating (spatially inconsistent) Gaussians while resampling new points onto the stereo-derived surface. The goal is to improve surface reconstruction quality in sparse-view (3-view) settings.

## Floater Pipeline (6 Stages)

| Stage | Name | Description |
|-------|------|-------------|
| **S1** | Spatial Constraint | Restrict Gaussians within scene bounding box |
| **S2** | Detection | Adaptive relative-error depth outlier detection: per-pixel threshold τ(p) = max(ε_abs, D_s(p)·θ), θ = max(η_min, median(r) + k·1.4826·MAD(r)) where r = \|D_r - D_s\|/D_s |
| **S3** | Soft Opacity Decay | Gradually reduce opacity of detected outlier Gaussians |
| **S4** | Hard Pruning | Remove Gaussians flagged as front-conflict (in front of stereo surface) or near-transparent, ranked by exceedance ratio; capped at `max_prune_ratio` per round |
| **S5A** | Stereo Resampling | Sample new points from high-confidence stereo pixels (confidence > 0.7), back-project to 3D, color from GT |
| **S5B** | 2D IDW Inpainting | Fill pruned holes by IDW-interpolating depth from healthy neighbors (rel_err < 2%) |
| **S6** | Color Down-weighting | Down-weight color loss for outlier regions |

**Outlier attribution (S2→Gaussian)**: Each Gaussian's own camera-space depth z_i is compared against the stereo depth at its projected pixel D_s(p_i). A Gaussian is flagged only when it lies clearly **in front** of the stereo surface (f_i = D_s - z_i > τ), the geometric signature of a floater. Gaussians behind the surface are not flagged (may be occluded structure).

## ConfSurf Innovations (A1–A4, currently disabled in experiments)

The codebase also contains confidence-driven extensions inherited from the original ConfSurf design. These are **disabled by default** in the Floater experiments:

| Module | Idea | Switch |
|--------|------|--------|
| **A1** Continuous confidence | Soft confidence-weighted stereo losses instead of binary mask | `--use_confidence_weighting {0,1}` |
| **A2** Adaptive multi-baseline | Cross-baseline depth fusion with uncertainty | `--adaptive_baseline {0,1}` |
| **A3** Confidence-aware schedule | Quality-EMA-gated stereo prior scheduling | `--confidence_aware_schedule {0,1}` |
| **B** Normal-guided propagation | Propagate depth along surface tangent into occluded regions | `--enable_depth_propagation {0,1}` |

## Key Floater Hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable_floater` | 1 | Master switch for the Floater pipeline |
| `--enable_hard_prune` | 1 | Enable Stage 4 hard pruning |
| `--prune_depth_diff_thresh` | 0.3 | Absolute floor ε_abs (m) for pixel threshold |
| `--prune_rel_floor` | 0.01 | Minimum relative threshold η_min (1%) |
| `--prune_mad_k` | 2.5 | k in θ = median + k·1.4826·MAD |
| `--max_prune_ratio` | 0.1 | Max fraction of Gaussians pruned per round |
| `--prune_interval` | 500 | Pruning/resampling interval (iterations) |
| `--resample_enabled` | 1 | Enable Stage 5A stereo resampling |
| `--resample_conf_thresh` | 0.7 | Stereo confidence threshold for resampling |
| `--inpaint_enabled` | 1 | Enable Stage 5B 2D inpainting |

## Installation

Tested with Python `3.8`, PyTorch `2.4.1`, and CUDA `11.8`.

```bash
git clone --recursive https://github.com/zeroonew/ReSurf.git
cd ReSurf
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate confsurf
```

Build the CUDA submodules (diff-gaussian-rasterization, simple-knn):

```bash
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Required Data

### 1. DTU dataset (sparse-view 3-view)

Place under `data/DTU/`:

```text
data/
├── DTU/
│   ├── submission_data_little/      # 3-view sparse DTU for training
│   │   ├── scan24/
│   │   │   ├── images/              # 3 training images
│   │   │   ├── sparse/0/            # COLMAP (cameras.txt, images.txt, points3D.txt)
│   │   │   ├── mask/                # evaluation masks
│   │   │   └── dense/               # stereo / fused.ply
│   │   ├── scan40/
│   │   └── ...
│   └── submission_data/             # (optional) full set
├── SampleSet/
│   └── MVS Data/                    # DTU ground-truth meshes for Chamfer eval
└── DTU_EVAL/                        # (optional) eval output cache
```

The 3-view sparse DTU data can be obtained from [FatesGS](https://github.com/yulunwu0108/FatesGS) or [DNGaussian](https://github.com/Fictionarry/DNGaussian).

### 2. FoundationStereo checkpoint

Download the FoundationStereo pretrained model and place at:

```text
utils/FoundationStereo/pretrained_models/23-51-11/model_best_bp2-001.pth
```

This path is git-ignored; set `FOUNDATION_STEREO_CKPT` if you place it elsewhere:

```bash
export FOUNDATION_STEREO_CKPT=/path/to/model_best_bp2-001.pth
```

## Running

### Quick start — Floater-only experiment

```bash
bash scripts/run_compare_confsurf_vs_sparsesurf.sh [gpu_id] [scan|all]
```

This runs the full Floater pipeline (A1–A4 disabled) on each scan:
train (7000 iters) → extract mesh → DTU Chamfer evaluation.

Example:
```bash
bash scripts/run_compare_confsurf_vs_sparsesurf.sh 0 scan24
```

### Manual training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/DTU/submission_data_little/scan24 \
  -m outputs/floater/scan24 \
  -r 2 \
  --n_views 3 \
  --iterations 7000 \
  --enable_floater 1 \
  --use_confidence_weighting 0 \
  --adaptive_baseline 0 \
  --confidence_aware_schedule 0 \
  --enable_depth_propagation 0 \
  --foundation_stereo_ckpt utils/FoundationStereo/pretrained_models/23-51-11/model_best_bp2-001.pth
```

### Mesh extraction

```bash
CUDA_VISIBLE_DEVICES=0 python extract_mesh.py \
  -s data/DTU/submission_data_little/scan24 \
  -m outputs/floater/scan24 \
  -r 2 \
  --skip_test
```

### DTU Chamfer evaluation

```bash
python scripts/eval_dtu/run_dtu_eval.py \
  --scene_dir data/DTU/submission_data_little/scan24 \
  --mask_dir data/DTU/submission_data_little/scan24/mask \
  --DTU "data/SampleSet/MVS Data" \
  --ply_path outputs/floater/scan24/point_cloud.ply
```

### Rendering

```bash
CUDA_VISIBLE_DEVICES=0 python render.py \
  -m outputs/floater/scan24 \
  --skip_train
```

## Project Structure

```text
ReSurf/
├── train.py                  # Main training loop (Floater 6 stages)
├── extract_mesh.py           # Mesh extraction from trained Gaussians
├── render.py                 # Novel view rendering
├── arguments/__init__.py     # All CLI arguments
├── utils/
│   ├── floater_utils.py      # Floater pipeline: detection, pruning, resampling
│   ├── point_utils.py        # 2D IDW inpainting (Stage 5B)
│   ├── sh_utils.py           # Spherical harmonics utilities
│   └── FoundationStereo/     # Stereo matching (checkpoint git-ignored)
├── scene/                    # Gaussian model, dataset, camera
├── gaussian_renderer/        # Diff rasterization wrapper
├── submodules/               # CUDA extensions (diff-plane-rasterization, simple-knn)
├── scripts/                  # Experiment scripts and evaluation
└── metrics_dtu.py            # DTU Chamfer distance metrics
```

## Citation

If you find this work useful, please cite SparseSurf:

```bibtex
@inproceedings{gu2026sparsesurf,
  title={SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction},
  author={Gu, Meiying and Zhang, Jiawei and Li, Jiahe and Yu, Xiaohan and Luo, Haonan and Zheng, Jin and Bai, Xiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={6},
  pages={4311--4319},
  year={2026}
}
```

## Acknowledgement

We thank the authors of [SparseSurf](https://arxiv.org/abs/2511.14633), [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), [PGSR](https://github.com/zju3dv/PGSR), [FoundationStereo](https://github.com/NVlabs/FoundationStereo/), [FatesGS](https://github.com/yulunwu0108/FatesGS), [DNGaussian](https://github.com/Fictionarry/DNGaussian), and [CoR-GS](https://github.com/jiaw-z/CoR-GS) for releasing their code and data resources.
