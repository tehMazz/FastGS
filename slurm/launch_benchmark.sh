#!/usr/bin/env bash
# Benchmark scope: all Mip-NeRF 360 scenes + one scene each of 3D-OVS, LERF, LLFF.
# Usage (from repo root): bash slurm/launch_benchmark.sh
set -euo pipefail
G=/leonardo_work/IscrC_EditGS/gabri/datasets
MIP=/leonardo_work/IscrC_EditGS/datasets/mip360
MM=$G/mipnerf360_hd_masks/masks
declare -A MIP_ARGS=(
  [bicycle]="--grad_abs_thresh 0.0012"
  [garden]="--highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0008"
  [stump]="--dense 0.004 --grad_abs_thresh 0.0015"
  [room]="--highfeature_lr 0.02 --grad_abs_thresh 0.0008"
  [counter]="--highfeature_lr 0.02 --grad_abs_thresh 0.0008"
  [kitchen]="--highfeature_lr 0.02 --grad_abs_thresh 0.0006"
  [bonsai]="--highfeature_lr 0.02 --grad_abs_thresh 0.0006"
)
for s in "${!MIP_ARGS[@]}"; do
  # shellcheck disable=SC2086
  sbatch -J "beags_mip_$s" slurm/beags_scene.sh "mip360/$s" "$MIP/$s" "$MM/training/sam2/$s/masks" "$MM/gt_masks/$s/masks" -i images_4 ${MIP_ARGS[$s]}
done
sbatch -J beags_bed slurm/beags_scene.sh 3DOVS/bed data/3DOVS/bed "$G/3DOVS/masks/training/sam2/bed/masks" "$G/3DOVS/masks/gt_masks/bed/masks"
sbatch -J beags_figurines slurm/beags_scene.sh lerf/figurines data/lerf/figurines "$G/lerf/masks/training/sam2/figurines/masks" "$G/lerf/masks/gt_masks/figurines/masks"
sbatch -J beags_fern slurm/beags_scene.sh llff/fern data/llff/fern "$G/nerf_llff_data_original/masks/training/sam2/fern/masks" "$G/nerf_llff_data_original/masks/gt_masks/fern/masks"
