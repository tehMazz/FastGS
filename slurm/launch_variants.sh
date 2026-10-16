#!/usr/bin/env bash
# Ablations on the benchmark scenes (needs output/beags/<scene> from launch_benchmark.sh):
#   dw0.1 : SAG with dilated-boundary weight 0.1 (BEA-GS --dilation_weight)
#   occbg : occupancy loss also on background Gaussians
# Usage (from repo root): bash slurm/launch_variants.sh
set -euo pipefail
G=/leonardo_work/IscrC_EditGS/gabri/datasets
MM=$G/mipnerf360_hd_masks/masks
declare -A MASKS GT
for s in bicycle bonsai counter garden kitchen room stump; do
  MASKS[mip360/$s]=$MM/training/sam2/$s/masks; GT[mip360/$s]=$MM/gt_masks/$s/masks
done
MASKS[3DOVS/bed]=$G/3DOVS/masks/training/sam2/bed/masks;  GT[3DOVS/bed]=$G/3DOVS/masks/gt_masks/bed/masks
MASKS[lerf/figurines]=$G/lerf/masks/training/sam2/figurines/masks; GT[lerf/figurines]=$G/lerf/masks/gt_masks/figurines/masks
MASKS[llff/fern]=$G/nerf_llff_data_original/masks/training/sam2/fern/masks; GT[llff/fern]=$G/nerf_llff_data_original/masks/gt_masks/fern/masks
for scene in "${!MASKS[@]}"; do
  tag=${scene#*/}
  sbatch -J "beags_dw01_$tag"  slurm/beags_variant.sh dw0.1 "$scene" "${MASKS[$scene]}" "${GT[$scene]}" --dilation_weight 0.1
  sbatch -J "beags_occbg_$tag" slurm/beags_variant.sh occbg "$scene" "${MASKS[$scene]}" "${GT[$scene]}" --occ_background
done
