#!/bin/bash
#SBATCH --job-name=beags_variant
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# Finetune variant on top of an existing prepared model (output/beags/<name>, from beags_scene.sh).
# The 30k model and prepare outputs are symlinked into output/beags_<variant>/<name>; nothing is overwritten.
# Usage (from repo root):
#   sbatch -J beags_<variant>_<x> slurm/beags_variant.sh <variant> <name> <sam2_masks_dir> <gt_masks_dir> [beags_finetune.py args]
set -euo pipefail
VARIANT=$1; NAME=$2; MASKS=$3; GT=$4; shift 4
module load gcc/12.2.0 cuda/12.2
PY=/leonardo_scratch/fast/IscrC_EditGS/envs/beags_fsplat/bin/python
BASE=$(realpath output/beags/$NAME)
OUT=output/beags_$VARIANT/$NAME
mkdir -p "$OUT/point_cloud" "$OUT/beags"
for f in cfg_args cameras.json input.ply point_cloud/iteration_30000 beags/masks beags/labeled_points.npz; do
  ln -sfn "$BASE/$f" "$OUT/$f"
done
set -x
$PY beags_finetune.py -m "$OUT" --iteration 30000 "$@"
$PY beags_eval.py -m "$OUT" --iteration 33000 --gt_masks "$GT" --masks "$MASKS" --render
$PY render.py -m "$OUT" --iteration 33000 --skip_train
$PY metrics.py -m "$OUT"
