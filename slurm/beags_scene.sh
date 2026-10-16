#!/bin/bash
#SBATCH --job-name=beags_scene
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
# Full BEA-GS pipeline on one scene: FastGS 30k -> prepare -> finetune -> eval (+ photometric).
# Usage (from repo root):
#   sbatch -J beags_<name> slurm/beags_scene.sh <name> <scene_dir> <sam2_masks_dir> <gt_masks_dir> [extra train.py args]
set -euo pipefail
NAME=$1; SRC=$2; MASKS=$3; GT=$4; shift 4
module load gcc/12.2.0 cuda/12.2
PY=/leonardo_scratch/fast/IscrC_EditGS/envs/beags_fsplat/bin/python
OUT=output/beags/$NAME
set -x
[[ -f $OUT/point_cloud/iteration_30000/point_cloud.ply ]] || \
  $PY train.py -s "$SRC" -m "$OUT" --eval --densification_interval 500 --optimizer_type default \
      --test_iterations 30000 --save_iterations 30000 --checkpoint_iterations -1 "$@"
$PY beags_prepare.py -m "$OUT" --iteration 30000 --masks "$MASKS"
$PY beags_eval.py -m "$OUT" --iteration 30000 --gt_masks "$GT" --masks "$MASKS"         # labels before finetune
$PY beags_finetune.py -m "$OUT" --iteration 30000
$PY beags_eval.py -m "$OUT" --iteration 33000 --gt_masks "$GT" --masks "$MASKS" --render
for it in 30000 33000; do $PY render.py -m "$OUT" --iteration $it --skip_train; done
$PY metrics.py -m "$OUT"
