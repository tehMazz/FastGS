#!/bin/bash
#SBATCH --job-name=beags_test_rast
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# Usage (from repo root): sbatch slurm/test_rasterizer.sh
set -euo pipefail
module load gcc/12.2.0 cuda/12.2
ENVS=/leonardo_scratch/fast/IscrC_EditGS/envs
OUT=logs/rast_dumps_${SLURM_JOB_ID}
mkdir -p "$OUT"
$ENVS/fastgs++/bin/python tests/dump_render.py "$OUT/stock.pt"      # stock FastGS build
$ENVS/beags_fsplat/bin/python tests/dump_render.py "$OUT/patched.pt"  # BEA-GS build
STOCK_DUMP="$OUT/stock.pt" PATCHED_DUMP="$OUT/patched.pt" \
  $ENVS/beags_fsplat/bin/python -m pytest -q tests/test_rasterizer_labels.py
