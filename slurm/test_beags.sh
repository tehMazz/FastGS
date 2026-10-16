#!/bin/bash
#SBATCH --job-name=beags_test
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# Usage (from repo root): sbatch slurm/test_beags.sh
set -euo pipefail
module load gcc/12.2.0 cuda/12.2
/leonardo_scratch/fast/IscrC_EditGS/envs/beags_fsplat/bin/python -m pytest -q tests/test_beags_utils.py tests/test_rasterizer_labels.py
