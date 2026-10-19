#!/bin/bash
#SBATCH --job-name=beags_af
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=logs/%x_%j.out
# AuraFusion360 removal + inpainting (submodules/aurafusion_3dgs) of one BEA-GS object on a finished BEA-GS scene.
#   beags:   object Gaussians from labels.pt (FlashSplat skipped)
#   vanilla: AuraFusion's own setup, FlashSplat on the SAM2 masks of the same object
# Usage (from repo root):
#   sbatch -J af_<mode>_<scene> slurm/beags_aurafusion.sh <name> <label> <beags|vanilla> [sam2_masks_dir]
#   <name> as in beags_scene.sh (output/beags/<name>). ITER (default 33000) and STEPS (a+b+c) override.
# Output: output/aurafusion/<scene>/<mode>/lama_ref/final/{point_cloud.ply,rgb/}
set -euo pipefail
NAME=$1; LABEL=$2; MODE=$3; MASKS=${4:-}
module load gcc/12.2.0 cuda/12.2
PY=/leonardo_scratch/fast/IscrC_EditGS/envs/beags_fsplat/bin/python
MODEL=$(pwd)/output/beags/$NAME
export OUT=$(pwd)/output/aurafusion/$(basename "$NAME")/$MODE
set -x
$PY beags_aurafusion.py -m "$MODEL" --iteration "${ITER:-33000}" --label "$LABEL" --mode "$MODE" --out "$OUT" ${MASKS:+--masks "$MASKS"}
source "$OUT/env.sh"
[[ $MODE == beags ]] && export STEPS=${STEPS:-render,contour,sam2,agdd,init,sdedit,finetune}

# AuraFusion envs and checkpoints (as in submodules/aurafusion_3dgs/scripts/leonardo/run.sbatch)
export GS_PY=/leonardo_work/IscrC_EditGS/envs_inpaint/aurafusion_3dgs/bin/python
export AF_PY=/leonardo_scratch/fast/IscrC_EditGS/envs/aurafusion/bin/python
export AF360_REPO=/leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official
export CKPT_DIR=/leonardo_work/IscrC_EditGS/repos/CVPR2027/AuraFusion-3DGS/checkpoints
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp/job_${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor TRITON_CACHE_DIR=$TMPDIR/triton
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd submodules/aurafusion_3dgs   # keeps this repo's utils/ and scene/ off sys.path (LeftRefill imports utils.*)
bash scripts/run.sh
