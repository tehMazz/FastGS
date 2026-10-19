# Shared settings for the BakerH VM scripts (setup.sh, run_scene.sh, run_all.sh). Sourced, not run.
# Every value is an env knob. Paths follow the ArtiFixer repo's scripts/bakerh on the same VM.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=${GPU:-1}
SCENES=${SCENES:-"TA_BLUE_MOTOR PCV TA_TURBINE Compressor G35 COGEN1 COGEN2"}
DW=${DW:-0.1}                 # BEA-GS finetune --dilation_weight
LABEL=${LABEL:-1}             # the SAM3 masks hold one object: label 1
FBK=${FBK:-/workspace/amazzucchelli/fbk-3dworld}
MODELS=${MODELS:-$FBK/models}                               # LaMa, SAM2, OpenCLIP, SD2-inpainting, Marigold
AF360_REPO=${AF360_REPO:-$FBK/AuraFusion360_official}       # its utils/LeftRefill (sdedit imports it)
SAM_MASKS=${SAM_MASKS:-$FBK/FlashSplat/sam3_single_object_anchor_tracking}   # <scene>/{full,0_400}/masks_npy
DATA=${DATA:-$ROOT/datasets/BakerH_undis}                   # <scene>/{images_1600,sparse} (train_bakerh.sh)
IMAGES=${IMAGES:-images_1600}
OUTPUT=${OUTPUT:-$ROOT/output}
LOG_ROOT=$OUTPUT/bakerh_logs
CKPT_DIR=$OUTPUT/aurafusion/checkpoints                     # links into MODELS, the layout run.sh expects
PY=${PY:-$ROOT/.venv/bin/python}                            # FastGS / BEA-GS, plus graphdeco's rasterizer (setup.sh)

model_dir() { echo "$OUTPUT/beags_dw$DW/bakerh/$1"; }      # BEA-GS model of scene $1
old_model() { echo "$OUTPUT/${1,,}_bakerh"; }               # FastGS model from train_bakerh.sh, reused if it has 30k
mask_src() {  # SAM3 masks of scene $1: full capture, else the first 400 frames
  local d
  for d in "$SAM_MASKS/$1/full/masks_npy" "$SAM_MASKS/$1/0_400/masks_npy"; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  return 1
}
find_af_py() {  # the AuraFusion360 2D venv (SAM2, Marigold, LeftRefill) that ArtiFixer's setup_baselines.sh built
  local p
  if [ -n "${AF_PY:-}" ]; then echo "$AF_PY"; return 0; fi
  for p in "$FBK/ArtiFixer/.venv_af2d" "$FBK/artifixer/.venv_af2d" /workspace/artifixer/.venv_af2d \
           "$ROOT/../ArtiFixer/.venv_af2d" "$HOME/ArtiFixer/.venv_af2d" "$FBK"/*/.venv_af2d; do
    [ -x "$p/bin/python" ] && { echo "$p/bin/python"; return 0; }
  done
  return 1
}
activate() {  # repo venv + CUDA (scripts/activate_env.sh), quietly
  set +u
  # shellcheck disable=SC1091
  source "$ROOT/scripts/activate_env.sh" > /dev/null || { set -u; return 1; }
  set -u
  export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
