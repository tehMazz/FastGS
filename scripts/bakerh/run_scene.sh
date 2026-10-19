#!/usr/bin/env bash
# One BakerH scene on the VM, one GPU, one log per phase: BEA-GS (dilation weight 0.1) on FastGS, vanilla
# AuraFusion360 on the BEA-GS model, and FlashSplat (vanilla) vs BEA-GS object extraction side by side.
#
# Usage, from any directory:  bash scripts/bakerh/run_scene.sh SCENE
#   SCENE  TA_BLUE_MOTOR | PCV | TA_TURBINE | Compressor | G35 | COGEN1 | COGEN2
# Phases, in order (PHASES=a,b runs a subset; a finished phase is skipped in seconds, FORCE=1 redoes it):
#   check          dataset, SAM3 masks, venvs, models present
#   masks          SAM3 id maps -> BEA-GS (2, H, W) masks                       <model>/beags/input_masks
#   train          FastGS 30k with train_bakerh.sh's settings; output/<scene>_bakerh is reused (linked)
#                  when it has iteration 30000 (RETRAIN=1 trains a fresh one)
#   prepare        beags_prepare.py: consensus masks, per-Gaussian labels
#   finetune       beags_finetune.py --dilation_weight $DW (0.1) -> iteration 33000
#   metrics        render.py + metrics.py: PSNR/SSIM/LPIPS of the test views at 33000
#   afinputs       label 1 of the SAM3 masks as AuraFusion's 2D masks
#   af_flashsplat  FlashSplat object Gaussians from those masks (vanilla AuraFusion removal)
#   compare        beags_extraction_compare.py: FlashSplat vs BEA-GS object and removal, per view
#   af_render .. af_finetune   the other vanilla AuraFusion360 stages (submodules/aurafusion_3dgs/README.md);
#                  the reference view is the one with the largest unseen mask (REF_INDEX overrides). When no
#                  view has an unseen region there is nothing to inpaint: the stages after af_sam2 are
#                  skipped and the removal (vanilla/base/removed.ply) is the result.
# Outputs: model output/beags_dw0.1/bakerh/<scene>; AuraFusion output/aurafusion/<scene>/vanilla
# (final: lama_ref/final/{point_cloud.ply,rgb/}); comparison output/aurafusion/<scene>/extraction/{panels,summary.json}.
# Logs: $BAKERH_RUN_DIR/<scene>/ (run_all.sh) or output/bakerh_logs/<stamp>_<scene>/: 00_config.log,
# NN_<phase>.log, summary.log (start, end, duration, peak GPU memory of every phase), job_summary.txt
# (what each phase found: mask counts, Gaussians per label, reference view, comparison numbers), FAILED.
# Knobs (common.sh): GPU (1), DW (0.1), LABEL (1), PANEL_EVERY (10), REF_INDEX, RETRAIN, FORCE, PHASES, ...
set -euo pipefail
S=${1:?usage: run_scene.sh SCENE   (TA_BLUE_MOTOR|PCV|TA_TURBINE|Compressor|G35|COGEN1|COGEN2)}
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
activate || { echo "scripts/activate_env.sh failed"; exit 1; }
AF_PY=$(find_af_py || echo "<no .venv_af2d: set AF_PY>")
M=$(model_dir "$S")
MASKS=$M/beags/input_masks
MASK_SRC=$(mask_src "$S" || echo "<no masks_npy under $SAM_MASKS/$S>")
AFO=$OUTPUT/aurafusion/$S/vanilla
CMP=$OUTPUT/aurafusion/$S/extraction
FT_ITER=33000
PHASES=${PHASES:-check,masks,train,prepare,finetune,metrics,afinputs,af_flashsplat,compare,af_render,af_contour,af_sam2,af_agdd,af_init,af_sdedit,af_finetune}

STAMP=$(date +%Y%m%d_%H%M%S)
LOGS=${BAKERH_RUN_DIR:+$BAKERH_RUN_DIR/$S}
LOGS=${LOGS:-$LOG_ROOT/${STAMP}_$S}
mkdir -p "$LOGS" "$M"
SUMMARY=$LOGS/summary.log
JS=$LOGS/job_summary.txt
N=0
note() { echo "[$(date '+%F %T')] $S $*" | tee -a "$SUMMARY"; }
js() { echo "$*" >> "$JS"; echo "[job summary] $*"; }
hms() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
done_or_force() { [ -z "${FORCE:-}" ] && [ -e "$1" ]; }   # done_or_force PATH: skip the phase?
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# run_phase NAME: runs phase_NAME with its output in NN_NAME.log (and on screen), samples GPU memory every
# 5 s, records OK/FAILED + duration + peak memory in summary.log, stops the scene on failure.
run_phase() {
  local name=$1 log start status peak sampler
  N=$((N + 1))
  log=$LOGS/$(printf %02d "$N")_$name.log
  note "START $name   log: $log"
  start=$(date +%s)
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 5000 > "$log.gpumem" 2>/dev/null &
  sampler=$!
  set +e
  ( set -euo pipefail; "phase_$name" ) 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  peak=$(sort -n "$log.gpumem" 2>/dev/null | tail -1)
  if [ "$status" -eq 0 ]; then
    note "END   $name   OK   $(hms $(($(date +%s) - start)))   peak GPU mem ${peak:-?} MiB"
    return
  fi
  note "END   $name   FAILED (exit $status)   $(hms $(($(date +%s) - start)))   peak GPU mem ${peak:-?} MiB"
  echo "phase=$name log=$log" > "$LOGS/FAILED"
  if grep -q -i -E 'out of memory|OutOfMemoryError' "$log"; then
    note "      cause: CUDA out of memory (peak ${peak:-?} MiB on GPU $GPU)"
  fi
  echo "---- last 30 lines of $log"
  tail -30 "$log"
  exit "$status"
}

phase_check() {
  local missing=() f
  for f in "$PY" "$DATA/$S/$IMAGES" "$DATA/$S/sparse" "$MASK_SRC" "$AF_PY" "$AF360_REPO/utils/LeftRefill" \
           "$CKPT_DIR/big-lama.pt" "$CKPT_DIR/marigold-v1-0/model_index.json" "$CKPT_DIR/sd2-inpainting/512-inpainting-ema.ckpt" \
           "$ROOT/submodules/aurafusion_3dgs/scripts/run.sh"; do
    [ -e "$f" ] || missing+=("$f")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing: %s\n' "${missing[@]}"
    echo "(models / venvs / submodule: run scripts/bakerh/setup.sh; masks: $SAM_MASKS/$S/{full,0_400}/masks_npy)"
    return 1
  fi
  js "photos: $(ls "$DATA/$S/$IMAGES" | wc -l) in $DATA/$S/$IMAGES; masks: $MASK_SRC"
}

phase_masks() {
  if done_or_force "$MASKS/.done"; then echo "found $MASKS"; return; fi
  "$PY" scripts/bakerh/masks_to_beags.py --mask_dir "$MASK_SRC" --out "$MASKS" | tee "$MASKS.txt"
  js "masks: $(tail -1 "$MASKS.txt")"
  touch "$MASKS/.done"
}

phase_train() {
  local old ply=$M/point_cloud/iteration_30000/point_cloud.ply
  old=$(old_model "$S")
  if done_or_force "$ply"; then echo "found $ply"; return; fi
  if [ -z "${RETRAIN:-}" ] && [ -f "$old/point_cloud/iteration_30000/point_cloud.ply" ] && [ -f "$old/cfg_args" ]; then
    # Link only the files: prepare writes labels.pt into iteration_30000, which must stay in this model dir.
    mkdir -p "$M/point_cloud/iteration_30000"
    for f in cfg_args cameras.json input.ply; do [ ! -e "$old/$f" ] || ln -sfn "$old/$f" "$M/$f"; done
    ln -sfn "$old/point_cloud/iteration_30000/point_cloud.ply" "$ply"
    js "FastGS 30k: reused $old ($(cat "$old/cfg_args"))"
    return
  fi
  "$PY" train.py -s "$DATA/$S" -m "$M" --images "$IMAGES" --eval --densification_interval 100 --optimizer_type default \
      --test_iterations 30000 --grad_abs_thresh 0.001 --data_device cpu
  [ -f "$ply" ] || { echo "train.py wrote no $ply"; return 1; }
  js "FastGS 30k: trained into $M"
}

label_counts() {  # label_counts IT: Gaussians per label at iteration IT
  "$PY" -c "import torch, sys; l = torch.load(sys.argv[1], map_location='cpu'); print(len(l), 'Gaussians, per label', torch.bincount(l.long()).tolist())" \
      "$M/point_cloud/iteration_$1/labels.pt"
}

phase_prepare() {
  if done_or_force "$M/point_cloud/iteration_30000/labels.pt" && [ -f "$M/beags/labeled_points.npz" ]; then echo "found labels"; return; fi
  "$PY" beags_prepare.py -m "$M" --iteration 30000 --masks "$MASKS"
  js "prepare (30000): $(label_counts 30000)"
}

phase_finetune() {
  if done_or_force "$M/point_cloud/iteration_$FT_ITER/labels.pt"; then echo "found iteration $FT_ITER"; return; fi
  "$PY" beags_finetune.py -m "$M" --iteration 30000 --dilation_weight "$DW"
  [ -f "$M/point_cloud/iteration_$FT_ITER/labels.pt" ] || { echo "no iteration_$FT_ITER/labels.pt"; return 1; }
  js "finetune dw $DW ($FT_ITER): $(label_counts $FT_ITER)"
}

phase_metrics() {
  if [ -z "${FORCE:-}" ] && grep -q "ours_$FT_ITER" "$M/results.json" 2>/dev/null; then echo "found $M/results.json"; return; fi
  "$PY" render.py -m "$M" --iteration $FT_ITER --skip_train
  "$PY" metrics.py -m "$M"
  js "test views at $FT_ITER: $("$PY" -c "import json, sys; r = json.load(open(sys.argv[1]))['ours_$FT_ITER']; print(', '.join(f'{k} {v:.4f}' for k, v in r.items()))" "$M/results.json")"
}

phase_afinputs() {
  if done_or_force "$AFO/env.sh"; then echo "found $AFO/env.sh"; return; fi
  "$PY" beags_aurafusion.py -m "$M" --iteration $FT_ITER --label "$LABEL" --mode vanilla --masks "$MASKS" --out "$AFO"
  js "aurafusion inputs: $(tr '\n' ' ' < "$AFO/env.sh")"
}

af_run() {  # af_run STEP: one stage of submodules/aurafusion_3dgs/scripts/run.sh, marked done in $AFO/.done_STEP
  if done_or_force "$AFO/.done_$1"; then echo "found $AFO/.done_$1"; return; fi
  if [ -f "$AFO/NO_UNSEEN" ] && [[ $1 =~ ^(agdd|init|sdedit|finetune)$ ]]; then
    echo "skipped: no view has an unseen region ($AFO/NO_UNSEEN); the result is $AFO/base/removed.ply"; return
  fi
  (
    set -a; source "$AFO/env.sh"; set +a
    export OUT=$AFO GS_PY=$PY AF_PY AF360_REPO CKPT_DIR STEPS=$1 IMAGES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    [ ! -f "$AFO/ref_index.txt" ] || export REF_INDEX=${REF_INDEX:-$(cat "$AFO/ref_index.txt")}
    cd "$ROOT/submodules/aurafusion_3dgs"   # keeps this repo's utils/ and scene/ off sys.path
    bash scripts/run.sh
  )
  touch "$AFO/.done_$1"
}

phase_af_flashsplat() {
  af_run flashsplat
  js "FlashSplat (vanilla): $("$PY" -c "import numpy as np, sys; o = np.load(sys.argv[1]); print(int(o.sum()), 'of', len(o), 'Gaussians')" "$AFO/base/flashsplat/object.npy")"
}

phase_compare() {
  if done_or_force "$CMP/summary.json"; then echo "found $CMP/summary.json"; return; fi
  "$PY" beags_extraction_compare.py -m "$M" --iteration $FT_ITER --label "$LABEL" \
      --flashsplat "$AFO/base/flashsplat/object.npy" --masks "$MASKS" --out "$CMP" --every "${PANEL_EVERY:-10}" | tee "$CMP.txt"
  js "$(grep '^\[compare\] Gaussians' "$CMP.txt")"
}

phase_af_render() {
  af_run render
  js "removal: $(cat "$AFO/base/removal.json")"
}

phase_af_contour() { af_run contour; }

phase_af_sam2() {
  af_run sam2
  # Reference view = the largest unseen mask (view 0 is often empty; AuraFusion asserts it is not).
  local stats
  stats=$("$AF_PY" - "$AFO/base/unseen_dilated" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
files = sorted(Path(sys.argv[1]).glob("*.png"))
area = np.array([(np.asarray(Image.open(f)) > 127).mean() for f in files])
best = int(area.argmax())
print(int(files[best].stem), len(files), int((area > 0).sum()), f"{area.max():.4f}")
EOF
)
  read -r ref views nonempty top <<< "$stats"
  js "unseen masks: $nonempty of $views views non-empty; largest $top of the frame in view $ref"
  if [ "$nonempty" -eq 0 ]; then
    touch "$AFO/NO_UNSEEN"
    js "no unseen region in any view: nothing to inpaint, AuraFusion stops at the removal ($AFO/base/removed.ply)"
  else
    echo "$ref" > "$AFO/ref_index.txt"
    js "reference view: ${REF_INDEX:-$ref}$([ -z "${REF_INDEX:-}" ] || echo ' (REF_INDEX)')"
  fi
}

phase_af_agdd() { af_run agdd; grep -h "AGDD: final" "$LOGS"/*_af_agdd.log 2>/dev/null | tail -1 | sed 's/^/AGDD: /' >> "$JS" || true; }
phase_af_init() { af_run init; }
phase_af_sdedit() { af_run sdedit; }
phase_af_finetune() {
  af_run finetune
  [ -f "$AFO/NO_UNSEEN" ] || js "AuraFusion final: $AFO/lama_ref/final ($(ls "$AFO/lama_ref/final/rgb" 2>/dev/null | wc -l) views)"
}

{
  echo "scene=$S phases=$PHASES gpu=$GPU dw=$DW label=$LABEL"
  echo "data=$DATA/$S images=$IMAGES masks=$MASK_SRC -> $MASKS"
  echo "model=$M (reuse $(old_model "$S") unless RETRAIN=1)"
  echo "aurafusion=$AFO compare=$CMP ckpt=$CKPT_DIR af360_repo=$AF360_REPO"
  echo "python=$PY af_py=$AF_PY venv=${VIRTUAL_ENV:-<not active>}"
  echo "on PATH: python=$(command -v python) nvcc=$(command -v nvcc) CUDA_HOME=${CUDA_HOME:-}"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) changed files; submodule $(git -C submodules/aurafusion_3dgs rev-parse --short HEAD 2>/dev/null)"
  nvidia-smi -i "$GPU" 2>&1 | head -20 || true
} > "$LOGS/00_config.log"
note "RUN $STAMP   phases=$PHASES   config: $LOGS/00_config.log"

IFS=, read -ra phase_list <<< "$PHASES"
for p in "${phase_list[@]}"; do
  declare -F "phase_$p" > /dev/null || { echo "unknown phase '$p'" >&2; exit 2; }
done
for p in "${phase_list[@]}"; do
  run_phase "$p"
done
note "DONE  phases=$PHASES"
echo
echo "Extraction comparison: $CMP/panels   numbers: $CMP/summary.json"
echo "AuraFusion (vanilla):  $AFO/lama_ref/final/rgb"
