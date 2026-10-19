#!/usr/bin/env bash
# Every BakerH scene on the VM, one after another on GPU 1: setup.sh, then per scene run_scene.sh
# (BEA-GS with dilation weight 0.1 on FastGS, vanilla AuraFusion360, FlashSplat vs BEA-GS extraction side by
# side; run_scene.sh's header documents the phases and knobs).
#
# Usage (from any directory):
#   bash scripts/bakerh/run_all.sh        starts in the background, keeps running after you log out,
#                                         prints the commands to follow and stop it
#   bash scripts/bakerh/run_all.sh --fg   runs in this terminal instead
#
# Everything is logged under output/bakerh_logs/<timestamp>/ (also output/bakerh_logs/latest/):
#   all.log     full console output of setup and every scene and phase
#   REPORT.txt  environment, setup result (SETUP OK / FAIL and why), status + duration of each scene,
#               per-phase durations and peak GPU memory, what each phase found (job_summary.txt: mask counts,
#               Gaussians per label, reference view, comparison numbers), and the last 80 lines of every
#               failed phase. Updated after each scene. If anything fails, send this file back.
#   <scene>/    per-phase logs NN_<phase>.log, summary.log, job_summary.txt
# A failed scene is skipped and the next one runs. Rerunning run_all.sh redoes only what is missing
# (finished phases are skipped in seconds). SCENES="PCV Compressor" runs a subset; GPU (1), DW (0.1),
# FORCE=1, PHASES, ... pass through to run_scene.sh. SKIP_SETUP=1 skips setup.sh.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/common.sh"
cd "$ROOT"
export GPU SCENES DW

LATEST=$LOG_ROOT/latest
if [ -z "${BAKERH_CHILD:-}" ] && [ -f "$LATEST/pgid" ] && kill -0 -- "-$(cat "$LATEST/pgid")" 2>/dev/null; then
  echo "a run is already going (log: $LATEST/all.log). Stop it first: kill -- -$(cat "$LATEST/pgid")"
  exit 1
fi

STAMP=${BAKERH_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$LOG_ROOT/$STAMP
mkdir -p "$RUN_DIR"
ln -sfn "$STAMP" "$LATEST"

if [ -z "${BAKERH_CHILD:-}" ] && [ "${1:-}" != "--fg" ]; then
  # Relaunch detached (own session, immune to hangup) with all output going to all.log.
  launcher=(nohup); command -v setsid > /dev/null && launcher=(setsid nohup)
  BAKERH_CHILD=1 BAKERH_STAMP=$STAMP "${launcher[@]}" bash "$HERE/run_all.sh" >> "$RUN_DIR/all.log" 2>&1 < /dev/null &
  sleep 3
  echo "Started in the background; it keeps running if you close this terminal."
  echo "  follow:  tail -f $LATEST/all.log"
  echo "  report:  $LATEST/REPORT.txt   (send this back if something fails)"
  echo "  stop:    kill -- -\$(cat $LATEST/pgid)"
  exit 0
fi
if [ -n "${BAKERH_CHILD:-}" ]; then
  ps -o pgid= -p $$ | tr -d ' ' > "$RUN_DIR/pgid"   # kill -- -<pgid> stops this run and its python jobs
else
  exec > >(tee -a "$RUN_DIR/all.log") 2>&1
fi
export BAKERH_RUN_DIR=$RUN_DIR
REPORT=$RUN_DIR/REPORT.txt
log() { echo "[$(date '+%F %T')] $*"; }
section() { printf '\n==================== %s ====================\n' "$*" >> "$REPORT"; }

log "run_all $STAMP   scenes: $SCENES   dw: $DW   gpu: $GPU"
log "report: $REPORT"
{
  echo "bakerh BEA-GS + AuraFusion report   started $(date -Is)   run $STAMP"
  echo "scenes: $SCENES"
  echo "dw=$DW label=$LABEL gpu=$GPU images=$IMAGES repo=$ROOT"
  echo "data=$DATA masks=$SAM_MASKS models=$MODELS af360_repo=$AF360_REPO"
} > "$REPORT"

section "git"
{
  git log --oneline -3
  git submodule status 2>&1
  git status --short | head -40
} >> "$REPORT" 2>&1

section "GPU"
nvidia-smi >> "$REPORT" 2>&1 || echo "nvidia-smi failed" >> "$REPORT"

section "setup"
if [ -n "${SKIP_SETUP:-}" ]; then
  echo "skipped (SKIP_SETUP=1)" >> "$REPORT"
else
  log "setup.sh"
  timeout 3600 bash "$HERE/setup.sh"
  code=$?
  cat "$LOG_ROOT/setup/SETUP_REPORT.txt" >> "$REPORT" 2>/dev/null
  if [ "$code" -ne 0 ]; then
    section "summary   finished $(date -Is)"
    echo "setup failed (exit $code): no scene was run. The setup section above says which step and why." >> "$REPORT"
    log "setup failed; report: $REPORT"
    exit 1
  fi
fi

results=()
for scene in $SCENES; do
  start=$(date +%s)
  log "scene $scene start"
  bash "$HERE/run_scene.sh" "$scene"
  code=$?
  took=$(($(date +%s) - start))
  if [ "$code" -eq 0 ]; then status=OK; else status="FAILED(exit $code)"; fi
  results+=("$(printf '%-16s %-18s %dh%02dm' "$scene" "$status" $((took / 3600)) $((took % 3600 / 60)))")
  log "scene $scene $status after $((took / 60)) min"

  logs=$RUN_DIR/$scene
  section "scene $scene: $status ($((took / 60)) min)"
  cat "$logs/summary.log" >> "$REPORT" 2>/dev/null
  if [ -s "$logs/job_summary.txt" ]; then
    echo "--- $scene/job_summary.txt" >> "$REPORT"
    tail -60 "$logs/job_summary.txt" >> "$REPORT"
  fi
  if [ "$code" -ne 0 ] && [ -f "$logs/FAILED" ]; then
    failed_log=$(sed -n 's/.*log=//p' "$logs/FAILED")
    echo "--- last 80 lines of $failed_log" >> "$REPORT"
    tail -80 "$failed_log" >> "$REPORT" 2>&1
  elif [ "$code" -ne 0 ]; then
    echo "--- no phase log; last 60 lines of all.log" >> "$REPORT"
    tail -60 "$RUN_DIR/all.log" >> "$REPORT"
  fi
done

section "summary   finished $(date -Is)"
printf '%s\n' "${results[@]}" >> "$REPORT"
log "run_all done"
printf '%s\n' "${results[@]}"
log "report: $REPORT"
