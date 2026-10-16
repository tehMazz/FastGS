#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACTIVATE_SCRIPT="$PROJECT_ROOT/scripts/activate_env.sh"

# Set to true to interpolate only between test cameras.
INTERPOLATE_TEST_ONLY="${INTERPOLATE_TEST_ONLY:-false}"
# Number of interpolated frames between consecutive key cameras.
FRAMES_BETWEEN="${FRAMES_BETWEEN:-5}"
FPS=24
VIDEO_NAME="trajectory.mp4"
ITERATION=-1

# Change to "$PROJECT_ROOT/output_distort" if needed.
MODEL_ROOT="$PROJECT_ROOT/output"
SCENES=(
  "cogen1_bakerh"
  "cogen2_bakerh"
  "compressor_bakerh"
  "g35_bakerh"
  "pcv_bakerh"
  "ta_blue_motor_bakerh"
  "ta_turbine_bakerh"
)

cd "$PROJECT_ROOT"
for scene in "${SCENES[@]}"; do
  model_path="$MODEL_ROOT/$scene"
  if [[ ! -d "$model_path" ]]; then
    echo "[skip] Missing model path: $model_path"
    continue
  fi

  cmd=(
    python render_video.py
    -m "$model_path"
    --iteration "$ITERATION"
    --frames_between "$FRAMES_BETWEEN"
    --fps "$FPS"
    --video_name "$VIDEO_NAME"
  )

  if [[ "$INTERPOLATE_TEST_ONLY" == "true" ]]; then
    cmd+=(--interpolate_test_only)
  fi

  if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "[error] Missing activation script: $ACTIVATE_SCRIPT"
    exit 1
  fi
  source "$ACTIVATE_SCRIPT"

  echo "[run] ${cmd[*]}"
  "${cmd[@]}"
done
