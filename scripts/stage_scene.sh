#!/usr/bin/env bash
# Stage a read-only COLMAP scene: symlink everything, but make sparse/0 a real dir
# (FastGS writes points3D.ply there on first load).
# Usage: scripts/stage_scene.sh <src_scene_dir> <dst_scene_dir>
set -euo pipefail
src=$(realpath "$1"); dst=$2
mkdir -p "$dst/sparse/0"
for f in "$src"/*; do
  [[ $(basename "$f") == sparse ]] || ln -sfn "$f" "$dst/$(basename "$f")"
done
sparse="$src/sparse/0"; [[ -d $sparse ]] || sparse="$src/sparse"
for f in "$sparse"/*; do ln -sfn "$f" "$dst/sparse/0/$(basename "$f")"; done
echo "staged $src -> $dst"
