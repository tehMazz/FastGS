#!/usr/bin/env bash
# One-time VM setup + self-test for the BakerH runs (run_all.sh calls it first; also runnable alone).
# Needs PyPI and GitHub (for graphdeco's rasterizer); never Hugging Face:
#   1. repo venv + CUDA (scripts/activate_env.sh), torch sees the GPU; 1b the FastGS rasterizer is the BEA-GS
#      build (rebuilt from submodules/ if the installed one is stock); 1c LPIPS weights for metrics.py
#   2. AuraFusion-3DGS code (submodules/aurafusion_3dgs)
#   3. graphdeco's diff_gaussian_rasterization (dr_aa: returns colour, radii, inverse depth) + scipy,
#      torchmetrics, plyfile in the repo venv: the AuraFusion 3D stages run there. FastGS's own rasterizer
#      is a different package (diff_gaussian_rasterization_fastgs), untouched
#   4. the AuraFusion360 2D venv (.venv_af2d from ArtiFixer's setup_baselines.sh): imports, and SAM2 /
#      OpenCLIP ViT-H resolve offline from the HF cache
#   5. model files under MODELS, linked into CKPT_DIR in the layout run.sh expects
#   6. official AuraFusion360 repo: LeftRefill code + learned prompt checkpoint
#   7. per scene: dataset, SAM3 masks, existing FastGS model
# Safe to rerun: finished steps are skipped. Writes output/bakerh_logs/setup/SETUP_REPORT.txt; the last line
# says SETUP OK or SETUP FAIL: <steps>. Step 7 only reports (a scene without masks fails on its own).
#
# Usage (from any directory):  bash scripts/bakerh/setup.sh
# Knobs (see common.sh): GPU, FBK, MODELS, AF360_REPO, SAM_MASKS, DATA, AF_PY, SCENES.
set -uo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
mkdir -p "$LOG_ROOT/setup"
exec > >(tee "$LOG_ROOT/setup/SETUP_REPORT.txt") 2>&1

failed=()
step() {  # step NAME CMD...: run, print OK/FAIL, remember failures
  local name=$1; shift
  echo; echo "==================== $name ===================="
  if "$@"; then echo "---- $name: OK"; else echo "---- $name: FAIL"; failed+=("$name"); fi
}
echo "setup $(date -Is)   repo=$ROOT   git=$(git rev-parse --short HEAD 2>/dev/null)"
echo "GPU=$GPU FBK=$FBK MODELS=$MODELS AF360_REPO=$AF360_REPO"
echo "SAM_MASKS=$SAM_MASKS DATA=$DATA PY=$PY"

uv_or_pip() {  # uv_or_pip ARGS...: install into the repo venv (uv venvs have no pip)
  if command -v uv > /dev/null; then uv pip install --python "$PY" "$@"; else "$PY" -m pip install "$@"; fi
}

venv_gpu() {
  activate || { echo "scripts/activate_env.sh failed (no .venv?)"; return 1; }
  echo "python=$(command -v python) nvcc=$(command -v nvcc) CUDA_HOME=${CUDA_HOME:-} CC=${CC:-}"
  "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch sees no GPU"
free, total = torch.cuda.mem_get_info()
print("device", torch.cuda.get_device_name(0), f"free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
import diff_gaussian_rasterization_fastgs, simple_knn, fused_ssim  # noqa: F401  (FastGS itself)
print("FastGS rasterizer, simple_knn, fused_ssim: ok")
EOF
}

submodule() {
  if [ ! -f submodules/aurafusion_3dgs/scripts/run.sh ]; then
    git submodule update --init submodules/aurafusion_3dgs || {
      echo "could not fetch git@github.com:IVIazz97/Aurafusion-modules.git (no GitHub SSH key on this VM?):"
      echo "copy the AuraFusion-3DGS repo to submodules/aurafusion_3dgs by hand"; return 1; }
  fi
  git -C submodules/aurafusion_3dgs log --oneline -1
}

gs_deps() {
  activate || return 1
  local test='
import inspect, torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as S, GaussianRasterizer as R
import scipy, torchmetrics, plyfile, fused_ssim  # noqa: F401
assert "antialiasing" in inspect.signature(S).parameters, "not the dr_aa rasterizer (no antialiasing)"
d = "cuda"; v = torch.eye(4, device=d); v[3, 2] = 3.0
s = S(image_height=8, image_width=8, tanfovx=1.0, tanfovy=1.0, bg=torch.zeros(3, device=d), scale_modifier=1.0,
      viewmatrix=v, projmatrix=v, sh_degree=0, campos=torch.zeros(3, device=d), prefiltered=False, debug=False,
      antialiasing=False)
out = R(s)(means3D=torch.zeros(1, 3, device=d), means2D=torch.zeros(1, 3, device=d), opacities=torch.ones(1, 1, device=d),
           colors_precomp=torch.ones(1, 3, device=d), scales=torch.full((1, 3), 0.1, device=d),
           rotations=torch.tensor([[1.0, 0, 0, 0]], device=d))
assert len(out) == 3, f"rasterizer returns {len(out)} outputs, AuraFusion-3DGS needs (colour, radii, invdepth)"
print("graphdeco rasterizer (dr_aa) ok; scipy, torchmetrics, plyfile, fused_ssim ok")'
  "$PY" -c "$test" 2>/dev/null && return 0
  echo "installing into $PY: scipy torchmetrics plyfile, graphdeco diff-gaussian-rasterization@dr_aa"
  uv_or_pip scipy torchmetrics plyfile || return 1
  uv_or_pip --no-build-isolation "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@dr_aa" || return 1
  "$PY" -c "$test"
}

af2d() {
  local af_py
  af_py=$(find_af_py) || { echo "no .venv_af2d found (set AF_PY): run ArtiFixer's scripts/bakerh/setup_baselines.sh first"; return 1; }
  echo "AF_PY=$af_py"
  HF_HUB_OFFLINE=1 "$af_py" - <<'EOF'
import sam2, diffusers, open_clip, pytorch_lightning, transformers, cv2  # noqa: F401
from huggingface_hub import snapshot_download
print("imports ok: diffusers", diffusers.__version__, "transformers", transformers.__version__)
for repo in ("facebook/sam2-hiera-large", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
    print(repo, "->", snapshot_download(repo, local_files_only=True))
EOF
}

models() {
  local ok=0 f
  for f in LaMa/big-lama.pt stable-diffusion-2-inpainting/512-inpainting-ema.ckpt marigold-depth-v1-0/model_index.json; do
    if [ -s "$MODELS/$f" ]; then echo "  ok       $MODELS/$f"; else echo "  MISSING  $MODELS/$f"; ok=1; fi
  done
  [ $ok -eq 0 ] || return 1
  mkdir -p "$CKPT_DIR/sd2-inpainting"
  ln -sfn "$MODELS/LaMa/big-lama.pt" "$CKPT_DIR/big-lama.pt"
  ln -sfn "$MODELS/marigold-depth-v1-0" "$CKPT_DIR/marigold-v1-0"
  ln -sfn "$MODELS/stable-diffusion-2-inpainting/512-inpainting-ema.ckpt" "$CKPT_DIR/sd2-inpainting/512-inpainting-ema.ckpt"
  ls -la "$CKPT_DIR" "$CKPT_DIR/sd2-inpainting"
}

af360_repo() {
  local prompt=$AF360_REPO/utils/LeftRefill/check_points/ref_guided_inpainting/ckpts/epoch=7-step=6039.ckpt
  [ -d "$AF360_REPO/utils/LeftRefill" ] || { echo "no $AF360_REPO/utils/LeftRefill"; return 1; }
  ls -la "$prompt" || return 1
  [ "$(stat -c %s "$prompt")" -gt 100000 ] || { echo "LeftRefill prompt checkpoint looks like a git-lfs pointer"; return 1; }
}

scenes() {
  local s d m old
  for s in $SCENES; do
    d=$DATA/$s; old=$(old_model "$s")
    printf '%-14s data %s (%s %s photos, sparse %s)\n' "$s" "$([ -d "$d" ] && echo ok || echo MISSING)" \
      "$IMAGES" "$(ls "$d/$IMAGES" 2>/dev/null | wc -l)" "$([ -d "$d/sparse" ] && echo ok || echo MISSING)"
    if m=$(mask_src "$s"); then echo "               masks $m ($(ls "$m" | grep -c '\.npy$') .npy)"
    else echo "               masks MISSING (no $SAM_MASKS/$s/{full,0_400}/masks_npy): this scene will fail"; fi
    if [ -f "$old/point_cloud/iteration_30000/point_cloud.ply" ]; then echo "               FastGS 30k: reuse $old"
    else echo "               FastGS 30k: none at $old, will train"; fi
  done
}

lpips_weights() {  # metrics.py (LPIPS vgg) loads them from lpipsPyTorch/weights, never from the internet
  local w=$ROOT/lpipsPyTorch/weights
  mkdir -p "$w/v0.1"
  [ -s "$w/v0.1/vgg.pth" ] || curl -fL -o "$w/v0.1/vgg.pth" \
      https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth || return 1
  [ -s "$w/vgg16-397923af.pth" ] || curl -fL -o "$w/vgg16-397923af.pth" \
      https://download.pytorch.org/models/vgg16-397923af.pth || return 1
  ls -la "$w/v0.1/vgg.pth" "$w/vgg16-397923af.pth"
}

fastgs_labels() {  # the repo's FastGS rasterizer carries the BEA-GS labels (votes, SAG); a stock build lacks them
  activate || return 1
  local src=$ROOT/submodules/diff-gaussian-rasterization_fastgs installed
  installed=$("$PY" -c 'import diff_gaussian_rasterization_fastgs as d; print(d.__file__)' 2>/dev/null)
  if [ -n "$installed" ] && cmp -s "$installed" "$src/diff_gaussian_rasterization_fastgs/__init__.py"; then
    echo "installed build matches $src ($installed)"; return 0
  fi
  echo "installed: ${installed:-none}; rebuilding $src (BEA-GS labels) into $PY"
  rm -rf "$src/build"
  uv_or_pip --no-build-isolation --force-reinstall --no-deps "$src" || return 1
  installed=$("$PY" -c 'import diff_gaussian_rasterization_fastgs as d; print(d.__file__)') || return 1
  cmp "$installed" "$src/diff_gaussian_rasterization_fastgs/__init__.py"
}

step "1 venv + GPU" venv_gpu
step "1b FastGS rasterizer with BEA-GS labels" fastgs_labels
step "1c LPIPS weights" lpips_weights
step "2 AuraFusion-3DGS submodule" submodule
step "3 graphdeco rasterizer + deps in the repo venv" gs_deps
step "4 AuraFusion360 2D venv + offline HF cache" af2d
step "5 model files + checkpoint links" models
step "6 official AuraFusion360 repo (LeftRefill)" af360_repo
echo; echo "==================== 7 scenes ===================="
scenes

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "SETUP OK"
else
  echo "SETUP FAIL: ${failed[*]}"
  exit 1
fi
