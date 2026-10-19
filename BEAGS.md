# BEA-GS on FastGS (3DGS)

Port of BEA-GS (tehMazz/2dgs, branch `3D`, final method `train_finetune_size.py`) to 3DGS on upstream FastGS.

## Pipeline
```bash
python train.py -s <scene> -m <model> --eval ...                   # stock FastGS, 30k
python beags_prepare.py  -m <model> --masks <sam2 mask dir>       # consensus masks, labels, labelled point cloud
python beags_finetune.py -m <model>                               # 3k it: photo + 0.5 SAG + 10 occupancy -> iteration 33000
python beags_eval.py     -m <model> --iteration 33000 --gt_masks <gt dir> --masks <sam2 mask dir> [--render]
```
One scene end to end on SLURM: `sbatch -J <job> slurm/beags_scene.sh <name> <scene> <masks> <gt> [train.py args]`.
Benchmark (Mip-NeRF 360 + 3D-OVS bed + LERF figurines + LLFF fern): `bash slurm/launch_benchmark.sh`, then `python scripts/collect_results.py`.
Env: `/leonardo_scratch/fast/IscrC_EditGS/envs/beags_fsplat`, `module load gcc/12.2.0 cuda/12.2`.

## AuraFusion360 removal + inpainting
`submodules/aurafusion_3dgs` = git submodule `IVIazz97/Aurafusion-modules` (AuraFusion360 on a 3DGS PLY;
`git submodule update --init`). Two variants on the same FastGS PLY (iteration 33000) and the same object (label k):
- `beags`: object Gaussians = `labels.pt == k` (FlashSplat skipped).
- `vanilla`: AuraFusion's own setup, FlashSplat on the SAM2 masks of object k.
```bash
sbatch -J af_beags_<scene>   slurm/beags_aurafusion.sh <name> <k> beags
sbatch -J af_vanilla_<scene> slurm/beags_aurafusion.sh <name> <k> vanilla <sam2 mask dir>
```
Output: `output/aurafusion/<scene>/<mode>/lama_ref/final/{point_cloud.ply,rgb/}`. AuraFusion runs in its own
envs (graphdeco rasterizer, not `_fastgs`); paths and checkpoints are set in the sbatch script.

## BakerH on the VM
`bash scripts/bakerh/run_all.sh` (detaches; `--fg` to stay in the terminal): `setup.sh` once, then per scene
`run_scene.sh`: SAM3 masks → FastGS 30k (reuses `output/<scene>_bakerh`) → BEA-GS prepare + finetune with dilation
weight 0.1 → test metrics → vanilla AuraFusion360 → FlashSplat vs BEA-GS extraction panels.
Send back `output/bakerh_logs/latest/REPORT.txt`. Outputs: `output/beags_dw0.1/bakerh/<scene>`,
`output/aurafusion/<scene>/{vanilla,extraction}`. Paths (models, masks, `.venv_af2d`) follow ArtiFixer's
`scripts/bakerh` on the same VM; knobs in `scripts/bakerh/common.sh`.

## Code
| where | what |
|---|---|
| `submodules/diff-gaussian-rasterization_fastgs` | optional labels: `sag` image + `votes` [P, L]; SAG alpha gradient only for mismatched Gaussians (as BEA-GS). No labels = stock FastGS |
| `gaussian_renderer` | `render_fastgs(..., gaussian_labels, pixel_labels, num_labels, mask)`, `render_depth_alpha` |
| `beags/masks.py` | SAM2 (C, H, W) masks -> pixel labels, boundary weights; GT masks |
| `beags/consensus.py` | multi-view mask consensus through rendered depth |
| `beags/labels.py` | label voting, label renders, labelled point cloud |
| `beags/occupancy.py` | labelled-voxel occupancy loss |
| `beags/metrics.py` | IoU / BIoU / EIoU / BEIoU, GT-id to label-id matching |
| `beags/splits.py` | 3D-OVS test frames, `*test*` views |
| `tests/` | rasterizer (stock equivalence, votes, SAG gradient) and helper tests |

## Differences from the 2DGS code
- 3DGS: no distortion/normal losses, expected (not median) depth, occupancy samples all 3 axes.
- Optimizers step every finetune iteration (FastGS steps every 64 it after 20k).
- Not carried over: SAG weighted by camera 0's boundary map (bug), test views in training < 16k it.
- Eval matches GT object ids to label ids with the SAM2 input masks (3D-OVS ids are permuted).
- Data: masks/GT in `/leonardo_work/IscrC_EditGS/gabri/datasets`; read-only scenes are staged with `scripts/stage_scene.sh`.
