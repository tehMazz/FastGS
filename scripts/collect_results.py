"""Collect BEA-GS benchmark numbers into one markdown table.

    python scripts/collect_results.py [output/beags]

Per scene and iteration (30000 = after prepare, 33000 = after finetune): segmentation metrics
from beags/metrics_<it>.json (test split, or train when the scene has GT only there) and
PSNR/SSIM/LPIPS from results.json.
"""
import glob
import json
import os
import sys

SEG = ("IoU", "BIoU", "EIoU", "BEIoU")
PHOTO = ("PSNR", "SSIM", "LPIPS")


def scene_rows(model_dir):
    photo = {}
    if os.path.exists(os.path.join(model_dir, "results.json")):
        photo = json.load(open(os.path.join(model_dir, "results.json")))
    for path in sorted(glob.glob(os.path.join(model_dir, "beags", "metrics_*.json"))):
        it = path.rsplit("_", 1)[1].split(".")[0]
        seg = json.load(open(path))
        split = "test" if "test" in seg else next(iter(seg), None)
        values = [seg[split][k] if split else None for k in SEG]
        values += [photo.get(f"ours_{it}", {}).get(k) for k in PHOTO]
        yield it, split, values


def main(root):
    print("| scene | it | split | " + " | ".join(SEG + PHOTO) + " |")
    print("|" + "---|" * (3 + len(SEG) + len(PHOTO)))
    for model_dir in sorted(glob.glob(os.path.join(root, "*", "*"))):
        scene = os.path.relpath(model_dir, root)
        for it, split, values in scene_rows(model_dir):
            cells = ["-" if v is None else f"{v:.3f}" for v in values]
            print(f"| {scene} | {it} | {split} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/beags")
