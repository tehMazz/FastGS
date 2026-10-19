"""Inputs for AuraFusion360 removal + inpainting (submodules/aurafusion_3dgs) on a BEA-GS FastGS model.

    python beags_aurafusion.py -m <model> --iteration 33000 --label <k> --mode beags
    python beags_aurafusion.py -m <model> --iteration 33000 --label <k> --mode vanilla --masks <SAM2 mask dir>

beags:   object Gaussians = labels.pt == k             -> output/aurafusion/<scene>/beags/object.npy (bool [P])
vanilla: AuraFusion's own setup, 2D masks of object k  -> output/aurafusion/<scene>/vanilla/masks/<image_name>.png
         (pixel label as in beags.masks.soft_to_labels; FlashSplat then picks the Gaussians)
Both write inputs.json and the run.sh variables (env.sh). slurm/beags_aurafusion.sh runs everything.
"""
import json
import os
import shlex
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from beags.common import labels_path
from beags.masks import soft_to_labels


def main():
    parser = ArgumentParser(description="BEA-GS -> AuraFusion360 inputs")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=33000)
    parser.add_argument("--label", type=int, required=True, help="object label k (0 = background)")
    parser.add_argument("--mode", choices=["beags", "vanilla"], required=True)
    parser.add_argument("--masks", default="", help="vanilla: dir with <image_name>.npy (C, H, W) SAM2 masks")
    parser.add_argument("--out", default="", help="default: output/aurafusion/<scene>/<mode>")
    args = parser.parse_args()
    assert args.label > 0, "label 0 is the background"

    with open(os.path.join(args.model_path, "cfg_args")) as f:
        cfg = eval(f.read(), {"Namespace": Namespace})
    ply = os.path.abspath(os.path.join(args.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    num_gaussians = PlyData.read(ply)["vertex"].count
    out = args.out or os.path.join("output", "aurafusion", os.path.basename(os.path.normpath(args.model_path)), args.mode)
    os.makedirs(out, exist_ok=True)
    env = {"PLY": ply, "COLMAP": cfg.source_path, "IMAGES": cfg.images}

    if args.mode == "beags":
        labels = torch.load(labels_path(args.model_path, args.iteration), map_location="cpu")
        assert len(labels) == num_gaussians, f"labels.pt has {len(labels)} entries, {ply} {num_gaussians} Gaussians"
        obj = (labels == args.label).numpy()
        assert obj.any(), f"no Gaussian has label {args.label}; per label: {torch.bincount(labels.long()).tolist()}"
        env["OBJECT_MASK"] = os.path.abspath(os.path.join(out, "object.npy"))
        np.save(env["OBJECT_MASK"], obj)
        print(f"[beags] label {args.label}: {int(obj.sum())} of {num_gaussians} Gaussians")
    else:
        assert args.masks, "--mode vanilla needs --masks"
        mask_dir = os.path.join(out, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        names = sorted(f for f in os.listdir(args.masks) if f.endswith(".npy"))
        present = 0
        for name in names:
            soft = np.load(os.path.join(args.masks, name))
            obj = soft_to_labels(torch.from_numpy(np.nan_to_num(soft.astype(np.float32)))) == args.label
            present += bool(obj.any())
            Image.fromarray(obj.numpy().astype(np.uint8) * 255).save(os.path.join(mask_dir, name[:-4] + ".png"))
        env["MASK_DIR"] = os.path.abspath(mask_dir)
        print(f"[beags] label {args.label}: {len(names)} masks from {args.masks}, object in {present}")

    with open(os.path.join(out, "inputs.json"), "w") as f:
        json.dump({"mode": args.mode, "label": args.label, "iteration": args.iteration, "masks": args.masks, **env}, f,
                  indent=1)
    with open(os.path.join(out, "env.sh"), "w") as f:
        f.writelines(f"export {k}={shlex.quote(str(v))}\n" for k, v in env.items())
    print(f"[beags] wrote {out}/env.sh")


if __name__ == "__main__":
    main()
