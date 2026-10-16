"""BEA-GS step 2: label-aware finetune of a prepared FastGS model (2dgs train_finetune_size.py).

    python beags_finetune.py -m <model>

loss = 0.8 L1 + 0.2 D-SSIM + lambda_sag * mean(SAG * boundary_weight) + lambda_occ * L_occ
No densification; opacity pruning every `prune_interval`; both optimizers step every iteration.
Saves point_cloud/iteration_<loaded + ft_iterations>/{point_cloud.ply, labels.pt}.
"""
import os
from random import randint

import numpy as np
import torch
from fused_ssim import fused_ssim
from tqdm import tqdm

from beags.common import beags_path, labels_path, load_model, parse_args
from beags.masks import attach_masks
from beags.occupancy import LabeledVoxels, occupancy_loss, voxel_size
from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss


def add_args(parser):
    parser.add_argument("--masks", default="", help="default: <model>/beags/masks (consensus masks)")
    parser.add_argument("--ft_iterations", type=int, default=3000)
    parser.add_argument("--lambda_sag", type=float, default=0.5)
    parser.add_argument("--lambda_occ", type=float, default=10.0)
    parser.add_argument("--samples_per_gaussian", type=int, default=20)
    parser.add_argument("--voxel_vicinity", type=float, default=4.30)
    parser.add_argument("--dilation_weight", type=float, default=1.0)
    parser.add_argument("--prune_opacity", type=float, default=0.05)
    parser.add_argument("--prune_interval", type=int, default=100)


def load_voxels(model_path, vicinity):
    data = np.load(beags_path(model_path, "labeled_points.npz"))
    xyz = torch.from_numpy(data["xyz"]).cuda()
    size = voxel_size(xyz) * vicinity
    print(f"[beags] voxel size {size:.5f} from {xyz.shape[0]} labelled points")
    return LabeledVoxels.build(xyz, torch.from_numpy(data["labels"]).cuda(), size)


def finetune(dataset, pipe, opt, args):
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")  # load_ply leaves it empty
    cameras = scene.getTrainCameras()
    num_labels = attach_masks(cameras, args.masks or beags_path(dataset.model_path, "masks"), args.dilation_weight)
    voxels = load_voxels(dataset.model_path, args.voxel_vicinity)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    optimizers = [gaussians.optimizer, gaussians.shoptimizer]

    start = scene.loaded_iter
    stack = []
    progress = tqdm(range(1, args.ft_iterations + 1), desc="BEA-GS finetune")
    for it in progress:
        gaussians.update_learning_rate(start - args.ft_iterations + it)  # continue the tail of the schedule
        if not stack:
            stack = list(cameras)
        cam = stack.pop(randint(0, len(stack) - 1))

        out = render_fastgs(cam, gaussians, pipe, bg, opt.mult, gaussian_labels=gaussians.labels,
                            pixel_labels=cam.masks.labels, num_labels=num_labels)
        gt = cam.original_image.cuda()
        photo = (1 - opt.lambda_dssim) * l1_loss(out["render"], gt) + \
            opt.lambda_dssim * (1 - fused_ssim(out["render"][None], gt[None]))
        sag = (out["sag"] * cam.masks.boundary).mean()
        occ = occupancy_loss(gaussians, gaussians.labels, voxels, args.samples_per_gaussian)
        (photo + args.lambda_sag * sag + args.lambda_occ * occ).backward()

        with torch.no_grad():
            for o in optimizers:
                o.step()
                o.zero_grad(set_to_none=True)
            if it % args.prune_interval == 0:
                gaussians.prune_points((gaussians.get_opacity < args.prune_opacity).squeeze(1))
            if it % 10 == 0:
                progress.set_postfix(photo=f"{photo.item():.4f}", sag=f"{sag.item():.5f}",
                                     occ=f"{occ.item():.5f}", n=gaussians.get_xyz.shape[0])

    end = start + args.ft_iterations
    scene.save(end)
    torch.save(gaussians.labels, labels_path(dataset.model_path, end))
    print(f"[beags] saved iteration {end} to {os.path.join(dataset.model_path, 'point_cloud')}")


if __name__ == "__main__":
    finetune(*parse_args("BEA-GS finetune", add_args))
