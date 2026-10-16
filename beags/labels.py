"""Per-Gaussian labels: voting from 2D masks, label renders, labelled point cloud."""
import torch

from beags.consensus import backproject
from gaussian_renderer import render_depth_alpha, render_fastgs


@torch.no_grad()
def assign_labels(gaussians, cameras, pipe, mult, num_labels):
    """label = argmax_l sum_views sum_pixels(label l) alpha*T. Never-hit Gaussians get 0 (background)."""
    n = gaussians.get_xyz.shape[0]
    votes = torch.zeros((n, num_labels), device="cuda")
    dummy = torch.zeros(n, dtype=torch.int32, device="cuda")  # labels only switch voting on here
    bg = torch.zeros(3, device="cuda")
    for cam in cameras:
        if (cam.masks.labels >= 0).any():
            votes += render_fastgs(cam, gaussians, pipe, bg, mult, gaussian_labels=dummy,
                                   pixel_labels=cam.masks.labels, num_labels=num_labels)["votes"]
    return votes.argmax(dim=1).int()


@torch.no_grad()
def label_coverage(cam, gaussians, pipe, mult, labels, label, extracted=False):
    """(H, W) weight of `label` Gaussians: rendered with all Gaussians (occluded by others),
    or alone (`extracted=True`, the object cut out of the scene)."""
    ones = torch.ones_like(gaussians.get_xyz)
    colors = ones * (labels == label)[:, None]
    mask = (labels == label) if extracted else None
    bg = torch.zeros(3, device="cuda")
    return render_fastgs(cam, gaussians, pipe, bg, mult, override_color=colors, mask=mask)["render"].mean(dim=0)


@torch.no_grad()
def labeled_point_cloud(gaussians, cameras, pipe, mult, labels, threshold=0.5, label_ids=None):
    """Back-project every pixel where a label covers >= threshold. Returns (xyz, label).
    label_ids: labels to back-project; default all object labels (1..max), i.e. no background."""
    if label_ids is None:
        label_ids = range(1, int(labels.max()) + 1)
    xyz, lab = [], []
    for cam in cameras:
        depth, _ = render_depth_alpha(cam, gaussians, pipe, mult)
        world = backproject(depth, cam)
        for label in label_ids:
            hit = (label_coverage(cam, gaussians, pipe, mult, labels, label) >= threshold) & (depth > 0)
            xyz.append(world[hit])
            lab.append(torch.full((int(hit.sum()),), label, dtype=torch.int32, device="cuda"))
    return torch.cat(xyz), torch.cat(lab)
