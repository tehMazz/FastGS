"""Unit tests for beags/ helpers (GPU). Run: python -m pytest -q tests/test_beags_utils.py"""
from types import SimpleNamespace

import torch

from beags.consensus import backproject, project, reproject_masks
from beags.masks import boundary_weight, soft_to_labels
from beags.occupancy import LabeledVoxels, occupancy_loss
from tests.synthetic import make_camera


def test_soft_to_labels():
    soft = torch.zeros(3, 2, 2)
    soft[1, 0, 0] = 0.9
    soft[2, 0, 0] = 0.6          # lower than channel 1 -> label 1
    soft[2, 1, 1] = 0.4          # below 0.5 -> background
    assert soft_to_labels(soft).tolist() == [[1, 0], [0, 0]]


def test_boundary_weight_marks_pixels_next_to_other_label():
    labels = torch.zeros(5, 5, dtype=torch.int32)
    labels[:, 3:] = 1
    w = boundary_weight(labels, 0.1)[0]
    assert torch.all(w[:, 2] == 0.1) and torch.all(w[:, 3] == 0.1)
    assert torch.all(w[:, :2] == 1) and torch.all(w[:, 4] == 1)


def _voxels():
    pts = torch.rand(2000, 3, device="cuda")                       # unit cube, label 1
    mixed = torch.tensor([[5.05, 5.05, 5.05], [5.06, 5.06, 5.06]], device="cuda")
    pts = torch.cat([pts, mixed])
    labels = torch.cat([torch.ones(2000, dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32)]).cuda()
    return LabeledVoxels.build(pts, labels, 0.1)


def test_voxel_membership():
    vox = _voxels()
    q = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [3.0, 3.0, 3.0], [5.05, 5.05, 5.05]], device="cuda")
    lab = torch.tensor([1, 2, 1, 1], device="cuda")
    # inside+same label, inside+other label, far away, only an impure (mixed-label) voxel nearby
    assert vox.contains(q, lab).tolist() == [True, False, False, False]


def _gaussians(center, scale):
    n = center.shape[0]
    return SimpleNamespace(
        get_xyz=center.cuda(), get_scaling=torch.full((n, 3), scale, device="cuda"),
        get_rotation=torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1),
        get_opacity=torch.full((n, 1), 0.8, device="cuda"))


def test_occupancy_loss_zero_inside_positive_outside():
    vox = _voxels()
    labels = torch.ones(1, dtype=torch.int32, device="cuda")
    inside = occupancy_loss(_gaussians(torch.tensor([[0.5, 0.5, 0.5]]), 0.01), labels, vox)
    outside = occupancy_loss(_gaussians(torch.tensor([[3.0, 3.0, 3.0]]), 0.01), labels, vox)
    background = occupancy_loss(_gaussians(torch.tensor([[3.0, 3.0, 3.0]]), 0.01), labels * 0, vox)
    assert inside.item() == 0 and outside.item() > 0 and background.item() == 0


def test_backproject_project_roundtrip():
    cam = make_camera()
    depth = torch.full((cam.image_height, cam.image_width), 3.0, device="cuda")
    uv, z = project(backproject(depth, cam), cam)
    ys, xs = torch.meshgrid(torch.arange(cam.image_height), torch.arange(cam.image_width), indexing="ij")
    torch.testing.assert_close(uv[..., 0].cpu(), xs.float(), atol=1e-3, rtol=0)
    torch.testing.assert_close(uv[..., 1].cpu(), ys.float(), atol=1e-3, rtol=0)
    torch.testing.assert_close(z, depth)


def test_consensus_of_single_view_is_identity():
    cam = make_camera()
    H, W = cam.image_height, cam.image_width
    depth = torch.full((1, H, W), 3.0, device="cuda")
    masks = torch.zeros(1, 2, H, W, device="cuda")
    masks[0, 1, 20:50, 30:90] = 1
    masks[0, 0] = 1 - masks[0, 1]
    out = reproject_masks([cam], depth, masks, torch.ones(1, 2, dtype=torch.bool))
    # the outermost pixel ring counts as out of bounds (same as BEA-GS)
    torch.testing.assert_close(out[..., 1:-1, 1:-1], masks.cpu()[..., 1:-1, 1:-1])
