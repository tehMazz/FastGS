"""Per-view object masks for BEA-GS.

Mask files are `<dir>/<image_name>.npy`, shape (C, H, W), channel 0 = background,
channels 1..C-1 = objects with ids consistent across views (SAM2 tracking).
"""
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ViewMasks:
    soft: Optional[torch.Tensor]   # (C, H, W) float32, CPU; None when the view has no mask file
    labels: torch.Tensor           # (H, W) int32, CUDA; -1 = no mask, 0 = background
    present: torch.Tensor          # (C,) bool, CPU; channel has any pixel
    boundary: torch.Tensor         # (1, H, W) float32, CUDA; SAG pixel weight


def load_npy_mask(path, height, width):
    """(C, H, W) float32 resized to (height, width) with bilinear interpolation, or None."""
    if not os.path.exists(path):
        return None
    masks = np.load(path)
    if masks.dtype == bool:
        masks = masks.astype(np.uint8)
    elif masks.dtype == np.float16:  # consensus masks; cv2 cannot resize float16
        masks = masks.astype(np.float32)
    masks = np.stack([cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR) for m in masks])
    return torch.from_numpy(np.nan_to_num(masks.astype(np.float32)))


def soft_to_labels(soft):
    """Pixel label = 1 + argmax over object channels where its value > 0.5, else 0 (background)."""
    objects = soft[1:]
    best, idx = objects.max(dim=0)
    return torch.where(best > 0.5, idx + 1, torch.zeros_like(idx)).int()


def boundary_weight(labels, weight, kernel_size=3):
    """1 everywhere, `weight` on pixels within a kernel_size window of a different (valid) label."""
    valid = labels >= 0
    near_other = torch.zeros_like(valid)
    for label in torch.unique(labels[valid]).tolist():
        one_hot = (labels == label).float()[None, None]
        dilated = F.max_pool2d(one_hot, kernel_size, stride=1, padding=kernel_size // 2)[0, 0] > 0
        near_other |= dilated & (labels != label)
    out = torch.ones_like(labels, dtype=torch.float32)
    out[near_other] = weight
    return out[None]


def make_view_masks(soft, height, width, num_labels, dilation_weight):
    if soft is None:
        labels = torch.full((height, width), -1, dtype=torch.int32, device="cuda")
        present = torch.zeros(num_labels, dtype=torch.bool)
    else:
        labels = soft_to_labels(soft).cuda()
        present = soft.flatten(1).amax(dim=1) > 0
    return ViewMasks(soft, labels, present, boundary_weight(labels, dilation_weight))


def attach_masks(cameras, mask_dir, dilation_weight=1.0):
    """Sets `cam.masks` (ViewMasks) on every camera; returns the number of labels C."""
    softs = [load_npy_mask(os.path.join(mask_dir, f"{c.image_name}.npy"), c.image_height, c.image_width)
             for c in cameras]
    found = [s for s in softs if s is not None]
    if not found:
        raise FileNotFoundError(f"no masks for any camera in {mask_dir}")
    num_labels = max(s.shape[0] for s in found)
    for cam, soft in zip(cameras, softs):
        cam.masks = make_view_masks(soft, cam.image_height, cam.image_width, num_labels, dilation_weight)
    print(f"[beags] masks for {len(found)}/{len(cameras)} views from {mask_dir}, {num_labels} labels")
    return num_labels


def attach_gt_masks(cameras, gt_dir):
    """Sets `cam.gt` and `cam.gt_extracted` ((C, H, W) bool CPU, or None).

    `gt_extracted` lives next to `gt_dir`, with 'gt_masks' replaced by 'gt_masks_extracted'."""
    extracted_dir = gt_dir.replace("gt_masks", "gt_masks_extracted")
    for cam in cameras:
        for attr, folder in (("gt", gt_dir), ("gt_extracted", extracted_dir)):
            soft = load_npy_mask(os.path.join(folder, f"{cam.image_name}.npy"), cam.image_height, cam.image_width)
            setattr(cam, attr, None if soft is None else soft > 0.5)
