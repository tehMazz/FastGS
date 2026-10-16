"""BEA-GS segmentation metrics (2dgs utils/general_utils.py + train_finetune_size.segmentation_logging).

IoU / BIoU:   label render (all Gaussians, coverage >= 0.5)  vs  gt_masks[c]
EIoU / BEIoU: object rendered alone (coverage >= 0.1)          vs  gt_masks_extracted[c]
Averaged over (view, object) pairs of views that have GT.
GT object ids can be a permutation of the SAM2 ids (3D-OVS): see `match_gt_to_labels`.
"""
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from beags.labels import label_coverage
from beags.masks import load_npy_mask


def iou(a, b):
    union = np.logical_or(a, b).sum()
    inter = np.logical_and(a, b).sum()
    return 1.0 if union == 0 else inter / union


def _boundary(mask, dilation_ratio):
    h, w = mask.shape
    dilation = max(1, int(round(dilation_ratio * np.sqrt(h ** 2 + w ** 2))))
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=dilation)[1:h + 1, 1:w + 1]
    return mask - eroded


def boundary_iou(gt, pred, dilation_ratio=0.02):
    gb = _boundary(gt.astype(np.uint8), dilation_ratio)
    pb = _boundary(pred.astype(np.uint8), dilation_ratio)
    union = ((gb + pb) > 0).sum()
    inter = ((gb * pb) > 0).sum()
    return 1.0 if union == 0 else inter / union


def match_gt_to_labels(cameras, sam_dir):
    """{gt channel: SAM2 label} from the SAM2 *input* masks of the GT views (prediction-independent),
    by Hungarian matching on IoU summed over views. Identity when no GT view has a SAM2 mask."""
    num_gt = max(c.gt.shape[0] for c in cameras)
    score = None
    for cam in cameras:
        sam = load_npy_mask(os.path.join(sam_dir, f"{cam.image_name}.npy"), cam.image_height, cam.image_width)
        if sam is None:
            continue
        sam = (sam > 0.5).numpy()
        if score is None:
            score = np.zeros((num_gt, sam.shape[0]))
        for j in range(1, cam.gt.shape[0]):
            for k in range(1, sam.shape[0]):
                score[j, k] += iou(sam[k], cam.gt[j].numpy())
    if score is None:
        return {j: j for j in range(1, num_gt)}
    rows, cols = linear_sum_assignment(-score[1:, 1:])
    return {int(r) + 1: int(c) + 1 for r, c in zip(rows, cols)}


def segmentation_metrics(cameras, gaussians, pipe, mult, labels, gt_to_label):
    """Returns {"IoU", "BIoU", "EIoU", "BEIoU", "per_view": [...]} over views with GT."""
    rows = []
    for cam in cameras:
        gt, gt_ext = cam.gt, getattr(cam, "gt_extracted", None)
        for j, c in gt_to_label.items():
            if j >= gt.shape[0]:
                continue
            g = gt[j].numpy()
            pred = (label_coverage(cam, gaussians, pipe, mult, labels, c) >= 0.5).cpu().numpy()
            row = dict(view=cam.image_name, gt=j, label=c, IoU=iou(pred, g), BIoU=boundary_iou(g, pred))
            if gt_ext is not None and j < gt_ext.shape[0]:
                ge = gt_ext[j].numpy()
                pe = (label_coverage(cam, gaussians, pipe, mult, labels, c, extracted=True) >= 0.1).cpu().numpy()
                row.update(EIoU=iou(pe, ge), BEIoU=boundary_iou(ge, pe))
            rows.append(row)
    summary = {k: float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
               for k in ("IoU", "BIoU", "EIoU", "BEIoU")}
    summary["per_view"] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                           for r in rows]
    return summary
