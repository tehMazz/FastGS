"""BEA-GS segmentation metrics (2dgs utils/general_utils.py + train_finetune_size.segmentation_logging).

IoU / BIoU:   label render (all Gaussians, coverage >= 0.5)  vs  gt_masks[c]
EIoU / BEIoU: object rendered alone (coverage >= 0.1)          vs  gt_masks_extracted[c]
Averaged over (view, object) pairs of views that have GT.
"""
import cv2
import numpy as np

from beags.labels import label_coverage


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


def segmentation_metrics(cameras, gaussians, pipe, mult, labels, num_labels):
    """Returns {"IoU", "BIoU", "EIoU", "BEIoU", "per_view": [...]}; skips views without GT."""
    rows = []
    for cam in cameras:
        gt = getattr(cam, "gt", None)
        if gt is None:
            continue
        gt_ext = getattr(cam, "gt_extracted", None)
        for c in range(1, min(num_labels, gt.shape[0])):
            g = gt[c].numpy()
            pred = (label_coverage(cam, gaussians, pipe, mult, labels, c) >= 0.5).cpu().numpy()
            row = dict(view=cam.image_name, label=c, IoU=iou(pred, g), BIoU=boundary_iou(g, pred))
            if gt_ext is not None and c < gt_ext.shape[0]:
                ge = gt_ext[c].numpy()
                pe = (label_coverage(cam, gaussians, pipe, mult, labels, c, extracted=True) >= 0.1).cpu().numpy()
                row.update(EIoU=iou(pe, ge), BEIoU=boundary_iou(ge, pe))
            rows.append(row)
    summary = {k: float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None
               for k in ("IoU", "BIoU", "EIoU", "BEIoU")}
    summary["per_view"] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                           for r in rows]
    return summary
