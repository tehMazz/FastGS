"""BEA-GS evaluation: segmentation metrics against GT masks, optional per-object renders.

    python beags_eval.py -m <model> --gt_masks <dir with <image_name>.npy> [--iteration N] [--render]

Writes <model>/beags/metrics_<it>.json (IoU, BIoU, EIoU, BEIoU per split).
--render saves, for views with GT: extracted_<label>/ (object alone, white bg) and removal_<label>/.
Photometric metrics: use the stock `render.py` + `metrics.py` on the same iteration.
"""
import json
import os

import torch
import torchvision

from beags.common import beags_path, load_model, parse_args
from beags.masks import attach_gt_masks
from beags.metrics import segmentation_metrics
from gaussian_renderer import render_fastgs


def add_args(parser):
    parser.add_argument("--gt_masks", required=True)
    parser.add_argument("--render", action="store_true")


@torch.no_grad()
def render_objects(cameras, gaussians, pipe, mult, out_dir):
    white = torch.ones(3, device="cuda")
    for label in range(1, int(gaussians.labels.max()) + 1):
        for name, mask in (("extracted", gaussians.labels == label), ("removal", gaussians.labels != label)):
            folder = os.path.join(out_dir, f"{name}_{label}")
            os.makedirs(folder, exist_ok=True)
            for cam in cameras:
                img = render_fastgs(cam, gaussians, pipe, white, mult, mask=mask)["render"].clamp(0, 1)
                torchvision.utils.save_image(img, os.path.join(folder, f"{cam.image_name}.png"))


def main():
    dataset, pipe, opt, args = parse_args("BEA-GS eval", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    results = {}
    for split, cams in (("test", scene.getTestCameras()), ("train", scene.getTrainCameras())):
        attach_gt_masks(cams, args.gt_masks)
        cams = [c for c in cams if c.gt is not None]
        if not cams:
            continue
        num_labels = max(c.gt.shape[0] for c in cams)  # from GT: objects without Gaussians count as IoU 0
        results[split] = segmentation_metrics(cams, gaussians, pipe, opt.mult, gaussians.labels, num_labels)
        summary = {k: v for k, v in results[split].items() if k != "per_view"}
        print(f"[beags] {split} ({len(cams)} GT views): {summary}")
        if args.render:
            render_objects(cams, gaussians, pipe, opt.mult, beags_path(dataset.model_path, f"renders_{scene.loaded_iter}"))
    with open(beags_path(dataset.model_path, f"metrics_{scene.loaded_iter}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
