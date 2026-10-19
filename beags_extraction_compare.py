"""FlashSplat (vanilla AuraFusion360) vs BEA-GS object extraction on the same model.

    python beags_extraction_compare.py -m <model> --iteration 33000 --label 1 \
        --flashsplat <object.npy> --masks <dir with (C, H, W) .npy> --out <dir> [--every 10]

Per train view (every --every-th one), <out>/panels/<image_name>.jpg:
    top:    photo            | FlashSplat object alone | BEA-GS object alone   (white background)
    bottom: SAM mask overlay | FlashSplat removal      | BEA-GS removal
<out>/summary.json: Gaussians per method and their overlap, and per method the IoU of the object-alone
coverage (> 0.5) with the SAM mask, over every train view that has a mask.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from beags.common import load_model, parse_args
from beags.masks import load_npy_mask, soft_to_labels
from gaussian_renderer import render_fastgs

TILE_WIDTH = 800


def add_args(parser):
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--flashsplat", required=True, help="bool .npy over the model's Gaussians")
    parser.add_argument("--masks", required=True, help="dir with <image_name>.npy (C, H, W) masks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=int, default=10, help="panel for every N-th train view")


def tile(img, title):
    """(3, H, W) in [0, 1] -> uint8 (h, TILE_WIDTH, 3) with the title in the corner."""
    h = round(img.shape[1] * TILE_WIDTH / img.shape[2])
    img = F.interpolate(img[None].clamp(0, 1), size=(h, TILE_WIDTH), mode="area")[0]
    pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, 8 * len(title) + 10, 18), fill=(0, 0, 0))
    draw.text((5, 3), title, fill=(255, 255, 255))
    return np.asarray(pil)


def iou(pred, gt):
    union = (pred | gt).sum().item()
    return (pred & gt).sum().item() / union if union else None


@torch.no_grad()
def main():
    dataset, pipe, opt, args = parse_args("FlashSplat vs BEA-GS extraction", add_args)
    scene, gaussians = load_model(dataset, args.iteration, with_labels=True)
    cameras = scene.getTrainCameras()
    beags = gaussians.labels == args.label
    flash = torch.from_numpy(np.load(args.flashsplat)).cuda()
    assert flash.dtype == torch.bool and len(flash) == len(beags), \
        f"{args.flashsplat}: {flash.dtype} [{len(flash)}], model has {len(beags)} Gaussians"
    methods = {"FlashSplat": flash, "BEA-GS": beags}

    os.makedirs(os.path.join(args.out, "panels"), exist_ok=True)
    white, black = torch.ones(3, device="cuda"), torch.zeros(3, device="cuda")
    ones = torch.ones_like(gaussians.get_xyz)
    per_view = []
    for n, cam in enumerate(cameras):
        soft = load_npy_mask(os.path.join(args.masks, f"{cam.image_name}.npy"), cam.image_height, cam.image_width)
        sam = (soft_to_labels(soft) == args.label).cuda() if soft is not None else None
        row = {"view": cam.image_name}
        for name, m in methods.items():
            if sam is not None:
                cover = render_fastgs(cam, gaussians, pipe, black, opt.mult, override_color=ones, mask=m)["render"][0]
                row[name] = iou(cover > 0.5, sam)
        per_view.append(row)
        if n % args.every:
            continue
        photo = cam.original_image.cuda()
        overlay = photo.clone()
        if sam is not None:
            overlay[:, sam] = 0.5 * overlay[:, sam] + 0.5 * torch.tensor([1.0, 0.0, 0.0], device="cuda")[:, None]
        top, bottom = [tile(photo, "photo")], [tile(overlay, "SAM mask" if sam is not None else "no SAM mask")]
        for name, m in methods.items():
            top.append(tile(render_fastgs(cam, gaussians, pipe, white, opt.mult, mask=m)["render"], f"{name} object"))
            bottom.append(tile(render_fastgs(cam, gaussians, pipe, white, opt.mult, mask=~m)["render"], f"{name} removal"))
        Image.fromarray(np.concatenate([np.concatenate(top, 1), np.concatenate(bottom, 1)], 0)).save(
            os.path.join(args.out, "panels", f"{cam.image_name}.jpg"), quality=90)

    both = (flash & beags).sum().item()
    summary = {"model": dataset.model_path, "iteration": args.iteration, "label": args.label,
               "gaussians": len(beags), "flashsplat": flash.sum().item(), "beags": beags.sum().item(),
               "both": both, "only_flashsplat": (flash & ~beags).sum().item(),
               "only_beags": (beags & ~flash).sum().item(),
               "jaccard": both / max((flash | beags).sum().item(), 1),
               "views": len(cameras), "views_with_mask": sum("BEA-GS" in r for r in per_view)}
    for name in methods:
        values = [r[name] for r in per_view if r.get(name) is not None]
        summary[f"iou_vs_sam_{name}"] = float(np.mean(values)) if values else None
    summary["per_view"] = per_view
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[compare] Gaussians: FlashSplat {summary['flashsplat']}, BEA-GS {summary['beags']}, both {both}, "
          f"Jaccard {summary['jaccard']:.3f}; mean IoU vs SAM masks over {summary['views_with_mask']} views: "
          f"FlashSplat {summary['iou_vs_sam_FlashSplat']}, BEA-GS {summary['iou_vs_sam_BEA-GS']}")
    print(f"[compare] panels: {os.path.join(args.out, 'panels')}")


if __name__ == "__main__":
    main()
