"""Render the synthetic scene and dump outputs + grads, calling the rasterizer directly.

Works with both the stock FastGS build (3 outputs) and the BEA-GS build (5 outputs), so
the patched build can be compared with the stock one:
    <stock env>/python tests/dump_render.py stock.pt
    <beags env>/python tests/dump_render.py patched.pt
"""
import math
import sys

import torch
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

sys.path.insert(0, ".")
from tests.synthetic import as_model, make_camera, make_gaussians  # noqa: E402


def render(cam, model, get_flag, metric_map):
    settings = GaussianRasterizationSettings(
        image_height=cam.image_height, image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=torch.tensor([0.1, 0.2, 0.3], device="cuda"), scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=model.active_sh_degree, campos=cam.camera_center, mult=0.5,
        prefiltered=False, debug=False, get_flag=get_flag, metric_map=metric_map,
    )
    means2D = torch.zeros((model.get_xyz.shape[0], 4), device="cuda", requires_grad=True)
    out = GaussianRasterizer(settings)(
        means3D=model.get_xyz, means2D=means2D, opacities=model.get_opacity,
        dc=model.get_features_dc, shs=model.get_features_rest,
        scales=model.get_scaling, rotations=model.get_rotation)
    return out[0], out[1], out[2], means2D


def main(path):
    cam = make_camera()
    params, deg = make_gaussians()
    metric_map = (torch.arange(cam.image_width * cam.image_height, device="cuda") % 3 == 0).int()
    result = {}
    for get_flag in (False, True):
        for p in params.values():
            p.grad = None
        color, radii, counts, means2D = render(cam, as_model(params, deg), get_flag, metric_map)
        (color * torch.linspace(0.5, 1.5, color.numel(), device="cuda").view_as(color)).sum().backward()
        tag = f"flag{int(get_flag)}"
        result[f"{tag}/color"] = color.detach().cpu()
        result[f"{tag}/radii"] = radii.cpu()
        result[f"{tag}/counts"] = counts.cpu()
        result[f"{tag}/grad_means2D"] = means2D.grad.cpu()
        for k, p in params.items():
            result[f"{tag}/grad_{k}"] = p.grad.cpu()
    torch.save(result, path)
    print(f"saved {len(result)} tensors to {path}")


if __name__ == "__main__":
    main(sys.argv[1])
