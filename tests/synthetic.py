"""Tiny synthetic camera + Gaussian cloud for rasterizer tests (no dataset needed)."""
import math
from types import SimpleNamespace

import numpy as np
import torch

from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def make_camera(width=128, height=96, fov_deg=60.0):
    fovx = math.radians(fov_deg)
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)
    view = torch.tensor(getWorld2View2(np.eye(3), np.zeros(3)), dtype=torch.float32).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    return SimpleNamespace(
        FoVx=fovx, FoVy=fovy, image_width=width, image_height=height,
        world_view_transform=view,
        full_proj_transform=(view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0),
        camera_center=view.inverse()[3, :3],
    )


def make_gaussians(n=2000, seed=0, sh_degree=3):
    """Random Gaussians in front of the camera; all tensors are leaf params with grad."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=g, device="cuda")
    xyz = torch.stack([(rand(n) - 0.5) * 2.0, (rand(n) - 0.5) * 1.5, 2.0 + 2.0 * rand(n)], dim=1)
    params = dict(
        xyz=xyz,
        scaling=torch.log(0.02 + 0.08 * rand(n, 3)),
        rotation=torch.nn.functional.normalize(rand(n, 4) - 0.5, dim=1),
        opacity=torch.logit(0.2 + 0.6 * rand(n, 1)),
        f_dc=(rand(n, 1, 3) - 0.5),
        f_rest=(rand(n, (sh_degree + 1) ** 2 - 1, 3) - 0.5) * 0.1,
    )
    return {k: v.clone().requires_grad_(True) for k, v in params.items()}, sh_degree


def as_model(p, sh_degree):
    """Duck-typed GaussianModel for render_fastgs."""
    return SimpleNamespace(
        get_xyz=p["xyz"],
        get_scaling=torch.exp(p["scaling"]),
        get_rotation=torch.nn.functional.normalize(p["rotation"], dim=1),
        get_opacity=torch.sigmoid(p["opacity"]),
        get_features_dc=p["f_dc"],
        get_features_rest=p["f_rest"],
        get_features=torch.cat([p["f_dc"], p["f_rest"]], dim=1),
        active_sh_degree=sh_degree,
        max_sh_degree=sh_degree,
    )


PIPE = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
