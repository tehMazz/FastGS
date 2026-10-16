#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fastgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None,
                  gaussian_labels=None, pixel_labels=None, num_labels=0, mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    BEA-GS extras:
      gaussian_labels [P] int, pixel_labels [H, W] int (-1 = unlabeled), num_labels
        -> adds "sag" [1, H, W] (weight of mislabeled Gaussians) and "votes" [P, L].
      mask [P] bool -> render only that subset (override_color / labels are given for all P).
    """
    def sel(t):
        return t if (mask is None or t is None) else t[mask]

    n_points = pc.get_xyz.shape[0] if mask is None else int(mask.sum())
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((n_points, 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map,
        gaussian_labels=sel(gaussian_labels),
        pixel_labels=pixel_labels,
        num_labels=num_labels,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = sel(pc.get_xyz)
    means2D = screenspace_points
    opacity = sel(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = sel(pc.get_covariance(scaling_modifier))
    else:
        scales = sel(pc.get_scaling)
        rotations = sel(pc.get_rotation)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = sel(pc.get_features).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = sel(pc.get_features_dc), sel(pc.get_features_rest)
    else:
        colors_precomp = sel(override_color)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts, sag, votes = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts,
            "sag": sag if sag.numel() else None,
            "votes": votes if votes.numel() else None}


@torch.no_grad()
def render_depth_alpha(viewpoint_camera, pc, pipe, mult, mask=None):
    """Expected depth [H, W] (normalized by alpha) and alpha [H, W], via colors_precomp = [z_view, 1, 0]."""
    view = viewpoint_camera.world_view_transform
    z = pc.get_xyz @ view[:3, 2:3] + view[3, 2]
    colors = torch.cat([z, torch.ones_like(z), torch.zeros_like(z)], dim=1)
    bg = torch.zeros(3, device="cuda")
    out = render_fastgs(viewpoint_camera, pc, pipe, bg, mult, override_color=colors, mask=mask)["render"]
    alpha = out[1]
    return out[0] / alpha.clamp_min(1e-6), alpha