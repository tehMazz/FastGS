"""Multi-view mask consensus (BEA-GS `reprojection_batch`).

For every view, the masks of all views are warped into it through the rendered depth.
A warp is kept only where the forward/backward reprojection agrees; the new mask is the
average of the kept warps, per channel, over views where that channel is present.
"""
import torch
import torch.nn.functional as F

from utils.graphics_utils import fov2focal


def _intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    return fov2focal(cam.FoVx, W), fov2focal(cam.FoVy, H), (W - 1) / 2, (H - 1) / 2


def _pixel_coords(H, W):
    ys, xs = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
    return torch.stack([xs, ys], dim=-1).float()  # (H, W, 2)


def backproject(depth, cam):
    """(H, W) depth -> (H, W, 3) world points."""
    fx, fy, cx, cy = _intrinsics(cam)
    uv = _pixel_coords(*depth.shape)
    cam_pts = torch.stack([(uv[..., 0] - cx) / fx * depth, (uv[..., 1] - cy) / fy * depth, depth, torch.ones_like(depth)], -1)
    return (cam_pts @ cam.world_view_transform.inverse())[..., :3]


def project(points, cam):
    """(..., 3) world points -> (..., 2) pixel coords and (...) camera depth."""
    fx, fy, cx, cy = _intrinsics(cam)
    p = torch.cat([points, torch.ones_like(points[..., :1])], -1) @ cam.world_view_transform
    z = p[..., 2]
    zs = z.clamp_min(1e-6)
    return torch.stack([p[..., 0] / zs * fx + cx, p[..., 1] / zs * fy + cy], -1), z


def _to_grid(uv, H, W):
    """Pixel coords -> grid_sample coords (align_corners=False)."""
    return torch.stack([(2 * uv[..., 0] + 1) / W - 1, (2 * uv[..., 1] + 1) / H - 1], -1)


@torch.no_grad()
def reproject_masks(cameras, depths, masks, present, threshold=0.01, chunk=16):
    """Consensus masks.

    cameras: list of N cameras (same resolution); depths: (N, H, W) CUDA, 0 = no surface;
    masks: (N, C, H, W) CUDA; present: (N, C) bool.
    Returns (N, C, H, W) float32 CPU (0 where no view could vote).
    """
    N, C, H, W = masks.shape
    world = torch.stack([backproject(d, c) for d, c in zip(depths, cameras)])     # (N, H, W, 3)
    pix = _to_grid(_pixel_coords(H, W), H, W)                                      # (H, W, 2)
    limit = 1 - 1 / torch.tensor([W, H], device="cuda")
    inside = lambda g: (g.abs() < limit).all(-1)
    out = torch.zeros((N, C, H, W), dtype=torch.float32)
    for i, cur in enumerate(cameras):
        votes = torch.zeros((C, H, W), device="cuda")
        count = torch.zeros((C, H, W), device="cuda")
        for s in range(0, N, chunk):
            refs = cameras[s:s + chunk]
            fwd, back, ok = [], [], []
            for r, ref in enumerate(refs):
                uv, z = project(world[i], ref)                                    # cur pixels seen from ref
                uv_back, _ = project(world[s + r], cur)                            # ref pixels seen from cur
                fwd.append(_to_grid(uv, H, W))
                back.append(_to_grid(uv_back, H, W))
                ok.append((z > 1e-4) & (depths[i] > 0))
            fwd, back, ok = torch.stack(fwd), torch.stack(back), torch.stack(ok)
            back = back.masked_fill((~inside(back) | (depths[s:s + len(refs)] <= 0))[..., None], 10.0)
            round_trip = F.grid_sample(back.permute(0, 3, 1, 2), fwd, mode="nearest", align_corners=False).permute(0, 2, 3, 1)
            valid = ok & inside(fwd) & ((round_trip - pix).norm(dim=-1) < threshold)  # (B, H, W)
            warped = F.grid_sample(masks[s:s + len(refs)].float(), fwd, mode="nearest", align_corners=False)
            votes += (warped * valid[:, None]).sum(0)
            count += (valid[:, None] & present[s:s + len(refs), :, None, None].cuda()).sum(0)
        out[i] = torch.where(count > 0, votes / count.clamp_min(1), torch.zeros_like(votes)).cpu()
    return out
