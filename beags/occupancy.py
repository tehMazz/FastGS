"""Labelled-voxel occupancy loss (BEA-GS `get_sampling_loss` + utils/occ_utils.py).

Samples drawn inside each object Gaussian (+-3 sigma) are penalised by opacity * kernel value
when no voxel of the same label is within the 27-voxel neighbourhood. Only voxels whose
points all carry one label ("pure" voxels) count.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.general_utils import build_rotation

_BIAS = 1 << 20  # keys pack 3 x 21 bits: each axis index must be in [-2^20, 2^20)


def _pack(ijk):
    ijk = ijk + _BIAS
    return (ijk[..., 0] << 42) | (ijk[..., 1] << 21) | ijk[..., 2]


def _neighbour_offsets():
    o = torch.tensor([-1, 0, 1], device="cuda")
    return torch.stack(torch.meshgrid(o, o, o, indexing="ij"), -1).reshape(27, 3)


def voxel_size(points, k=25, target=25, trim=0.05, max_queries=200_000, seed=0):
    """Cube edge expected to hold `target` points, from the median k-NN density (5% trimmed)."""
    pts = points.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    queries = pts[rng.choice(len(pts), min(len(pts), max_queries), replace=False)]
    radius = cKDTree(pts).query(queries, k=k, workers=-1)[0][:, -1]
    lo, hi = np.quantile(radius, [trim, 1 - trim])
    r = np.median(radius[(radius >= lo) & (radius <= hi)])
    density = k / ((4.0 / 3.0) * np.pi * r ** 3 + 1e-12)
    return float((target / (density + 1e-12)) ** (1 / 3))


@dataclass
class LabeledVoxels:
    keys: torch.Tensor    # (M,) int64 sorted keys of pure voxels
    labels: torch.Tensor  # (M,) int64 label of each voxel
    origin: torch.Tensor  # (1, 3)
    size: float

    @classmethod
    def build(cls, points, labels, size):
        origin = points.min(dim=0).values[None]
        keys = _pack(torch.floor((points - origin) / size).long())
        keys, order = torch.sort(keys)
        labels = labels[order].long()
        uniq, inverse = torch.unique_consecutive(keys, return_inverse=True)
        lo = torch.full_like(uniq, torch.iinfo(torch.int64).max).scatter_reduce(0, inverse, labels, "amin")
        hi = torch.full_like(uniq, torch.iinfo(torch.int64).min).scatter_reduce(0, inverse, labels, "amax")
        pure = lo == hi
        if not pure.any():
            raise ValueError("no single-label voxel: point cloud empty or voxel size too large")
        return cls(uniq[pure], lo[pure], origin, float(size))

    def contains(self, points, labels):
        """(N,) bool: a pure voxel with the same label lies in the 27-neighbourhood of each point."""
        ijk = torch.floor((points - self.origin) / self.size).long()
        keys = _pack(ijk[:, None, :] + _neighbour_offsets()[None])            # (N, 27)
        pos = torch.searchsorted(self.keys, keys).clamp_max(self.keys.numel() - 1)
        hit = (self.keys[pos] == keys) & (self.labels[pos] == labels[:, None].long())
        return hit.any(dim=1)


def occupancy_loss(gaussians, gaussian_labels, voxels, samples_per_gaussian=20, limit=3.0):
    """mean(opacity * G(x) * [x outside same-label voxels]) over samples of object Gaussians."""
    obj = gaussian_labels != 0
    if not obj.any():
        return gaussians.get_xyz.sum() * 0
    mu = gaussians.get_xyz[obj]
    rot = build_rotation(gaussians.get_rotation[obj])
    scale = gaussians.get_scaling[obj]
    gid = torch.arange(mu.shape[0], device="cuda").repeat_interleave(samples_per_gaussian)
    with torch.no_grad():
        u = (torch.rand(gid.numel(), 3, device="cuda") * 2 - 1) * limit
        x = mu[gid] + torch.einsum("nij,nj->ni", rot[gid], u * scale[gid])
        outside = ~voxels.contains(x, gaussian_labels[obj][gid])
    q = torch.einsum("nji,nj->ni", rot[gid], x - mu[gid]) / scale[gid]       # R^T (x - mu) / s
    kernel = torch.exp(-0.5 * (q ** 2).sum(dim=1))
    return (gaussians.get_opacity[obj][gid, 0] * kernel * outside).mean()
