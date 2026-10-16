"""BEA-GS label extension of the FastGS rasterizer. Run on a GPU node:
    pytest -q tests/test_rasterizer_labels.py
Optional stock comparison: set STOCK_DUMP / PATCHED_DUMP to files written by tests/dump_render.py.
"""
import os

import pytest
import torch

from gaussian_renderer import render_depth_alpha, render_fastgs
from tests.synthetic import PIPE, as_model, make_camera, make_gaussians

BG = torch.tensor([0.1, 0.2, 0.3])


def _render(params, deg, cam, **kw):
    return render_fastgs(cam, as_model(params, deg), PIPE, BG.cuda(), 0.5, **kw)


def _split_labels(cam, n_gauss, num_labels=3, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gl = torch.randint(0, num_labels, (n_gauss,), generator=g, device="cuda", dtype=torch.int32)
    pl = torch.randint(-1, num_labels, (cam.image_height, cam.image_width), generator=g, device="cuda", dtype=torch.int32)
    return gl, pl


@pytest.mark.skipif(not (os.environ.get("STOCK_DUMP") and os.environ.get("PATCHED_DUMP")), reason="no dumps")
def test_matches_stock_build():
    stock, patched = torch.load(os.environ["STOCK_DUMP"]), torch.load(os.environ["PATCHED_DUMP"])
    assert stock.keys() == patched.keys()
    for k in stock:
        if "grad" in k:  # float atomics: order-dependent
            torch.testing.assert_close(patched[k], stock[k], rtol=1e-4, atol=1e-6, msg=k)
        else:
            assert torch.equal(patched[k], stock[k]), k


def test_no_labels_returns_none():
    cam = make_camera()
    params, deg = make_gaussians()
    out = _render(params, deg, cam)
    assert out["sag"] is None and out["votes"] is None


def test_rgb_and_grads_unchanged_by_labels():
    cam = make_camera()
    params, deg = make_gaussians()
    gl, pl = _split_labels(cam, params["xyz"].shape[0])
    grads = []
    for kw in ({}, dict(gaussian_labels=gl, pixel_labels=pl, num_labels=3)):
        for p in params.values():
            p.grad = None
        img = _render(params, deg, cam, **kw)["render"]
        img.sum().backward()
        grads.append((img.detach(), {k: p.grad.clone() for k, p in params.items()}))
    assert torch.equal(grads[0][0], grads[1][0])
    for k in params:
        torch.testing.assert_close(grads[1][1][k], grads[0][1][k], rtol=1e-4, atol=1e-6)


def test_votes_sum_to_alpha_per_label():
    cam = make_camera()
    params, deg = make_gaussians()
    gl, pl = _split_labels(cam, params["xyz"].shape[0])
    with torch.no_grad():
        votes = _render(params, deg, cam, gaussian_labels=gl, pixel_labels=pl, num_labels=3)["votes"]
        _, alpha = render_depth_alpha(cam, as_model(params, deg), PIPE, 0.5)
    for label in range(3):
        torch.testing.assert_close(votes[:, label].sum(), alpha[pl == label].sum(), rtol=1e-4, atol=1e-3)


def test_sag_limits():
    cam = make_camera()
    params, deg = make_gaussians()
    n = params["xyz"].shape[0]
    _, pl = _split_labels(cam, n)
    with torch.no_grad():
        _, alpha = render_depth_alpha(cam, as_model(params, deg), PIPE, 0.5)
        all_wrong = torch.full((n,), -1, dtype=torch.int32, device="cuda")
        sag = _render(params, deg, cam, gaussian_labels=all_wrong, pixel_labels=pl, num_labels=3)["sag"][0]
        torch.testing.assert_close(sag, alpha * (pl >= 0), rtol=1e-4, atol=1e-5)

        same = torch.ones(n, dtype=torch.int32, device="cuda")
        ones = torch.ones_like(pl)
        sag = _render(params, deg, cam, gaussian_labels=same, pixel_labels=ones, num_labels=3)["sag"]
        assert torch.count_nonzero(sag) == 0


def test_sag_gradient_mismatch_only():
    """All pixels have label 1. Label-1 Gaussians are matched (zero SAG grad);
    for label-0 Gaussians the mismatch-only gradient is the exact derivative."""
    cam = make_camera(width=32, height=32)
    params, deg = make_gaussians(n=6, seed=3)
    with torch.no_grad():
        params["xyz"][:, :2] *= 0.15                       # stack them in view
        params["scaling"].fill_(torch.log(torch.tensor(0.15)).item())
    gl = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int32, device="cuda")
    pl = torch.ones(32, 32, dtype=torch.int32, device="cuda")
    kw = dict(gaussian_labels=gl, pixel_labels=pl, num_labels=2)

    def sag_sum():
        return _render(params, deg, cam, **kw)["sag"].double().sum()

    params["opacity"].grad = None
    sag_sum().backward()
    analytic = params["opacity"].grad[:, 0].clone()
    assert torch.all(analytic[gl == 1] == 0)

    eps = 1e-2
    for i in torch.nonzero(gl == 0).flatten().tolist():
        with torch.no_grad():
            params["opacity"][i] += eps
            up = sag_sum()
            params["opacity"][i] -= 2 * eps
            down = sag_sum()
            params["opacity"][i] += eps
        numeric = ((up - down) / (2 * eps)).float()
        torch.testing.assert_close(analytic[i], numeric, rtol=2e-2, atol=1e-2)
