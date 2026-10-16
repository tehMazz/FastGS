import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
SCENE_DATASETS = {
    "cogen1_bakerh": "COGEN1",
    "cogen2_bakerh": "COGEN2",
    "compressor_bakerh": "Compressor",
    "g35_bakerh": "G35",
    "pcv_bakerh": "PCV",
    "ta_blue_motor_bakerh": "TA_BLUE_MOTOR",
    "ta_turbine_bakerh": "TA_TURBINE",
    "tta_bakerh": "TTA",
}


def score_views(counts, slackness=0.0):
    if counts.ndim != 2:
        raise ValueError(f"Expected [classes, gaussians] counts, got {tuple(counts.shape)}")
    raw = counts.float()
    totals = raw.sum(dim=0)
    shares = raw / totals.clamp_min(torch.finfo(raw.dtype).eps)
    shares[:, totals == 0] = 0

    if raw.shape[0] == 2:
        assignment = F.normalize(raw, dim=0)
        assignment[0] += slackness
        labels = assignment.argmax(dim=0)
        masks = torch.stack([labels == class_index for class_index in range(2)])
        return raw, shares, assignment, masks

    assignment = torch.zeros_like(raw)
    masks = torch.zeros_like(raw, dtype=torch.bool)
    for class_index in range(raw.shape[0]):
        binary_scores = torch.stack((shares.sum(dim=0) - shares[class_index], shares[class_index]))
        if slackness != 0:
            binary_scores = F.normalize(binary_scores, dim=0)
            binary_scores[0] += slackness
        assignment[class_index] = binary_scores[1]
        masks[class_index] = binary_scores.argmax(dim=0) == 1
    return raw, shares, assignment, masks


def test_score_views():
    counts = torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 2.0]])
    raw, shares, assignment, masks = score_views(counts)
    assert torch.equal(raw, counts)
    assert torch.allclose(shares[:, 0], torch.tensor([3 / 7, 4 / 7]))
    assert torch.equal(shares[:, 1], torch.zeros(2))
    assert torch.allclose(assignment[:, 0], torch.tensor([0.6, 0.8]))
    assert torch.equal(masks[:, 0], torch.tensor([False, True]))
    multi = torch.tensor([[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
    _, _, _, multi_masks = score_views(multi)
    assert torch.equal(multi_masks[:, 0], torch.tensor([False, True, False]))
    assert not multi_masks[:, 1].any()
    print("FastGS score self-test passed.")


def load_scene(model_path, iteration, source_path, resolution):
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    dataset = SimpleNamespace(
        model_path=str(model_path),
        source_path=str(source_path),
        sh_degree=3,
        images="images_1600",
        eval=True,
        white_background=False,
        resolution=resolution,
        data_device="cuda",
    )
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return dataset, gaussians, scene


def load_count_file(path, device="cpu"):
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"Expected a tensor shaped [classes, gaussians] in {path}")
    return value.to(device=device, dtype=torch.float32)


def write_plots(output_dir, xyz, raw, shares, assignment, masks, max_points):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir.mkdir(parents=True, exist_ok=True)
    count = raw.shape[1]
    indices = np.linspace(0, count - 1, min(count, max_points), dtype=np.int64)
    xyz_np = xyz.detach().cpu().numpy()
    metrics = (
        ("raw", raw, "Accumulated alpha contribution"),
        ("share", shares, "Normalized class share"),
        ("assignment", assignment, "FlashSplat assignment score"),
    )
    for class_index in range(raw.shape[0]):
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=("Raw contribution by Gaussian", "Normalized share by Gaussian", "Assignment score by Gaussian", "Score distributions"),
            vertical_spacing=0.08,
        )
        class_mask = masks[class_index].detach().cpu().numpy()
        for row, (metric, values, label) in enumerate(metrics, start=1):
            val = values[class_index].detach().cpu().numpy()
            fig.add_trace(go.Bar(
                x=indices,
                y=val[indices],
                name=metric,
                marker_color=np.where(class_mask[indices], "#d45b36", "#2878a5"),
                hovertemplate=f"Gaussian %{{x}}<br>{label}: %{{y:.6g}}<extra></extra>",
            ), row=row, col=1)
            fig.update_xaxes(title_text="Gaussian index", row=row, col=1)
            fig.update_yaxes(title_text=label, row=row, col=1)
        for metric, values, _ in metrics:
            fig.add_trace(go.Histogram(
                x=values[class_index].detach().cpu().numpy(),
                name=metric,
                opacity=0.55,
                nbinsx=80,
            ), row=4, col=1)
        fig.update_layout(
            title=f"Class {class_index}: used-count scores (orange bars are assigned)",
            barmode="overlay",
            height=1250,
            template="plotly_white",
            legend={"orientation": "h", "y": -0.08},
        )
        fig.write_html(output_dir / f"class_{class_index:03d}_distributions.html", include_plotlyjs=True)

        for metric, values, label in metrics:
            score = values[class_index].detach().cpu().numpy()
            color_max = max(float(np.percentile(score, 99)), np.finfo(np.float32).eps) if metric == "raw" else 1.0
            cloud = go.Figure(data=go.Scatter3d(
                x=xyz_np[indices, 0],
                y=xyz_np[indices, 1],
                z=xyz_np[indices, 2],
                mode="markers",
                customdata=np.column_stack((indices, class_mask[indices], score[indices])),
                marker={
                    "size": 2,
                    "opacity": 0.85,
                    "color": np.clip(score[indices], 0, color_max),
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": color_max,
                    "showscale": True,
                    "colorbar": {"title": metric},
                },
                hovertemplate=f"Gaussian %{{customdata[0]}}<br>{label}: %{{customdata[2]:.6g}}<br>Assigned: %{{customdata[1]}}<extra></extra>",
            ))
            cloud.update_layout(
                title=f"Class {class_index}: 3D centers colored by {label.lower()}",
                scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
                template="plotly_white",
            )
            cloud.write_html(output_dir / f"class_{class_index:03d}_{metric}_3d.html", include_plotlyjs=True)


def make_colors(values, metric):
    import matplotlib

    scores = values.detach().float().cpu().numpy()
    upper = max(float(np.percentile(scores, 99)), np.finfo(np.float32).eps) if metric == "raw" else 1.0
    normalized = np.clip(scores / upper, 0.0, 1.0)
    rgb = matplotlib.colormaps["viridis"](normalized)[..., :3]
    return torch.as_tensor(rgb, dtype=torch.float32, device=values.device), upper


def render_views(args, gaussians, scene, score_views, masks, output_dir):
    from gaussian_renderer import render_fastgs

    views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if not views:
        raise ValueError(f"No {args.split} cameras available")
    selected_views = np.linspace(0, len(views) - 1, min(args.render_views, len(views)), dtype=np.int64)
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    metrics = (("raw", score_views[0]), ("share", score_views[1]), ("assignment", score_views[2]))
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    with torch.no_grad():
        rgb_views = {}
        for view_index in selected_views:
            rgb_render = render_fastgs(
                views[int(view_index)], gaussians, pipeline, bg_color, args.mult,
            )["render"]
            rgb_views[int(view_index)] = (
                rgb_render.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
            ).round().astype(np.uint8)

        for metric, values in metrics:
            for class_index in range(values.shape[0]):
                mask = masks[class_index].to(device=gaussians.get_xyz.device)
                if not mask.any():
                    continue
                all_colors, upper = make_colors(values[class_index].to(device=gaussians.get_xyz.device), metric)
                subset = SimpleNamespace(
                    get_xyz=gaussians.get_xyz[mask],
                    get_opacity=gaussians.get_opacity[mask],
                    get_scaling=gaussians.get_scaling[mask],
                    get_rotation=gaussians.get_rotation[mask],
                    active_sh_degree=gaussians.active_sh_degree,
                )
                class_dir = output_dir / metric / f"class_{class_index:03d}"
                class_dir.mkdir(parents=True, exist_ok=True)
                for view_index in selected_views:
                    rendered = render_fastgs(
                        views[int(view_index)], subset, pipeline, bg_color, args.mult,
                        override_color=all_colors[mask],
                    )["render"]
                    score_rgb = (rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    paired = np.concatenate((rgb_views[int(view_index)], score_rgb), axis=1)
                    Image.fromarray(paired).save(class_dir / f"view_{int(view_index):04d}.png")
                fig, axis = plt.subplots(figsize=(5, 0.7))
                sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=upper), cmap="viridis")
                bar = fig.colorbar(sm, cax=axis, orientation="horizontal")
                label = "Accumulated alpha contribution" if metric == "raw" else metric.replace("_", " ").title()
                bar.set_label(label)
                fig.savefig(class_dir / "colorbar.png", dpi=160, bbox_inches="tight")
                plt.close(fig)
                (class_dir / "color_scale.json").write_text(json.dumps({
                    "metric": metric,
                    "class_index": class_index,
                    "upper_color_limit": upper,
                    "assigned_gaussians": int(mask.sum().item()),
                    "color_map": "viridis",
                    "render_layout": "full RGB render left; score-colored class render right",
                }, indent=2))


def process_count(args, scene_name, source_path, count_path):
    from plyfile import PlyData

    model_path = ROOT / "output" / "flashsplat" / scene_name
    checkpoints = sorted(
        model_path.glob("point_cloud/iteration_*/point_cloud.ply"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No PLY checkpoints found under {model_path / 'point_cloud'}")
    ply_path = checkpoints[0]
    iteration = int(ply_path.parent.name.split("_")[-1])
    vertex = PlyData.read(str(ply_path))["vertex"]
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32)
    counts = load_count_file(count_path)
    if counts.shape[1] != xyz.shape[0]:
        raise ValueError(
            f"{count_path} has {counts.shape[1]} Gaussian columns but {scene_name} iteration {iteration} has "
            f"{xyz.shape[0]} points"
        )
    raw, shares, assignment, masks = score_views(counts, args.slackness)
    output_dir = ROOT / "output" / "flashsplat_debug" / scene_name / f"ours_{iteration}" / args.split / count_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "scores.npz",
        raw=raw.cpu().numpy(),
        class_share=shares.cpu().numpy(),
        assignment_score=assignment.cpu().numpy(),
        assigned=masks.cpu().numpy(),
        xyz=xyz,
    )
    write_plots(output_dir / "plots", torch.from_numpy(xyz), raw, shares, assignment, masks, args.max_plot_gaussians)
    if args.render_views:
        if not torch.cuda.is_available():
            raise RuntimeError("FastGS camera rendering requires CUDA")
        dataset, gaussians, scene = load_scene(model_path, iteration, source_path, args.resolution)
        render_views(args, gaussians, scene, (raw, shares, assignment), masks, output_dir / "renders")
        del scene, gaussians
        torch.cuda.empty_cache()
    summary = {
        "scene": scene_name,
        "count_file": str(count_path),
        "iteration": iteration,
        "split": args.split,
        "classes": int(raw.shape[0]),
        "gaussians": int(raw.shape[1]),
        "count_semantics": "accumulated alpha * transmittance contribution by class and Gaussian",
        "slackness": args.slackness,
        "assigned_per_class": [int(mask.sum().item()) for mask in masks],
        "outputs": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {scene_name} ({count_path.name}) to {output_dir}")
    del counts


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FastGS FlashSplat used-count tensors.")
    parser.add_argument("--scene", choices=sorted(SCENE_DATASETS))
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--count-file", type=Path)
    parser.add_argument("--slackness", type=float, default=0.0)
    parser.add_argument("--render-views", type=int, default=1)
    parser.add_argument("--max-plot-gaussians", type=int, default=50000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        test_score_views()
        return
    if args.all_scenes == (args.scene is not None):
        raise ValueError("Choose exactly one of --scene SCENE or --all-scenes")
    if args.render_views < 0 or args.max_plot_gaussians < 1:
        raise ValueError("render-views must be non-negative and max-plot-gaussians positive")
    scenes = sorted(SCENE_DATASETS) if args.all_scenes else [args.scene]
    for scene_name in scenes:
        model_path = ROOT / "output" / "flashsplat" / scene_name
        source_path = ROOT / "datasets" / "BakerH_undis" / SCENE_DATASETS[scene_name]
        count_dir = model_path / args.split / "ours_30000" / "used_count"
        count_files = [args.count_file] if args.count_file else sorted(count_dir.glob("*.pth"))
        if not count_files:
            print(f"Skipping {scene_name}: no used_count .pth files in {count_dir}")
            continue
        for count_path in count_files:
            process_count(args, scene_name, source_path, count_path)


if __name__ == "__main__":
    main()