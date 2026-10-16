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

import os
import shutil
import subprocess
import time
from argparse import ArgumentParser
from os import makedirs
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_fastgs
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


FRAMES_BETWEEN = 5
OUTPUT_WIDTH = 960


def _interpolate_rotation(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation, Slerp

        key_rots = Rotation.from_matrix(np.stack([r0, r1], axis=0))
        slerp = Slerp([0.0, 1.0], key_rots)
        return slerp([t]).as_matrix()[0].astype(np.float32)
    except Exception:
        # Fallback when scipy is not available: lerp then project to SO(3).
        r_lerp = (1.0 - t) * r0 + t * r1
        u, _, vt = np.linalg.svd(r_lerp)
        r_ortho = u @ vt
        if np.linalg.det(r_ortho) < 0:
            u[:, -1] *= -1.0
            r_ortho = u @ vt
        return r_ortho.astype(np.float32)


def _build_interpolated_view(src_view, dst_view, t: float):
    r_interp = _interpolate_rotation(np.asarray(src_view.R), np.asarray(dst_view.R), t)
    t_interp = ((1.0 - t) * np.asarray(src_view.T) + t * np.asarray(dst_view.T)).astype(np.float32)

    trans = np.asarray(getattr(src_view, "trans", np.array([0.0, 0.0, 0.0])), dtype=np.float32)
    scale = float(getattr(src_view, "scale", 1.0))

    world_view_transform = torch.tensor(
        getWorld2View2(r_interp, t_interp, trans, scale), dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=float(getattr(src_view, "znear", 0.01)),
        zfar=float(getattr(src_view, "zfar", 100.0)),
        fovX=float(src_view.FoVx),
        fovY=float(src_view.FoVy),
    ).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    )

    return SimpleNamespace(
        FoVx=float(src_view.FoVx),
        FoVy=float(src_view.FoVy),
        image_width=int(src_view.image_width),
        image_height=int(src_view.image_height),
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=world_view_transform.inverse()[3, :3],
    )


def _collect_sorted_key_views(scene: Scene, skip_train: bool, skip_test: bool):
    key_views = []
    if not skip_train:
        key_views.extend(scene.getTrainCameras())
    if not skip_test:
        key_views.extend(scene.getTestCameras())

    key_views.sort(key=lambda cam: getattr(cam, "image_name", ""))

    return key_views


def _build_trajectory_views(key_views, frames_between: int):
    if len(key_views) == 0:
        raise ValueError("No cameras selected. Disable at least one of --skip_train/--skip_test.")

    trajectory = []
    for idx, view in enumerate(key_views):
        trajectory.append(view)
        if idx == len(key_views) - 1:
            continue

        next_view = key_views[idx + 1]
        for step in range(1, frames_between + 1):
            t = step / float(frames_between + 1)
            trajectory.append(_build_interpolated_view(view, next_view, t))

    return trajectory


def _start_ffmpeg_writer(input_width: int, input_height: int, video_path: str, fps: int):
    if OUTPUT_WIDTH <= 0:
        raise ValueError(f"OUTPUT_WIDTH must be > 0, got {OUTPUT_WIDTH}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise FileNotFoundError("ffmpeg is required for MP4 output but was not found in PATH.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(input_width)}x{int(input_height)}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        f"scale={OUTPUT_WIDTH}:-2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        video_path,
    ]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _finalize_ffmpeg_writer(proc: subprocess.Popen, video_path: str):
    if proc.stdin is not None and not proc.stdin.closed:
        proc.stdin.close()

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while encoding {video_path}:\n{stderr.decode('utf-8', errors='ignore')}")


def _render_to_uint8_frame(view, gaussians, pipeline, background, mult):
    rendering = render_fastgs(view, gaussians, pipeline, background, mult)["render"]
    frame_rgb = (
        rendering[0:3]
        .detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return frame_rgb


def _fit_frame_to_shape(frame_rgb: np.ndarray, expected_shape):
    target_h, target_w, target_c = expected_shape
    if frame_rgb.shape[2] != target_c:
        raise ValueError(
            f"Inconsistent channel count. Expected {target_c}, got {frame_rgb.shape[2]}"
        )

    src_h, src_w = frame_rgb.shape[:2]

    # Center-crop if current frame is larger than the target.
    if src_h > target_h:
        start_h = (src_h - target_h) // 2
        frame_rgb = frame_rgb[start_h:start_h + target_h, :, :]
        src_h = target_h
    if src_w > target_w:
        start_w = (src_w - target_w) // 2
        frame_rgb = frame_rgb[:, start_w:start_w + target_w, :]
        src_w = target_w

    # Symmetrically pad if current frame is smaller than the target.
    if src_h < target_h or src_w < target_w:
        pad_top = max((target_h - src_h) // 2, 0)
        pad_bottom = max(target_h - src_h - pad_top, 0)
        pad_left = max((target_w - src_w) // 2, 0)
        pad_right = max(target_w - src_w - pad_left, 0)
        frame_rgb = np.pad(
            frame_rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )

    return frame_rgb


def render_video_set(model_path, iteration, views, gaussians, pipeline, background, args):
    output_root = os.path.join(model_path, "video", "ours_{}".format(iteration))
    video_path = os.path.join(output_root, args.video_name)
    makedirs(output_root, exist_ok=True)

    if len(views) == 0:
        raise ValueError("No trajectory views to render.")

    total_time = 0.0
    ffmpeg_proc = None
    expected_shape = None
    try:
        for view in tqdm(views, desc="Rendering + encoding video"):
            start_time = time.time()
            frame_rgb = _render_to_uint8_frame(view, gaussians, pipeline, background, args.mult)
            end_time = time.time()
            total_time += (end_time - start_time)

            if ffmpeg_proc is None:
                frame_height, frame_width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
                ffmpeg_proc = _start_ffmpeg_writer(
                    input_width=frame_width,
                    input_height=frame_height,
                    video_path=video_path,
                    fps=args.fps,
                )
                expected_shape = frame_rgb.shape
            elif frame_rgb.shape != expected_shape:
                frame_rgb = _fit_frame_to_shape(frame_rgb, expected_shape)
                if frame_rgb.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent frame shape after normalization. Expected {expected_shape}, got {frame_rgb.shape}"
                    )

            if ffmpeg_proc.stdin is None:
                raise RuntimeError("ffmpeg stdin is not available for writing frame data.")
            ffmpeg_proc.stdin.write(frame_rgb.tobytes())
    finally:
        if ffmpeg_proc is not None:
            with tqdm(total=1, desc="Finalizing video file", unit="step") as finalize_bar:
                _finalize_ffmpeg_writer(ffmpeg_proc, video_path)
                finalize_bar.update(1)

    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0.0
    render_fps = 1.0 / avg_time if avg_time > 0 else 0.0
    print(f"[video] Rendered {num_frames} frames in {total_time:.2f}s. Avg render FPS: {render_fps:.2f}")
    print(f"[video] Output width fixed to {OUTPUT_WIDTH}px (aspect ratio preserved)")
    print(f"[video] Saved MP4: {video_path}")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        if args.interpolate_test_only:
            if skip_test:
                raise ValueError("--interpolate_test_only requires test cameras, but --skip_test was provided.")
            key_views = list(scene.getTestCameras())
            key_views.sort(key=lambda cam: getattr(cam, "image_name", ""))
            print("[video] Interpolating only between test cameras.")
        else:
            key_views = _collect_sorted_key_views(scene, skip_train=skip_train, skip_test=skip_test)

        trajectory_views = _build_trajectory_views(key_views, frames_between=args.frames_between)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_video_set(
            dataset.model_path,
            scene.loaded_iter,
            trajectory_views,
            gaussians,
            pipeline,
            background,
            args,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Video rendering with interpolated train/test cameras")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video_name", type=str, default="trajectory.mp4")
    parser.add_argument(
        "--frames_between",
        type=int,
        default=FRAMES_BETWEEN,
        help="Number of interpolated frames inserted between consecutive key cameras.",
    )
    parser.add_argument(
        "--interpolate_test_only",
        action="store_true",
        help="Build interpolation trajectory only from test cameras.",
    )
    args = get_combined_args(parser)

    if args.frames_between < 0:
        raise ValueError("--frames_between must be >= 0")

    print("Rendering video " + args.model_path)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args,
    )
