"""Shared helpers for the beags_* scripts: argument parsing, model loading, file layout.

Layout inside a FastGS model dir:
    point_cloud/iteration_<it>/{point_cloud.ply, labels.pt}
    beags/masks/<image_name>.npy      consensus masks (float16, C x H x W)
    beags/labeled_points.npz          xyz (N, 3), labels (N,)
    beags/metrics_<it>.json
"""
import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def parse_args(description, add_args):
    """FastGS args (merged with <model>/cfg_args) + script-specific ones -> dataset, pipe, opt, args."""
    parser = ArgumentParser(description=description)
    model, pipeline, optim = ModelParams(parser, sentinel=True), PipelineParams(parser), OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    add_args(parser)
    args = get_combined_args(parser)
    safe_state(args.quiet)
    return model.extract(args), pipeline.extract(args), optim.extract(args), args


def load_model(dataset, iteration, with_labels):
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    if with_labels:
        gaussians.labels = torch.load(labels_path(dataset.model_path, scene.loaded_iter), map_location="cuda").int()
    return scene, gaussians


def labels_path(model_path, iteration):
    return os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "labels.pt")


def beags_path(model_path, name):
    """<model>/beags/<name>; creates <model>/beags."""
    os.makedirs(os.path.join(model_path, "beags"), exist_ok=True)
    return os.path.join(model_path, "beags", name)
