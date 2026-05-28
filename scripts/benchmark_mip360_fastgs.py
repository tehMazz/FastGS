import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


scenes = [
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
]

# Predefined 3-way split for parallel SLURM submission (3-2-2).
SCENE_GROUPS = {
    0: ["bicycle", "bonsai", "counter"],
    1: ["garden", "kitchen"],
    2: ["room", "stump"],
}

# Per-scene FastGS "big" hyperparameters, lifted verbatim from train_big.sh.
# Image subdir is "images" (full resolution) for all scenes, as the per-scene
# grad_abs_thresh / dense / highfeature_lr values were tuned at full res.
SCENE_CONFIG = {
    "bicycle": {"images": "images", "extra": "--grad_abs_thresh 0.0008"},
    "bonsai":  {"images": "images", "extra": "--highfeature_lr 0.02 --grad_abs_thresh 0.0002"},
    "counter": {"images": "images", "extra": "--highfeature_lr 0.02 --grad_abs_thresh 0.0004"},
    "garden":  {"images": "images", "extra": "--highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0003"},
    "kitchen": {"images": "images", "extra": "--highfeature_lr 0.02 --grad_abs_thresh 0.0002"},
    "room":    {"images": "images", "extra": "--highfeature_lr 0.02 --grad_abs_thresh 0.0004"},
    "stump":   {"images": "images", "extra": "--dense 0.004 --grad_abs_thresh 0.001"},
}

COMMON_TRAIN_ARGS = (
    "--eval "
    "--densification_interval 100 "
    "--optimizer_type default "
    "--test_iterations 30000"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end MipNeRF 360 benchmark for FastGS: train + render + metrics."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing one subdirectory per MipNeRF 360 scene.")
    parser.add_argument("--output_dir", type=str, default="./output/mip360",
                        help="Where to save trained models.")
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="CUDA device index.")
    parser.add_argument("--scenes_id", type=int, default=None, choices=sorted(SCENE_GROUPS.keys()),
                        help="Run only the subset of scenes from the predefined 3-way split "
                             "(0/1/2). If omitted, runs all scenes.")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated explicit scene list (overrides --scenes_id).")
    args = parser.parse_args()

    if args.scenes:
        scene_subset = [s.strip() for s in args.scenes.split(",") if s.strip()]
    elif args.scenes_id is not None:
        scene_subset = SCENE_GROUPS[args.scenes_id]
    else:
        scene_subset = scenes

    print(f"[INFO] Running {len(scene_subset)} scene(s): {scene_subset}")

    for scene_name in scene_subset:
        print(f"\n[INFO] ===== {scene_name} =====")

        if scene_name not in SCENE_CONFIG:
            print(f"[WARN] No SCENE_CONFIG entry for '{scene_name}', skipping.")
            continue

        cfg = SCENE_CONFIG[scene_name]
        scene_data_dir = os.path.join(args.data_dir, scene_name)
        scene_output_dir = os.path.join(args.output_dir, f"{scene_name}_big")

        train_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            f"OAR_JOB_ID={scene_name}_big",
            "python train.py",
            f"-s {scene_data_dir}",
            f"-i {cfg['images']}",
            f"-m {scene_output_dir}",
            COMMON_TRAIN_ARGS,
            cfg["extra"],
        ])

        print("\n[INFO] Running train command:", train_command, sep="\n")
        os.system(train_command)

        render_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python render.py",
            f"-m {scene_output_dir}",
            "--skip_train",
        ])

        print("\n[INFO] Running render command:", render_command, sep="\n")
        os.system(render_command)

        metrics_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python metrics.py",
            f"-m {scene_output_dir}",
        ])

        print("\n[INFO] Running metrics command:", metrics_command, sep="\n")
        os.system(metrics_command)
