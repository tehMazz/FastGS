import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


scenes = [
    "Barn",
    "Caterpillar",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck",
]

# Predefined 2-way split for parallel SLURM submission (3-3).
SCENE_GROUPS = {
    0: ["Barn", "Caterpillar", "Courthouse"],
    1: ["Ignatius", "Meetingroom", "Truck"],
}

# Vanilla FastGS config for TNT (no published per-scene hyperparameters).
# Downsample by 2 to match the CoMe / GaussianWrapping TNT convention.
SCENE_CONFIG = {scene: {"images": "images", "extra": ""} for scene in scenes}

COMMON_TRAIN_ARGS = (
    "--eval "
    "--densification_interval 100 "
    "--optimizer_type default "
    "--test_iterations 30000 "
    "-r 2"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Tanks & Temples benchmark for FastGS: train + render + metrics."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing one subdirectory per TNT scene.")
    parser.add_argument("--output_dir", type=str, default="./output/tnt",
                        help="Where to save trained models.")
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="CUDA device index.")
    parser.add_argument("--scenes_id", type=int, default=None, choices=sorted(SCENE_GROUPS.keys()),
                        help="Run only the subset of scenes from the predefined 2-way split "
                             "(0/1). If omitted, runs all scenes.")
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
        scene_output_dir = os.path.join(args.output_dir, scene_name)

        train_command = " ".join(filter(None, [
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            f"OAR_JOB_ID={scene_name}",
            "python train.py",
            f"-s {scene_data_dir}",
            f"-i {cfg['images']}",
            f"-m {scene_output_dir}",
            COMMON_TRAIN_ARGS,
            cfg["extra"],
        ]))

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
