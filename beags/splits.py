"""Train/test split used by BEA-GS (2dgs/scene/dataset_readers.py)."""
import os

# 3D-OVS test frames, keyed by scene folder name.
OVS_TEST_FRAMES = {
    "bed": ["00", "04", "10", "23", "30"],
    "bench": ["02", "05", "25", "27", "32"],
    "blue_sofa": ["03", "05", "13", "24", "27"],
    "covered_desk": ["00", "01", "11", "26", "29"],
    "lawn": ["01", "03", "09", "13", "29"],
    "office_desk": ["03", "07", "12", "14", "20"],
    "room": ["00", "04", "19", "25", "30"],
    "snacks": ["04", "08", "18", "26", "40"],
    "sofa": ["02", "04", "10", "15", "22"],
    "table": ["00", "02", "14", "26", "30"],
}


def split_cameras(cam_infos, path, eval, llffhold=8):
    """Every llffhold-th view + views named '*test*' go to test; 3D-OVS uses fixed frames."""
    scene = os.path.basename(os.path.normpath(path))
    if eval and "3DOVS" in path and scene in OVS_TEST_FRAMES:
        frames = set(OVS_TEST_FRAMES[scene])
        return ([c for c in cam_infos if c.image_name not in frames],
                [c for c in cam_infos if c.image_name in frames])
    if not eval:
        return ([c for c in cam_infos if "test" not in c.image_name],
                [c for c in cam_infos if "test" in c.image_name])
    train = [c for i, c in enumerate(cam_infos) if i % llffhold != 0 and "test" not in c.image_name]
    test = [c for i, c in enumerate(cam_infos) if i % llffhold == 0 or "test" in c.image_name]
    return train, test
