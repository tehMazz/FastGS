from collections import OrderedDict
from pathlib import Path

import torch

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    weight_path = _WEIGHTS_DIR / f"v{version}" / f"{net_type}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"LPIPS linear weights not found at {weight_path}. "
            f"Download them on a login node with:\n"
            f"  python -c \"import torch; torch.hub.load_state_dict_from_url("
            f"'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
            f"master/lpips/weights/v{version}/{net_type}.pth')\"\n"
            f"then copy ~/.cache/torch/hub/checkpoints/{net_type}.pth "
            f"to {weight_path}"
        )

    old_state_dict = torch.load(weight_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
