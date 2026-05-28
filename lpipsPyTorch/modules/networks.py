from typing import Sequence
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .utils import normalize_activation

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

_WEIGHT_FILES = {
    "vgg16": "vgg16-397923af.pth",
    "alexnet": "alexnet-owt-7be5be79.pth",
    "squeezenet1_1": "squeezenet1_1-b8a52dc0.pth",
}


def _load_state_dict(model: nn.Module, name: str) -> nn.Module:
    weight_path = _WEIGHTS_DIR / _WEIGHT_FILES[name]
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=False)
    else:
        raise FileNotFoundError(
            f"Pretrained weights not found at {weight_path}. "
            f"Download them on a login node with:\n"
            f"  python -c \"from torchvision.models import {name}; {name}(pretrained=True)\"\n"
            f"then copy ~/.cache/torch/hub/checkpoints/{_WEIGHT_FILES[name]} "
            f"to {_WEIGHTS_DIR}/"
        )
    return model


def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = _load_state_dict(models.squeezenet1_1(weights=None), "squeezenet1_1").features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = _load_state_dict(models.alexnet(weights=None), "alexnet").features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = _load_state_dict(models.vgg16(weights=None), "vgg16").features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
