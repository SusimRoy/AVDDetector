import re
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, L1Loss
from torch.optim import Adam
from torchmetrics.classification import AUROC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, RandomRotation, RandomAffine
from torchvision.transforms import ColorJitter, RandomGrayscale, RandomPerspective, RandomErasing
from torchvision.transforms import RandomRotation, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomGrayscale, RandomPerspective, RandomErasing
from torchvision.transforms import RandomRotation, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip
from lightning.pytorch import Callback, Trainer, LightningModule
from abc import ABC

class _ConvNd(Module, ABC):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
        build_activation: Optional[Callable] = None
    ):
        super().__init__()
        self.conv = self.PtConv(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        if build_activation is not None:
            self.activation = build_activation()
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv1d(_ConvNd):
    PtConv = torch.nn.Conv1d


class Conv2d(_ConvNd):
    PtConv = torch.nn.Conv2d


class Conv3d(_ConvNd):
    PtConv = torch.nn.Conv3d