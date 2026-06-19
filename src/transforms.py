import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Overlay pipeline transforms — no ImageNet normalization (raw [0,1] pixels)
# ---------------------------------------------------------------------------

def get_overlay_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


def get_overlay_val_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# BayarConv — learnable constrained high-pass filter
# ---------------------------------------------------------------------------

class BayarConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = kernel_size // 2

    def _constrained_weights(self):
        w = self.weight.clone()
        # zero the center, normalize the rest to sum to 1, then set center to -1
        w[:, :, self.center, self.center] = 0
        # sum over spatial dims per (out, in) pair
        s = w.sum(dim=(2, 3), keepdim=True)
        # avoid division by zero
        s = s + (s == 0).float() * 1e-8
        w = w / s
        w[:, :, self.center, self.center] = -1
        return w

    def forward(self, x):
        return F.conv2d(x, self._constrained_weights(), self.bias, padding=self.padding)


# ---------------------------------------------------------------------------
# SRM fixed-filter bank — 3 standard forensic kernels, not learnable
# ---------------------------------------------------------------------------

class SRMConv2d(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # 3 SRM kernels (5x5)
        srm1 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        srm2 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  1, -4,  1,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        srm3 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float32) / 12.0

        kernels = np.stack([srm1, srm2, srm3])  # (3, 5, 5)
        # tile across input channels: (3*in_channels, in_channels, 5, 5)
        weight = np.zeros((3 * in_channels, in_channels, 5, 5), dtype=np.float32)
        for i, k in enumerate(kernels):
            for c in range(in_channels):
                weight[i * in_channels + c, c] = k

        self.out_channels = 3 * in_channels
        self.register_buffer("weight", torch.from_numpy(weight))
        self.register_buffer("bias", torch.zeros(self.out_channels))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=2)
