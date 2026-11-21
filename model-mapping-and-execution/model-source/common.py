"""
models/common.py
------------------
Basic building blocks for the from-scratch GELAN-style backbone/neck:
- Conv: conv + BatchNorm + SiLU
- ELANBlock: the CSP-ELAN-like block used throughout YOLOv7/v9/GELAN --
  split into two branches, one goes through a stack of 3x3 convs, all
  intermediate outputs get concatenated and fused with a 1x1 "transition"
  conv. This is what the AnytimeYOLO paper calls a GELAN block.
- Down: strided 3x3 conv for downsampling (a real GELAN uses an avgpool+
  conv down block; a single strided conv is functionally equivalent and
  simpler to implement correctly from scratch).
- SPPF: spatial pyramid pooling (fast variant, 3x maxpool k=5 concatenated)
  used at the deepest backbone stage to enlarge the receptive field
  before the neck (this is the "SPP-ELAN" tap in the AnytimeYOLO paper).

All modules are standard nn.Module subclasses; nothing here is
AnytimeYOLO-repo-specific, so this has zero dependency on the real
DKuhse/anytime-yolo checkout -- it's a genuine from-scratch model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELANBlock(nn.Module):
    """Simplified GELAN/CSP-ELAN block.

    x -> cv1 (1x1) -> y1  \
    x -> cv2 (1x1) -> y2 --> [y2, conv(y2), conv(conv(y2)), ...] -> concat -> cv3 (1x1) -> out
    """

    def __init__(self, c_in, c_out, c_mid=None, n=2):
        super().__init__()
        c_mid = c_mid or c_out // 2
        self.cv1 = Conv(c_in, c_mid, 1)
        self.cv2 = Conv(c_in, c_mid, 1)
        self.blocks = nn.ModuleList([Conv(c_mid, c_mid, 3) for _ in range(n)])
        self.cv3 = Conv(c_mid * (2 + n), c_out, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        outs = [y1, y2]
        cur = y2
        for blk in self.blocks:
            cur = blk(cur)
            outs.append(cur)
        return self.cv3(torch.cat(outs, dim=1))


class Down(nn.Module):
    """Strided-conv downsample (stride 2, halves H and W)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = Conv(c_in, c_out, k=3, s=2)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Nearest-neighbor upsample (x2), no learnable params (matches YOLOv9's Up block)."""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.up(x)


class SPPF(nn.Module):
    """Fast SPP: 3x sequential maxpool(k=5) concatenated, then 1x1 fuse.
    Equivalent receptive field to SPP with kernels 5/9/13 but cheaper."""

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_mid = c_in // 2
        self.cv1 = Conv(c_in, c_mid, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_mid * 4, c_out, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
