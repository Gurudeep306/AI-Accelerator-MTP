"""
models/detect_head.py
-----------------------
Anchor-free, decoupled detection head (YOLOX/YOLOv8-style): for each
feature-map scale, a small conv stack splits into a classification
branch and a regression branch. No anchors -- each grid cell directly
predicts:
  - cls_logits: (num_classes,)      one-vs-rest logits
  - obj_logit:  (1,)                "is there an object here" logit
  - reg:        (4,)                raw (l, t, r, b) distances to the
                                     box edges, in stride units, decoded
                                     as exp(raw) * stride (always positive)

This is intentionally simpler than the real YOLOv9's DFL-based
regression ("from-scratch with slight modification" -- direct ltrb
regression trains fine at small/tiny model scale and is far easier to
get correct without a live environment to test against).

Each anytime "tap" (Head1..Head3, Final) gets its OWN DetectHead
instance -- i.e. its own trainable weights -- since it must learn to
produce a useful prediction from whatever depth of feature processing
has completed by that point (brief section 21: intermediate heads
should also learn useful detections, not just copy the final head).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .common import Conv


class ScaleHead(nn.Module):
    """One scale's (e.g. P3) cls/obj/reg branches."""

    def __init__(self, c_in, num_classes, c_mid=None):
        super().__init__()
        c_mid = c_mid or c_in
        self.cls_stem = nn.Sequential(Conv(c_in, c_mid, 3), Conv(c_mid, c_mid, 3))
        self.reg_stem = nn.Sequential(Conv(c_in, c_mid, 3), Conv(c_mid, c_mid, 3))
        self.cls_pred = nn.Conv2d(c_mid, num_classes, 1)
        self.reg_pred = nn.Conv2d(c_mid, 4, 1)
        self.obj_pred = nn.Conv2d(c_mid, 1, 1)

        # Initialize obj/cls biases so training starts from "mostly
        # background" (standard detector trick -- avoids early loss blowup).
        nn.init.constant_(self.obj_pred.bias, -4.595)  # sigmoid(-4.6) ~= 0.01
        nn.init.constant_(self.cls_pred.bias, -4.595)

    def forward(self, x):
        cls_feat = self.cls_stem(x)
        reg_feat = self.reg_stem(x)
        cls_logits = self.cls_pred(cls_feat)
        obj_logit = self.obj_pred(reg_feat)
        reg_raw = self.reg_pred(reg_feat)
        return cls_logits, obj_logit, reg_raw


class DetectHead(nn.Module):
    """Wraps one ScaleHead per FPN level (P3, P4, P5) for a single tap."""

    def __init__(self, in_channels_list: Sequence[int], num_classes: int, strides: Sequence[int]):
        super().__init__()
        assert len(in_channels_list) == len(strides)
        self.num_classes = num_classes
        self.strides = list(strides)
        self.scale_heads = nn.ModuleList([ScaleHead(c, num_classes) for c in in_channels_list])

    def forward(self, feats: Sequence[torch.Tensor]):
        """Returns a list (per scale) of (cls_logits, obj_logit, reg_raw),
        each shaped (B, C, H, W)."""
        outs = []
        for feat, head in zip(feats, self.scale_heads):
            outs.append(head(feat))
        return outs


@torch.no_grad()
def make_grid(h: int, w: int, stride: int, device, dtype):
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    return cx, cy  # each (H, W)


def decode_scale(cls_logits, obj_logit, reg_raw, stride: int, reg_max_ratio: float = 8.0):
    """Decode one scale's raw predictions into (boxes_xyxy, scores, cls_ids)
    per batch item, in ORIGINAL image pixel coordinates.

    reg_raw -> ltrb distances = exp(clamp(reg_raw)) * stride. Clamping the
    exponent avoids numerical blow-up early in training/inference.
    """
    b, c, h, w = cls_logits.shape
    cx, cy = make_grid(h, w, stride, cls_logits.device, cls_logits.dtype)

    max_exp = float(torch.log(torch.tensor(reg_max_ratio)).item() + 2.0)
    reg_raw = torch.clamp(reg_raw, min=-4.0, max=max_exp)
    ltrb = torch.exp(reg_raw) * stride  # (B, 4, H, W)
    l, t, r, btm = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]

    x1 = cx.unsqueeze(0) - l
    y1 = cy.unsqueeze(0) - t
    x2 = cx.unsqueeze(0) + r
    y2 = cy.unsqueeze(0) + btm
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, H, W, 4)

    obj = torch.sigmoid(obj_logit).squeeze(1)            # (B, H, W)
    cls_prob = torch.sigmoid(cls_logits)                 # (B, C, H, W)
    cls_score, cls_id = cls_prob.max(dim=1)              # (B, H, W) each
    score = obj * cls_score                              # (B, H, W)

    return boxes, score, cls_id


def greedy_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """Simple, dependency-free per-class-agnostic greedy NMS (torchvision-free,
    so this has no extra install requirement). boxes: (N,4) xyxy, scores: (N,)."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)
