"""
models/gelan_anytime.py
--------------------------
A from-scratch, small ("tiny"-scale) YOLOv9/GELAN-*style* detector with
FOUR anytime detection taps, matching the project brief's narrative:

    Head 1 (shallow)    -- straight off the backbone, before any FPN/PAN fusion
    Head 2 (deeper)     -- after top-down FPN fusion
    Head 3 (near-final) -- after bottom-up PAN fusion
    Final  (complete)   -- after one further refinement pass

Every tap is a real, independently-trained DetectHead (own weights) --
this is deep supervision repurposed for anytime inference, exactly as
AnytimeYOLO's paper (Sec. 4) describes, but implemented from scratch
here rather than by modifying the DKuhse/anytime-yolo checkout (which I
had no network access to clone -- see README).

CRITICAL: forward() computes ALL FOUR taps in one pass and returns all
of them -- it never returns early. Early/anytime *use* of these results
(saving to ResultManager, honoring interrupts) is the job of
`inference/gelan_adapter.py`, not this file. This file is pure model
architecture.

Architecture (channels chosen to comfortably fit a 4GB GTX 1650 at
small batch sizes and 640x640 input):

    Input (3, H, W)
      -> stem: Conv s2 (3->16), Conv s2 (16->32), ELAN(32->32)      stride 4
      -> Down s2 (32->64),  ELAN(64->64)   = P3_bb                   stride 8
      -> Down s2 (64->96),  ELAN(96->96)   = P4_bb                   stride 16
      -> Down s2 (96->128), ELAN(128->128), SPPF(128->128) = P5_bb   stride 32

    ---- Head 1 taps directly here: (P3_bb, P4_bb, P5_bb) ----

    Top-down FPN:
      P5_td = P5_bb
      P4_td = ELAN(concat(Up(reduce(P5_td)), P4_bb))
      P3_td = ELAN(concat(Up(reduce(P4_td)), P3_bb))

    ---- Head 2 taps here: (P3_td, P4_td, P5_td) ----

    Bottom-up PAN:
      P3_bu = ELAN(P3_td)                                  (self-refine, no fusion partner)
      P4_bu = ELAN(concat(Down(P3_bu), P4_td))
      P5_bu = ELAN(concat(Down(P4_bu), P5_td))

    ---- Head 3 taps here: (P3_bu, P4_bu, P5_bu) ----

    Final refinement:
      P3_f = ELAN(P3_bu); P4_f = ELAN(P4_bu); P5_f = ELAN(P5_bu)

    ---- Final head taps here: (P3_f, P4_f, P5_f) ----
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .common import Conv, ELANBlock, Down, Up, SPPF
from .detect_head import DetectHead

# Channels at P3/P4/P5 (kept identical across all 4 taps, so every
# DetectHead has the same input-channel signature).
P_CHANNELS = (64, 96, 128)
STRIDES = (8, 16, 32)


@dataclass
class HeadOutput:
    head_id: int
    head_name: str
    is_final: bool
    # per scale: (cls_logits, obj_logit, reg_raw), matching STRIDES order
    scale_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class GelanAnytimeModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.strides = STRIDES

        # ---- Stem ----
        self.stem = nn.Sequential(
            Conv(3, 16, 3, 2),
            Conv(16, 32, 3, 2),
            ELANBlock(32, 32, c_mid=16, n=2),
        )

        # ---- Backbone pyramid ----
        self.down_p3 = Down(32, 64)
        self.elan_p3_bb = ELANBlock(64, 64, c_mid=32, n=2)

        self.down_p4 = Down(64, 96)
        self.elan_p4_bb = ELANBlock(96, 96, c_mid=48, n=2)

        self.down_p5 = Down(96, 128)
        self.elan_p5_bb = ELANBlock(128, 128, c_mid=64, n=2)
        self.sppf_p5 = SPPF(128, 128)

        # ---- Top-down FPN ----
        self.reduce_p5_to_p4 = Conv(128, 96, 1)
        self.up1 = Up()
        self.elan_p4_td = ELANBlock(96 + 96, 96, c_mid=48, n=2)

        self.reduce_p4_to_p3 = Conv(96, 64, 1)
        self.up2 = Up()
        self.elan_p3_td = ELANBlock(64 + 64, 64, c_mid=32, n=2)

        # ---- Bottom-up PAN ----
        self.elan_p3_bu = ELANBlock(64, 64, c_mid=32, n=1)  # self-refine, no fusion partner
        self.down_p3_bu = Down(64, 96)
        self.elan_p4_bu = ELANBlock(96 + 96, 96, c_mid=48, n=2)

        self.down_p4_bu = Down(96, 128)
        self.elan_p5_bu = ELANBlock(128 + 128, 128, c_mid=64, n=2)

        # ---- Final refinement ----
        self.elan_p3_final = ELANBlock(64, 64, c_mid=32, n=1)
        self.elan_p4_final = ELANBlock(96, 96, c_mid=48, n=1)
        self.elan_p5_final = ELANBlock(128, 128, c_mid=64, n=1)

        # ---- Four independent detection heads (own weights each) ----
        self.head1 = DetectHead(P_CHANNELS, num_classes, STRIDES)
        self.head2 = DetectHead(P_CHANNELS, num_classes, STRIDES)
        self.head3 = DetectHead(P_CHANNELS, num_classes, STRIDES)
        self.head_final = DetectHead(P_CHANNELS, num_classes, STRIDES)

        self.head_names = ["early_head_1", "early_head_2", "early_head_3", "final"]

    # ------------------------------------------------------------------
    # Incremental sub-methods -- these are what let the REAL anytime
    # adapter (inference/gelan_adapter.py) time each tap separately and
    # correctly, instead of timing one monolithic forward() call. Each
    # method does genuinely more work than the previous one, and their
    # outputs chain together exactly like forward() below does.
    # ------------------------------------------------------------------
    def forward_backbone(self, x: torch.Tensor):
        s = self.stem(x)
        p3_bb = self.elan_p3_bb(self.down_p3(s))
        p4_bb = self.elan_p4_bb(self.down_p4(p3_bb))
        p5_bb = self.sppf_p5(self.elan_p5_bb(self.down_p5(p4_bb)))
        return p3_bb, p4_bb, p5_bb

    def forward_topdown(self, p3_bb, p4_bb, p5_bb):
        p5_td = p5_bb
        p4_td = self.elan_p4_td(torch.cat([self.up1(self.reduce_p5_to_p4(p5_td)), p4_bb], dim=1))
        p3_td = self.elan_p3_td(torch.cat([self.up2(self.reduce_p4_to_p3(p4_td)), p3_bb], dim=1))
        return p3_td, p4_td, p5_td

    def forward_bottomup(self, p3_td, p4_td, p5_td):
        p3_bu = self.elan_p3_bu(p3_td)
        p4_bu = self.elan_p4_bu(torch.cat([self.down_p3_bu(p3_bu), p4_td], dim=1))
        p5_bu = self.elan_p5_bu(torch.cat([self.down_p4_bu(p4_bu), p5_td], dim=1))
        return p3_bu, p4_bu, p5_bu

    def forward_final_refine(self, p3_bu, p4_bu, p5_bu):
        p3_f = self.elan_p3_final(p3_bu)
        p4_f = self.elan_p4_final(p4_bu)
        p5_f = self.elan_p5_final(p5_bu)
        return p3_f, p4_f, p5_f

    def forward(self, x: torch.Tensor) -> List[HeadOutput]:
        """ONE forward pass, computed via the incremental sub-methods
        above (so training and the anytime adapter always compute
        IDENTICAL feature maps -- no drift between the two code paths).
        Computes and returns all four taps' raw (cls_logits, obj_logit,
        reg_raw) per scale. Never exits early."""
        outputs: List[HeadOutput] = []

        p3_bb, p4_bb, p5_bb = self.forward_backbone(x)
        outputs.append(HeadOutput(
            head_id=1, head_name=self.head_names[0], is_final=False,
            scale_outputs=self.head1([p3_bb, p4_bb, p5_bb]),
        ))

        p3_td, p4_td, p5_td = self.forward_topdown(p3_bb, p4_bb, p5_bb)
        outputs.append(HeadOutput(
            head_id=2, head_name=self.head_names[1], is_final=False,
            scale_outputs=self.head2([p3_td, p4_td, p5_td]),
        ))

        p3_bu, p4_bu, p5_bu = self.forward_bottomup(p3_td, p4_td, p5_td)
        outputs.append(HeadOutput(
            head_id=3, head_name=self.head_names[2], is_final=False,
            scale_outputs=self.head3([p3_bu, p4_bu, p5_bu]),
        ))

        p3_f, p4_f, p5_f = self.forward_final_refine(p3_bu, p4_bu, p5_bu)
        outputs.append(HeadOutput(
            head_id=4, head_name=self.head_names[3], is_final=True,
            scale_outputs=self.head_final([p3_f, p4_f, p5_f]),
        ))

        return outputs


def build_model(num_classes: int) -> GelanAnytimeModel:
    return GelanAnytimeModel(num_classes)
