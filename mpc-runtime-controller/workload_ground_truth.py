#!/usr/bin/env python3
"""Workload Ground Truth Generation for YOLOv8n on Configurable Accelerators.

This module establishes the ground truth for MAC calculations and resolution tiers:
1. Analytically profiles the YOLOv8n execution graph via PyTorch forward hooks.
2. Verifies static Conv2d MAC counts for resolutions R1 (320x192), R2 (480x288), R3 (640x384).
3. Verifies Layer 0 (model.0.conv) MAC count and fraction.
4. Generates a reproducible per-frame, per-resolution MAC table (1,212 rows).
5. Generates a resolution-tier label dataset (404 frames) with explicitly documented proxy labels.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO

# Standard Resolution Configurations
RESOLUTIONS = {
    "R1_320x192": {"width": 320, "height": 192, "label": "R1", "tier_idx": 0},
    "R2_480x288": {"width": 480, "height": 288, "label": "R2", "tier_idx": 1},
    "R3_640x384": {"width": 640, "height": 384, "label": "R3", "tier_idx": 2},
}

# Baseline file paths
PROJECT_ROOT = Path("C:/AI accelerator project")
WEIGHTS_PATH = PROJECT_ROOT / "yolov8n.pt"
COLAB_PER_FRAME_CSV = (
    PROJECT_ROOT
    / "results"
    / "colab_experiment"
    / "extracted"
    / "colab_yolov8n_per_frame_latency.csv"
)
TRACES_DIR = PROJECT_ROOT / "traces"
TRACES_REAL_YOLO_DIR = TRACES_DIR / "real_yolo"


def compute_layer_by_layer_conv2d_macs(
    model: nn.Module, input_size: Tuple[int, int]
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """Profile Conv2d MACs layer by layer using PyTorch forward hooks.

    Args:
        model: PyTorch model module.
        input_size: (width, height) tuple.

    Returns:
        total_macs: Total multiply-accumulate operations across all Conv2d layers.
        layer0_macs: MACs in the first Conv2d layer (model.0.conv).
        layer0_fraction: layer0_macs / total_macs.
        layer_details: List of per-layer profiling dictionaries.
    """
    w, h = input_size
    layer_details: List[Dict[str, Any]] = []
    total_macs = 0
    layer0_macs = 0
    hooks = []

    def make_hook(name: str):
        def hook(module: nn.Conv2d, inp: Tuple[torch.Tensor], outp: torch.Tensor):
            nonlocal total_macs, layer0_macs
            x = inp[0]
            _, cin, hin, win = x.shape
            _, cout, hout, wout = outp.shape
            kh, kw = module.kernel_size
            groups = module.groups
            # Standard Conv2d MACs = H_out * W_out * C_out * (C_in / groups) * K_h * K_w
            macs = hout * wout * cout * (cin // groups) * kh * kw
            total_macs += macs
            if len(layer_details) == 0:
                layer0_macs = macs
            layer_details.append(
                {
                    "layer_name": name,
                    "layer_idx": len(layer_details),
                    "cin": cin,
                    "cout": cout,
                    "hin": hin,
                    "win": win,
                    "hout": hout,
                    "wout": wout,
                    "kernel": (kh, kw),
                    "groups": groups,
                    "macs": macs,
                    "gmac": macs / 1e9,
                }
            )

        return hook

    # Register forward hooks on all nn.Conv2d modules
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, h, w, device=device)

    with torch.no_grad():
        _ = model(dummy)

    # Clean up hooks
    for h_hook in hooks:
        h_hook.remove()

    layer0_fraction = layer0_macs / total_macs if total_macs > 0 else 0.0
    return total_macs, layer0_macs, layer0_fraction, layer_details


def verify_static_mac_budgets(model_path: Path) -> Dict[str, Dict[str, Any]]:
    """Verify Conv2d MACs against the YOLOv8n PyTorch model graph."""
    yolo = YOLO(str(model_path))
    net = yolo.model
    net.eval()

    results = {}
    for res_name, cfg in RESOLUTIONS.items():
        w, h = cfg["width"], cfg["height"]
        total_macs, l0_macs, l0_frac, details = compute_layer_by_layer_conv2d_macs(net, (w, h))
        results[res_name] = {
            "width": w,
            "height": h,
            "total_macs": total_macs,
            "total_gmac": total_macs / 1e9,
            "layer0_macs": l0_macs,
            "layer0_gmac": l0_macs / 1e9,
            "layer0_fraction": l0_frac,
            "num_conv2d_layers": len(details),
        }
    return results


def build_reproducible_mac_table(
    colab_csv_path: Path, mac_budgets: Dict[str, Dict[str, Any]], output_csv: Path
) -> pd.DataFrame:
    """Build the reproducible 1,212-row per-frame, per-resolution MAC table."""
    df_raw = pd.read_csv(colab_csv_path)

    records = []
    for _, row in df_raw.iterrows():
        res_label = str(row["resolution_label"])
        budget = mac_budgets[res_label]

        records.append(
            {
                "frame_id": int(row["frame_id"]),
                "scene_id": str(row["scene_id"]),
                "scene_split": str(row["scene_split"]),
                "environment_tag": str(row["environment_tag"]),
                "frame_in_scene": int(row["frame_in_scene"]),
                "resolution_label": res_label,
                "input_width": budget["width"],
                "input_height": budget["height"],
                "conv2d_macs": budget["total_macs"],
                "conv2d_gmac": round(budget["total_gmac"], 4),
                "layer0_macs": budget["layer0_macs"],
                "layer0_gmac": round(budget["layer0_gmac"], 6),
                "layer0_mac_fraction": round(budget["layer0_fraction"], 6),
                "forward_repeat_raw_mean_ms": round(float(row["forward_repeat_mean_ms"]), 4),
                "forward_repeat_std_ms": round(float(row["forward_repeat_std_ms"]), 4),
                "compute_only_ms": round(float(row["compute_only_ms"]), 4),
                "final_detection_count": int(row["final_detection_count"]),
                "mean_detection_conf": (
                    round(float(row["mean_detection_conf"]), 4)
                    if pd.notna(row["mean_detection_conf"])
                    else 0.0
                ),
                "mac_provenance": "ANALYTICAL_GRAPH_HOOKS",
                "timing_provenance": "MEASURED_TESLA_T4",
            }
        )

    df_out = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return df_out


def build_resolution_tier_dataset(
    colab_csv_path: Path, mac_budgets: Dict[str, Dict[str, Any]], output_csv: Path
) -> pd.DataFrame:
    """Generate resolution-tier labels for the 404 unique frames.

    Target classes:
      0 = R1_320x192 (0.6558 GMAC) - Low-power tier
      1 = R2_480x288 (1.4755 GMAC) - Balanced tier
      2 = R3_640x384 (2.6232 GMAC) - High-resolution / High-demand tier

    Labeling methodology:
      We use relative detection retention against R3 as a clear proxy label.
      If R1 retains >= 80% of R3 detections (or R3 <= 2 total detections): Tier 0 (R1).
      Else if R2 retains >= 80% of R3 detections: Tier 1 (R2).
      Else (demanding complex scene with small objects): Tier 2 (R3).

    NOTE: This is explicitly labeled as PROXY_DETECTION_RETENTION.
    """
    df_raw = pd.read_csv(colab_csv_path)

    # Extract image preview features from existing real_yolo scene traces
    preview_features: Dict[Tuple[str, int], Dict[str, float]] = {}
    for scene_file in sorted(TRACES_REAL_YOLO_DIR.glob("scene-*.csv")):
        scene_df = pd.read_csv(scene_file)
        for _, row in scene_df.iterrows():
            key = (str(row["sequence_id"]), int(row["frame_id"]))
            preview_features[key] = {
                "brightness_mean": float(row["brightness_mean"]),
                "brightness_std": float(row["brightness_std"]),
                "edge_density": float(row["edge_density"]),
                "histogram_entropy": float(row["histogram_entropy"]),
                "temporal_difference": float(row["temporal_difference"]),
                "aspect_ratio": float(row["aspect_ratio"]),
                "normalized_pixel_count": float(row["normalized_pixel_count"]),
            }

    # Group by frame_id
    grouped = df_raw.groupby("frame_id")
    tier_records = []

    for fid, group in grouped:
        first = group.iloc[0]
        scene_id = str(first["scene_id"])
        split = str(first["scene_split"])
        env_tag = str(first["environment_tag"])

        # Extract detection counts and confidences for each resolution
        r1_row = group[group["resolution_label"] == "R1_320x192"].iloc[0]
        r2_row = group[group["resolution_label"] == "R2_480x288"].iloc[0]
        r3_row = group[group["resolution_label"] == "R3_640x384"].iloc[0]

        d1 = int(r1_row["final_detection_count"])
        d2 = int(r2_row["final_detection_count"])
        d3 = int(r3_row["final_detection_count"])

        c1 = float(r1_row["mean_detection_conf"]) if pd.notna(r1_row["mean_detection_conf"]) else 0.0
        c2 = float(r2_row["mean_detection_conf"]) if pd.notna(r2_row["mean_detection_conf"]) else 0.0
        c3 = float(r3_row["mean_detection_conf"]) if pd.notna(r3_row["mean_detection_conf"]) else 0.0

        # Relative retention ratios
        ratio_r1 = (d1 / d3) if d3 > 0 else 1.0
        ratio_r2 = (d2 / d3) if d3 > 0 else 1.0

        # Proxy Tier Label Assignment
        # Tier 0 (R1): Simple scene (d3 <= 2 or R1 preserves >= 80% of R3 detections)
        # Tier 1 (R2): Moderate scene (R2 preserves >= 80% of R3 detections)
        # Tier 2 (R3): Demanding scene (requires maximum spatial resolution)
        if d3 <= 2 or ratio_r1 >= 0.80:
            tier_label = 0
            tier_name = "R1_320x192"
            tier_gmac = mac_budgets["R1_320x192"]["total_gmac"]
        elif ratio_r2 >= 0.80:
            tier_label = 1
            tier_name = "R2_480x288"
            tier_gmac = mac_budgets["R2_480x288"]["total_gmac"]
        else:
            tier_label = 2
            tier_name = "R3_640x384"
            tier_gmac = mac_budgets["R3_640x384"]["total_gmac"]

        feat = preview_features[(scene_id, int(first["frame_in_scene"]))]

        tier_records.append(
            {
                "frame_id": fid,
                "scene_id": scene_id,
                "scene_split": split,
                "environment_tag": env_tag,
                "r1_detections": d1,
                "r2_detections": d2,
                "r3_detections": d3,
                "r1_mean_conf": round(c1, 4),
                "r2_mean_conf": round(c2, 4),
                "r3_mean_conf": round(c3, 4),
                "r1_retention_ratio": round(ratio_r1, 4),
                "r2_retention_ratio": round(ratio_r2, 4),
                "proxy_tier_label": tier_label,
                "proxy_tier_name": tier_name,
                "proxy_tier_gmac": round(tier_gmac, 4),
                "label_provenance": "PROXY_DETECTION_RETENTION",
                "label_limitation_note": (
                    "Label derived from relative YOLO detection retention against R3; "
                    "does not represent ground-truth 2D bounding box mAP against nuScenes annotations."
                ),
                **{k: round(v, 6) for k, v in feat.items()},
            }
        )

    df_tier = pd.DataFrame(tier_records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_tier.to_csv(output_csv, index=False)
    return df_tier


def main():
    print("=" * 80)
    print("PHASE 1: MAC AND RESOLUTION-TIER GROUND TRUTH GENERATION")
    print("=" * 80)

    print(f"Loading weights from: {WEIGHTS_PATH}")
    mac_budgets = verify_static_mac_budgets(WEIGHTS_PATH)

    print("\n[VERIFICATION] YOLOv8n Static Conv2d MAC Budgets (PyTorch Forward Hooks):")
    for res_name, b in mac_budgets.items():
        print(
            f"  {res_name:12s} ({b['width']:3d}x{b['height']:3d}): "
            f"Total Conv2d MACs = {b['total_macs']:,} ({b['total_gmac']:.4f} GMAC), "
            f"Layer 0 MACs = {b['layer0_macs']:,} ({b['layer0_gmac']:.6f} GMAC, {b['layer0_fraction']*100:.4f}%)"
        )

    # Build reproducible per-frame MAC table
    mac_table_path = TRACES_DIR / "yolov8n_per_frame_mac_table.csv"
    print(f"\nBuilding reproducible MAC table -> {mac_table_path}")
    df_mac = build_reproducible_mac_table(COLAB_PER_FRAME_CSV, mac_budgets, mac_table_path)
    print(f"  Successfully wrote {len(df_mac)} rows across 3 resolutions.")

    # Build resolution tier dataset
    tier_dataset_path = TRACES_DIR / "yolov8n_resolution_tier_labels.csv"
    print(f"\nBuilding resolution tier label dataset -> {tier_dataset_path}")
    df_tier = build_resolution_tier_dataset(COLAB_PER_FRAME_CSV, mac_budgets, tier_dataset_path)
    print(f"  Successfully wrote {len(df_tier)} frame labels.")
    print("\nProxy Tier Distribution:")
    print(df_tier["proxy_tier_name"].value_counts())
    print("\nSplit Distribution by Tier:")
    print(pd.crosstab(df_tier["scene_split"], df_tier["proxy_tier_name"]))

    print("\nPhase 1 ground-truth generation complete.")


if __name__ == "__main__":
    main()
