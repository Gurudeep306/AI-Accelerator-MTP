#!/usr/bin/env python3
"""Unit tests for Phase 1 MAC ground truth and resolution tier generation."""
import pytest
from pathlib import Path
import pandas as pd
from workload_ground_truth import (
    RESOLUTIONS,
    verify_static_mac_budgets,
    WEIGHTS_PATH,
    TRACES_DIR,
)

def test_static_mac_verification():
    """Verify that YOLOv8n Conv2d MACs match the theoretical model graph."""
    budgets = verify_static_mac_budgets(WEIGHTS_PATH)

    # R1: 320x192
    assert budgets["R1_320x192"]["total_macs"] == 655_799_040
    assert budgets["R1_320x192"]["layer0_macs"] == 6_635_520
    assert abs(budgets["R1_320x192"]["layer0_fraction"] - 0.010118) < 1e-5

    # R2: 480x288
    assert budgets["R2_480x288"]["total_macs"] == 1_475_547_840
    assert budgets["R2_480x288"]["layer0_macs"] == 14_929_920
    assert abs(budgets["R2_480x288"]["layer0_fraction"] - 0.010118) < 1e-5

    # R3: 640x384
    assert budgets["R3_640x384"]["total_macs"] == 2_623_196_160
    assert budgets["R3_640x384"]["layer0_macs"] == 26_542_080
    assert abs(budgets["R3_640x384"]["layer0_fraction"] - 0.010118) < 1e-5

def test_reproducible_mac_table_structure():
    """Verify the generated per-frame MAC table integrity."""
    mac_table_path = TRACES_DIR / "yolov8n_per_frame_mac_table.csv"
    assert mac_table_path.exists()

    df = pd.read_csv(mac_table_path)
    assert len(df) == 1212
    assert df["frame_id"].nunique() == 404
    assert set(df["resolution_label"].unique()) == {"R1_320x192", "R2_480x288", "R3_640x384"}
    assert (df["mac_provenance"] == "ANALYTICAL_GRAPH_HOOKS").all()

def test_resolution_tier_dataset_structure():
    """Verify the generated resolution tier label dataset integrity."""
    tier_path = TRACES_DIR / "yolov8n_resolution_tier_labels.csv"
    assert tier_path.exists()

    df = pd.read_csv(tier_path)
    assert len(df) == 404
    assert set(df["proxy_tier_label"].unique()) == {0, 1, 2}
    assert set(df["proxy_tier_name"].unique()) == {"R1_320x192", "R2_480x288", "R3_640x384"}
    assert (df["label_provenance"] == "PROXY_DETECTION_RETENTION").all()
