#!/usr/bin/env python3
"""Focused tests for Milestone B real DNN preprocessing, inference recording,
dataset schema integrity, missing-data handling, and causal predictor splitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from milestone_b_pilot_dataset import (
    load_cam_front_keyframes,
    select_smoke_test_frames,
)

WEIGHTS_PATH = Path("results/milestone_b/weights/yolov8n.pt")
PILOT_CSV_PATH = Path("real_dnn_pilot_measurements.csv")

EXPECTED_PILOT_COLUMNS = [
    "frame_id", "scene_id", "scene_split", "environment_tag", "frame_in_scene",
    "sample_token", "sample_data_token", "timestamp_us", "scene_elapsed_ms",
    "dt_from_prev_ms", "image_path", "orig_width", "orig_height",
    "input_width", "input_height", "letterbox_scale", "pad_top_px", "pad_bottom_px",
    "model_name", "model_version", "device", "precision", "num_threads",
    "conf_thres", "iou_thres", "image_load_ms", "preprocess_ms", "inference_ms",
    "postprocess_ms", "compute_only_ms", "total_pipeline_ms", "repeat_count",
    "inference_repeat_mean_ms", "inference_repeat_median_ms", "inference_repeat_std_ms",
    "postprocess_repeat_median_ms", "static_params", "static_macs", "static_flops_2x",
    "measured_activation_sparsity", "raw_anchor_count", "candidates_before_nms",
    "final_detection_count", "mean_detection_conf", "max_detection_conf",
    "detected_class_counts", "cpu_percent_proc", "rss_memory_mb", "rss_delta_mb",
    "preview_extract_ms", "preview_brightness_mean", "preview_brightness_std",
    "preview_edge_density", "preview_histogram_entropy", "preview_temporal_difference",
]


def test_pilot_dataset_preserves_scenes_and_2hz_timestamps():
    records = load_cam_front_keyframes(frames_per_scene=10)
    assert len(records) == 100
    assert len({r.scene_id for r in records}) == 10
    dts = [r.dt_from_prev_ms for r in records if r.dt_from_prev_ms is not None]
    assert len(dts) == 90
    assert all(350.0 <= dt <= 650.0 for dt in dts)
    for r in records:
        assert Path(r.image_path).is_file()
        assert (r.meta_width, r.meta_height) == (1600, 900)
        if r.frame_in_scene == 0:
            assert r.dt_from_prev_ms is None
            assert r.scene_elapsed_ms == 0.0


def test_preprocessing_letterbox_640x384_preserves_aspect_ratio():
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    from milestone_b_static_profiler import preprocess_bgr_image

    # Synthetic 1600x900 RGB test image (constant blue in BGR)
    img_bgr = np.zeros((900, 1600, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = 200
    tensor, meta = preprocess_bgr_image(img_bgr, target_hw=(384, 640))

    assert tensor.shape == (1, 3, 384, 640)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert meta["orig_width"] == 1600 and meta["orig_height"] == 900
    assert meta["input_width"] == 640 and meta["input_height"] == 384
    assert meta["letterbox_scale"] == pytest.approx(0.4, rel=1e-6)
    assert meta["pad_top_px"] == 12 and meta["pad_bottom_px"] == 12
    assert meta["pad_left_px"] == 0 and meta["pad_right_px"] == 0

    # Verify top gray border equals 114/255 and center RGB matches channel conversion
    expected_pad = 114.0 / 255.0
    assert float(tensor[0, 0, 0, 0]) == pytest.approx(expected_pad, abs=1e-3)
    assert float(tensor[0, 2, 192, 320]) == pytest.approx(200.0 / 255.0, abs=1e-3)
    assert cv2.__version__ is not None


def test_model_load_postprocess_empty_and_repeated_reproducibility():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    if not WEIGHTS_PATH.is_file():
        pytest.skip("Pretrained weights not present")

    from milestone_b_pilot_inference import (
        configure_cpu_environment,
        execute_single_frame_stages,
        run_single_postprocess,
    )
    from milestone_b_static_profiler import load_yolo_weights

    configure_cpu_environment(num_threads=4)
    yolo = load_yolo_weights(WEIGHTS_PATH, fuse=True)
    assert sum(p.numel() for p in yolo.model.parameters()) == 3151904

    # Missing-detection empty-output handling
    zero_raw = torch.zeros((1, 84, 5040), dtype=torch.float32)
    empty_post = run_single_postprocess(
        zero_raw, (384, 640), (900, 1600), yolo.names, 0.25, 0.45, 300
    )
    assert empty_post["raw_anchor_count"] == 5040
    assert empty_post["candidates_before_nms"] == 0
    assert empty_post["final_detection_count"] == 0
    assert empty_post["mean_detection_conf"] is None
    assert empty_post["detected_class_counts"] == {}

    # Repeated-run reproducibility on a real pilot image
    sample_frame = select_smoke_test_frames(load_cam_front_keyframes(frames_per_scene=1), 1)[0]
    rec1, _, _ = execute_single_frame_stages(yolo.model, sample_frame.image_path, yolo.names)
    rec2, _, _ = execute_single_frame_stages(yolo.model, sample_frame.image_path, yolo.names)

    assert rec1["candidates_before_nms"] == rec2["candidates_before_nms"]
    assert rec1["final_detection_count"] == rec2["final_detection_count"]
    assert rec1["detected_class_counts"] == rec2["detected_class_counts"]
    assert rec1["mean_detection_conf"] == rec2["mean_detection_conf"]
    for k in ("image_load_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "total_pipeline_ms"):
        assert rec1[k] >= 0.0 and rec2[k] >= 0.0


def test_measurement_recording_schema_and_no_fabricated_values():
    if not PILOT_CSV_PATH.is_file():
        pytest.skip("real_dnn_pilot_measurements.csv not yet generated")

    df = pd.read_csv(PILOT_CSV_PATH, keep_default_na=True)
    assert list(df.columns) == EXPECTED_PILOT_COLUMNS
    assert len(df) == 100
    assert df["scene_id"].nunique() == 10

    # Static MACs & parameters must be invariant across fixed-shape frames
    assert df["static_params"].nunique() == 1
    assert df["static_macs"].nunique() == 1
    assert int(df["static_macs"].iloc[0]) == 2623196160
    assert int(df["raw_anchor_count"].iloc[0]) == 5040

    # Missing activation sparsity must remain explicitly missing (NaN), never fabricated
    assert df["measured_activation_sparsity"].isna().all()

    # Frame 0 of each scene must have missing dt_from_prev_ms; t>=1 must be ~2 Hz (~500 ms)
    f0 = df[df["frame_in_scene"] == 0]
    assert len(f0) == 10 and f0["dt_from_prev_ms"].isna().all()
    f_rest = df[df["frame_in_scene"] >= 1]
    assert f_rest["dt_from_prev_ms"].notna().all()
    assert ((f_rest["dt_from_prev_ms"] >= 350.0) & (f_rest["dt_from_prev_ms"] <= 650.0)).all()

    # Every row timing & count must be valid
    assert (df["total_pipeline_ms"] >= df["compute_only_ms"] - 1e-3).all()
    assert (df["candidates_before_nms"] >= df["final_detection_count"]).all()
    for raw_json in df["detected_class_counts"]:
        parsed = json.loads(raw_json)
        assert isinstance(parsed, dict)


def test_causal_lag_features_prevent_cross_scene_and_future_leakage():
    pytest.importorskip("sklearn")
    from milestone_b_workload_analysis import add_causal_lag_features

    toy = pd.DataFrame({
        "scene_id": ["s1", "s1", "s1", "s2", "s2"],
        "frame_in_scene": [0, 1, 2, 0, 1],
        "inference_ms": [10.0, 20.0, 30.0, 100.0, 200.0],
        "postprocess_ms": [1.0, 2.0, 3.0, 4.0, 5.0],
        "candidates_before_nms": [5, 6, 7, 50, 60],
        "final_detection_count": [1, 2, 3, 10, 20],
    })
    lagged = add_causal_lag_features(toy, "inference_ms")

    # Start of each scene must have NaN lags (never leaked from prior scene!)
    assert pd.isna(lagged.loc[lagged["scene_id"].eq("s1") & lagged["frame_in_scene"].eq(0), "lag1_target"]).all()
    assert pd.isna(lagged.loc[lagged["scene_id"].eq("s2") & lagged["frame_in_scene"].eq(0), "lag1_target"]).all()

    # Check exact lag1 within s2 at t=1 is 100.0 (not s1's 30.0 or future 200.0)
    s2_t1 = lagged[lagged["scene_id"].eq("s2") & lagged["frame_in_scene"].eq(1)].iloc[0]
    assert s2_t1["lag1_target"] == pytest.approx(100.0)
    assert s2_t1["lag1_candidates"] == pytest.approx(50.0)
