#!/usr/bin/env python3
"""Long-run deterministic stability test (180 frames across 4 dynamic phases).
Evaluates queue draining, host/accelerator utilization, and thermal stability.
"""
import pytest
import numpy as np
from typing import List
from unified_replay_harness import (
    FrameArrivalEvent, ReplaySimulator, FixedBaselineController,
    RuleBasedBaselineController, compute_summary_metrics
)
from practical_mpc_controller import MPCController, Constraints, Action
from history_mpc_controller import HistoryBasedMPC
from frame_preview_mpc import FramePreviewMPCPipeline, ProfilePerformancePredictor
from within_frame_mpc_controller import WithinFrameAdaptiveMPC, RemainingWorkloadEstimator


def generate_180_frame_trace(frame_rate_fps: float = 30.0,
                             deadline_ms: float = 50.0,
                             seed: int = 123) -> List[FrameArrivalEvent]:
    """Generates a 180-frame stress trace:
    Phase 1 (0-44): Nominal cruise
    Phase 2 (45-89): Sustained urban clutter
    Phase 3 (90-134): Extreme burst overload
    Phase 4 (135-179): Clear road recovery
    """
    np.random.seed(seed)
    events: List[FrameArrivalEvent] = []
    inter_frame_ms = 1000.0 / frame_rate_fps

    for fid in range(180):
        arr_ms = fid * inter_frame_ms

        if fid < 45:
            # Phase 1: Nominal
            base_scale = 0.85 + 0.10 * np.sin(fid / 3.0)
            edge_dens = 0.10 + np.random.uniform(0, 0.05)
            entropy = 0.50 + np.random.uniform(0, 0.05)
            temp_diff = 0.05 + np.random.uniform(0, 0.03)
        elif fid < 90:
            # Phase 2: Sustained urban clutter
            base_scale = 1.35 + 0.15 * np.sin(fid / 4.0)
            edge_dens = 0.25 + np.random.uniform(0, 0.08)
            entropy = 0.75 + np.random.uniform(0, 0.08)
            temp_diff = 0.15 + np.random.uniform(0, 0.05)
        elif fid < 135:
            # Phase 3: Extreme burst overload
            base_scale = 1.75 + np.random.uniform(-0.05, 0.15)
            edge_dens = 0.38 + np.random.uniform(0, 0.05)
            entropy = 0.90 + np.random.uniform(0, 0.04)
            temp_diff = 0.25 + np.random.uniform(0, 0.04)
        else:
            # Phase 4: Clear road recovery
            base_scale = 0.80 + 0.08 * np.cos(fid / 3.0)
            edge_dens = 0.09 + np.random.uniform(0, 0.04)
            entropy = 0.48 + np.random.uniform(0, 0.04)
            temp_diff = 0.04 + np.random.uniform(0, 0.02)

        wl_scale = max(0.5, float(base_scale + np.random.normal(0, 0.02)))
        features = {
            "brightness_mean": 0.5 + float(np.random.normal(0, 0.05)),
            "brightness_std": 0.2 + float(np.random.normal(0, 0.02)),
            "edge_density": float(edge_dens),
            "histogram_entropy": float(entropy),
            "temporal_difference": float(temp_diff),
            "aspect_ratio": 16.0 / 9.0,
            "normalized_pixel_count": 1.0
        }
        event = FrameArrivalEvent(
            frame_id=fid,
            arrival_time_ms=arr_ms,
            deadline_ms=deadline_ms,
            actual_workload_scale=wl_scale,
            first_layer_mac_fraction=0.15,
            image_features=features
        )
        events.append(event)

    return events


def test_180_frame_long_run_stability():
    """Verify stability across a 180-frame stress run:
    1. Queue stability & post-burst draining.
    2. Accelerator and Host utilization sanity.
    3. Forward causality and thermal evolution bounded.
    """
    trace = generate_180_frame_trace(frame_rate_fps=30.0, deadline_ms=50.0, seed=42)
    predictor = ProfilePerformancePredictor(
        profiles_csv="synthetic_calibrated_profiles.csv",
        allow_uncalibrated_quality=True
    )
    constraints = Constraints(
        deadline_ms=50.0,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        ambient_c=35.0,
        thermal_tau_s=27.0,
        thermal_resistance_c_per_w=2.0,
        horizon=5,
        beam_width=64
    )

    actions = sorted(predictor.profiles.keys())
    fallback_action = Action(0.50, 1, 4)
    nominal_act = Action(1.00, 3, 8)
    low_power_act = Action(0.50, 1, 4)

    core_mpc_a = MPCController(predictor, actions, fallback=fallback_action, constraints=constraints)
    ctrl_a = HistoryBasedMPC(core_mpc_a, cold_start_nominal_scale=1.0)

    ctrl_b = FramePreviewMPCPipeline(
        workload_model_path="synthetic_workload_model.json",
        profiles_csv="synthetic_calibrated_profiles.csv",
        constraints=constraints,
        fallback=fallback_action,
        require_thermal_calibration=False,
        allow_uncalibrated_quality=True
    )

    core_mpc_c = MPCController(predictor, actions, fallback=fallback_action, constraints=constraints)
    est_c = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(core_mpc_c, est_c, measured_switch_cost_mj=0.02)

    controllers = [
        ("controller_a", "Controller A (History)", ctrl_a),
        ("controller_b", "Controller B (Frame Preview)", ctrl_b),
        ("controller_c", "Controller C (Within-Frame)", ctrl_c),
        ("fixed_nominal", "Fixed Nominal (W1.00-E3-INT8)", FixedBaselineController(nominal_act, "Fixed Nominal (W1.00-E3-INT8)")),
        ("fixed_low_power", "Fixed Low-Power (W0.50-E1-INT4)", FixedBaselineController(low_power_act, "Fixed Low-Power (W0.50-E1-INT4)")),
        ("rule_based", "Rule-Based Baseline", RuleBasedBaselineController(actions, "Rule-Based Baseline"))
    ]

    for key, name, ctrl in controllers:
        sim = ReplaySimulator(predictor=predictor, constraints=constraints)
        results = sim.run_trace(key, ctrl, trace)
        summary = compute_summary_metrics(results)

        # 1. Verification of frame completion count
        assert summary["total_frames"] == 180, f"[{name}] incomplete frame count: {summary['total_frames']}"

        # 2. Queue draining check: By frame 179 (end of Phase 4 recovery), queue must be drained
        final_queue_depth = results[-1].queue_depth_at_arrival
        assert final_queue_depth <= 2, f"[{name}] Queue did not drain after burst, final depth: {final_queue_depth}"

        # 3. Maximum temperature bound
        assert summary["max_temp_c"] <= 85.0, f"[{name}] Unphysical thermal runaway: {summary['max_temp_c']:.1f} C"

        # 4. Host vs Accelerator utilization
        assert 0.0 <= summary["host_utilization_pct"] <= 100.0, f"[{name}] Invalid host util: {summary['host_utilization_pct']}"
        assert 0.0 <= summary["accel_utilization_pct"] <= 100.0, f"[{name}] Invalid accel util: {summary['accel_utilization_pct']}"

        # 5. Controller C specific checks:
        # Controller C must exhibit significantly higher accelerator utilization due to within-frame stalls
        if "Controller C" in name:
            assert summary["accel_utilization_pct"] > 15.0, (
                f"Controller C expected higher accelerator occupancy due to L1 stalls, got {summary['accel_utilization_pct']:.2f}%"
            )
