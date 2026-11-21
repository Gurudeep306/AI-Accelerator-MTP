#!/usr/bin/env python3
"""Unit tests for Controller C bug fixes and Strategy 3 asymmetric horizon."""
import pytest
from practical_mpc_controller import Action, Constraints, MPCController, Prediction, Telemetry, DEFAULT_ACTIONS
from within_frame_mpc_controller import (
    WithinFrameAdaptiveMPC, RemainingWorkloadEstimator, FirstLayerObservation,
    YOLO_R3_LAYER0_FRACTION
)

class MockQualityPredictor:
    """Predictor with distinct action qualities."""
    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        scale = float(workload["workload_scale"]) if hasattr(workload, "__getitem__") else float(workload)
        # Low-width actions have low quality (0.50), high-width actions have high quality (0.80)
        qual = 0.5036 if action.width < 0.70 else 0.7884
        lat = action.width * 5.0 * scale
        pwr = action.width * 0.5
        return Prediction(latency_ms=lat, power_w=pwr, energy_mj=lat * pwr, validated_quality=qual)


def test_controller_c_yolo_graph_constants():
    """Verify that Controller C defaults to YOLOv8n graph constants."""
    estimator = RemainingWorkloadEstimator()
    obs = FirstLayerObservation(frame_id=0, layer_id=0, mac_count=None, measured_latency_ms=0.15)
    est = estimator.estimate(obs)

    assert abs(est.layer0_mac_fraction - 0.010118) < 1e-5
    assert abs(est.workload_scale - (1.0 - 0.010118)) < 1e-5
    assert est.measured_l0_latency_ms == 0.15
    assert est.observed_throughput_gmac_per_s is not None
    assert est.projected_remaining_latency_ms is not None
    assert "measured_layer0_latency_ms" in est.features_used


def test_controller_c_strictly_blocks_infeasible_keep_current():
    """Verify that KEEP_CURRENT never bypasses an active quality floor constraint."""
    # Constrain quality floor to 0.70
    constraints = Constraints(quality_floor=0.70)
    predictor = MockQualityPredictor()
    mpc = MPCController(predictor, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[6], constraints=constraints)
    estimator = RemainingWorkloadEstimator()

    # Even with very high switch cost (100 mJ), infeasible action must NOT be kept!
    ctrl_c = WithinFrameAdaptiveMPC(mpc, estimator, measured_switch_cost_mj=100.0)

    # Current action has low quality (0.5036 < 0.70 floor)
    infeasible_action = DEFAULT_ACTIONS[0]  # W0.50-E1-INT4 (qual 0.5036)
    obs = FirstLayerObservation(frame_id=0, layer_id=0, mac_count=None, measured_latency_ms=0.15)
    telem = Telemetry(temperature_c=40.0)

    res = ctrl_c.decide_after_first_layer(
        obs, telem, current_action=infeasible_action, current_remaining_energy_mj=10.0
    )

    # Must switch to an action that meets the quality floor (>= 0.70)
    assert res.decision.action != infeasible_action
    assert res.decision.action.width >= 0.75
    assert "switch enforced" in res.note
    assert "infeasible" in res.note


def test_controller_c_asymmetric_horizon_strategy3():
    """Verify Strategy 3 asymmetric horizon handling."""
    constraints = Constraints(horizon=3, quality_floor=0.70)
    predictor = MockQualityPredictor()
    mpc = MPCController(predictor, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[6], constraints=constraints)
    estimator = RemainingWorkloadEstimator()
    ctrl_c = WithinFrameAdaptiveMPC(mpc, estimator)

    obs = FirstLayerObservation(frame_id=1, layer_id=0, mac_count=None, measured_latency_ms=0.20)
    telem = Telemetry(temperature_c=40.0)

    # Provide explicit full-frame forecast for future steps 1 and 2
    full_forecast = [1.2, 1.2]
    res = ctrl_c.decide_after_first_layer(
        obs, telem, current_action=DEFAULT_ACTIONS[6], full_frame_forecast=full_forecast
    )

    assert res.decision.action is not None
    # Telemetry should reflect Layer 0 timing
    assert telem.layer1_latency_ms == 0.20
    assert abs(telem.layer1_mac_fraction - YOLO_R3_LAYER0_FRACTION) < 1e-5
