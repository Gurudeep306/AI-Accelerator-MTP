#!/usr/bin/env python3
"""Integration and unit tests for Phase 2: Controller Repair and Standardization."""
import pytest
from pathlib import Path
from practical_mpc_controller import (
    Action, Constraints, MPCController, Prediction, Telemetry, DEFAULT_ACTIONS
)
from workload_prediction_interface import (
    NOMINAL_R3_GMAC,
    PredictionScope,
    WorkloadPrediction,
    TIER_GMAC_MAP,
)
from history_mpc_controller import (
    HistoryBasedMPC,
    HistoryWorkloadPredictor,
    WorkloadObservation,
)
from frame_preview_mpc import (
    ProfilePerformancePredictor,
    TierClassifierWorkloadModel,
    FramePreviewMPCPipeline,
)
from within_frame_mpc_controller import (
    FirstLayerObservation,
    RemainingWorkloadEstimator,
    WithinFrameAdaptiveMPC,
    YOLO_R3_CONV2D_MACS,
    YOLO_R3_LAYER0_MACS,
    YOLO_R3_LAYER0_FRACTION,
)


class MockGroundedPredictor:
    """Predictor using realistic quality and latency scaling."""
    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        if hasattr(workload, "workload_scale"):
            scale = float(workload.workload_scale)
        elif isinstance(workload, dict):
            scale = float(workload["workload_scale"])
        else:
            scale = float(workload)

        # Quality grounded on width (0.50 -> 0.5036, 0.75 -> 0.7884, 1.00 -> 0.9500)
        if action.width <= 0.55:
            qual = 0.5036
        elif action.width <= 0.80:
            qual = 0.7884
        else:
            qual = 0.9500

        lat = action.width * 6.5 * scale
        pwr = 0.20 + action.width * 0.30
        energy = lat * pwr
        return Prediction(latency_ms=lat, power_w=pwr, energy_mj=energy, validated_quality=qual)


def test_standardized_workload_prediction_interface():
    """Verify WorkloadPrediction interface, normalization, and duck-typing."""
    pred = WorkloadPrediction(
        expected_gmac=1.4755,
        gmac_std=0.10,
        scope=PredictionScope.FULL_FRAME,
        overhead_ms=0.35,
        confidence=0.85,
        tier_idx=1,
        tier_name="R2_480x288",
    )

    # Workload scale relative to R3 nominal (2.6232 GMAC)
    assert abs(pred.workload_scale - (1.4755 / NOMINAL_R3_GMAC)) < 1e-4
    assert float(pred) == pred.workload_scale
    assert pred["workload_scale"] == pred.workload_scale
    assert pred["expected_gmac"] == 1.4755
    assert pred.remaining_ratio == 1.0

    # Remaining frame scope
    pred_rem = WorkloadPrediction(
        expected_gmac=2.5966,
        scope=PredictionScope.REMAINING_FRAME,
        layer_boundary=1,
    )
    assert abs(pred_rem.remaining_ratio - (1.0 - YOLO_R3_LAYER0_FRACTION)) < 1e-4


def test_controller_a_gmac_and_tier_prediction():
    """Verify Controller A historical prediction operates on GMACs and predicts tiers."""
    forecaster = HistoryWorkloadPredictor()

    # Feed 5 observations at R2 scale (~1.4755 GMAC)
    for i in range(5):
        obs = WorkloadObservation(frame_id=i, gmac=1.4755)
        forecaster.update(obs)

    fc = forecaster.predict_next()
    assert abs(fc.expected_gmac - 1.4755) < 0.1
    assert fc.tier_idx == 1
    assert fc.tier_name == "R2_480x288"

    pred_obj = forecaster.predict_next_prediction()
    assert isinstance(pred_obj, WorkloadPrediction)
    assert pred_obj.tier_idx == 1
    assert pred_obj.scope == PredictionScope.FULL_FRAME


def test_controller_b_tier_classifier_integration():
    """Verify Controller B 3-class classifier loading and inference."""
    model_path = Path("C:/AI accelerator project/workload_tier_model_yolo.json")
    assert model_path.exists()

    clf_model = TierClassifierWorkloadModel(model_path)
    sample_features = {
        "brightness_mean": 0.45,
        "brightness_std": 0.21,
        "edge_density": 0.12,
        "histogram_entropy": 0.88,
        "temporal_difference": 0.05,
        "aspect_ratio": 1.7778,
        "normalized_pixel_count": 0.6944,
    }

    pred = clf_model.predict(sample_features)
    assert isinstance(pred, WorkloadPrediction)
    assert pred.tier_idx in (0, 1, 2)
    assert pred.expected_gmac in (0.655799040, 1.475547840, 2.623196160)
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.overhead_ms == 0.35


def test_controller_c_repaired_pipeline():
    """Verify Controller C full pipeline: graph constants, latency throughput, feasibility gate."""
    constraints = Constraints(quality_floor=0.70)
    predictor = MockGroundedPredictor()
    mpc = MPCController(predictor, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[6], constraints=constraints)
    estimator = RemainingWorkloadEstimator()
    ctrl_c = WithinFrameAdaptiveMPC(mpc, estimator, measured_switch_cost_mj=50.0)

    # Layer 0 execution observation
    obs = FirstLayerObservation(
        frame_id=10,
        layer_id=0,
        mac_count=YOLO_R3_LAYER0_MACS,
        measured_latency_ms=0.18,
        total_model_macs=YOLO_R3_CONV2D_MACS,
    )
    telem = Telemetry(temperature_c=42.0)

    # Current action is low-quality: W0.50-E1-INT4 (quality 0.5036 < 0.70)
    res = ctrl_c.decide_after_first_layer(
        obs, telem, current_action=DEFAULT_ACTIONS[0], current_remaining_energy_mj=5.0
    )

    # Must switch to quality-compliant action (width >= 0.75)
    assert res.decision.action.width >= 0.75
    assert res.decision.mode == "MPC"
    assert "switch enforced" in res.note
    assert abs(res.estimate.layer0_mac_fraction - YOLO_R3_LAYER0_FRACTION) < 1e-5


def test_mpc_fallback_quality_compliance():
    """Verify that MPC fallback selects quality-compliant action under quality floors."""
    constraints = Constraints(quality_floor=0.70, power_cap_w=0.01)  # impossible power cap forces fallback
    predictor = MockGroundedPredictor()
    # Default fallback is low-quality DEFAULT_ACTIONS[0] (quality 0.5036)
    mpc = MPCController(predictor, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0], constraints=constraints)

    telem = Telemetry(temperature_c=40.0)
    decision = mpc.choose(1.0, telem)

    # Since DEFAULT_ACTIONS[0] fails quality floor 0.70, safe_fallback must be promoted to an action >= 0.70
    assert decision.action.width >= 0.75
