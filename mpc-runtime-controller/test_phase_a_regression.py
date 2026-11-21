#!/usr/bin/env python3
"""Phase A regression tests and pre-change controller B/C baselines."""
from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import pytest

from practical_mpc_controller import (
    Action, Constraints, DEFAULT_ACTIONS, MPCController, Prediction, Telemetry
)
from history_mpc_controller import (
    HistoryBasedMPC, WorkloadObservation
)
from frame_preview_mpc import FramePreviewMPCPipeline, ProfilePerformancePredictor
from within_frame_mpc_controller import (
    FirstLayerObservation, RemainingWorkloadEstimator, WithinFrameAdaptiveMPC
)

ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "tests" / "baselines" / "baseline_controller_b_c_phase4.json"
DATA_ROOT = ROOT / "data" / "nuscenes_mini"
TABLE_ROOT = DATA_ROOT / "v1.0-mini"
PROFILES = ROOT / "synthetic_calibrated_profiles.csv"
WORKLOAD_MODEL = ROOT / "synthetic_workload_model.json"
BASELINE = json.loads(BASELINE_PATH.read_text())


class SimplePredictor:
    def predict(self, action, workload, telemetry):
        scale = float(workload["workload_scale"]) if isinstance(workload, dict) else float(workload)
        # Deterministic, action-dependent predictions for unit tests only.
        latency = (1.0 + action.width * 0.5 + action.exit * 0.1 + action.bits * 0.01) * scale
        power = 0.5 + action.width * 0.2
        energy = latency * power
        quality = 0.95 - (1.0 - action.width) * 0.10 - (8 - action.bits) * 0.002
        return Prediction(latency, power, energy, quality)


def _constraints(horizon=5):
    return Constraints(
        deadline_ms=33.33, power_cap_w=8.0, quality_floor=0.65,
        thermal_resistance_c_per_w=None, horizon=horizon, beam_width=32
    )


def _telemetry():
    return Telemetry(temperature_c=35.0)


def _scene_0061_images():
    sensors = {x["token"]: x for x in json.loads((TABLE_ROOT / "sensor.json").read_text())}
    calibrated = {x["token"]: x for x in json.loads((TABLE_ROOT / "calibrated_sensor.json").read_text())}
    sample_data = json.loads((TABLE_ROOT / "sample_data.json").read_text())
    sample_by_token = {x["token"]: x for x in json.loads((TABLE_ROOT / "sample.json").read_text())}
    scene = next(x for x in json.loads((TABLE_ROOT / "scene.json").read_text())
                 if x["name"] == "scene-0061")
    cam_front = {}
    for row in sample_data:
        sensor_token = calibrated[row["calibrated_sensor_token"]]["sensor_token"]
        if row["is_key_frame"] and sensors[sensor_token]["channel"] == "CAM_FRONT":
            cam_front[row["sample_token"]] = row
    paths = []
    token = scene["first_sample_token"]
    while token:
        sample = sample_by_token[token]
        if token in cam_front:
            paths.append(DATA_ROOT / cam_front[token]["filename"])
        if token == scene["last_sample_token"]:
            break
        token = sample["next"]
    assert len(paths) == 39
    assert all(path.is_file() for path in paths)
    return paths


def _capture_b_condition_a():
    pipeline = FramePreviewMPCPipeline(
        str(WORKLOAD_MODEL), str(PROFILES), _constraints(),
        require_thermal_calibration=False
    )
    rows = []
    for frame_id, image_path in enumerate(_scene_0061_images()):
        result = pipeline.process_frame(frame_id, image_path, _telemetry())
        rows.append({
            "frame_id": frame_id, "action": result.action, "mode": result.mode,
            "planned_sequence": list(result.planned_sequence),
            "predicted_latency_ms": result.predicted_latency_ms,
            "predicted_energy_mj": result.predicted_energy_mj,
            "predicted_e2e_latency_ms": result.predicted_e2e_latency_ms,
        })
    return rows


def _capture_c_condition_a():
    predictor = ProfilePerformancePredictor(str(PROFILES))
    mpc = MPCController(
        predictor, predictor.profiles.keys(), fallback=Action(0.50, 1, 4),
        constraints=_constraints()
    )
    estimator = RemainingWorkloadEstimator(total_model_macs=105_420_000.0)
    controller = WithinFrameAdaptiveMPC(mpc, estimator)
    current_action = Action(0.50, 1, 4)
    rows = []
    for frame_id in range(39):
        observation = FirstLayerObservation(
            frame_id=frame_id, layer_id=1, mac_count=4_016_000.0,
            measured_latency_ms=0.21
        )
        decision = controller.decide_after_first_layer(
            observation, _telemetry(), current_action
        ).decision
        rows.append({
            "frame_id": frame_id, "action": decision.action.name,
            "mode": decision.mode, "planned_sequence": list(decision.planned_sequence),
            "predicted_latency_ms": decision.predicted_latency_ms,
            "predicted_energy_mj": decision.predicted_energy_mj,
            "predicted_e2e_latency_ms": decision.predicted_e2e_latency_ms,
            "feasible_actions": decision.feasible_actions,
        })
        current_action = decision.action
    return rows


def _assert_records_match(actual, expected):
    assert len(actual) == len(expected)
    float_fields = (
        "predicted_latency_ms", "predicted_energy_mj", "predicted_e2e_latency_ms"
    )
    for got, want in zip(actual, expected):
        for key in want:
            if key in float_fields:
                if want[key] is None:
                    assert got[key] is None
                else:
                    assert got[key] == pytest.approx(want[key], rel=1e-10, abs=1e-10)
            else:
                assert got[key] == want[key]


def test_controller_b_condition_a_matches_pre_phase_a_baseline():
    expected = BASELINE["condition_a"]["controller_b"]["frames"]
    _assert_records_match(_capture_b_condition_a(), expected)


def test_controller_c_condition_a_matches_pre_phase_a_baseline():
    expected = BASELINE["condition_a"]["controller_c"]["frames"]
    _assert_records_match(_capture_c_condition_a(), expected)


def test_controller_b_future_horizon_spike_changes_to_partial_sequence():
    pipeline = FramePreviewMPCPipeline(
        str(WORKLOAD_MODEL), str(PROFILES), _constraints(),
        require_thermal_calibration=False
    )
    # Controlled forecast fixture: retain real image preview, inject only future workload.
    pipeline.forecaster.forecast = MethodType(
        lambda self, steps, min_scale, max_scale: [1.0, 1.0, 25.0, 25.0][:steps],
        pipeline.forecaster
    )
    result = pipeline.process_frame(0, _scene_0061_images()[0], _telemetry())
    assert result.mode == "PARTIAL_HORIZON_FEASIBLE"
    assert len(result.planned_sequence) == 3
    assert result.action == result.planned_sequence[0]


def test_shared_mpc_future_horizon_spike_preserves_partial_sequence():
    predictor = ProfilePerformancePredictor(str(PROFILES))
    controller = MPCController(
        predictor, predictor.profiles.keys(), fallback=Action(0.50, 1, 4),
        constraints=_constraints()
    )
    decision = controller.choose(
        1.0, _telemetry(), workload_forecast=(1.0, 1.0, 25.0, 25.0)
    )
    assert decision.mode == "PARTIAL_HORIZON_FEASIBLE"
    assert len(decision.planned_sequence) == 3
    assert decision.action.name == decision.planned_sequence[0]


def test_shared_mpc_step_zero_infeasibility_uses_configured_fallback():
    predictor = ProfilePerformancePredictor(str(PROFILES))
    fallback = Action(0.50, 1, 4)
    controller = MPCController(
        predictor, predictor.profiles.keys(), fallback=fallback,
        constraints=_constraints()
    )
    decision = controller.choose(25.0, _telemetry())
    assert decision.action == fallback
    assert decision.mode in {"INFEASIBLE_SAFE_STATE", "SAFETY_FALLBACK"}


def test_quality_horizon_validation_and_runtime_override_immutability():
    controller = MPCController(
        SimplePredictor(), DEFAULT_ACTIONS, constraints=Constraints(horizon=3)
    )
    original = controller.c
    valid = Constraints(horizon=3, quality_floor=0.70,
                        quality_floor_horizon=(0.70, 0.80, 0.90))
    controller.validate_constraints(valid)
    with pytest.raises(ValueError, match="length"):
        controller.validate_constraints(Constraints(
            horizon=3, quality_floor_horizon=(0.7, 0.8)
        ))
    with pytest.raises(ValueError, match="horizon"):
        controller.validate_constraints(Constraints(horizon=2))
    with pytest.raises(ValueError, match="finite"):
        controller.validate_constraints(Constraints(
            horizon=3, quality_floor_horizon=(0.7, float("nan"), 0.9)
        ))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        controller.validate_constraints(Constraints(
            horizon=3, quality_floor_horizon=(0.7, 1.2, 0.9)
        ))
    decision = controller.choose(1.0, _telemetry(), constraints_override=valid)
    assert decision.mode == "MPC"
    assert controller.c is original
    assert controller.c.quality_floor == 0.85
    # A subsequent call without override must use the original scalar floor.
    controller.choose(1.0, _telemetry())
    assert controller.c is original



def test_horizon_quality_floors_are_applied_at_their_matching_steps():
    controller = MPCController(
        SimplePredictor(), DEFAULT_ACTIONS,
        constraints=Constraints(
            deadline_ms=50.0, horizon=2, quality_floor=0.65,
            quality_floor_horizon=(0.65, 0.99)
        )
    )
    decision = controller.choose(1.0, _telemetry())
    # All actions meet the first-step floor; none meet the second-step floor.
    assert decision.mode == "PARTIAL_HORIZON_FEASIBLE"
    assert len(decision.planned_sequence) == 1
    assert controller.last_rejection_counts[1]["quality"] == len(DEFAULT_ACTIONS)
    assert controller.last_rejection_reasons

def test_override_horizon_mismatch_rejected_before_predictor_evaluation():
    class CountingPredictor(SimplePredictor):
        def __init__(self):
            self.calls = 0
        def predict(self, action, workload, telemetry):
            self.calls += 1
            return super().predict(action, workload, telemetry)

    predictor = CountingPredictor()
    controller = MPCController(predictor, DEFAULT_ACTIONS, constraints=Constraints(horizon=3))
    with pytest.raises(ValueError, match="horizon"):
        controller.choose(
            1.0, _telemetry(),
            constraints_override=Constraints(horizon=2)
        )
    assert predictor.calls == 0


def test_history_controller_uses_next_frame_forecast_at_step_zero():
    class CaptureMPC:
        c = Constraints(horizon=3)
        fallback = DEFAULT_ACTIONS[0]
        previous_action = None
        def validate_constraints(self, constraints):
            return MPCController(SimplePredictor(), DEFAULT_ACTIONS,
                                 constraints=self.c).validate_constraints(constraints)
        def choose(self, workload_now, telemetry, workload_forecast=(),
                   constraints_override=None):
            self.captured = (workload_now, tuple(workload_forecast), constraints_override)
            from practical_mpc_controller import Decision
            return Decision(DEFAULT_ACTIONS[0], "MPC", "captured")

    mpc = CaptureMPC()
    controller = HistoryBasedMPC(mpc)
    controller.observe_completed_frame(WorkloadObservation(0, 0.8))
    controller.observe_completed_frame(WorkloadObservation(1, 1.4))
    forecast = controller.forecaster.predict_next()
    controller.decide_before_next_frame(_telemetry())
    assert mpc.captured[0] == pytest.approx(forecast.workload_scale)
    assert mpc.captured[0] != pytest.approx(1.4)
    assert mpc.captured[1] == pytest.approx((forecast.workload_scale,) * 2)


def test_history_constraint_override_is_scoped_and_cold_start_is_preserved():
    controller = HistoryBasedMPC(MPCController(
        SimplePredictor(), DEFAULT_ACTIONS, constraints=Constraints(horizon=3)
    ))
    override = Constraints(horizon=3, quality_floor=0.70,
                           quality_floor_horizon=(0.7, 0.8, 0.9))
    cold = controller.decide_before_next_frame(_telemetry(), constraints_override=override)
    assert cold.mode == "COLD_START_FALLBACK"
    assert "No workload history available" in cold.reason
    assert "configured fallback action applied" in cold.reason
    assert controller.mpc.c.quality_floor == 0.85
    with pytest.raises(ValueError, match="length"):
        controller.decide_before_next_frame(
            _telemetry(),
            constraints_override=Constraints(horizon=3, quality_floor_horizon=(0.7,))
        )
    controller.observe_completed_frame(WorkloadObservation(0, 1.0))
    result = controller.decide_before_next_frame(
        _telemetry(), constraints_override=override
    )
    assert result.mode in {"MPC", "PARTIAL_HORIZON_FEASIBLE"}
    assert controller.mpc.c.quality_floor == 0.85


def test_step_indexed_rejection_diagnostics_keep_legacy_reason_strings():
    controller = MPCController(
        SimplePredictor(), DEFAULT_ACTIONS,
        constraints=Constraints(horizon=3, deadline_ms=0.1)
    )
    decision = controller.choose(1.0, _telemetry())
    assert decision.mode in {"INFEASIBLE_SAFE_STATE", "SAFETY_FALLBACK"}
    assert controller.last_rejection_counts
    assert 0 in controller.last_rejection_counts
    assert "e2e_latency" in controller.last_rejection_counts[0]
    assert any("e2e_latency (" in reason for reason in controller.last_rejection_reasons)
