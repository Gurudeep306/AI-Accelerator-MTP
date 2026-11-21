#!/usr/bin/env python3
"""Unit tests for Controller C action normalization, validation, and switching policy (Requirement 5)."""
import pytest
from practical_mpc_controller import Action, Constraints, MPCController, Prediction, Telemetry, DEFAULT_ACTIONS
from within_frame_mpc_controller import (
    WithinFrameAdaptiveMPC, RemainingWorkloadEstimator, FirstLayerObservation
)

class MockPredictor:
    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        scale = float(workload["workload_scale"]) if isinstance(workload, dict) else float(workload)
        # Higher width -> more energy
        lat = action.width * 5.0
        pwr = action.width * 1.0
        return Prediction(latency_ms=lat * scale, power_w=pwr, energy_mj=lat * pwr * scale, validated_quality=0.90)

def test_controller_c_action_normalization():
    mpc = MPCController(MockPredictor(), DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0])
    estimator = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(mpc, estimator, measured_switch_cost_mj=0.01)

    obs = FirstLayerObservation(frame_id=0, layer_id=1, mac_count=200.0, measured_latency_ms=1.5)
    telem = Telemetry(temperature_c=40.0)

    # 1. Accepts Action instance
    res1 = ctrl_c.decide_after_first_layer(obs, telem, current_action=DEFAULT_ACTIONS[0])
    assert isinstance(res1.decision.action, Action)

    # 2. Accepts string action name
    res2 = ctrl_c.decide_after_first_layer(obs, telem, current_action="W0.50-E1-INT4")
    assert isinstance(res2.decision.action, Action)
    assert res2.decision.action.name == "W0.50-E1-INT4"

    # 3. Rejects unknown string action name
    with pytest.raises(ValueError, match="Unknown action name"):
        ctrl_c.decide_after_first_layer(obs, telem, current_action="W9.99-E9-INT99")

    # 4. Rejects unvalidated Action instance
    unvalidated_action = Action(0.12, 1, 2)
    with pytest.raises(ValueError, match="not in validated action set"):
        ctrl_c.decide_after_first_layer(obs, telem, current_action=unvalidated_action)

    # 5. Rejects invalid type
    with pytest.raises(TypeError, match="current_action must be an Action or valid action name"):
        ctrl_c.decide_after_first_layer(obs, telem, current_action=12345)

def test_controller_c_switch_overhead_keep_current():
    mpc = MPCController(MockPredictor(), DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0])
    estimator = RemainingWorkloadEstimator(total_model_macs=1000.0)
    # Very high switch cost: 100 mJ
    ctrl_c = WithinFrameAdaptiveMPC(mpc, estimator, measured_switch_cost_mj=100.0)

    obs = FirstLayerObservation(frame_id=0, layer_id=1, mac_count=200.0, measured_latency_ms=1.5)
    telem = Telemetry(temperature_c=40.0)
    # Start with high-energy action
    high_action = DEFAULT_ACTIONS[-1]

    # Energy remaining if unchanged is small, so saving cannot exceed 100 mJ
    res = ctrl_c.decide_after_first_layer(obs, telem, current_action=high_action, current_remaining_energy_mj=5.0)
    assert res.decision.action == high_action
    assert res.decision.mode == "KEEP_CURRENT"
    assert "switch overhead exceeds estimated benefit" in res.decision.reason
