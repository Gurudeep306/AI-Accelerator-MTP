#!/usr/bin/env python3
"""Unit tests for MPC horizon feasibility, partial-sequence handling, and fallback."""
import pytest
from practical_mpc_controller import Action, Constraints, MPCController, Prediction, Telemetry, DEFAULT_ACTIONS

class FeasibilityMockPredictor:
    def __init__(self, step_fail: int = -1):
        self.step_fail = step_fail
        self.call_count = 0

    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        self.call_count += 1
        # Workload scale can signal step index in test
        scale = float(workload["workload_scale"]) if isinstance(workload, dict) else float(workload)
        # If scale indicates step that fails constraints
        if scale >= 10.0:
            # Latency exceeds deadline
            return Prediction(latency_ms=100.0, power_w=1.0, energy_mj=100.0, validated_quality=0.90)
        return Prediction(latency_ms=10.0, power_w=1.0, energy_mj=10.0, validated_quality=0.90)

def test_mpc_preserves_partial_sequence_when_future_step_is_infeasible():
    """A future-step failure preserves the best feasible prefix for receding-horizon replanning."""
    pred = FeasibilityMockPredictor()
    c = Constraints(deadline_ms=50.0, horizon=5)
    ctrl = MPCController(pred, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0], constraints=c)
    telem = Telemetry(temperature_c=40.0)

    # Future forecast has an infeasible step at horizon index 4 (5th frame)
    future = [1.0, 1.0, 1.0, 10.0]
    decision = ctrl.choose(1.0, telem, workload_forecast=future)

    assert decision.action == DEFAULT_ACTIONS[0]
    assert decision.mode == "PARTIAL_HORIZON_FEASIBLE"
    assert len(decision.planned_sequence) == 4
    assert decision.action.name == decision.planned_sequence[0]
    assert "executing first action and replanning" in decision.reason

def test_mpc_admitted_sequence_is_complete():
    """Verify that when MPC succeeds, a complete sequence of length 5 was planned."""
    pred = FeasibilityMockPredictor()
    c = Constraints(deadline_ms=50.0, horizon=5)
    ctrl = MPCController(pred, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0], constraints=c)
    telem = Telemetry(temperature_c=40.0)

    future = [1.0, 1.0, 1.0, 1.0]
    decision = ctrl.choose(1.0, telem, workload_forecast=future)

    assert decision.mode == "MPC"
    assert len(decision.planned_sequence) == 5
