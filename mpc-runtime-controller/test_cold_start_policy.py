#!/usr/bin/env python3
"""Unit tests for Controller A cold-start policy and frame 0 handling (Requirement 3)."""
import pytest
from practical_mpc_controller import Action, Constraints, MPCController, Prediction, Telemetry, DEFAULT_ACTIONS
from history_mpc_controller import HistoryBasedMPC, HistoryWorkloadPredictor, WorkloadObservation

class MockPredictor:
    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        scale = float(workload["workload_scale"]) if isinstance(workload, dict) else float(workload)
        return Prediction(latency_ms=10.0 * scale, power_w=1.0, energy_mj=10.0 * scale, validated_quality=0.90)

def test_controller_a_cold_start_fallback():
    mpc = MPCController(MockPredictor(), DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0])
    ctrl_a = HistoryBasedMPC(mpc)
    telem = Telemetry(temperature_c=40.0)

    # Frame 0 before any observation: should NOT crash with RuntimeError; must apply cold-start fallback
    decision = ctrl_a.decide_before_next_frame(telem)
    assert decision.action == DEFAULT_ACTIONS[0]
    assert decision.mode == "COLD_START_FALLBACK"
    assert "No workload history available" in decision.reason

    # Observe Frame 0 completion
    ctrl_a.observe_completed_frame(WorkloadObservation(frame_id=0, workload_scale=1.2))

    # Frame 1: Now history is available; should use receding-horizon MPC
    decision_f1 = ctrl_a.decide_before_next_frame(telem)
    assert decision_f1.mode == "MPC"
    assert decision_f1.feasible_actions > 0

def test_controller_a_cold_start_nominal_scale():
    mpc = MPCController(MockPredictor(), DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0])
    # Configured with nominal scale 1.0 for cold start
    ctrl_a = HistoryBasedMPC(mpc, cold_start_nominal_scale=1.0)
    telem = Telemetry(temperature_c=40.0)

    decision = ctrl_a.decide_before_next_frame(telem)
    assert decision.mode == "MPC"
