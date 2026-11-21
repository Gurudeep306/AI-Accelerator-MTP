#!/usr/bin/env python3
"""Deterministic algebraic verification of the execution timeline and scheduling semantics (Section 1 & 3).

Verifies algebraically and numerically that:
1. Every interval (Layer 1, decision, reconfiguration, remaining layers) is represented exactly once.
2. No double-counting or omission occurs.
3. Controller C's decision occurs strictly AFTER Layer 1 completes.
4. E2E Latency = Completion Time - Arrival Time.
"""
import pytest
from practical_mpc_controller import Action, Constraints, Telemetry
from unified_replay_harness import (
    FrameArrivalEvent, ReplaySimulator
)
from within_frame_mpc_controller import (
    WithinFrameAdaptiveMPC, RemainingWorkloadEstimator
)

class DeterministicMockPredictor:
    """Predictor returning fixed, known latencies and energies for algebraic verification."""
    def __init__(self):
        self.profiles = {
            # High-energy / high-latency action (e.g. W1.00-E3-INT8)
            Action(1.0, 3, 8): {"latency": 20.0, "power": 2.0, "energy": 40.0, "quality": 0.95,
                                "latency_std": 0, "power_std": 0, "energy_std": 0, "quality_std": 0},
            # Low-energy / low-latency action (e.g. W0.50-E1-INT4)
            Action(0.5, 1, 4): {"latency": 10.0, "power": 1.0, "energy": 10.0, "quality": 0.86,
                                "latency_std": 0, "power_std": 0, "energy_std": 0, "quality_std": 0}
        }

    def predict(self, action, workload, telemetry):
        p = self.profiles[action]
        scale = float(workload) if isinstance(workload, (int, float)) else float(workload.get("workload_scale", 1.0))
        return type("Pred", (), {
            "latency_ms": p["latency"] * scale,
            "power_w": p["power"],
            "energy_mj": p["energy"] * scale,
            "validated_quality": p["quality"]
        })()

class DeterministicMockMPC:
    """Mock MPC controller that returns a fixed chosen action and predicted energy."""
    def __init__(self, chosen_action, predicted_energy_mj=5.0, validated_actions=None):
        self.chosen_action = chosen_action
        self.predicted_energy_mj = predicted_energy_mj
        self.c = Constraints()
        self.fallback = chosen_action
        self.validated_actions = validated_actions or [chosen_action]

    def choose(self, workload, telemetry, workload_forecast=None):
        return type("Dec", (), {
            "action": self.chosen_action,
            "mode": "MPC",
            "reason": "Deterministic algebraic test",
            "predicted_latency_ms": 10.0,
            "predicted_power_w": 1.0,
            "predicted_energy_mj": self.predicted_energy_mj,
            "predicted_temperature_c": 35.0,
            "feasible_actions": len(self.validated_actions),
            "planned_sequence": (self.chosen_action.name,)
        })()

def test_controller_c_algebraic_timeline_with_switch():
    """Verify Controller C timeline when a valid, beneficial switch occurs:
    Starts with high-energy Action(1.0, 3, 8), switches to low-energy Action(0.5, 1, 4).
    """
    initial_action = Action(1.0, 3, 8)   # Latency = 20.0 ms, Energy = 40.0 mJ
    switched_action = Action(0.5, 1, 4)  # Latency = 10.0 ms, Energy = 10.0 mJ
    pred = DeterministicMockPredictor()
    c = Constraints(ambient_c=35.0, thermal_tau_s=27.0, thermal_resistance_c_per_w=2.0)

    switch_lat_ms = 0.5
    switch_energy_mj = 0.02
    sim = ReplaySimulator(pred, c, switch_latency_ms=switch_lat_ms, switch_energy_mj=switch_energy_mj)

    arr_ms = 10.0
    wl_scale = 1.0
    l1_frac = 0.20
    event = FrameArrivalEvent(
        frame_id=0,
        arrival_time_ms=arr_ms,
        deadline_ms=50.0,
        actual_workload_scale=wl_scale,
        first_layer_mac_fraction=l1_frac,
        image_features={}
    )

    # Initial remaining energy without switch = 40.0 * 0.8 = 32.0 mJ.
    # Switched action predicts remaining energy = 10.0 * 0.8 = 8.0 mJ.
    # Saving = 32.0 - 8.0 = 24.0 mJ > 0.02 mJ switch cost -> Switch occurs!
    mock_mpc = DeterministicMockMPC(switched_action, predicted_energy_mj=8.0,
                                    validated_actions=[initial_action, switched_action])
    mock_mpc.fallback = initial_action
    estimator = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(mock_mpc, estimator, measured_switch_cost_mj=switch_energy_mj)

    results = sim.run_trace("controller_c", ctrl_c, [event])
    assert len(results) == 1
    r = results[0]

    # Verify algebraic identity:
    # 1. t_l1_start = max(10.0, 0.0) = 10.0 ms
    assert r.dispatch_time_ms == 10.0

    # 2. Layer 1 latency = 20.0 * 0.20 = 4.0 ms
    # Decision starts strictly at t_l1_end = 10.0 + 4.0 = 14.0 ms
    assert r.decision_start_ms == pytest.approx(14.0)

    # 3. Decision finishes at t_dec_end = 14.0 + decision_ms
    dec_ms = r.decision.decision_latency_ms
    assert r.decision_end_ms == pytest.approx(14.0 + dec_ms)

    # 4. Reconfiguration occurs: t_reconfig_end = t_dec_end + switch_lat_ms (0.5 ms)
    # 5. Remaining layers with switched action: L_rem = 10.0 * 0.80 = 8.0 ms
    expected_completion = 14.0 + dec_ms + switch_lat_ms + 8.0
    assert r.completion_time_ms == pytest.approx(expected_completion)

    # 6. E2E Latency = completion - arrival
    expected_e2e = expected_completion - arr_ms
    assert r.end_to_end_latency_ms == pytest.approx(expected_e2e)

    # 7. Total inference latency = L1 (4.0) + L_rem (8.0) = 12.0 ms
    assert r.inference_latency_ms == pytest.approx(12.0)

    # 8. Total energy = L1_energy (40 * 0.2 = 8.0) + rem_energy (10 * 0.8 = 8.0) + switch_energy (0.02) = 16.02 mJ
    assert r.total_energy_mj == pytest.approx(16.02)
    assert r.decision.action == switched_action
    assert r.decision.reconfig_latency_ms == switch_lat_ms

def test_controller_c_algebraic_timeline_keep_current():
    """Verify Controller C timeline when switch cost exceeds savings (KEEP_CURRENT)."""
    initial_action = Action(0.5, 1, 4)   # Latency = 10.0 ms, Energy = 10.0 mJ
    expensive_action = Action(1.0, 3, 8) # Latency = 20.0 ms, Energy = 40.0 mJ
    pred = DeterministicMockPredictor()
    c = Constraints(ambient_c=35.0, thermal_tau_s=27.0, thermal_resistance_c_per_w=2.0)

    switch_lat_ms = 0.5
    switch_energy_mj = 0.02
    sim = ReplaySimulator(pred, c, switch_latency_ms=switch_lat_ms, switch_energy_mj=switch_energy_mj)

    arr_ms = 10.0
    wl_scale = 1.0
    l1_frac = 0.20
    event = FrameArrivalEvent(
        frame_id=0,
        arrival_time_ms=arr_ms,
        deadline_ms=50.0,
        actual_workload_scale=wl_scale,
        first_layer_mac_fraction=l1_frac,
        image_features={}
    )

    # Candidate action has higher energy -> Saving is negative -> KEEP_CURRENT
    mock_mpc = DeterministicMockMPC(expensive_action, predicted_energy_mj=32.0,
                                    validated_actions=[initial_action, expensive_action])
    mock_mpc.fallback = initial_action
    estimator = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(mock_mpc, estimator, measured_switch_cost_mj=switch_energy_mj)

    results = sim.run_trace("controller_c", ctrl_c, [event])
    r = results[0]

    # Action kept: initial_action
    assert r.decision.action == initial_action
    assert r.decision.mode == "KEEP_CURRENT"
    # No reconfiguration latency charged
    assert r.decision.reconfig_latency_ms == 0.0
    assert r.decision.reconfig_energy_mj == 0.0

    # L1 = 10.0 * 0.2 = 2.0 ms. t_l1_end = 12.0 ms
    dec_ms = r.decision.decision_latency_ms
    # L_rem = 10.0 * 0.8 = 8.0 ms
    expected_completion = 12.0 + dec_ms + 0.0 + 8.0
    assert r.completion_time_ms == pytest.approx(expected_completion)
    assert r.end_to_end_latency_ms == pytest.approx(expected_completion - arr_ms)
    # Total energy = 10.0 * 0.2 + 10.0 * 0.8 + 0.0 = 10.0 mJ
    assert r.total_energy_mj == pytest.approx(10.0)
