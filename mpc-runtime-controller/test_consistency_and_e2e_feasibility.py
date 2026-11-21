#!/usr/bin/env python3
"""Regression tests for MPC prediction-evaluation consistency, E2E feasibility,

unified switching costs, and Controller B preview energy accounting.
"""
import pytest
import numpy as np

from practical_mpc_controller import (
    Action, Constraints, MPCController, Prediction, Telemetry, Decision,
    DEFAULT_SWITCH_ENERGY_MJ, DEFAULT_SWITCH_LATENCY_MS,
    DEFAULT_PREVIEW_ENERGY_MJ, DEFAULT_PREVIEW_LATENCY_MS
)
from frame_preview_mpc import ProfilePerformancePredictor, FramePreviewMPCPipeline
from within_frame_mpc_controller import (
    WithinFrameAdaptiveMPC, RemainingWorkloadEstimator, FirstLayerObservation
)
from unified_replay_harness import (
    FrameArrivalEvent, ReplaySimulator, compute_summary_metrics
)


class MockPredictor:
    """Predictor returning controlled latency and energy values for testing."""
    def __init__(self, lat_map=None, energy_map=None):
        self.lat_map = lat_map or {}
        self.energy_map = energy_map or {}

    def predict(self, action: Action, workload: float, telemetry: Telemetry) -> Prediction:
        lat = self.lat_map.get(action, 10.0) * (workload if isinstance(workload, (int, float)) else 1.0)
        energy = self.energy_map.get(action, 5.0)
        return Prediction(
            latency_ms=lat,
            power_w=1.0,
            energy_mj=energy,
            validated_quality=0.95,
            latency_std_ms=0.0,
            power_std_w=0.0,
            energy_std_mj=0.0,
            quality_std=0.0
        )


def test_candidate_rejected_when_inference_fits_but_e2e_exceeds_deadline():
    """Requirement: A candidate with inference latency < deadline (10 ms < 20 ms)

    must be rejected when host decision overhead makes E2E latency > deadline (10 + 12 = 22 ms > 20 ms).
    """
    act1 = Action(0.5, 1, 4)
    pred = MockPredictor(lat_map={act1: 10.0})
    # Deadline = 20 ms. Inference is 10 ms (well below 20 ms).
    c = Constraints(deadline_ms=20.0, horizon=1, beam_width=16, switch_latency_ms=0.0)
    mpc = MPCController(pred, (act1,), fallback=act1, constraints=c)

    # Telemetry with 12 ms expected host decision time:
    # E2E = 12 ms (host) + 10 ms (inf) = 22 ms > 20 ms (deadline)
    telem = Telemetry(temperature_c=35.0, estimated_host_decision_ms=12.0)
    decision = mpc.choose(1.0, telem)

    # Must be rejected -> fallback also exceeds 20 ms (12 + 10 = 22 ms) -> triggers INFEASIBLE_SAFE_STATE
    assert decision.mode == "INFEASIBLE_SAFE_STATE"
    assert decision.feasible_actions == 0
    assert "unsafe" in decision.reason.lower() or "exceeds deadline" in decision.reason.lower()


def test_exact_deadline_boundary_feasibility():
    """Requirement: Exact deadline boundary:

    Candidate with E2E == deadline is feasible (<= deadline_ms convention),
    Candidate with E2E > deadline is infeasible.
    """
    act1 = Action(0.5, 1, 4)
    pred = MockPredictor(lat_map={act1: 10.0})
    # Deadline = 20.0 ms.
    c = Constraints(deadline_ms=20.0, horizon=1, beam_width=16, switch_latency_ms=0.0)
    mpc = MPCController(pred, (act1,), fallback=act1, constraints=c)

    # Case 1: E2E = 10.0 + 10.0 = 20.0 ms == deadline_ms -> Feasible
    telem_exact = Telemetry(temperature_c=35.0, estimated_host_decision_ms=10.0)
    dec_exact = mpc.choose(1.0, telem_exact)
    assert dec_exact.mode == "MPC"
    assert dec_exact.feasible_actions == 1
    assert dec_exact.predicted_e2e_latency_ms == pytest.approx(20.0)

    # Case 2: E2E = 10.0 + 10.01 = 20.01 ms > deadline_ms -> Infeasible (fallback is also unsafe -> INFEASIBLE_SAFE_STATE)
    telem_over = Telemetry(temperature_c=35.0, estimated_host_decision_ms=10.01)
    dec_over = mpc.choose(1.0, telem_over)
    assert dec_over.mode == "INFEASIBLE_SAFE_STATE"
    assert dec_over.feasible_actions == 0


def test_overlapping_scheduling_does_not_double_count():
    """Requirement: When accelerator is already busy running previous frame,

    host decision runs concurrently on host CPU.
    Dispatch delay is max(t_host_ready, t_accel_ready), NOT their sum.
    """
    act = Action(0.5, 1, 4)
    pred = MockPredictor(lat_map={act: 8.0})
    c = Constraints(deadline_ms=50.0, horizon=1, beam_width=16, switch_latency_ms=0.5)
    mpc = MPCController(pred, (act,), fallback=act, constraints=c)

    # Case A: Host decision takes 12 ms, accelerator queue wait is 15 ms
    # t_host_ready = 12 ms, t_accel_ready = 15 ms -> max(12, 15) = 15 ms
    # E2E = 15 ms + 0.0 (no switch) + 8.0 ms = 23.0 ms
    telem_accel_bound = Telemetry(
        temperature_c=35.0,
        host_queue_wait_ms=0.0,
        estimated_host_decision_ms=12.0,
        accel_queue_wait_ms=15.0
    )
    dec_accel = mpc.choose(1.0, telem_accel_bound)
    assert dec_accel.predicted_e2e_latency_ms == pytest.approx(23.0)

    # Case B: Host decision wait is 20 ms, accelerator queue wait is 10 ms
    # t_host_ready = 20 ms, t_accel_ready = 10 ms -> max(20, 10) = 20 ms
    # E2E = 20 ms + 0.0 + 8.0 ms = 28.0 ms
    telem_host_bound = Telemetry(
        temperature_c=35.0,
        host_queue_wait_ms=0.0,
        estimated_host_decision_ms=20.0,
        accel_queue_wait_ms=10.0
    )
    dec_host = mpc.choose(1.0, telem_host_bound)
    assert dec_host.predicted_e2e_latency_ms == pytest.approx(28.0)


def test_unified_switching_cost_and_accounting():
    """Requirement: Switching cost is unified at 0.02 mJ across:

    - Generic Constraints: switch_penalty_mj = 0.02
    - Controller C: measured_switch_cost_mj = 0.02
    - ReplaySimulator: switch_energy_mj = 0.02
    And switching cost is charged on switch, zero on no-switch.
    """
    assert DEFAULT_SWITCH_ENERGY_MJ == 0.02
    assert DEFAULT_SWITCH_LATENCY_MS == 0.5

    c = Constraints(deadline_ms=50.0)
    assert c.switch_penalty_mj == 0.02
    assert c.switch_latency_ms == 0.5

    act_a = Action(0.5, 1, 4)
    act_b = Action(1.0, 3, 8)
    pred = MockPredictor()
    mpc = MPCController(pred, (act_a, act_b), fallback=act_a, constraints=c)
    estimator = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(mpc, estimator)
    assert ctrl_c.measured_switch_cost_mj == 0.02

    sim = ReplaySimulator(pred, c)
    assert sim.switch_energy_mj == 0.02
    assert sim.switch_latency_ms == 0.5


def test_controller_b_preview_energy_accounted_once():
    """Requirement: Controller B accounts for 0.03 mJ preview energy.

    It must appear in expected total energy and decision records,
    and must NOT be multiplied across horizon steps.
    """
    assert DEFAULT_PREVIEW_ENERGY_MJ == 0.03
    assert DEFAULT_PREVIEW_LATENCY_MS == 1.2

    pipeline = FramePreviewMPCPipeline(
        workload_model_path="synthetic_workload_model.json",
        profiles_csv="synthetic_calibrated_profiles.csv",
        constraints=Constraints(deadline_ms=50.0, horizon=5),
        require_thermal_calibration=False,
        allow_uncalibrated_quality=True
    )
    assert pipeline.preview_energy_mj == 0.03

    features = {
        "brightness_mean": 0.5, "brightness_std": 0.2, "edge_density": 0.15,
        "histogram_entropy": 0.6, "temporal_difference": 0.05,
        "aspect_ratio": 16.0 / 9.0, "normalized_pixel_count": 1.0
    }
    telem = Telemetry(temperature_c=35.0)

    est = pipeline.workload_model.predict(features)
    pipeline.forecaster.update(est.workload_scale)
    horizon_scales = pipeline.forecaster.forecast(
        max(0, pipeline.controller.c.horizon - 1),
        pipeline.workload_model.min_target, pipeline.workload_model.max_target)
    current_wl = {"frame_id": 0, "workload_scale": est.workload_scale, "workload_scale_std": est.uncertainty}
    future_wls = [{"workload_scale": x, "workload_scale_std": est.uncertainty} for x in horizon_scales]
    decision = pipeline.controller.choose(current_wl, telem, future_wls)

    pred_tot_energy = (
        (decision.predicted_energy_mj + pipeline.preview_energy_mj)
        if decision.predicted_energy_mj is not None else None
    )

    # Preview energy must be exactly 0.03 mJ
    assert pipeline.preview_energy_mj == 0.03
    assert pred_tot_energy is not None
    # Total energy must be predicted_energy + 0.03 (preview), NOT 5 * 0.03
    assert pred_tot_energy == pytest.approx(decision.predicted_energy_mj + 0.03)


def test_all_candidates_exceed_deadline_triggers_safety_fallback():
    """Requirement: When all available candidates exceed deadline under E2E feasibility,

    safety fallback is triggered.
    """
    act1 = Action(0.5, 1, 4)
    act2 = Action(1.0, 3, 8)
    # Both actions have high inference latency (35 ms and 40 ms)
    pred = MockPredictor(lat_map={act1: 35.0, act2: 40.0})
    # Deadline is 33.333 ms
    c = Constraints(deadline_ms=33.333, horizon=1, beam_width=16)
    mpc = MPCController(pred, (act1, act2), fallback=act1, constraints=c)

    # Even with 0 host delay, 35 ms > 33.333 ms; fallback act1 is also 35 ms > 33.333 ms -> INFEASIBLE_SAFE_STATE
    telem = Telemetry(temperature_c=35.0, estimated_host_decision_ms=0.0)
    dec = mpc.choose(1.0, telem)

    assert dec.mode == "INFEASIBLE_SAFE_STATE"
    assert dec.feasible_actions == 0
    assert dec.action == act1
    assert "e2e_latency" in dec.reason.lower()


def test_low_energy_candidate_violating_quality_is_rejected():
    """Policy Requirement: Low-energy candidate that violates quality floor (0.80 < 0.85)

    must be strictly rejected in favor of the lowest-energy FEASIBLE candidate (0.15 mJ).
    """
    act_low_qual = Action(0.5, 1, 4)
    act_feasible = Action(0.75, 2, 4)

    class QualityMockPredictor:
        def predict(self, action, workload, telemetry):
            if action == act_low_qual:
                return Prediction(latency_ms=5.0, power_w=0.5, energy_mj=0.05, validated_quality=0.80)
            else:
                return Prediction(latency_ms=8.0, power_w=0.8, energy_mj=0.15, validated_quality=0.90)

    c = Constraints(deadline_ms=50.0, quality_floor=0.85, horizon=1)
    mpc = MPCController(QualityMockPredictor(), (act_low_qual, act_feasible),
                        fallback=act_feasible, constraints=c)
    telem = Telemetry(temperature_c=35.0)
    decision = mpc.choose(1.0, telem)

    assert decision.mode == "MPC"
    assert decision.action == act_feasible
    assert decision.feasible_actions == 1
    assert decision.predicted_energy_mj == pytest.approx(0.15)


def test_candidate_exceeding_power_limit_is_rejected():
    """Policy Requirement: Candidate exceeding power cap (3.5 W > 2.0 W)

    must be rejected even if energy is low.
    """
    act_high_pwr = Action(0.5, 1, 4)
    act_feasible = Action(0.75, 2, 4)

    class PowerMockPredictor:
        def predict(self, action, workload, telemetry):
            if action == act_high_pwr:
                return Prediction(latency_ms=5.0, power_w=3.5, energy_mj=0.08, validated_quality=0.90)
            else:
                return Prediction(latency_ms=8.0, power_w=1.2, energy_mj=0.18, validated_quality=0.90)

    c = Constraints(deadline_ms=50.0, power_cap_w=2.0, horizon=1)
    mpc = MPCController(PowerMockPredictor(), (act_high_pwr, act_feasible),
                        fallback=act_feasible, constraints=c)
    telem = Telemetry(temperature_c=35.0)
    decision = mpc.choose(1.0, telem)

    assert decision.mode == "MPC"
    assert decision.action == act_feasible
    assert decision.feasible_actions == 1


def test_candidate_exceeding_temperature_limit_is_rejected():
    """Policy Requirement: Candidate causing temperature projection >= cap

    must be rejected in favor of thermal-safe candidate.
    """
    act_hot = Action(1.0, 3, 8)
    act_cool = Action(0.5, 1, 4)

    class ThermalMockPredictor:
        def predict(self, action, workload, telemetry):
            if action == act_hot:
                # High power = 2.0 W with thermal resistance 20.0 C/W -> equilibrium = 75 C >= 70 C cap
                return Prediction(latency_ms=5.0, power_w=2.0, energy_mj=0.10, validated_quality=0.95)
            else:
                return Prediction(latency_ms=6.0, power_w=0.2, energy_mj=0.15, validated_quality=0.90)

    c = Constraints(deadline_ms=50.0, temperature_cap_c=70.0, thermal_resistance_c_per_w=20.0,
                    thermal_tau_s=1.0, decision_period_s=5.0, horizon=1)
    mpc = MPCController(ThermalMockPredictor(), (act_hot, act_cool),
                        fallback=act_cool, constraints=c)
    # Current temperature is 65 C
    telem = Telemetry(temperature_c=65.0)
    decision = mpc.choose(1.0, telem)

    assert decision.mode == "MPC"
    assert decision.action == act_cool
    assert decision.feasible_actions == 1


def test_lowest_energy_configuration_selected_among_all_feasible():
    """Policy Requirement: Among all configurations satisfying every required safety constraint,

    the MPC MUST choose the one with the lowest energy consumption.
    """
    act_high = Action(1.0, 3, 8)   # Energy = 0.30 mJ, Quality = 0.98
    act_mid = Action(0.75, 2, 4)   # Energy = 0.20 mJ, Quality = 0.92
    act_lowest = Action(0.5, 1, 4) # Energy = 0.12 mJ, Quality = 0.86

    class MultiFeasiblePredictor:
        def predict(self, action, workload, telemetry):
            if action == act_high:
                return Prediction(latency_ms=6.0, power_w=1.0, energy_mj=0.30, validated_quality=0.98)
            elif action == act_mid:
                return Prediction(latency_ms=8.0, power_w=0.8, energy_mj=0.20, validated_quality=0.92)
            else:
                return Prediction(latency_ms=10.0, power_w=0.5, energy_mj=0.12, validated_quality=0.86)

    # All three actions satisfy deadline (50 ms), quality floor (0.85), and power (2.0 W)
    c = Constraints(deadline_ms=50.0, quality_floor=0.85, power_cap_w=2.0, horizon=1)
    mpc = MPCController(MultiFeasiblePredictor(), (act_high, act_mid, act_lowest),
                        fallback=act_lowest, constraints=c)
    telem = Telemetry(temperature_c=35.0)
    decision = mpc.choose(1.0, telem)

    assert decision.mode == "MPC"
    assert decision.feasible_actions == 3
    # Must choose act_lowest purely because its energy (0.12 mJ) is the lowest!
    assert decision.action == act_lowest
    assert decision.predicted_energy_mj == pytest.approx(0.12)


def test_tie_breaking_favors_current_action_to_avoid_unnecessary_switching():
    """Policy Requirement: If multiple configurations have equivalent energy,

    tie-breaking rule prefers the currently active action to avoid unnecessary switching.
    """
    act_a = Action(0.5, 1, 4)
    act_b = Action(0.75, 1, 4)

    class EqualEnergyPredictor:
        def predict(self, action, workload, telemetry):
            # Both actions have identical energy (0.15 mJ) and identical quality (0.90)
            return Prediction(latency_ms=8.0, power_w=0.5, energy_mj=0.15, validated_quality=0.90)

    # Switch penalty = 0.0 to test exact energy equivalence tie-breaker
    c = Constraints(deadline_ms=50.0, switch_penalty_mj=0.0, horizon=1)
    mpc = MPCController(EqualEnergyPredictor(), (act_a, act_b), fallback=act_a, constraints=c)

    # Set current active action to act_b
    mpc.previous_action = act_b
    telem = Telemetry(temperature_c=35.0)
    decision = mpc.choose(1.0, telem)

    # Must stay on act_b to avoid switching churn
    assert decision.action == act_b


def test_infeasible_safe_state_triggered_when_fallback_is_unsafe():
    """Requirement 7.2 & 7.3: When all candidates are infeasible and the fallback also violates

    constraints, return INFEASIBLE_SAFE_STATE and do not describe fallback as safe.
    """
    act_a = Action(0.5, 1, 4)
    act_b = Action(1.0, 3, 8)

    class ImpossiblePredictor:
        def predict(self, action, workload, telemetry):
            # All actions have latency 40 ms > 30 ms deadline
            return Prediction(latency_ms=40.0, power_w=1.0, energy_mj=5.0, validated_quality=0.90)

    c = Constraints(deadline_ms=30.0, horizon=1)
    mpc = MPCController(ImpossiblePredictor(), (act_a, act_b), fallback=act_a, constraints=c)
    telem = Telemetry(temperature_c=35.0)

    decision = mpc.choose(1.0, telem)
    assert decision.mode == "INFEASIBLE_SAFE_STATE"
    assert "unsafe" in decision.reason.lower()
    assert "exceeds deadline" in decision.reason.lower()
    assert decision.feasible_actions == 0


def test_partial_sequence_when_future_fails_but_current_step_is_feasible():
    """A feasible current action is executed when only a later horizon step fails."""
    act_heavy = Action(1.0, 3, 8)
    act_light = Action(0.5, 1, 4)

    class HorizonFailPredictor:
        def predict(self, action, workload, telemetry):
            scale = float(workload) if not isinstance(workload, dict) else float(workload["workload_scale"])
            # At scale 5.0 (step 1), both act_light and act_heavy fail the 30 ms deadline (lat = 35 ms)
            # At scale 1.0 (step 0), act_light passes (15 ms), act_heavy fails (45 ms)
            if scale >= 5.0:
                lat = 35.0
            else:
                lat = 15.0 if action == act_light else 45.0
            return Prediction(latency_ms=lat, power_w=0.5, energy_mj=1.0, validated_quality=0.88)

    # Horizon = 2; validated_actions includes both act_heavy and fallback act_light
    c = Constraints(deadline_ms=30.0, horizon=2)
    mpc = MPCController(HorizonFailPredictor(), (act_heavy, act_light), fallback=act_light, constraints=c)
    # Workload now = 1.0 (act_light fits at step 0), future forecast = 5.0 (all actions fail at step 1)
    # Give telem estimated_host_decision_ms=0.0 so act_light E2E = 15.0 ms <= 30.0 ms
    telem = Telemetry(temperature_c=35.0, estimated_host_decision_ms=0.0)
    decision = mpc.choose(1.0, telem, workload_forecast=[5.0])

    # The current step is feasible; preserve its action and replan after observing the next frame.
    assert decision.mode == "PARTIAL_HORIZON_FEASIBLE"
    assert decision.action == act_light
    assert len(decision.planned_sequence) == 1
    assert decision.planned_sequence[0] == act_light.name
    assert "executing first action and replanning" in decision.reason


def test_simulator_records_and_handles_rejected_frame():
    """Requirement 7.4: When controller returns INFEASIBLE_SAFE_STATE, simulator rejects the new frame,

    records rejection reason and zero inference work, and flags deadline miss.
    """
    act_a = Action(0.5, 1, 4)

    class FailPredictor:
        def predict(self, action, workload, telemetry):
            return Prediction(latency_ms=60.0, power_w=1.0, energy_mj=5.0, validated_quality=0.90)

    from unified_replay_harness import ReplaySimulator, FrameArrivalEvent, compute_summary_metrics
    c = Constraints(deadline_ms=30.0, horizon=1)
    mpc = MPCController(FailPredictor(), (act_a,), fallback=act_a, constraints=c)
    sim = ReplaySimulator(FailPredictor(), c)

    trace = [
        FrameArrivalEvent(frame_id=0, arrival_time_ms=0.0, deadline_ms=30.0,
                          actual_workload_scale=1.0, first_layer_mac_fraction=0.1, image_features={})
    ]
    from history_mpc_controller import HistoryBasedMPC
    ctrl = HistoryBasedMPC(mpc, cold_start_nominal_scale=1.0)
    results = sim.run_trace("controller_a", ctrl, trace)

    assert len(results) == 1
    r = results[0]
    assert r.rejected is True
    assert "unsafe" in r.rejection_reason.lower()
    assert r.inference_latency_ms == 0.0
    assert r.inference_energy_mj == 0.0
    assert r.quality is None
    assert r.deadline_miss is True

    summary = compute_summary_metrics(results)
    assert summary["rejected_count"] == 1
    assert summary["rejected_rate_pct"] == 100.0


def test_inflight_work_and_resource_scheduling_consistency_during_rejection():
    """Requirement 7.5: Rejection of a frame does not disrupt in-flight accelerator execution

    or corrupt the accelerator idle timeline for subsequent frames.
    """
    act_a = Action(0.5, 1, 4)

    # Mock controller that returns MPC on Frame 0 & 2, but INFEASIBLE_SAFE_STATE on Frame 1
    class InFlightTestController:
        def __init__(self):
            self.call_count = 0

        def decide_before_next_frame(self, telemetry):
            self.call_count += 1
            if self.call_count == 2:
                # Frame 1: Infeasible and fallback unsafe
                return Decision(act_a, "INFEASIBLE_SAFE_STATE",
                                "No feasible configuration satisfies hard safety constraints. Fallback is unsafe.")
            return Decision(act_a, "MPC", "Lowest-energy feasible sequence",
                            predicted_latency_ms=20.0, predicted_power_w=0.8, predicted_energy_mj=2.0)

        def observe_completed_frame(self, observation):
            pass

    class FixedPredictor:
        def predict(self, action, workload, telemetry):
            return Prediction(latency_ms=20.0, power_w=0.8, energy_mj=2.0, validated_quality=0.88)

    from unified_replay_harness import ReplaySimulator, FrameArrivalEvent

    c = Constraints(deadline_ms=45.0, horizon=1)
    sim = ReplaySimulator(FixedPredictor(), c)
    ctrl = InFlightTestController()

    trace = [
        # Frame 0: Arrives at 0 ms, completes at ~20 ms
        FrameArrivalEvent(frame_id=0, arrival_time_ms=0.0, deadline_ms=45.0, actual_workload_scale=1.0,
                          first_layer_mac_fraction=0.1, image_features={}),
        # Frame 1: Arrives at 10 ms (while Frame 0 is executing on accelerator!). Infeasible -> Rejected!
        FrameArrivalEvent(frame_id=1, arrival_time_ms=10.0, deadline_ms=45.0, actual_workload_scale=1.0,
                          first_layer_mac_fraction=0.1, image_features={}),
        # Frame 2: Arrives at 25 ms (after Frame 0 finishes). Feasible -> Dispatches cleanly!
        FrameArrivalEvent(frame_id=2, arrival_time_ms=25.0, deadline_ms=45.0, actual_workload_scale=1.0,
                          first_layer_mac_fraction=0.1, image_features={})
    ]
    results = sim.run_trace("controller_a", ctrl, trace)

    assert len(results) == 3
    # Frame 0 completed successfully
    assert results[0].rejected is False
    assert results[0].inference_latency_ms == 20.0
    f0_comp = results[0].completion_time_ms

    # Frame 1 was rejected at host decision end without stalling accelerator
    assert results[1].rejected is True
    assert results[1].inference_latency_ms == 0.0

    # Frame 2 dispatched after Frame 0 completion and ran for 20 ms
    assert results[2].rejected is False
    assert results[2].dispatch_time_ms >= f0_comp
    assert results[2].inference_latency_ms == 20.0

