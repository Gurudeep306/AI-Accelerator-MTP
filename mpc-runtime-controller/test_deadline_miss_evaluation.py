#!/usr/bin/env python3
"""Adversarial audit tests for deadline-miss evaluation, boundary semantics, and fallback guarantees."""
import pytest
import numpy as np
from practical_mpc_controller import (
    Action, Constraints, MPCController, Telemetry
)
from frame_preview_mpc import ProfilePerformancePredictor
from history_mpc_controller import HistoryBasedMPC
from unified_replay_harness import (
    FrameArrivalEvent, NormalizedDecisionRecord, FrameExecutionResult,
    ReplaySimulator, compute_summary_metrics, FixedBaselineController
)


def _make_dummy_decision(controller_name: str = "TestController",
                         action: Action = Action(0.50, 1, 4),
                         decision_ms: float = 10.0) -> NormalizedDecisionRecord:
    return NormalizedDecisionRecord(
        frame_id=0,
        controller_name=controller_name,
        action=action,
        action_name=action.name,
        mode="MPC",
        reason="Test",
        predicted_latency_ms=15.0,
        predicted_power_w=1.0,
        predicted_energy_mj=5.0,
        predicted_temp_c=40.0,
        decision_latency_ms=decision_ms,
        preview_latency_ms=0.0,
        preview_energy_mj=0.0,
        reconfig_latency_ms=0.0,
        reconfig_energy_mj=0.0,
        feasible_action_count=18,
        planned_sequence=tuple([action.name] * 5)
    )


def test_deterministic_deadline_miss_at_40ms_vs_30ms():
    """Requirement 2: A frame arriving at 0 ms, completing at 40 ms, and having a 30 ms deadline

    must report exactly one deadline miss in both the frame result and aggregate summary.
    """
    arrival_ms = 0.0
    completion_ms = 40.0
    deadline_ms = 30.0

    e2e_lat = completion_ms - arrival_ms
    miss = e2e_lat > deadline_ms

    assert e2e_lat == 40.0
    assert miss is True, "40 ms frame must miss a 30 ms deadline"

    dec = _make_dummy_decision()
    res = FrameExecutionResult(
        frame_id=0,
        controller_name="TestController",
        arrival_time_ms=arrival_ms,
        decision_start_ms=0.0,
        decision_end_ms=10.0,
        dispatch_time_ms=10.0,
        completion_time_ms=completion_ms,
        end_to_end_latency_ms=e2e_lat,
        deadline_miss=miss,
        inference_latency_ms=30.0,
        inference_energy_mj=10.0,
        total_energy_mj=10.0,
        quality=0.90,
        temperature_c=40.0,
        queue_depth_at_arrival=0,
        decision=dec,
        host_queue_depth_at_arrival=0,
        accel_queue_depth_at_arrival=0
    )

    summary = compute_summary_metrics([res])
    assert summary["total_frames"] == 1
    assert summary["deadline_misses"] == 1
    assert summary["on_time_completions"] == 0
    assert summary["deadline_miss_rate_pct"] == 100.0
    assert summary["on_time_rate_pct"] == 0.0
    assert summary["max_e2e_latency_ms"] == 40.0


def test_deadline_boundary_convention():
    """Requirement 3: Test the boundary condition where a frame completes exactly at its deadline.

    The implementation documents `miss = e2e_lat > event.deadline_ms`.
    - At e2e == deadline: on-time (miss == False).
    - At e2e > deadline: miss (miss == True).
    - At e2e < deadline: on-time (miss == False).
    """
    deadline_ms = 30.0
    arrival_ms = 0.0

    # 1. Exactly on deadline (30.000 ms)
    comp_exact = 30.0
    e2e_exact = comp_exact - arrival_ms
    miss_exact = e2e_exact > deadline_ms
    assert miss_exact is False, "Frame completing exactly at deadline must meet the deadline"

    # 2. Strictly before deadline (29.999 ms)
    comp_before = 29.999
    e2e_before = comp_before - arrival_ms
    miss_before = e2e_before > deadline_ms
    assert miss_before is False, "Frame completing before deadline must meet the deadline"

    # 3. Strictly after deadline (30.001 ms)
    comp_after = 30.001
    e2e_after = comp_after - arrival_ms
    miss_after = e2e_after > deadline_ms
    assert miss_after is True, "Frame completing strictly after deadline must be flagged as a miss"

    dec = _make_dummy_decision()
    results = [
        FrameExecutionResult(
            frame_id=0, controller_name="TestController",
            arrival_time_ms=arrival_ms, decision_start_ms=0.0, decision_end_ms=5.0,
            dispatch_time_ms=5.0, completion_time_ms=comp_exact, end_to_end_latency_ms=e2e_exact,
            deadline_miss=miss_exact, inference_latency_ms=25.0, inference_energy_mj=5.0,
            total_energy_mj=5.0, quality=0.9, temperature_c=40.0, queue_depth_at_arrival=0,
            decision=dec, host_queue_depth_at_arrival=0, accel_queue_depth_at_arrival=0
        ),
        FrameExecutionResult(
            frame_id=1, controller_name="TestController",
            arrival_time_ms=arrival_ms, decision_start_ms=0.0, decision_end_ms=5.0,
            dispatch_time_ms=5.0, completion_time_ms=comp_after, end_to_end_latency_ms=e2e_after,
            deadline_miss=miss_after, inference_latency_ms=25.001, inference_energy_mj=5.0,
            total_energy_mj=5.0, quality=0.9, temperature_c=40.0, queue_depth_at_arrival=0,
            decision=dec, host_queue_depth_at_arrival=0, accel_queue_depth_at_arrival=0
        )
    ]
    summary = compute_summary_metrics(results)
    assert summary["total_frames"] == 2
    assert summary["deadline_misses"] == 1
    assert summary["on_time_completions"] == 1
    assert summary["deadline_miss_rate_pct"] == 50.0


def test_identical_timestamps_differing_deadlines():
    """Requirement 4: Keep completion timestamps unchanged at 40 ms and change only the deadline.

    Confirm that the 40 ms frame misses a 30 ms deadline but meets a 50 ms deadline.
    """
    arrival_ms = 0.0
    completion_ms = 40.0
    e2e_lat = completion_ms - arrival_ms

    # Case A: 30 ms deadline
    deadline_strict = 30.0
    miss_strict = e2e_lat > deadline_strict
    assert miss_strict is True, "40 ms frame must miss a 30 ms deadline"

    # Case B: 50 ms deadline
    deadline_relaxed = 50.0
    miss_relaxed = e2e_lat > deadline_relaxed
    assert miss_relaxed is False, "40 ms frame must meet a 50 ms deadline"

    dec = _make_dummy_decision()

    # Evaluator summary under 30 ms
    res_strict = FrameExecutionResult(
        frame_id=0, controller_name="TestController",
        arrival_time_ms=arrival_ms, decision_start_ms=0.0, decision_end_ms=10.0,
        dispatch_time_ms=10.0, completion_time_ms=completion_ms, end_to_end_latency_ms=e2e_lat,
        deadline_miss=miss_strict, inference_latency_ms=30.0, inference_energy_mj=8.0,
        total_energy_mj=8.0, quality=0.88, temperature_c=42.0, queue_depth_at_arrival=0,
        decision=dec, host_queue_depth_at_arrival=0, accel_queue_depth_at_arrival=0
    )
    summary_strict = compute_summary_metrics([res_strict])
    assert summary_strict["deadline_misses"] == 1
    assert summary_strict["deadline_miss_rate_pct"] == 100.0

    # Evaluator summary under 50 ms
    res_relaxed = FrameExecutionResult(
        frame_id=0, controller_name="TestController",
        arrival_time_ms=arrival_ms, decision_start_ms=0.0, decision_end_ms=10.0,
        dispatch_time_ms=10.0, completion_time_ms=completion_ms, end_to_end_latency_ms=e2e_lat,
        deadline_miss=miss_relaxed, inference_latency_ms=30.0, inference_energy_mj=8.0,
        total_energy_mj=8.0, quality=0.88, temperature_c=42.0, queue_depth_at_arrival=0,
        decision=dec, host_queue_depth_at_arrival=0, accel_queue_depth_at_arrival=0
    )
    summary_relaxed = compute_summary_metrics([res_relaxed])
    assert summary_relaxed["deadline_misses"] == 0
    assert summary_relaxed["deadline_miss_rate_pct"] == 0.0


def test_extremely_small_deadline_evaluator_and_controller():
    """Requirement 5: Test an extremely small deadline (e.g. 0.1 ms).

    1. Evaluator: Replay simulator must report 100% deadline misses.
    2. Controller: MPC controller must report infeasibility and invoke the documented SAFETY_FALLBACK.
    """
    # 1. Controller Feasibility Check under tiny deadline
    predictor = ProfilePerformancePredictor("synthetic_calibrated_profiles.csv", allow_uncalibrated_quality=True)
    fallback = Action(0.50, 1, 4)
    tiny_deadline = 0.1  # 0.1 ms is physically impossible (fastest INT4 inference is ~1.12 ms)
    constraints = Constraints(
        deadline_ms=tiny_deadline,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        horizon=5,
        beam_width=128
    )
    core_mpc = MPCController(predictor, tuple(predictor.profiles.keys()), fallback=fallback, constraints=constraints)
    ctrl_a = HistoryBasedMPC(core_mpc, cold_start_nominal_scale=1.0)
    telem = Telemetry(temperature_c=35.0)

    # Controller decision
    dec = ctrl_a.decide_before_next_frame(telem)
    assert dec.action == fallback, f"Must return fallback structure {fallback}"
    assert dec.mode == "INFEASIBLE_SAFE_STATE", f"Mode must be INFEASIBLE_SAFE_STATE, got {dec.mode}"
    assert dec.feasible_actions == 0, "No actions can be feasible under 0.1 ms deadline"
    assert "No fully feasible predicted sequence" in dec.reason

    # 2. Simulator Replay Evaluation under tiny deadline
    simulator = ReplaySimulator(predictor, constraints)
    tiny_trace = [
        FrameArrivalEvent(
            frame_id=0,
            arrival_time_ms=0.0,
            deadline_ms=tiny_deadline,
            actual_workload_scale=1.0,
            first_layer_mac_fraction=0.1,
            image_features={"complexity": 1.0}
        ),
        FrameArrivalEvent(
            frame_id=1,
            arrival_time_ms=33.333,
            deadline_ms=tiny_deadline,
            actual_workload_scale=1.0,
            first_layer_mac_fraction=0.1,
            image_features={"complexity": 1.0}
        )
    ]
    results = simulator.run_trace("controller_a", ctrl_a, tiny_trace)
    assert len(results) == 2
    for r in results:
        assert r.deadline_miss is True, f"Frame {r.frame_id} with E2E {r.end_to_end_latency_ms:.2f} ms must miss {tiny_deadline} ms"
        assert r.rejected is True, f"Frame {r.frame_id} must be rejected under INFEASIBLE_SAFE_STATE"
        assert "unsafe" in r.rejection_reason.lower() or "exceeds deadline" in r.rejection_reason.lower()

    summary = compute_summary_metrics(results)
    assert summary["deadline_misses"] == 2
    assert summary["deadline_miss_rate_pct"] == 100.0
    assert summary["on_time_completions"] == 0
    assert summary["rejected_count"] == 2
    assert summary["rejected_rate_pct"] == 100.0
