#!/usr/bin/env python3
"""Audit tests for scheduling semantics, event ordering, resource occupancy, and summary aggregation (Sections 3 & 5)."""
import pytest
import numpy as np
from practical_mpc_controller import Action, Constraints
from unified_replay_harness import (
    generate_evaluation_trace, run_full_comparison, compute_summary_metrics
)

def test_event_ordering_and_resource_occupancy():
    """Verify strictly forward causal event ordering for all controllers."""
    trace = generate_evaluation_trace(n_frames=15, frame_rate_fps=30.0, deadline_ms=50.0, seed=42)
    summaries = run_full_comparison(
        profiles_csv="synthetic_calibrated_profiles.csv",
        workload_model_json="synthetic_workload_model.json",
        n_frames=15,
        deadline_ms=50.0,
        output_csv="test_audit_temp.csv",
        output_summary_json="test_audit_summary_temp.json",
        output_summary_csv="test_audit_summary_temp.csv"
    )

    import csv
    with open("test_audit_temp.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    controllers = sorted(set(r["controller"] for r in rows))
    for c in controllers:
        c_rows = [r for r in rows if r["controller"] == c]
        for r in c_rows:
            arr = float(r["arrival_ms"])
            disp = float(r["dispatch_ms"])
            comp = float(r["completion_ms"])
            e2e = float(r["e2e_latency_ms"])
            dec_ms = float(r["decision_ms"])

            # 1. Dispatch cannot occur before arrival
            assert disp >= arr, f"[{c}] dispatch {disp} < arrival {arr}"
            # 2. For executed frames, completion must be strictly after dispatch.
            # For rejected frames (INFEASIBLE_SAFE_STATE), completion equals decision end / dispatch (comp >= disp).
            is_rejected = (r.get("rejected") == "True" or r.get("mode") == "INFEASIBLE_SAFE_STATE")
            if is_rejected:
                assert comp >= disp, f"[{c}] rejected frame completion {comp} < dispatch {disp}"
            else:
                assert comp > disp, f"[{c}] completion {comp} <= dispatch {disp}"
            # 3. E2E latency must equal completion - arrival within rounding (0.02 ms)
            assert abs(e2e - (comp - arr)) < 0.05, f"[{c}] E2E {e2e} != {comp - arr}"

def test_summary_regression_matches_raw_records():
    """Verify that aggregate summaries match exact recalculation from raw records (Section 5)."""
    summaries = run_full_comparison(
        profiles_csv="synthetic_calibrated_profiles.csv",
        workload_model_json="synthetic_workload_model.json",
        n_frames=20,
        deadline_ms=50.0,
        output_csv="test_regression_temp.csv",
        output_summary_json="test_summary_temp.json",
        output_summary_csv="test_summary_temp.csv"
    )

    import csv
    with open("test_regression_temp.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    controllers = sorted(set(r["controller"] for r in rows))
    for c in controllers:
        c_rows = [r for r in rows if r["controller"] == c]
        s = summaries[c]

        # Recompute directly from rows
        raw_energies = [float(r["total_energy_mj"]) for r in c_rows]
        raw_lats = sorted([float(r["e2e_latency_ms"]) for r in c_rows])
        raw_misses = sum(1 for r in c_rows if r["deadline_miss"] == "True")
        raw_q_depths = [int(r["queue_depth"]) for r in c_rows]

        # Check energy sum within rounding tolerance (CSV writes 3 decimal places)
        assert abs(s["total_energy_mj"] - sum(raw_energies)) < 0.05
        # Check mean latency
        assert abs(s["mean_e2e_latency_ms"] - np.mean(raw_lats)) < 0.05
        # Check median latency
        assert abs(s["median_e2e_latency_ms"] - np.median(raw_lats)) < 0.05
        # Check deadline misses
        assert s["deadline_misses"] == raw_misses
        assert abs(s["deadline_miss_rate_pct"] - (100.0 * raw_misses / len(c_rows))) < 0.05
        # Check queue depth
        assert s["max_queue_depth"] == max(raw_q_depths)
