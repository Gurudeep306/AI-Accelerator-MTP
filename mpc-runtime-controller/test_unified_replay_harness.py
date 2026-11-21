#!/usr/bin/env python3
"""Integration tests for the unified replay evaluation harness."""
import pytest
from unified_replay_harness import (
    generate_evaluation_trace, run_full_comparison, compute_summary_metrics
)

def test_evaluation_trace_generation():
    trace = generate_evaluation_trace(n_frames=20, frame_rate_fps=30.0, deadline_ms=50.0)
    assert len(trace) == 20
    assert trace[0].frame_id == 0
    assert trace[0].arrival_time_ms == 0.0
    assert trace[1].arrival_time_ms > 0.0
    for e in trace:
        assert e.deadline_ms == 50.0
        assert e.actual_workload_scale > 0
        assert "edge_density" in e.image_features


def test_full_comparison_execution(tmp_path):
    out_csv = tmp_path / "test_replay_results.csv"
    out_json = tmp_path / "test_replay_summary.json"
    out_sum_csv = tmp_path / "test_replay_summary.csv"
    summaries = run_full_comparison(
        profiles_csv="synthetic_calibrated_profiles.csv",
        workload_model_json="synthetic_workload_model.json",
        n_frames=10,
        deadline_ms=50.0,
        output_csv=str(out_csv),
        output_summary_json=str(out_json),
        output_summary_csv=str(out_sum_csv)
    )
    assert "Controller A (History)" in summaries
    assert "Controller B (Preview)" in summaries
    assert "Controller C (Within-Frame)" in summaries
    assert "Fixed Nominal (W1.00-E3-INT8)" in summaries
    assert "Fixed Low-Power (W0.50-E1-INT4)" in summaries
    assert "Rule-Based Baseline" in summaries

    # Verify CSV was written
    assert out_csv.exists()
    content = out_csv.read_text()
    assert "controller" in content
    assert "e2e_latency_ms" in content
