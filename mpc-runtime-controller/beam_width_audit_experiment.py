#!/usr/bin/env python3
"""Beam Width Validation and MPC Optimization — Strict Experimental Audit.

Strict Rules:
- Does NOT modify production code or existing benchmarks.
- Tests beam widths: 128 (reference), 64, 32, 16, 8.
- Covers 10 distinct, reproducible synthetic scenarios across multiple trials and seeds.
- Measures cold-start vs steady-state latency, energy, E2E timing, deadline misses,
  rejection rates, agreement with beam width 128, and constraint/safety guarantees.
- Emits dedicated CSV traces and summary JSON reports.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from practical_mpc_controller import (
    Action, Constraints, MPCController, Prediction, Telemetry, Decision,
    DEFAULT_SWITCH_ENERGY_MJ, DEFAULT_SWITCH_LATENCY_MS,
    DEFAULT_PREVIEW_ENERGY_MJ, DEFAULT_PREVIEW_LATENCY_MS
)
from frame_preview_mpc import ProfilePerformancePredictor
from history_mpc_controller import HistoryBasedMPC, WorkloadObservation
from unified_replay_harness import (
    FrameArrivalEvent, FrameExecutionResult, ReplaySimulator, compute_summary_metrics
)

# Test beam widths
BEAM_WIDTHS = [128, 64, 32, 16, 8]
REFERENCE_BW = 128

PROFILES_CSV = "synthetic_calibrated_profiles.csv"


@dataclass
class ScenarioConfig:
    name: str
    description: str
    n_frames: int
    deadline_ms: float
    base_ambient_c: float
    quality_floor: float
    power_cap_w: float
    # Generates a workload scale sequence for frame arrivals
    generator_func: Any


def make_trace_for_scenario(scenario: ScenarioConfig, seed: int, frame_rate_fps: float = 30.0) -> List[FrameArrivalEvent]:
    """Deterministically generates a FrameArrivalEvent sequence for a scenario."""
    np.random.seed(seed)
    scales = scenario.generator_func(scenario.n_frames)
    events: List[FrameArrivalEvent] = []
    inter_frame_ms = 1000.0 / frame_rate_fps

    for fid in range(scenario.n_frames):
        arr_ms = fid * inter_frame_ms
        scale = float(scales[fid])
        # Synthetic feature extraction representation
        features = {
            "brightness_mean": 0.5, "brightness_std": 0.2, "edge_density": 0.15 * scale,
            "histogram_entropy": 0.6, "temporal_difference": 0.05,
            "aspect_ratio": 16.0 / 9.0, "normalized_pixel_count": 1.0
        }
        event = FrameArrivalEvent(
            frame_id=fid,
            arrival_time_ms=arr_ms,
            deadline_ms=scenario.deadline_ms,
            actual_workload_scale=scale,
            first_layer_mac_fraction=0.15,
            image_features=features
        )
        events.append(event)
    return events


# 10 Distinct Workload Scenarios
def gen_low_stable(n: int):
    return [0.70 + 0.02 * np.sin(i) for i in range(n)]

def gen_high_stable(n: int):
    return [1.60 + 0.03 * np.cos(i) for i in range(n)]

def gen_sudden_spikes(n: int):
    # Base 0.8, spikes to 2.2 on periodic intervals
    out = [0.80 for _ in range(n)]
    for i in range(5, n, 8):
        out[i] = 2.20
        if i + 1 < n:
            out[i + 1] = 1.80
    return out

def gen_gradual_increasing(n: int):
    return [0.60 + (1.40 * i / max(1, n - 1)) for i in range(n)]

def gen_alternating(n: int):
    return [0.65 if (i % 2 == 0) else 1.55 for i in range(n)]

def gen_close_to_deadline(n: int):
    # Tuned scale so fastest actions are close to the threshold
    return [1.25 + 0.05 * np.sin(i / 2.0) for i in range(n)]

def gen_high_temp(n: int):
    return [1.00 + 0.05 * np.cos(i) for i in range(n)]

def gen_close_to_power_cap(n: int):
    return [1.35 + 0.10 * np.sin(i / 3.0) for i in range(n)]

def gen_quality_floor_boundary(n: int):
    return [0.90 + 0.04 * np.sin(i) for i in range(n)]

def gen_frequent_opt_changes(n: int):
    # Rapid step changes every 2 frames
    return [0.60 if (i // 2) % 3 == 0 else (1.10 if (i // 2) % 3 == 1 else 1.65) for i in range(n)]


SCENARIOS = [
    ScenarioConfig("1_low_stable", "Low and stable workload (scale ~0.70)", 30, 50.0, 35.0, 0.85, 2.0, gen_low_stable),
    ScenarioConfig("2_high_stable", "High and stable workload (scale ~1.60)", 30, 50.0, 35.0, 0.85, 2.0, gen_high_stable),
    ScenarioConfig("3_sudden_spikes", "Periodic sudden workload bursts up to 2.20", 30, 50.0, 35.0, 0.85, 2.0, gen_sudden_spikes),
    ScenarioConfig("4_gradual_increase", "Workload ramps from 0.60 to 2.00", 30, 50.0, 35.0, 0.85, 2.0, gen_gradual_increasing),
    ScenarioConfig("5_alternating_complexity", "Alternating easy (0.65) and complex (1.55) frames", 30, 50.0, 35.0, 0.85, 2.0, gen_alternating),
    ScenarioConfig("6_near_deadline", "Tight deadline boundary scenario (deadline=25ms)", 30, 25.0, 35.0, 0.85, 2.0, gen_close_to_deadline),
    ScenarioConfig("7_high_temperature", "High ambient temp (55C) near thermal cap (70C)", 30, 50.0, 55.0, 0.85, 2.0, gen_high_temp),
    ScenarioConfig("8_near_power_cap", "Strict power cap (0.15W) near action limits", 30, 50.0, 35.0, 0.85, 0.15, gen_close_to_power_cap),
    ScenarioConfig("9_quality_boundary", "Higher quality floor (0.91) filtering low INT4 options", 30, 50.0, 35.0, 0.91, 2.0, gen_quality_floor_boundary),
    ScenarioConfig("10_frequent_optimal_shifts", "Workload step shifts every 2 frames", 30, 50.0, 35.0, 0.85, 2.0, gen_frequent_opt_changes),
]


def run_single_simulation(predictor: ProfilePerformancePredictor,
                          constraints: Constraints,
                          trace: List[FrameArrivalEvent]) -> Tuple[List[FrameExecutionResult], List[float]]:
    """Runs a complete trace simulation while accurately logging raw decision latencies."""
    fallback_action = Action(0.50, 1, 4)
    validated_actions = tuple(predictor.profiles.keys())
    core_mpc = MPCController(predictor, validated_actions, fallback=fallback_action, constraints=constraints)
    ctrl = HistoryBasedMPC(core_mpc, cold_start_nominal_scale=1.0)

    simulator = ReplaySimulator(
        predictor, constraints,
        switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS,
        switch_energy_mj=DEFAULT_SWITCH_ENERGY_MJ,
        preview_latency_ms=DEFAULT_PREVIEW_LATENCY_MS,
        preview_energy_mj=DEFAULT_PREVIEW_ENERGY_MJ
    )

    results = simulator.run_trace("controller_a", ctrl, trace)
    raw_decision_latencies = [r.decision.decision_latency_ms for r in results]
    return results, raw_decision_latencies


def run_audit(trials_per_scenario: int = 3, seeds: Sequence[int] = (42, 101, 777)):
    print(f"Starting Strict Beam Width Validation Audit across {len(SCENARIOS)} scenarios...")
    print(f"Beam widths to test: {BEAM_WIDTHS}")
    print(f"Trials per scenario: {trials_per_scenario} (Seeds: {seeds[:trials_per_scenario]})")

    predictor = ProfilePerformancePredictor(PROFILES_CSV, allow_uncalibrated_quality=True)

    all_detailed_traces: List[Dict[str, Any]] = []
    scenario_bw_summaries: Dict[str, Dict[int, Dict[str, Any]]] = {}

    # Overall aggregated metrics per beam width
    overall_stats: Dict[int, Dict[str, List[float]]] = {
        bw: {
            "mean_dec_ms": [], "median_dec_ms": [], "p95_dec_ms": [], "p99_dec_ms": [], "max_dec_ms": [],
            "cold_dec_ms": [], "steady_dec_ms": [],
            "mean_energy_mj": [], "total_energy_mj": [], "mean_e2e_lat_ms": [],
            "deadline_misses": [], "rejected_count": [], "infeasible_safe_state_count": [],
            "agreement_pct": [], "switch_count": []
        }
        for bw in BEAM_WIDTHS
    }

    # Step-by-step discrepancy log
    discrepancy_log: List[Dict[str, Any]] = []

    for sc in SCENARIOS:
        print(f"\n---> Evaluating Scenario: {sc.name} ({sc.description})")
        scenario_bw_summaries[sc.name] = {}

        for trial_idx in range(trials_per_scenario):
            seed = seeds[trial_idx]
            trace = make_trace_for_scenario(sc, seed=seed)

            # First, run reference beam width 128
            ref_constraints = Constraints(
                deadline_ms=sc.deadline_ms,
                power_cap_w=sc.power_cap_w,
                temperature_cap_c=70.0,
                quality_floor=sc.quality_floor,
                ambient_c=sc.base_ambient_c,
                thermal_tau_s=27.0,
                thermal_resistance_c_per_w=2.0,
                horizon=5,
                beam_width=REFERENCE_BW,
                switch_penalty_mj=DEFAULT_SWITCH_ENERGY_MJ,
                switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS
            )
            ref_results, ref_dec_times = run_single_simulation(predictor, ref_constraints, trace)
            ref_actions = [r.decision.action_name for r in ref_results]
            ref_modes = [r.decision.mode for r in ref_results]

            # Now evaluate all beam widths under the identical trace & seed
            for bw in BEAM_WIDTHS:
                c = Constraints(
                    deadline_ms=sc.deadline_ms,
                    power_cap_w=sc.power_cap_w,
                    temperature_cap_c=70.0,
                    quality_floor=sc.quality_floor,
                    ambient_c=sc.base_ambient_c,
                    thermal_tau_s=27.0,
                    thermal_resistance_c_per_w=2.0,
                    horizon=5,
                    beam_width=bw,
                    switch_penalty_mj=DEFAULT_SWITCH_ENERGY_MJ,
                    switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS
                )
                results, dec_times = run_single_simulation(predictor, c, trace)
                summary = compute_summary_metrics(results)

                # Cold-start vs steady-state latency
                cold_dec = dec_times[0]
                steady_dec = float(np.mean(dec_times[1:])) if len(dec_times) > 1 else cold_dec

                sorted_dec = sorted(dec_times)
                p95_dec = float(sorted_dec[int(0.95 * len(sorted_dec))])
                p99_dec = float(sorted_dec[min(len(sorted_dec)-1, int(0.99 * len(sorted_dec)))])
                max_dec = float(max(dec_times))
                median_dec = float(np.median(dec_times))

                # Agreement with reference (BW=128)
                matches = sum(1 for a, r_a in zip(results, ref_results) if a.decision.action_name == r_a.decision.action_name)
                agreement_pct = 100.0 * matches / len(results)

                # Rejections & Infeasible Safe State
                infeasible_state_count = sum(1 for r in results if r.decision.mode == "INFEASIBLE_SAFE_STATE")
                rejected_count = summary.get("rejected_count", 0)
                misses = summary["deadline_misses"]
                tot_energy = summary["total_energy_mj"]
                mean_energy = summary["mean_energy_per_frame_mj"]
                mean_e2e = summary["mean_e2e_latency_ms"]

                # Switching churn count
                switches = 0
                for i in range(1, len(results)):
                    if results[i].decision.action_name != results[i-1].decision.action_name:
                        switches += 1

                # Record overall statistics
                stats = overall_stats[bw]
                stats["mean_dec_ms"].append(float(np.mean(dec_times)))
                stats["median_dec_ms"].append(median_dec)
                stats["p95_dec_ms"].append(p95_dec)
                stats["p99_dec_ms"].append(p99_dec)
                stats["max_dec_ms"].append(max_dec)
                stats["cold_dec_ms"].append(cold_dec)
                stats["steady_dec_ms"].append(steady_dec)
                stats["mean_energy_mj"].append(mean_energy)
                stats["total_energy_mj"].append(tot_energy)
                stats["mean_e2e_lat_ms"].append(mean_e2e)
                stats["deadline_misses"].append(misses)
                stats["rejected_count"].append(rejected_count)
                stats["infeasible_safe_state_count"].append(infeasible_state_count)
                stats["agreement_pct"].append(agreement_pct)
                stats["switch_count"].append(switches)

                # Log detailed frame rows
                for r_idx, r in enumerate(results):
                    ref_r = ref_results[r_idx]
                    is_diff = (r.decision.action_name != ref_r.decision.action_name)
                    all_detailed_traces.append({
                        "scenario": sc.name,
                        "trial": trial_idx,
                        "seed": seed,
                        "beam_width": bw,
                        "frame_id": r.frame_id,
                        "arrival_ms": f"{r.arrival_time_ms:.2f}",
                        "dispatch_ms": f"{r.dispatch_time_ms:.2f}",
                        "completion_ms": f"{r.completion_time_ms:.2f}",
                        "e2e_latency_ms": f"{r.end_to_end_latency_ms:.2f}",
                        "deadline_miss": r.deadline_miss,
                        "action": r.decision.action_name,
                        "ref_action": ref_r.decision.action_name,
                        "action_differs_from_ref": is_diff,
                        "mode": r.decision.mode,
                        "ref_mode": ref_r.decision.mode,
                        "decision_ms": f"{r.decision.decision_latency_ms:.3f}",
                        "total_energy_mj": f"{r.total_energy_mj:.3f}",
                        "quality": f"{r.quality:.3f}" if r.quality is not None else "N/A",
                        "temp_c": f"{r.temperature_c:.2f}",
                        "rejected": getattr(r, "rejected", False),
                        "rejection_reason": getattr(r, "rejection_reason", "") or ""
                    })

                    if is_diff:
                        discrepancy_log.append({
                            "scenario": sc.name,
                            "seed": seed,
                            "beam_width": bw,
                            "frame_id": r.frame_id,
                            "action": r.decision.action_name,
                            "ref_action": ref_r.decision.action_name,
                            "reason": r.decision.reason,
                            "ref_reason": ref_r.decision.reason,
                            "planned_sequence": r.decision.planned_sequence,
                            "ref_planned_sequence": ref_r.decision.planned_sequence,
                        })

    # Save detailed per-frame CSV trace
    out_csv = "beam_width_validation_detailed_traces.csv"
    if all_detailed_traces:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_detailed_traces[0].keys()))
            writer.writeheader()
            writer.writerows(all_detailed_traces)
        print(f"\nSaved detailed frame-level trace to: {out_csv} ({len(all_detailed_traces)} rows)")

    # Aggregate final metrics table per beam width
    final_table: List[Dict[str, Any]] = []
    for bw in BEAM_WIDTHS:
        st = overall_stats[bw]
        entry = {
            "beam_width": bw,
            "mean_dec_ms": float(np.mean(st["mean_dec_ms"])),
            "std_dec_ms": float(np.std(st["mean_dec_ms"])),
            "median_dec_ms": float(np.mean(st["median_dec_ms"])),
            "p95_dec_ms": float(np.mean(st["p95_dec_ms"])),
            "p99_dec_ms": float(np.mean(st["p99_dec_ms"])),
            "max_dec_ms": float(max(st["max_dec_ms"])),
            "cold_dec_ms": float(np.mean(st["cold_dec_ms"])),
            "steady_dec_ms": float(np.mean(st["steady_dec_ms"])),
            "mean_energy_mj": float(np.mean(st["mean_energy_mj"])),
            "total_energy_mj": float(np.mean(st["total_energy_mj"])),
            "mean_e2e_lat_ms": float(np.mean(st["mean_e2e_lat_ms"])),
            "total_deadline_misses": int(sum(st["deadline_misses"])),
            "mean_deadline_misses": float(np.mean(st["deadline_misses"])),
            "total_rejected_frames": int(sum(st["rejected_count"])),
            "mean_rejected_frames": float(np.mean(st["rejected_count"])),
            "total_infeasible_safe_states": int(sum(st["infeasible_safe_state_count"])),
            "agreement_pct": float(np.mean(st["agreement_pct"])),
            "mean_switches": float(np.mean(st["switch_count"]))
        }
        final_table.append(entry)

    # Save summary table CSV
    summary_csv = "beam_width_comparison_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_table[0].keys()))
        writer.writeheader()
        writer.writerows(final_table)
    print(f"Saved summary comparison CSV to: {summary_csv}")

    # Save scenario-wise summary JSON
    summary_json = "beam_width_scenarios_summary.json"
    json_data = {
        "metadata": {
            "trials_per_scenario": trials_per_scenario,
            "seeds": list(seeds[:trials_per_scenario]),
            "scenarios_count": len(SCENARIOS),
            "beam_widths": BEAM_WIDTHS,
            "reference_beam_width": REFERENCE_BW,
            "total_simulated_frame_evaluations": len(all_detailed_traces),
            "total_discrepancies_detected": len(discrepancy_log)
        },
        "aggregate_by_beam_width": {b["beam_width"]: b for b in final_table},
        "discrepancies": discrepancy_log
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved scenario-wise summary JSON to: {summary_json}")

    return final_table, discrepancy_log


if __name__ == "__main__":
    run_audit(trials_per_scenario=3, seeds=(42, 101, 777))
