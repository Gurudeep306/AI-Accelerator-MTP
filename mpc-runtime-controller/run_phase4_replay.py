#!/usr/bin/env python3
"""Execute Phase 4 Replay Evaluation on nuScenes CAM_FRONT Test Set.

Evaluates:
- Hardware Modes: Dense vs Sparse
- Deadlines: 33.333 ms (Strict 30 FPS) and 50.0 ms (Standard)
- Sequences: Held-out Test scenes (scene-1094, scene-1100)
- Controllers: Controller A, B, C, Fixed Nominal, Fixed Low-Power, Rule-Based
- Preserves all production constraints, 18 actions, H=5, beam width 32
"""
import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from practical_mpc_controller import Action, Constraints
from history_mpc_controller import HistoryBasedMPC
from frame_preview_mpc import FramePreviewMPCPipeline, ProfilePerformancePredictor
from within_frame_mpc_controller import WithinFrameAdaptiveMPC, RemainingWorkloadEstimator
from unified_replay_harness import (
    FrameArrivalEvent, ReplaySimulator, FixedBaselineController, RuleBasedBaselineController,
    compute_summary_metrics, DEFAULT_SWITCH_LATENCY_MS, DEFAULT_SWITCH_ENERGY_MJ,
    DEFAULT_PREVIEW_LATENCY_MS, DEFAULT_PREVIEW_ENERGY_MJ
)
from practical_mpc_controller import MPCController, Telemetry


def load_trace_events(csv_path: Path, deadline_override_ms: float = None) -> List[FrameArrivalEvent]:
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arr_ms = float(row["timestamp_ms"])
            dead_ms = deadline_override_ms if deadline_override_ms is not None else float(row["deadline_ms"])
            feats = {
                "brightness_mean": float(row["brightness_mean"]),
                "brightness_std": float(row["brightness_std"]),
                "edge_density": float(row["edge_density"]),
                "histogram_entropy": float(row["histogram_entropy"]),
                "temporal_difference": float(row["temporal_difference"]),
                "aspect_ratio": float(row["aspect_ratio"]),
                "normalized_pixel_count": float(row["normalized_pixel_count"]),
            }
            events.append(FrameArrivalEvent(
                frame_id=int(row["frame_id"]),
                arrival_time_ms=arr_ms,
                deadline_ms=dead_ms,
                actual_workload_scale=float(row["actual_workload_scale"]),
                first_layer_mac_fraction=float(row["first_layer_mac_fraction"]),
                image_features=feats
            ))
    return events


def build_controllers(profiles_csv: str, workload_model_json: str, constraints: Constraints):
    predictor = ProfilePerformancePredictor(profiles_csv, allow_uncalibrated_quality=True)
    validated_actions = tuple(predictor.profiles.keys())
    fallback_action = Action(0.50, 1, 4)

    # Controller A
    core_mpc_a = MPCController(predictor, validated_actions, fallback=fallback_action, constraints=constraints)
    ctrl_a = HistoryBasedMPC(core_mpc_a, cold_start_nominal_scale=1.0)

    # Controller B
    ctrl_b = FramePreviewMPCPipeline(
        workload_model_path=workload_model_json,
        profiles_csv=profiles_csv,
        constraints=constraints,
        fallback=fallback_action,
        require_thermal_calibration=False,
        allow_uncalibrated_quality=True
    )

    # Controller C
    core_mpc_c = MPCController(predictor, validated_actions, fallback=fallback_action, constraints=constraints)
    est_c = RemainingWorkloadEstimator(total_model_macs=1000.0)
    ctrl_c = WithinFrameAdaptiveMPC(core_mpc_c, est_c, measured_switch_cost_mj=DEFAULT_SWITCH_ENERGY_MJ)

    # Baselines
    ctrl_fixed_nom = FixedBaselineController(Action(1.00, 2, 8), name="Fixed (Nominal 1.0/2/8)")
    ctrl_fixed_low = FixedBaselineController(Action(0.50, 1, 4), name="Fixed (Low-Power 0.5/1/4)")
    ctrl_rule_based = RuleBasedBaselineController(validated_actions, name="Rule-Based (Heuristic)")

    controllers = [
        ("controller_a", ctrl_a),
        ("controller_b", ctrl_b),
        ("controller_c", ctrl_c),
        ("fixed_nominal", ctrl_fixed_nom),
        ("fixed_low_power", ctrl_fixed_low),
        ("rule_based", ctrl_rule_based)
    ]
    return predictor, controllers


def run_experiment(mode: str, deadline_ms: float, profiles_csv: str = "synthetic_calibrated_profiles.csv"):
    workload_model_json = f"real_driving_workload_model_{mode}.json"
    
    # Load test sequences
    s1094_path = Path(f"traces/{mode}/scene-1094.csv")
    s1100_path = Path(f"traces/{mode}/scene-1100.csv")
    
    e1094 = load_trace_events(s1094_path, deadline_override_ms=deadline_ms)
    e1100 = load_trace_events(s1100_path, deadline_override_ms=deadline_ms)
    
    # Combined continuous trace of 80 frames
    combined_trace = []
    for ev in e1094:
        combined_trace.append(ev)
    offset_ms = len(e1094) * 33.3333
    for ev in e1100:
        c_ev = FrameArrivalEvent(
            frame_id=ev.frame_id + len(e1094),
            arrival_time_ms=ev.arrival_time_ms + offset_ms,
            deadline_ms=deadline_ms,
            actual_workload_scale=ev.actual_workload_scale,
            first_layer_mac_fraction=ev.first_layer_mac_fraction,
            image_features=ev.image_features
        )
        combined_trace.append(c_ev)
        
    constraints = Constraints(
        deadline_ms=deadline_ms,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        ambient_c=35.0,
        thermal_tau_s=27.0,
        thermal_resistance_c_per_w=2.0,
        horizon=5,
        beam_width=32,
        switch_penalty_mj=DEFAULT_SWITCH_ENERGY_MJ,
        switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS,
    )
    
    predictor, controllers = build_controllers(profiles_csv, workload_model_json, constraints)
    
    # Warmup
    dummy_telem = Telemetry(
        temperature_c=35.0, observed_latency_ms=10.0, observed_power_w=1.0,
        observed_energy_mj=10.0, timestamp_s=0.0
    )
    for _ in range(5):
        controllers[0][1].mpc.choose(1.0, dummy_telem)
        controllers[1][1].controller.choose(
            {"frame_id": 0, "workload_scale": 1.0, "workload_scale_std": 0.05},
            dummy_telem,
            [{"workload_scale": 1.0, "workload_scale_std": 0.05}] * 4
        )
        controllers[2][1].mpc.choose(1.0, dummy_telem)
        
    simulator = ReplaySimulator(
        predictor, constraints,
        switch_latency_ms=DEFAULT_SWITCH_LATENCY_MS,
        switch_energy_mj=DEFAULT_SWITCH_ENERGY_MJ,
        preview_latency_ms=DEFAULT_PREVIEW_LATENCY_MS,
        preview_energy_mj=DEFAULT_PREVIEW_ENERGY_MJ
    )
    
    out_dir = Path("results/phase4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summaries = {}
    detailed_rows = []
    
    for key, c_obj in controllers:
        results = simulator.run_trace(key, c_obj, combined_trace)
        sm = compute_summary_metrics(results)
        
        # Calculate switching count
        switches = 0
        for i in range(1, len(results)):
            if results[i].decision.action != results[i-1].decision.action:
                switches += 1
        sm["switching_count"] = switches
        sm["switching_frequency_hz"] = switches / (sm["total_time_span_ms"] / 1000.0)
        
        # Calculate prediction accuracy metrics if available
        pred_errors = []
        for r in results:
            if hasattr(r.decision, "metadata") and "predicted_workload" in r.decision.metadata:
                pred_errors.append(r.decision.metadata["predicted_workload"] - combined_trace[r.frame_id].actual_workload_scale)
        if pred_errors:
            sm["workload_mae"] = float(np.mean(np.abs(pred_errors)))
            sm["workload_bias"] = float(np.mean(pred_errors))
            
        summaries[sm["controller"]] = sm
        
        for r in results:
            detailed_rows.append({
                "hardware_mode": mode,
                "deadline_ms": deadline_ms,
                "frame_id": r.frame_id,
                "controller": r.controller_name,
                "arrival_ms": f"{r.arrival_time_ms:.2f}",
                "dispatch_ms": f"{r.dispatch_time_ms:.2f}",
                "completion_ms": f"{r.completion_time_ms:.2f}",
                "e2e_latency_ms": f"{r.end_to_end_latency_ms:.2f}",
                "deadline_miss": r.deadline_miss,
                "action": r.decision.action_name,
                "mode": r.decision.mode,
                "reason": r.decision.reason,
                "decision_ms": f"{r.decision.decision_latency_ms:.2f}",
                "preview_ms": f"{r.decision.preview_latency_ms:.2f}",
                "reconfig_ms": f"{r.decision.reconfig_latency_ms:.2f}",
                "total_energy_mj": f"{r.total_energy_mj:.3f}",
                "quality": f"{r.quality:.3f}" if r.quality is not None else "UNKNOWN",
                "temp_c": f"{r.temperature_c:.2f}",
                "queue_depth": r.queue_depth_at_arrival,
                "host_queue_depth": r.host_queue_depth_at_arrival,
                "accel_queue_depth": r.accel_queue_depth_at_arrival,
            })
            
    # Save outputs
    dead_str = f"{int(deadline_ms)}ms"
    summary_json_path = out_dir / f"phase4_summary_{mode}_{dead_str}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
        
    summary_csv_path = out_dir / f"phase4_summary_{mode}_{dead_str}.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(list(summaries.values())[0].keys()))
        writer.writeheader()
        writer.writerows(list(summaries.values()))
        
    detail_csv_path = out_dir / f"phase4_detailed_{mode}_{dead_str}.csv"
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
        
    print(f"\n=========================================================================================")
    print(f"PHASE 4 REPLAY: {mode.upper()} MODE | DEADLINE: {deadline_ms:.1f} ms | 80 TEST FRAMES (scene-1094, 1100)")
    print(f"=========================================================================================")
    header = (
        f"{'Controller':<30} | {'Energy(mJ)':<10} | {'E2E Lat(ms)':<11} | "
        f"{'Misses(%)':<10} | {'On-Time FPS':<11} | {'Quality':<7} | {'Dec(ms)':<7} | {'Switches':<8}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        q_str = f"{s['mean_quality']:.3f}" if s["mean_quality"] is not None else "N/A"
        print(
            f"{name:<30} | {s['total_energy_mj']:10.2f} | "
            f"{s['mean_e2e_latency_ms']:11.2f} | "
            f"{s['deadline_miss_rate_pct']:9.1f}% | "
            f"{s.get('on_time_throughput_fps', s['throughput_fps']):11.2f} | "
            f"{q_str:<7} | "
            f"{s['mean_decision_ms']:7.2f} | "
            f"{s.get('switching_count', 0):<8}"
        )
    print("=" * len(header))
    return summaries


def main():
    print("Starting Phase 4 Replay Evaluation across Dual Modes and Central Deadlines...\n")
    
    # 1. Dense Mode @ 33.333 ms
    s_dense_33 = run_experiment("dense", 33.333)
    
    # 2. Dense Mode @ 50.0 ms
    s_dense_50 = run_experiment("dense", 50.0)
    
    # 3. Sparse Mode @ 33.333 ms
    s_sparse_33 = run_experiment("sparse", 33.333)
    
    # 4. Sparse Mode @ 50.0 ms
    s_sparse_50 = run_experiment("sparse", 50.0)
    
    print("\nAll 4 Replay Experiment Scenarios Completed Successfully!")


if __name__ == "__main__":
    main()
