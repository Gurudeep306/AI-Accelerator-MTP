#!/usr/bin/env python3
"""Execute Replay Evaluation on Grounded YOLOv8n Traces and Profiles.

Evaluates:
- Traces: Real YOLOv8n nuScenes test traces (scene-1094, scene-1100)
- Profiles: measured_yolo_profiles.csv (grounded on Colab GPU timing and detection sensitivity)
- Workload model: real_driving_workload_model_yolo.json
- Controllers: Controller A, B, C, Fixed Nominal, Fixed Low-Power, Rule-Based
- Evaluates behavior across quality floors: 0.85 (High), 0.70 (Medium), and 0.45 (Low)
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from practical_mpc_controller import Action, Constraints, MPCController
from history_mpc_controller import HistoryBasedMPC
from frame_preview_mpc import FramePreviewMPCPipeline, ProfilePerformancePredictor
from within_frame_mpc_controller import WithinFrameAdaptiveMPC, RemainingWorkloadEstimator
from unified_replay_harness import (
    FrameArrivalEvent, ReplaySimulator, FixedBaselineController, RuleBasedBaselineController,
    compute_summary_metrics, DEFAULT_SWITCH_LATENCY_MS, DEFAULT_SWITCH_ENERGY_MJ,
    DEFAULT_PREVIEW_LATENCY_MS, DEFAULT_PREVIEW_ENERGY_MJ
)


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
    # Fallback to Action(1.00, 2, 8) if high quality needed, or Action(0.50, 1, 4)
    fallback_action = Action(1.00, 2, 8) if constraints.quality_floor >= 0.80 else Action(0.50, 1, 4)

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


def run_yolo_experiment(q_floor: float, deadline_ms: float = 33.33):
    print(f"\n{'='*70}")
    print(f"RUNNING REAL YOLO REPLAY EXPERIMENT: Quality Floor = {q_floor:.2f}, Deadline = {deadline_ms:.1f} ms")
    print(f"{'='*70}")

    profiles_csv = "measured_yolo_profiles.csv"
    workload_model_json = "real_driving_workload_model_yolo.json"
    traces_dir = Path("traces/real_yolo")

    constraints = Constraints(
        deadline_ms=deadline_ms,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=q_floor,
        switch_penalty_mj=0.02,
        switch_latency_ms=0.5,
        horizon=3,
        beam_width=32
    )

    predictor, controllers = build_controllers(profiles_csv, workload_model_json, constraints)
    test_scenes = ["scene-1094", "scene-1100"]

    summary_records = []

    for ctrl_id, ctrl in controllers:
        all_results = []
        action_counts = {}

        for scene in test_scenes:
            csv_path = traces_dir / f"{scene}.csv"
            events = load_trace_events(csv_path, deadline_override_ms=deadline_ms)

            sim = ReplaySimulator(
                predictor=predictor,
                constraints=constraints,
                switch_latency_ms=constraints.switch_latency_ms,
                switch_energy_mj=constraints.switch_penalty_mj
            )

            # Reset controller state
            if hasattr(ctrl, "reset"):
                ctrl.reset()
            elif hasattr(ctrl, "core_mpc"):
                ctrl.core_mpc.previous_action = None

            results = sim.run_trace(ctrl_id, ctrl, events)
            all_results.extend(results)

            for r in results:
                act_name = r.decision.action_name
                action_counts[act_name] = action_counts.get(act_name, 0) + 1

        metrics = compute_summary_metrics(all_results)
        switches = sum(1 for i in range(1, len(all_results)) if all_results[i].decision.action_name != all_results[i-1].decision.action_name)

        summary_records.append({
            "controller": ctrl_id,
            "quality_floor": q_floor,
            "deadline_ms": deadline_ms,
            "total_frames": len(all_results),
            "deadline_misses": metrics["deadline_misses"],
            "mean_e2e_lat_ms": round(metrics["mean_e2e_latency_ms"], 2),
            "mean_energy_mj": round(metrics["mean_energy_per_frame_mj"], 3),
            "total_switches": switches,
            "top_actions": dict(sorted(action_counts.items(), key=lambda x: -x[1])[:3])
        })

        print(f"[{ctrl_id.upper()}] Mean Lat: {metrics['mean_e2e_latency_ms']:.2f} ms | Mean Energy: {metrics['mean_energy_per_frame_mj']:.3f} mJ | Switches: {switches} | Actions: {dict(sorted(action_counts.items(), key=lambda x: -x[1])[:3])}")

    return summary_records


def main():
    all_summaries = []
    # Test High Quality Floor (0.85) -> Should force Width 1.00 actions!
    all_summaries.extend(run_yolo_experiment(q_floor=0.85, deadline_ms=33.33))

    # Test Medium Quality Floor (0.70) -> Should pick Width 0.75 actions to save energy!
    all_summaries.extend(run_yolo_experiment(q_floor=0.70, deadline_ms=33.33))

    # Test Low Quality Floor (0.45) -> Should pick Width 0.50 actions for maximum energy savings!
    all_summaries.extend(run_yolo_experiment(q_floor=0.45, deadline_ms=33.33))

    out_json = Path("results/real_yolo_replay_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll replay experiment summaries saved to {out_json}")


if __name__ == "__main__":
    main()
