#!/usr/bin/env python3
"""Phase 6 Comprehensive Regression Benchmarks.

Compares:
- Old Controller (from practical_mpc_controller.py.bak with beam_width=128)
- New Optimized Controller (from practical_mpc_controller.py with beam_width=32)

Across 10 Operating Regimes:
1. 30 FPS / 33.333 ms deadline (Target streaming throughput)
2. 50 ms deadline (Relaxed throughput)
3. Stable workload (1.0x constant)
4. Sudden workload spikes (2.5x transient spike)
5. Increasing workload (Ramp from 0.6x to 2.0x)
6. Alternating workload (Square wave 0.6x <-> 1.6x)
7. High-temperature conditions (Initial 68.5C near 70C cap)
8. Near-power-cap conditions (Tight power cap 0.25W)
9. Quality-boundary conditions (Tight quality floor 0.88)
10. Infeasible conditions (Exceeding thermal and deadline caps)
"""
import copy
import importlib.util
import math
import os
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

import old_practical_mpc_controller as old_mod
import practical_mpc_controller as new_mod
from frame_preview_mpc import ProfilePerformancePredictor

def run_benchmarks():
    profiles_csv = "synthetic_calibrated_profiles.csv"
    
    predictor = ProfilePerformancePredictor(profiles_csv, allow_uncalibrated_quality=True)
    raw_actions = tuple(predictor.profiles.keys())
    actions_old = tuple(old_mod.Action(a.width, a.exit, a.bits) for a in raw_actions)
    actions_new = tuple(new_mod.Action(a.width, a.exit, a.bits) for a in raw_actions)
    fallback_old = actions_old[0]
    fallback_new = actions_new[0]

    base_constraints_old = old_mod.Constraints(
        deadline_ms=33.333,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        horizon=5,
        beam_width=128,
        ambient_c=35.0,
        thermal_tau_s=27.0,
        thermal_resistance_c_per_w=2.0
    )

    base_constraints_new = new_mod.Constraints(
        deadline_ms=33.333,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        horizon=5,
        beam_width=32,
        ambient_c=35.0,
        thermal_tau_s=27.0,
        thermal_resistance_c_per_w=2.0
    )

    scenarios = [
        ("30_FPS_33.3ms_Deadline", {"deadline_ms": 33.333, "workload": [1.0]*20, "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("50.0ms_Deadline",        {"deadline_ms": 50.0,   "workload": [1.0]*20, "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("Stable_Workload",        {"deadline_ms": 33.333, "workload": [1.0]*20, "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("Sudden_Spikes",          {"deadline_ms": 33.333, "workload": [1.0, 1.0, 2.5, 2.5, 1.0]*4, "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("Increasing_Workload",    {"deadline_ms": 33.333, "workload": list(np.linspace(0.6, 2.0, 20)), "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("Alternating_Workload",   {"deadline_ms": 33.333, "workload": [0.6, 1.6]*10, "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("High_Temperature_68.5C", {"deadline_ms": 33.333, "workload": [1.0]*20, "temp_init": 68.5, "pwr_cap": 2.0, "q_floor": 0.85}),
        ("Near_Power_Cap_0.25W",   {"deadline_ms": 33.333, "workload": [1.0]*20, "temp_init": 50.0, "pwr_cap": 0.25, "q_floor": 0.85}),
        ("Quality_Boundary_0.88",  {"deadline_ms": 33.333, "workload": [1.0]*20, "temp_init": 50.0, "pwr_cap": 2.0, "q_floor": 0.88}),
        ("Infeasible_Conditions",  {"deadline_ms": 33.333, "workload": [1.0]*20, "temp_init": 72.5, "pwr_cap": 2.0, "q_floor": 0.85}),
    ]

    # Map action name to raw Action
    action_dict = {a.name: a for a in raw_actions}
    
    class UnifiedPredictorWrapper:
        def __init__(self, raw_pred):
            self.raw_pred = raw_pred
        def predict(self, action, wl, telemetry):
            raw_act = action_dict[action.name]
            return self.raw_pred.predict(raw_act, wl, telemetry)

    unified_pred = UnifiedPredictorWrapper(predictor)

    results = []

    print(f"Running Phase 6 Regressions across {len(scenarios)} regimes...")

    for name, params in scenarios:
        wl_list = params["workload"]
        dline = params["deadline_ms"]
        t_init = params["temp_init"]
        p_cap = params["pwr_cap"]
        q_flr = params["q_floor"]

        # Instantiate Old Controller
        c_old = replace(base_constraints_old, deadline_ms=dline, power_cap_w=p_cap, quality_floor=q_flr)
        ctrl_old = old_mod.MPCController(unified_pred, actions_old, fallback=fallback_old, constraints=c_old)

        # Instantiate New Controller
        c_new = replace(base_constraints_new, deadline_ms=dline, power_cap_w=p_cap, quality_floor=q_flr)
        ctrl_new = new_mod.MPCController(unified_pred, actions_new, fallback=fallback_new, constraints=c_new)

        # Replay simulation with simulated frame arrivals and host queueing
        for label, ctrl, is_new in [("Old_BW128", ctrl_old, False), ("New_BW32", ctrl_new, True)]:
            curr_temp = t_init
            current_action = None
            prev_host_dec_end = 0.0
            accel_idle_at_ms = 0.0
            recent_dec_ms = []

            total_energy = 0.0
            deadline_misses = 0
            rejected_frames = 0
            dec_times = []
            e2e_latencies = []
            selected_actions = []

            for f_idx, wl in enumerate(wl_list):
                arrival_ms = f_idx * dline
                host_q = max(0.0, prev_host_dec_end - arrival_ms)
                accel_q = max(0.0, accel_idle_at_ms - arrival_ms)
                est_dec = float(np.mean(recent_dec_ms[-5:])) if recent_dec_ms else 10.0

                telem = old_mod.Telemetry(
                    temperature_c=curr_temp,
                    timestamp_s=arrival_ms / 1000.0,
                    host_queue_wait_ms=host_q,
                    accel_queue_wait_ms=accel_q,
                    estimated_host_decision_ms=est_dec,
                    switch_latency_ms=0.5
                )

                t0 = time.perf_counter()
                future_wls = wl_list[f_idx+1 : f_idx+5]
                dec = ctrl.choose(wl, telem, future_wls)
                dec_ms = (time.perf_counter() - t0) * 1000.0
                dec_times.append(dec_ms)
                recent_dec_ms.append(dec_ms)

                t_dec_start = max(arrival_ms, prev_host_dec_end)
                t_dec_end = t_dec_start + dec_ms
                prev_host_dec_end = t_dec_end

                if dec.mode == "INFEASIBLE_SAFE_STATE":
                    rejected_frames += 1
                    continue

                act = dec.action
                selected_actions.append(act.name)

                reconfig_lat = 0.5 if (current_action is not None and act != current_action) else 0.0
                reconfig_energy = 0.02 if (current_action is not None and act != current_action) else 0.0
                current_action = act

                t_dispatch = max(t_dec_end, accel_idle_at_ms) + reconfig_lat
                p_act = unified_pred.predict(act, wl, telem)
                t_comp = t_dispatch + p_act.latency_ms
                accel_idle_at_ms = t_comp

                e2e_lat = t_comp - arrival_ms
                e2e_latencies.append(e2e_lat)
                if e2e_lat > dline:
                    deadline_misses += 1

                total_energy += p_act.energy_mj + reconfig_energy

                # Thermal step
                dt_exec_s = p_act.latency_ms / 1000.0
                alpha = math.exp(-dt_exec_s / 27.0)
                curr_temp = alpha * curr_temp + (1.0 - alpha) * (35.0 + 2.0 * p_act.power_w)

            results.append({
                "scenario": name,
                "controller": label,
                "mean_decision_ms": float(np.mean(dec_times)),
                "median_decision_ms": float(np.median(dec_times)),
                "p95_decision_ms": float(np.percentile(dec_times, 95)),
                "p95_e2e_lat_ms": float(np.percentile(e2e_latencies, 95)) if e2e_latencies else 0.0,
                "deadline_misses": deadline_misses,
                "rejected_frames": rejected_frames,
                "total_energy_mj": total_energy,
                "actions_sampled": ";".join(selected_actions[:5])
            })

    df = pd.DataFrame(results)
    df.to_csv("phase6_regression_benchmark_results.csv", index=False)
    print("\n================== PHASE 6 REGRESSION BENCHMARK RESULTS ==================")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmarks()
