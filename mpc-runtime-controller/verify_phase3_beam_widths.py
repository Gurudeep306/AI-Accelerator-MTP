#!/usr/bin/env python3
"""Phase 3 Verification Script: Beam Width Robustness Audit across 128, 64, 32, 16, and 8.

Compares against Exhaustive Search Ground Truth across:
1. Standard Baseline Profile
2. Flat Energy Profile (Tie-breaking stress)
3. Extreme Energy-Latency Inversion Profile
4. Thermal Runaway Risk Profile
5. Tight Feasibility Boundary Profile
6. Additional dynamic stress workloads: High Frequency Oscillation, Shock Spike, Near-Deadline Stress.
"""
import math
import time
from dataclasses import replace
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from practical_mpc_controller import (
    Action, Constraints, MPCController, Prediction, Telemetry, DEFAULT_ACTIONS
)
from verify_phase2_mathematical_correctness import run_branch_and_bound

class CSVProfilePredictor:
    def __init__(self, profiles: Dict[Action, Dict[str, float]]):
        self.profiles = profiles

    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        p = self.profiles[action]
        wl = float(workload) if isinstance(workload, (int, float)) else 1.0
        lat = p["lat"] * wl
        pwr = p["pwr"] * (0.85 + 0.15 * wl)
        energy = (pwr * lat) / 1000.0
        return Prediction(
            latency_ms=lat,
            power_w=pwr,
            energy_mj=energy,
            validated_quality=p["qual"]
        )

def load_profile_families(csv_path: str = "stress_test_profile_families.csv") -> Dict[str, Dict[Action, Dict[str, float]]]:
    df = pd.read_csv(csv_path, comment="#")
    families = {}
    for fam in df["profile_family"].unique():
        fam_df = df[df["profile_family"] == fam]
        profiles = {}
        for _, row in fam_df.iterrows():
            act = Action(float(row["width"]), int(row["exit"]), int(row["bits"]))
            profiles[act] = {
                "lat": float(row["latency_ms"]),
                "pwr": float(row["power_w"]),
                "qual": float(row["validated_quality"])
            }
        families[fam] = profiles
    return families

def benchmark_beam_widths():
    families = load_profile_families()
    beam_widths = [128, 64, 32, 16, 8]

    # Dynamic workload stress traces (5 frames each)
    test_traces = {
        "Nominal": [1.0, 1.0, 1.0, 1.0, 1.0],
        "Rapid_Shock": [1.0, 2.5, 0.7, 1.8, 1.0],
        "High_Freq_Oscillation": [0.6, 1.6, 0.6, 1.6, 0.6],
        "Near_Deadline_Stress": [1.3, 1.35, 1.3, 1.35, 1.3],
        "Thermal_Soak": [1.2, 1.2, 1.2, 1.2, 1.2]
    }

    base_constraints = Constraints(
        deadline_ms=33.3,
        power_cap_w=15.0,
        temperature_cap_c=70.0,
        quality_floor=0.70,
        horizon=5,
        beam_width=128,
        ambient_c=35.0,
        thermal_tau_s=20.0,
        thermal_resistance_c_per_w=2.0
    )

    results = []
    timing_records = {w: [] for w in beam_widths}

    print(f"Auditing Beam Widths {beam_widths} against Exhaustive Ground Truth...")

    for fam_name, profiles in families.items():
        predictor = CSVProfilePredictor(profiles)
        actions = list(profiles.keys())
        fallback = actions[0]

        for trace_name, wl_trace in test_traces.items():
            init_temp = 65.0 if trace_name == "Thermal_Soak" else 55.0
            telem = Telemetry(temperature_c=init_temp)

            # 1. Compute Exhaustive Search Ground Truth (via verified Branch & Bound)
            c_bb = replace(base_constraints, horizon=5, beam_width=128)
            ctrl_bb = MPCController(predictor, actions, fallback=fallback, constraints=c_bb)
            dec_bb, seq_bb, score_bb, evals_bb = run_branch_and_bound(
                ctrl_bb, wl_trace[0], telem, wl_trace[1:]
            )
            exh_action = dec_bb.action if dec_bb else None
            exh_seq = seq_bb
            exh_energy = score_bb

            # 2. Test each beam width
            bw_decisions = {}
            for bw in beam_widths:
                c_bw = replace(base_constraints, horizon=5, beam_width=bw)
                ctrl_bw = MPCController(predictor, actions, fallback=fallback, constraints=c_bw)

                latencies = []
                for rep in range(15): # 15 iterations for timing distribution
                    t0 = time.perf_counter()
                    dec_bw = ctrl_bw.choose(wl_trace[0], telem, wl_trace[1:])
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    if rep > 0: # Drop cold start for timing array
                        latencies.append(dt_ms)
                        timing_records[bw].append(dt_ms)

                bw_decisions[bw] = (dec_bw, np.mean(latencies), np.median(latencies), np.percentile(latencies, 95))

            # Compare each beam width to Exhaustive and to BW=128
            dec_128, mean_128, med_128, p95_128 = bw_decisions[128]

            for bw in beam_widths:
                dec_curr, mean_lat, med_lat, p95_lat = bw_decisions[bw]
                curr_act = dec_curr.action if dec_curr else None
                curr_seq = dec_curr.planned_sequence if dec_curr else ()
                curr_energy = dec_curr.predicted_energy_mj if dec_curr else float("inf")

                match_exh_act = (curr_act == exh_action)
                match_128_act = (curr_act == dec_128.action)
                
                # Check seq agreement
                exh_seq_names = tuple(a.name for a in exh_seq) if exh_seq else ()
                seq_128_names = dec_128.planned_sequence if dec_128 else ()
                match_exh_seq = (curr_seq == exh_seq_names)
                match_128_seq = (curr_seq == seq_128_names)

                energy_diff_exh = curr_energy - (exh_energy if math.isfinite(exh_energy) else 0.0)

                results.append({
                    "family": fam_name,
                    "workload_trace": trace_name,
                    "beam_width": bw,
                    "selected_action": curr_act.name if curr_act else "None",
                    "exh_action": exh_action.name if exh_action else "None",
                    "match_exh_action": match_exh_act,
                    "match_128_action": match_128_act,
                    "match_exh_seq": match_exh_seq,
                    "match_128_seq": match_128_seq,
                    "energy_diff_exh_mj": energy_diff_exh,
                    "mean_latency_ms": mean_lat,
                    "median_latency_ms": med_lat,
                    "p95_latency_ms": p95_lat,
                    "feasible": (dec_curr.mode != "INFEASIBLE_SAFE_STATE") if dec_curr else False
                })

    df = pd.DataFrame(results)
    df.to_csv("phase3_beam_width_verification_results.csv", index=False)

    print("\n================== PHASE 3 SUMMARY BY BEAM WIDTH ==================")
    summary = []
    for bw in beam_widths:
        sub = df[df["beam_width"] == bw]
        exh_act_pct = sub["match_exh_action"].mean() * 100.0
        exh_seq_pct = sub["match_exh_seq"].mean() * 100.0
        ref_act_pct = sub["match_128_action"].mean() * 100.0
        ref_seq_pct = sub["match_128_seq"].mean() * 100.0
        max_e_diff = sub["energy_diff_exh_mj"].max()
        all_times = timing_records[bw]
        mean_t = np.mean(all_times)
        med_t = np.median(all_times)
        p95_t = np.percentile(all_times, 95)
        summary.append({
            "beam_width": bw,
            "exh_action_agreement_pct": exh_act_pct,
            "exh_seq_agreement_pct": exh_seq_pct,
            "bw128_action_agreement_pct": ref_act_pct,
            "bw128_seq_agreement_pct": ref_seq_pct,
            "max_energy_diff_mj": max_e_diff,
            "mean_latency_ms": mean_t,
            "median_latency_ms": med_t,
            "p95_latency_ms": p95_t
        })
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv("phase3_beam_width_summary.csv", index=False)
    print(sum_df.to_string(index=False))

if __name__ == "__main__":
    benchmark_beam_widths()
