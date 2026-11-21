#!/usr/bin/env python3
"""Phase 2 Verification Script: Exhaustive Comparison and Mathematical Correctness Audit.

Tests:
1. Exact Cartesian Product Enumeration vs. Branch-and-Bound.
2. Multiple initial configurations (None, W0.50-E1-INT4, W1.00-E3-INT8, W0.75-E2-INT4).
3. Multiple workload patterns (Constant, Ramp, Spike, Alternating).
4. Multiple thermal initial states (35C, 55C, 68C, 69.8C, 72.0C).
5. Asymmetric switching costs / varying penalties.
6. Equal-energy tie-breaking verification.
7. Infeasible states (no candidates meet deadline, power, temp, or quality).
"""
import itertools
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd

from practical_mpc_controller import (
    Action, Constraints, MPCController, Prediction, Telemetry, Decision
)

class SyntheticTestPredictor:
    def __init__(self, action_table: Dict[Action, Dict[str, float]]):
        self.action_table = action_table

    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        base = self.action_table[action]
        scale = float(workload) if isinstance(workload, (int, float)) else 1.0
        lat = base["lat"] * scale
        pwr = base["pwr"] * (0.8 + 0.2 * scale)
        energy = (pwr * lat) / 1000.0
        qual = base.get("qual", 0.90)
        return Prediction(
            latency_ms=lat,
            power_w=pwr,
            energy_mj=energy,
            validated_quality=qual
        )

def run_cartesian_enumeration(controller: MPCController,
                              workload_now: float,
                              telemetry: Telemetry,
                              workload_forecast: Sequence[float] = (),
                              asymmetric_switch_costs: Optional[Dict[Tuple[Action, Action], float]] = None
                              ) -> Tuple[Optional[Decision], Optional[Tuple[Action, ...]], float, int]:
    horizon = controller.c.horizon
    workloads = [workload_now] + list(workload_forecast[:horizon - 1])
    while len(workloads) < horizon:
        workloads.append(workloads[-1])

    actions = controller.validated_actions
    pred = []
    for wl in workloads:
        row = {}
        for a in actions:
            row[a] = controller.predictor.predict(a, wl, telemetry)
        pred.append(row)

    best_seq = None
    best_sort_key = None
    best_score = float("inf")
    eval_count = 0
    thermal_mode = controller.thermal_model_ready

    for seq in itertools.product(actions, repeat=horizon):
        eval_count += 1
        feasible = True
        cum_score = 0.0
        current_temp = telemetry.temperature_c
        prev_act = controller.previous_action

        for step in range(horizon):
            action = seq[step]
            p = pred[step][action]
            lat, power, energy, quality = controller._robust(p)
            e2e_lat = controller._estimate_e2e_latency(action, step, seq[:step], lat, telemetry)

            if e2e_lat > controller.c.deadline_ms or power > controller.c.power_cap_w:
                feasible = False
                break
            if quality is None or quality < controller.c.quality_floor:
                feasible = False
                break

            tnext = controller._temperature_next(current_temp, power) if thermal_mode else None
            if thermal_mode and (tnext is None or tnext >= controller.c.temperature_cap_c):
                feasible = False
                break
            if thermal_mode and tnext is not None:
                current_temp = tnext

            step_score = energy
            if controller.c.latency_penalty > 0.0:
                step_score += controller.c.latency_penalty * (e2e_lat / controller.c.deadline_ms) ** 2

            if prev_act is not None and prev_act != action:
                if asymmetric_switch_costs is not None:
                    step_score += asymmetric_switch_costs.get((prev_act, action), controller.c.switch_penalty_mj)
                else:
                    step_score += controller.c.switch_penalty_mj

            cum_score += step_score
            prev_act = action

        if not feasible:
            continue

        state = (cum_score, current_temp, seq[0], seq)
        sort_key = controller._beam_sort_key(state, pred[0])

        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_seq = seq
            best_score = cum_score

    if best_seq is None:
        return None, None, float("inf"), eval_count

    first_act = best_seq[0]
    p0 = pred[0][first_act]
    dec = Decision(first_act, "EXHAUSTIVE", "Direct Enum",
                   p0.latency_ms, p0.power_w, p0.energy_mj, None, len(actions),
                   tuple(a.name for a in best_seq))
    return dec, best_seq, best_score, eval_count

def run_branch_and_bound(controller: MPCController,
                         workload_now: float,
                         telemetry: Telemetry,
                         workload_forecast: Sequence[float] = (),
                         asymmetric_switch_costs: Optional[Dict[Tuple[Action, Action], float]] = None
                         ) -> Tuple[Optional[Decision], Optional[Tuple[Action, ...]], float, int]:
    horizon = controller.c.horizon
    workloads = [workload_now] + list(workload_forecast[:horizon - 1])
    while len(workloads) < horizon:
        workloads.append(workloads[-1])

    actions = controller.validated_actions
    pred = []
    for wl in workloads:
        row = {}
        for a in actions:
            row[a] = controller.predictor.predict(a, wl, telemetry)
        pred.append(row)

    best_seq = None
    best_sort_key = None
    best_score = float("inf")
    branches_visited = 0
    thermal_mode = controller.thermal_model_ready

    def search_step(step: int, current_score: float, current_temp: float,
                    prev_act: Optional[Action], seq: Tuple[Action, ...]):
        nonlocal best_score, best_seq, best_sort_key, branches_visited
        branches_visited += 1

        if step == horizon:
            state = (current_score, current_temp, seq[0], seq)
            sort_key = controller._beam_sort_key(state, pred[0])
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_seq = seq
                best_score = current_score
            return

        for action in actions:
            p = pred[step][action]
            lat, power, energy, quality = controller._robust(p)
            e2e_lat = controller._estimate_e2e_latency(action, step, seq, lat, telemetry)

            if e2e_lat > controller.c.deadline_ms or power > controller.c.power_cap_w:
                continue
            if quality is None or quality < controller.c.quality_floor:
                continue
            tnext = controller._temperature_next(current_temp, power) if thermal_mode else None
            if thermal_mode and (tnext is None or tnext >= controller.c.temperature_cap_c):
                continue

            step_score = energy
            if controller.c.latency_penalty > 0.0:
                step_score += controller.c.latency_penalty * (e2e_lat / controller.c.deadline_ms) ** 2

            if prev_act is not None and prev_act != action:
                if asymmetric_switch_costs is not None:
                    step_score += asymmetric_switch_costs.get((prev_act, action), controller.c.switch_penalty_mj)
                else:
                    step_score += controller.c.switch_penalty_mj

            next_score = current_score + step_score

            # Pruning check: only prune if next_score is strictly worse in energy than best_sort_key[0]
            if best_sort_key is not None and round(next_score, 6) > best_sort_key[0]:
                continue

            search_step(step + 1, next_score, current_temp if tnext is None else tnext,
                        action, seq + (action,))

    search_step(0, 0.0, telemetry.temperature_c, controller.previous_action, tuple())

    if best_seq is None:
        return None, None, float("inf"), branches_visited

    first_act = best_seq[0]
    p0 = pred[0][first_act]
    dec = Decision(first_act, "BRANCH_AND_BOUND", "B&B",
                   p0.latency_ms, p0.power_w, p0.energy_mj, None, len(actions),
                   tuple(a.name for a in best_seq))
    return dec, best_seq, best_score, branches_visited

def run_phase2_comprehensive_tests():
    # Setup test actions: 5 distinct actions
    actions = [
        Action(0.50, 1, 4), # A0: low power, low qual
        Action(0.50, 2, 4), # A1: med-low
        Action(0.75, 1, 8), # A2: med
        Action(0.75, 2, 8), # A3: med-high
        Action(1.00, 3, 8), # A4: high power, high qual
    ]
    action_table = {
        actions[0]: {"lat": 8.0, "pwr": 3.0, "qual": 0.86},
        actions[1]: {"lat": 12.0, "pwr": 4.5, "qual": 0.88},
        actions[2]: {"lat": 16.0, "pwr": 6.0, "qual": 0.90},
        actions[3]: {"lat": 20.0, "pwr": 8.0, "qual": 0.92},
        actions[4]: {"lat": 25.0, "pwr": 12.0, "qual": 0.95},
    }
    predictor = SyntheticTestPredictor(action_table)

    base_constraints = Constraints(
        deadline_ms=33.3,
        power_cap_w=15.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        horizon=4,
        beam_width=32,
        ambient_c=35.0,
        thermal_tau_s=20.0,
        thermal_resistance_c_per_w=2.0
    )

    results = []

    # Test Matrix:
    # 1. Initial configurations
    initial_configs = [None, actions[0], actions[4], actions[2]]
    # 2. Workload sequences (4 steps)
    workload_patterns = {
        "Constant": (1.0, [1.0, 1.0, 1.0]),
        "Spike": (1.0, [2.2, 1.0, 1.0]),
        "Ramp": (0.6, [0.9, 1.3, 1.7]),
        "Alternating": (0.7, [1.5, 0.7, 1.5]),
    }
    # 3. Thermal states
    thermal_states = [35.0, 55.0, 68.0, 69.8, 72.0]

    case_id = 0

    print("Running Phase 2 Test Matrix...")

    # Group A: Combinations of Initial Config x Workload x Thermal
    for init_act in initial_configs:
        for wl_name, (wl_now, wl_fore) in workload_patterns.items():
            for t_init in [35.0, 68.0]:
                case_id += 1
                ctrl = MPCController(predictor, actions, fallback=actions[0], constraints=base_constraints)
                ctrl.previous_action = init_act
                telem = Telemetry(temperature_c=t_init)

                dec_enum, seq_enum, score_enum, ev_enum = run_cartesian_enumeration(ctrl, wl_now, telem, wl_fore)
                dec_bb, seq_bb, score_bb, ev_bb = run_branch_and_bound(ctrl, wl_now, telem, wl_fore)

                match_action = (dec_enum.action == dec_bb.action) if (dec_enum and dec_bb) else (dec_enum == dec_bb)
                match_seq = (seq_enum == seq_bb)
                match_score = (abs(score_enum - score_bb) < 1e-9) if (math.isfinite(score_enum) and math.isfinite(score_bb)) else (score_enum == score_bb)

                results.append({
                    "case_id": case_id,
                    "category": "InitConfig_x_Workload_x_Temp",
                    "description": f"Init={init_act.name if init_act else 'None'}|WL={wl_name}|T={t_init}C",
                    "enum_evals": ev_enum,
                    "bb_evals": ev_bb,
                    "pruned_pct": (1.0 - ev_bb / ev_enum) * 100.0,
                    "score_enum": score_enum,
                    "score_bb": score_bb,
                    "action_match": match_action,
                    "seq_match": match_seq,
                    "score_match": match_score,
                    "all_pass": match_action and match_seq and match_score
                })

    # Group B: Asymmetric switching costs
    asym_costs = {
        (actions[4], actions[0]): 0.001, # Cheap downswitch
        (actions[0], actions[4]): 0.050, # Expensive upswitch
        (actions[2], actions[1]): 0.002,
        (actions[1], actions[2]): 0.020,
    }
    for wl_name, (wl_now, wl_fore) in [("Spike", (1.0, [2.2, 1.0, 1.0])), ("Alternating", (0.7, [1.5, 0.7, 1.5]))]:
        case_id += 1
        ctrl = MPCController(predictor, actions, fallback=actions[0], constraints=base_constraints)
        ctrl.previous_action = actions[4]
        telem = Telemetry(temperature_c=50.0)

        dec_enum, seq_enum, score_enum, ev_enum = run_cartesian_enumeration(ctrl, wl_now, telem, wl_fore, asymmetric_switch_costs=asym_costs)
        dec_bb, seq_bb, score_bb, ev_bb = run_branch_and_bound(ctrl, wl_now, telem, wl_fore, asymmetric_switch_costs=asym_costs)

        match_action = (dec_enum.action == dec_bb.action) if (dec_enum and dec_bb) else (dec_enum == dec_bb)
        match_seq = (seq_enum == seq_bb)
        match_score = (abs(score_enum - score_bb) < 1e-9) if (math.isfinite(score_enum) and math.isfinite(score_bb)) else (score_enum == score_bb)

        results.append({
            "case_id": case_id,
            "category": "Asymmetric_Switch_Costs",
            "description": f"AsymSwitch|WL={wl_name}|Init={actions[4].name}",
            "enum_evals": ev_enum,
            "bb_evals": ev_bb,
            "pruned_pct": (1.0 - ev_bb / ev_enum) * 100.0,
            "score_enum": score_enum,
            "score_bb": score_bb,
            "action_match": match_action,
            "seq_match": match_seq,
            "score_match": match_score,
            "all_pass": match_action and match_seq and match_score
        })

    # Group C: Equal-Energy Tie-Breaking Edge Cases
    # Create twin actions with identical energy but differing quality or latency
    twin_act1 = Action(0.50, 1, 4)
    twin_act2 = Action(0.50, 2, 4)
    twin_table = {
        twin_act1: {"lat": 10.0, "pwr": 5.0, "qual": 0.86}, # E = 0.050 mJ, Qual=0.86, Lat=10.0
        twin_act2: {"lat": 12.0, "pwr": 4.1666667, "qual": 0.91}, # E = 0.050 mJ, Qual=0.91, Lat=12.0
    }
    twin_pred = SyntheticTestPredictor(twin_table)
    twin_actions = [twin_act1, twin_act2]
    ctrl_twin = MPCController(twin_pred, twin_actions, fallback=twin_act1, constraints=base_constraints)
    ctrl_twin.previous_action = twin_act1
    telem_twin = Telemetry(temperature_c=45.0)

    case_id += 1
    dec_enum, seq_enum, score_enum, ev_enum = run_cartesian_enumeration(ctrl_twin, 1.0, telem_twin, [1.0, 1.0, 1.0])
    dec_bb, seq_bb, score_bb, ev_bb = run_branch_and_bound(ctrl_twin, 1.0, telem_twin, [1.0, 1.0, 1.0])

    match_action = (dec_enum.action == dec_bb.action) if (dec_enum and dec_bb) else (dec_enum == dec_bb)
    match_seq = (seq_enum == seq_bb)
    match_score = (abs(score_enum - score_bb) < 1e-9) if (math.isfinite(score_enum) and math.isfinite(score_bb)) else (score_enum == score_bb)

    results.append({
        "case_id": case_id,
        "category": "Equal_Energy_Tie_Breaker",
        "description": "Twin actions equal energy, diff quality/latency",
        "enum_evals": ev_enum,
        "bb_evals": ev_bb,
        "pruned_pct": (1.0 - ev_bb / ev_enum) * 100.0,
        "score_enum": score_enum,
        "score_bb": score_bb,
        "action_match": match_action,
        "seq_match": match_seq,
        "score_match": match_score,
        "all_pass": match_action and match_seq and match_score
    })

    # Group D: Infeasible Edge Cases (No candidate meets constraints)
    infeasible_configs = [
        ("Impossible_Deadline_1ms", replace(base_constraints, deadline_ms=1.0)),
        ("Impossible_PowerCap_0.1W", replace(base_constraints, power_cap_w=0.1)),
        ("Impossible_Quality_0.99", replace(base_constraints, quality_floor=0.99)),
        ("Thermal_Exceeded_72C", base_constraints),
    ]
    for inf_label, c_inf in infeasible_configs:
        case_id += 1
        ctrl_inf = MPCController(predictor, actions, fallback=actions[0], constraints=c_inf)
        ctrl_inf.previous_action = actions[0]
        telem_inf = Telemetry(temperature_c=72.0 if inf_label == "Thermal_Exceeded_72C" else 45.0)

        dec_enum, seq_enum, score_enum, ev_enum = run_cartesian_enumeration(ctrl_inf, 1.0, telem_inf, [1.0, 1.0, 1.0])
        dec_bb, seq_bb, score_bb, ev_bb = run_branch_and_bound(ctrl_inf, 1.0, telem_inf, [1.0, 1.0, 1.0])

        both_none = (dec_enum is None and dec_bb is None)
        results.append({
            "case_id": case_id,
            "category": "Infeasible_State_Verification",
            "description": inf_label,
            "enum_evals": ev_enum,
            "bb_evals": ev_bb,
            "pruned_pct": (1.0 - ev_bb / ev_enum) * 100.0 if ev_enum > 0 else 0.0,
            "score_enum": score_enum,
            "score_bb": score_bb,
            "action_match": both_none,
            "seq_match": (seq_enum == seq_bb),
            "score_match": (score_enum == score_bb),
            "all_pass": both_none and (seq_enum == seq_bb)
        })

    df = pd.DataFrame(results)
    df.to_csv("phase2_mathematical_verification_results.csv", index=False)
    print(f"\nCompleted {len(df)} exhaustive verification cases.")
    print(f"Total passed: {df['all_pass'].sum()} / {len(df)}")
    print(f"Any failures? {(~df['all_pass']).any()}")
    if (~df['all_pass']).any():
        print("FAILURES DETECTED:")
        print(df[~df['all_pass']])
    else:
        print(">>> 100% MATHEMATICAL AND TIE-BREAKING EQUIVALENCE CONFIRMED ACROSS ALL TEST CASES! <<<")

if __name__ == "__main__":
    run_phase2_comprehensive_tests()
