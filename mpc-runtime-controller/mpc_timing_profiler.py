#!/usr/bin/env python3
"""Detailed instrumentation and benchmark for the MPC decision core.

Measures sample count, median, mean, p95, max, and decomposes execution time into:
1. Predictor queries (per-action evaluation across horizon)
2. Feasibility constraint checks (deadline, power, quality)
3. Thermal state evolution (_temperature_next)
4. Objective scoring (energy + latency penalty + switch penalty)
5. Beam sorting and pruning
"""
from __future__ import annotations
import math
import time
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np

from practical_mpc_controller import (
    Action, Constraints, DEFAULT_ACTIONS, MPCController, Prediction, Telemetry, Decision
)

class MockCalibratedPredictor:
    """Deterministic synthetic predictor for benchmark reproducibility."""
    def __init__(self, actions: Tuple[Action, ...] = DEFAULT_ACTIONS):
        self.profiles = {}
        for a in actions:
            # Synthetic but realistic latency and power curves
            lat = a.width * 8.0 + a.exit * 2.5 + (16 - a.bits) * 0.5
            pwr = 0.4 + a.width * 0.8 + (a.bits / 8.0) * 0.3
            self.profiles[a] = {
                "latency": lat,
                "power": pwr,
                "energy": lat * pwr * 0.001 * 1000.0, # mJ
                "quality": 0.86 + (a.width * 0.08) - (3 - a.exit) * 0.02,
                "latency_std": 0.1,
                "power_std": 0.05,
                "energy_std": 0.1,
                "quality_std": 0.01,
            }

    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        p = self.profiles[action]
        scale = float(workload["workload_scale"]) if isinstance(workload, dict) else float(workload)
        return Prediction(
            latency_ms=p["latency"] * scale,
            power_w=p["power"],
            energy_mj=p["energy"] * scale,
            validated_quality=p["quality"],
            latency_std_ms=p["latency_std"],
            power_std_w=p["power_std"],
            energy_std_mj=p["energy_std"],
            quality_std=p["quality_std"]
        )


class InstrumentingMPCController(MPCController):
    """Subclass that instruments each phase of the MPC decision loop."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_timing_breakdown: Dict[str, float] = {}

    def choose_instrumented(self, workload_now: object, telemetry: Telemetry,
                            workload_forecast: Optional[List[object]] = None) -> Tuple[Decision, Dict[str, float]]:
        t_total_start = time.perf_counter()
        override = self._hard_override(telemetry)
        if override:
            self.previous_action = self.fallback
            tot = (time.perf_counter() - t_total_start) * 1000.0
            return Decision(self.fallback, "SAFETY_FALLBACK", override), {"total_ms": tot}

        thermal_mode = self.thermal_model_ready

        workloads = [workload_now]
        if workload_forecast:
            workloads.extend(list(workload_forecast[:max(0, self.c.horizon - 1)]))
        while len(workloads) < self.c.horizon:
            workloads.append(workloads[-1])

        # Phase 1: Prediction
        t_pred_start = time.perf_counter()
        pred: List[Dict[Action, Prediction]] = []
        for wl in workloads:
            row: Dict[Action, Prediction] = {}
            for action in self.validated_actions:
                try:
                    p = self.predictor.predict(action, wl, telemetry)
                    values = [p.latency_ms, p.power_w, p.energy_mj]
                    if any((not math.isfinite(v) or v < 0) for v in values):
                        continue
                    row[action] = p
                except Exception:
                    continue
            pred.append(row)
        t_pred = (time.perf_counter() - t_pred_start) * 1000.0

        # Phase 2-5: Beam search inner loops
        t_feasibility = 0.0
        t_thermal = 0.0
        t_scoring = 0.0
        t_sorting = 0.0

        beam = [(0.0, telemetry.temperature_c, None, tuple())]
        first_step_feasible = set()

        for step in range(self.c.horizon):
            next_beam = []
            for score, temp, first, seq in beam:
                for action, p in pred[step].items():
                    # Feasibility
                    t_f0 = time.perf_counter()
                    lat, power, energy, quality = self._robust(p)
                    feas = (lat <= self.c.deadline_ms and power <= self.c.power_cap_w and
                            quality is not None and quality >= self.c.quality_floor)
                    t_feasibility += (time.perf_counter() - t_f0) * 1000.0
                    if not feas:
                        continue

                    # Thermal
                    t_th0 = time.perf_counter()
                    tnext = self._temperature_next(temp, power) if thermal_mode else None
                    if thermal_mode and (tnext is None or tnext >= self.c.temperature_cap_c):
                        t_thermal += (time.perf_counter() - t_th0) * 1000.0
                        continue
                    t_thermal += (time.perf_counter() - t_th0) * 1000.0

                    if step == 0:
                        first_step_feasible.add(action)

                    # Scoring
                    t_sc0 = time.perf_counter()
                    step_score = energy + self.c.latency_penalty * (lat / self.c.deadline_ms) ** 2
                    if seq and seq[-1] != action:
                        step_score += self.c.switch_penalty_mj
                    elif not seq and self.previous_action is not None and self.previous_action != action:
                        step_score += self.c.switch_penalty_mj
                    next_beam.append((score + step_score, temp if tnext is None else tnext,
                                      action if first is None else first, seq + (action,)))
                    t_scoring += (time.perf_counter() - t_sc0) * 1000.0

            if not next_beam:
                break

            # Sorting & pruning
            t_so0 = time.perf_counter()
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:max(1, self.c.beam_width)]
            t_sorting += (time.perf_counter() - t_so0) * 1000.0

        complete_sequences = [state for state in beam if len(state[3]) == self.c.horizon]
        if not complete_sequences or not first_step_feasible:
            self.previous_action = self.fallback
            tot = (time.perf_counter() - t_total_start) * 1000.0
            comp_sum = t_pred + t_feasibility + t_thermal + t_scoring + t_sorting
            overhead = max(0.0, tot - comp_sum)
            breakdown = {
                "total_ms": tot, "prediction_ms": t_pred,
                "feasibility_ms": t_feasibility, "thermal_ms": t_thermal,
                "scoring_ms": t_scoring, "sorting_ms": t_sorting,
                "overhead_ms": overhead
            }
            return Decision(self.fallback, "SAFETY_FALLBACK", "No fully feasible sequence"), breakdown

        score, temp, first_action, sequence = min(complete_sequences, key=lambda x: x[0])
        p = pred[0][first_action]
        lat, power, energy, _ = self._robust(p)
        tnext = self._temperature_next(telemetry.temperature_c, power) if thermal_mode else None
        self.previous_action = first_action
        tot = (time.perf_counter() - t_total_start) * 1000.0

        # Account explicitly for inter-block overhead:
        # Loop control, dict iteration, branch evaluation, candidate tuple creation,
        # post-loop sequence selection, and timer call probe effect.
        comp_sum = t_pred + t_feasibility + t_thermal + t_scoring + t_sorting
        overhead = max(0.0, tot - comp_sum)

        breakdown = {
            "total_ms": tot, "prediction_ms": t_pred,
            "feasibility_ms": t_feasibility, "thermal_ms": t_thermal,
            "scoring_ms": t_scoring, "sorting_ms": t_sorting,
            "overhead_ms": overhead
        }
        dec = Decision(first_action, "MPC", "Lowest-energy feasible sequence",
                       lat, power, energy, tnext, len(first_step_feasible), tuple(a.name for a in sequence))
        return dec, breakdown


def run_benchmark(n_samples: int = 200, with_thermal: bool = True) -> Dict[str, float]:
    predictor = MockCalibratedPredictor(DEFAULT_ACTIONS)
    c = Constraints(
        thermal_resistance_c_per_w=2.0 if with_thermal else None,
        deadline_ms=50.0,
        power_cap_w=2.0,
        temperature_cap_c=70.0,
        quality_floor=0.85,
        horizon=5,
        beam_width=128
    )
    ctrl = InstrumentingMPCController(predictor, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0], constraints=c)
    plain_ctrl = MPCController(predictor, DEFAULT_ACTIONS, fallback=DEFAULT_ACTIONS[0], constraints=c)
    telem = Telemetry(temperature_c=45.0)

    # 1. Measure pure uninstrumented controller (no probe overhead)
    # Warmup 20 iterations
    for _ in range(20):
        plain_ctrl.choose(1.0, telem, [1.0]*4)

    plain_times = []
    for _ in range(n_samples):
        scale = 1.0 + (np.random.rand() - 0.5) * 0.2
        t0 = time.perf_counter()
        plain_ctrl.choose(scale, telem, [scale]*4)
        plain_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Measure instrumented controller with component breakdown
    # Warmup 20 iterations
    for _ in range(20):
        ctrl.choose_instrumented(1.0, telem, [1.0]*4)

    total_times = []
    pred_times = []
    feas_times = []
    therm_times = []
    score_times = []
    sort_times = []
    overhead_times = []

    for _ in range(n_samples):
        # vary workload scale slightly
        scale = 1.0 + (np.random.rand() - 0.5) * 0.2
        _, b = ctrl.choose_instrumented(scale, telem, [scale]*4)
        total_times.append(b["total_ms"])
        pred_times.append(b["prediction_ms"])
        feas_times.append(b["feasibility_ms"])
        therm_times.append(b["thermal_ms"])
        score_times.append(b["scoring_ms"])
        sort_times.append(b["sorting_ms"])
        overhead_times.append(b["overhead_ms"])

    total_times.sort()
    plain_times.sort()
    p95_idx = int(0.95 * len(total_times))

    stats = {
        "samples": float(n_samples),
        "uninstrumented_mean_ms": float(statistics.mean(plain_times)),
        "uninstrumented_median_ms": float(statistics.median(plain_times)),
        "uninstrumented_p95_ms": float(plain_times[p95_idx]),
        "mean_ms": float(statistics.mean(total_times)),
        "median_ms": float(statistics.median(total_times)),
        "p95_ms": float(total_times[p95_idx]),
        "max_ms": float(max(total_times)),
        "min_ms": float(min(total_times)),
        "stdev_ms": float(statistics.stdev(total_times)),
        "pred_mean_ms": float(statistics.mean(pred_times)),
        "feas_mean_ms": float(statistics.mean(feas_times)),
        "therm_mean_ms": float(statistics.mean(therm_times)),
        "score_mean_ms": float(statistics.mean(score_times)),
        "sort_mean_ms": float(statistics.mean(sort_times)),
        "overhead_mean_ms": float(statistics.mean(overhead_times)),
    }
    return stats


if __name__ == "__main__":
    def print_report(title: str, s: Dict[str, float]):
        print("=" * 65)
        print(title)
        print("=" * 65)
        print(f"  Samples                         : {s['samples']:8.0f}")
        print(f"  Uninstrumented Baseline Mean    : {s['uninstrumented_mean_ms']:8.3f} ms (p95: {s['uninstrumented_p95_ms']:.3f} ms)")
        print(f"  Instrumented Total Mean         : {s['mean_ms']:8.3f} ms")
        print(f"    - Median                      : {s['median_ms']:8.3f} ms")
        print(f"    - P95                         : {s['p95_ms']:8.3f} ms")
        print(f"    - Min                         : {s['min_ms']:8.3f} ms")
        print(f"    - Max                         : {s['max_ms']:8.3f} ms")
        print(f"    - Std Dev                     : {s['stdev_ms']:8.3f} ms")
        print("  Component Decomposition:")
        print(f"    1. Prediction                 : {s['pred_mean_ms']:8.3f} ms ({100.0 * s['pred_mean_ms'] / s['mean_ms']:5.1f}%)")
        print(f"    2. Feasibility Checks         : {s['feas_mean_ms']:8.3f} ms ({100.0 * s['feas_mean_ms'] / s['mean_ms']:5.1f}%)")
        print(f"    3. Thermal State Model        : {s['therm_mean_ms']:8.3f} ms ({100.0 * s['therm_mean_ms'] / s['mean_ms']:5.1f}%)")
        print(f"    4. Cost Function Scoring      : {s['score_mean_ms']:8.3f} ms ({100.0 * s['score_mean_ms'] / s['mean_ms']:5.1f}%)")
        print(f"    5. Beam Sorting & Pruning     : {s['sort_mean_ms']:8.3f} ms ({100.0 * s['sort_mean_ms'] / s['mean_ms']:5.1f}%)")
        print(f"    6. Inter-Block/Probe Overhead : {s['overhead_mean_ms']:8.3f} ms ({100.0 * s['overhead_mean_ms'] / s['mean_ms']:5.1f}%)")
        comp_sum = (s['pred_mean_ms'] + s['feas_mean_ms'] + s['therm_mean_ms'] +
                    s['score_mean_ms'] + s['sort_mean_ms'] + s['overhead_mean_ms'])
        print(f"  Reconciled Sum                  : {comp_sum:8.3f} ms (Delta vs Mean: {abs(s['mean_ms'] - comp_sum):.5f} ms)")

    s1 = run_benchmark(n_samples=200, with_thermal=True)
    print_report("BENCHMARK 1: Standard MPC (Horizon=5, Actions=18, Thermal Enabled)", s1)

    print()
    s2 = run_benchmark(n_samples=200, with_thermal=False)
    print_report("BENCHMARK 2: Standard MPC (Horizon=5, Actions=18, Thermal Disabled)", s2)
