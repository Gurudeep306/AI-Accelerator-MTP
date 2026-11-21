#!/usr/bin/env python3
"""Workload Prediction Benchmarking and Evaluation Framework.

Compares 6 workload prediction methods across 9 synthetic perception traces:
1. Persistence Predictor: w_hat[t+h] = w[t]
2. Pure EWMA Predictor: w_hat[t+h] = ewma[t]
3. Controller A Forecaster: EWMA (alpha=0.45) + clipped trend (flat replication across horizon)
4. Controller B Step-0 Preview: 7-feature Ridge Regression (evaluating immediate frame preview)
5. Controller B Multi-Step Forecaster: 2-point slope extrapolation (h=1..5)
6. Controller C Estimator: Static First-Layer MAC fraction (flat replication across horizon)

Evaluates accuracy across horizons h in {1, 2, 3, 4, 5}:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Maximum Absolute Error (MaxError)
- Step-by-step Horizon Degradation
- Execution Latency (microseconds per invocation)
"""
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from frame_preview_mpc import LinearWorkloadModel, TemporalWorkloadForecaster
from history_mpc_controller import HistoryWorkloadPredictor, WorkloadObservation

# ==============================================================================
# 1. EVALUATION-ONLY PREDICTOR IMPLEMENTATIONS
# ==============================================================================

class PersistencePredictor:
    """Baseline: Forecast equals the last observed ground-truth workload."""
    def __init__(self):
        self.last_val = 1.0

    def update(self, val: float):
        self.last_val = float(val)

    def forecast(self, steps: int) -> List[float]:
        return [self.last_val] * steps


class PureEWMAPredictor:
    """Standard Exponentially Weighted Moving Average (alpha = 0.45)."""
    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha
        self.ewma = None

    def update(self, val: float):
        val = float(val)
        if self.ewma is None:
            self.ewma = val
        else:
            self.ewma = self.alpha * val + (1.0 - self.alpha) * self.ewma

    def forecast(self, steps: int) -> List[float]:
        val = self.ewma if self.ewma is not None else 1.0
        return [val] * steps


class ControllerAPredictor:
    """Controller A: Production EWMA (alpha=0.45) + clipped recent trend."""
    def __init__(self, history_size: int = 12, alpha: float = 0.45, max_relative_step: float = 0.50):
        self.predictor = HistoryWorkloadPredictor(history_size=history_size, alpha=alpha, max_relative_step=max_relative_step)

    def update(self, frame_id: int, val: float):
        self.predictor.update(WorkloadObservation(frame_id, float(val)))

    def forecast(self, steps: int) -> List[float]:
        if not self.predictor.history:
            return [1.0] * steps
        pred = self.predictor.predict_next()
        # In Controller A, the single 1-step forecast is replicated flat across all future steps
        return [pred.workload_scale] * steps


class ControllerBPreviewPipeline:
    """Controller B: Ridge preview for step 0 + bounded slope extrapolation for steps 1..H."""
    def __init__(self, model_path: str = "synthetic_workload_model.json", history_size: int = 5):
        self.model = LinearWorkloadModel(model_path)
        self.forecaster = TemporalWorkloadForecaster(history_size=history_size, max_relative_step=0.25)

    def predict_current_and_forecast(self, features: Dict[str, float], steps: int) -> Tuple[float, List[float]]:
        # 1. Ridge regression on current frame preview features
        est = self.model.predict(features)
        # 2. Update sliding history deque
        self.forecaster.update(est.workload_scale)
        # 3. Forecast future steps using 2-point slope extrapolation
        future = self.forecaster.forecast(steps, self.model.min_target, self.model.max_target)
        return est.workload_scale, future


class ControllerCEstimator:
    """Controller C: Static MAC-based remaining work estimator (1.0 - L1_frac)."""
    def __init__(self, total_macs: float = 1000.0, l1_frac: float = 0.15):
        self.total_macs = total_macs
        self.l1_frac = l1_frac

    def estimate_current_and_forecast(self, steps: int) -> Tuple[float, List[float]]:
        # Remaining workload scale: (total - l1_macs) / total = 1.0 - l1_frac
        rem_scale = max(1e-3, (self.total_macs - self.l1_frac * self.total_macs) / self.total_macs)
        # Replicated flat across the horizon
        return rem_scale, [rem_scale] * steps


# ==============================================================================
# 2. WORKLOAD PATTERN TRACE GENERATOR
# ==============================================================================

def generate_test_traces(n_frames: int = 60, seed: int = 42) -> Dict[str, List[Tuple[float, Dict[str, float]]]]:
    """Generates 9 diverse workload traces pairing actual workload with synthetic preview features."""
    np.random.seed(seed)
    traces = {}

    def make_features(wl: float) -> Dict[str, float]:
        # Synthesize preview features correlated with workload scale (matching calibration regime)
        norm_wl = (wl - 0.8) / 0.8 # rough normalization
        return {
            "brightness_mean": float(np.clip(0.47 + 0.10 * norm_wl + np.random.normal(0, 0.02), 0.1, 0.9)),
            "brightness_std": float(np.clip(0.22 + 0.03 * norm_wl + np.random.normal(0, 0.01), 0.05, 0.4)),
            "edge_density": float(np.clip(0.20 + 0.12 * norm_wl + np.random.normal(0, 0.02), 0.05, 0.5)),
            "histogram_entropy": float(np.clip(0.60 + 0.15 * norm_wl + np.random.normal(0, 0.02), 0.3, 0.95)),
            "temporal_difference": float(np.clip(0.10 + 0.08 * norm_wl + np.random.normal(0, 0.01), 0.01, 0.35)),
            "aspect_ratio": 16.0 / 9.0,
            "normalized_pixel_count": 1.0
        }

    # 1. Stable Highway (calm road, low constant complexity)
    w_hw = [float(0.85 + 0.03 * np.sin(i / 3.0) + np.random.normal(0, 0.01)) for i in range(n_frames)]
    traces["Stable_Highway"] = [(w, make_features(w)) for w in w_hw]

    # 2. Stable Urban (steady city navigation, medium-high constant complexity)
    w_urb = [float(1.35 + 0.04 * np.sin(i / 3.0) + np.random.normal(0, 0.015)) for i in range(n_frames)]
    traces["Stable_Urban"] = [(w, make_features(w)) for w in w_urb]

    # 3. Sudden Workload Increase (step jump from 0.8 to 2.2 at frame 25)
    w_step_up = [0.80 if i < 25 else 2.20 for i in range(n_frames)]
    w_step_up = [float(w + np.random.normal(0, 0.02)) for w in w_step_up]
    traces["Sudden_Workload_Increase"] = [(w, make_features(w)) for w in w_step_up]

    # 4. Sudden Workload Decrease (step drop from 2.0 to 0.7 at frame 25)
    w_step_down = [2.00 if i < 25 else 0.70 for i in range(n_frames)]
    w_step_down = [float(w + np.random.normal(0, 0.02)) for w in w_step_down]
    traces["Sudden_Workload_Decrease"] = [(w, make_features(w)) for w in w_step_down]

    # 5. Gradual Workload Ramp (smooth linear ramp from 0.60 to 2.00)
    w_ramp = list(np.linspace(0.60, 2.00, n_frames))
    w_ramp = [float(w + np.random.normal(0, 0.02)) for w in w_ramp]
    traces["Gradual_Workload_Ramp"] = [(w, make_features(w)) for w in w_ramp]

    # 6. Alternating Workload (square wave switching every 5 frames between 0.70 and 1.60)
    w_alt = [1.60 if (i // 5) % 2 == 1 else 0.70 for i in range(n_frames)]
    w_alt = [float(w + np.random.normal(0, 0.02)) for w in w_alt]
    traces["Alternating_Workload"] = [(w, make_features(w)) for w in w_alt]

    # 7. Highway-to-Urban Transition (smooth sigmoid shift from 0.85 to 1.45 at frames 20..35)
    w_h2u = [float(0.85 + 0.60 / (1.0 + np.exp(-(i - 25) / 2.5)) + np.random.normal(0, 0.02)) for i in range(n_frames)]
    traces["Highway_to_Urban_Transition"] = [(w, make_features(w)) for w in w_h2u]

    # 8. Urban-to-Highway Transition (smooth sigmoid drop from 1.45 to 0.85 at frames 20..35)
    w_u2h = [float(1.45 - 0.60 / (1.0 + np.exp(-(i - 25) / 2.5)) + np.random.normal(0, 0.02)) for i in range(n_frames)]
    traces["Urban_to_Highway_Transition"] = [(w, make_features(w)) for w in w_u2h]

    # 9. Congestion Spikes (calm road with extreme bursts of 2.50 at frames 15..18 and 38..41)
    w_spike = [0.90]*n_frames
    for i in range(15, 19): w_spike[i] = 2.45
    for i in range(38, 42): w_spike[i] = 2.55
    w_spike = [float(w + np.random.normal(0, 0.02)) for w in w_spike]
    traces["Congestion_Spikes"] = [(w, make_features(w)) for w in w_spike]

    return traces


# ==============================================================================
# 3. BENCHMARK EXECUTION & ACCURACY METRICS
# ==============================================================================

def run_prediction_benchmark():
    print("================================================================================")
    print("RUNNING WORKLOAD PREDICTION ACCURACY & HORIZON BENCHMARK")
    print("================================================================================")
    traces = generate_test_traces(n_frames=60, seed=42)
    horizons = [1, 2, 3, 4, 5]

    predictor_names = [
        "Persistence",
        "Pure_EWMA",
        "Controller_A_EWMA_Trend",
        "Controller_B_Preview_Step0",
        "Controller_B_Slope_Forecaster",
        "Controller_C_Static_MAC"
    ]

    # Storage for detailed horizon-level errors: (predictor, pattern, horizon, error, abs_error, pct_error)
    horizon_records = []
    # Storage for aggregated predictor-level metrics:
    overall_records = []
    timing_records = {p: [] for p in predictor_names}

    for pattern_name, trace_data in traces.items():
        n_frames = len(trace_data)
        actuals = [w for w, _ in trace_data]

        # Instantiate fresh predictors per trace
        p_persist = PersistencePredictor()
        p_ewma = PureEWMAPredictor(alpha=0.45)
        p_ctrl_a = ControllerAPredictor(history_size=12, alpha=0.45, max_relative_step=0.50)
        p_ctrl_b = ControllerBPreviewPipeline(history_size=5)
        p_ctrl_c = ControllerCEstimator(total_macs=1000.0, l1_frac=0.15)

        for t in range(n_frames):
            actual_now = actuals[t]
            feats_now = trace_data[t][1]

            # 1. Persistence
            t0 = time.perf_counter_ns()
            pred_persist = p_persist.forecast(5)
            timing_records["Persistence"].append((time.perf_counter_ns() - t0) / 1000.0)
            p_persist.update(actual_now)

            # 2. Pure EWMA
            t0 = time.perf_counter_ns()
            pred_ewma = p_ewma.forecast(5)
            timing_records["Pure_EWMA"].append((time.perf_counter_ns() - t0) / 1000.0)
            p_ewma.update(actual_now)

            # 3. Controller A
            t0 = time.perf_counter_ns()
            pred_ctrl_a = p_ctrl_a.forecast(5)
            timing_records["Controller_A_EWMA_Trend"].append((time.perf_counter_ns() - t0) / 1000.0)
            p_ctrl_a.update(t, actual_now)

            # 4. Controller B (Preview Step 0 + Slope Extrapolation for steps 1..5)
            t0 = time.perf_counter_ns()
            est_b_now, future_b = p_ctrl_b.predict_current_and_forecast(feats_now, 5)
            dt_b_us = (time.perf_counter_ns() - t0) / 1000.0
            timing_records["Controller_B_Slope_Forecaster"].append(dt_b_us)
            timing_records["Controller_B_Preview_Step0"].append(dt_b_us)
            pred_ctrl_b = future_b

            # 5. Controller C (Static MAC ratio 1 - 0.15 = 0.85)
            t0 = time.perf_counter_ns()
            est_c_now, future_c = p_ctrl_c.estimate_current_and_forecast(5)
            timing_records["Controller_C_Static_MAC"].append((time.perf_counter_ns() - t0) / 1000.0)
            pred_ctrl_c = future_c

            # Evaluate predictions for horizons h in 1..5
            preds_map = {
                "Persistence": pred_persist,
                "Pure_EWMA": pred_ewma,
                "Controller_A_EWMA_Trend": pred_ctrl_a,
                "Controller_B_Preview_Step0": [est_b_now] * 5,
                "Controller_B_Slope_Forecaster": pred_ctrl_b,
                "Controller_C_Static_MAC": pred_ctrl_c
            }

            for p_name, forecast_list in preds_map.items():
                for h_idx, h in enumerate(horizons):
                    target_t = t + h
                    if target_t < n_frames:
                        y_true = actuals[target_t]
                        y_hat = forecast_list[h_idx]
                        err = y_hat - y_true
                        abs_err = abs(err)
                        pct_err = (abs_err / y_true) * 100.0

                        horizon_records.append({
                            "pattern": pattern_name,
                            "predictor": p_name,
                            "frame_t": t,
                            "horizon": h,
                            "y_true": y_true,
                            "y_pred": y_hat,
                            "error": err,
                            "abs_error": abs_err,
                            "pct_error": pct_err
                        })

    df_horizons = pd.DataFrame(horizon_records)
    df_horizons.to_csv("workload_prediction_horizon_results.csv", index=False)
    print("Saved horizon-level raw results to: workload_prediction_horizon_results.csv")

    # Aggregate by Predictor and Pattern
    summary_list = []
    for (p_name, pat_name), grp in df_horizons.groupby(["predictor", "pattern"]):
        mae = grp["abs_error"].mean()
        rmse = np.sqrt((grp["error"] ** 2).mean())
        mape = grp["pct_error"].mean()
        max_err = grp["abs_error"].max()
        summary_list.append({
            "predictor": p_name,
            "pattern": pat_name,
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape,
            "max_error": max_err
        })

    # Also compute aggregate across all patterns
    overall_list = []
    for p_name in predictor_names:
        sub = df_horizons[df_horizons["predictor"] == p_name]
        mae = sub["abs_error"].mean()
        rmse = np.sqrt((sub["error"] ** 2).mean())
        mape = sub["pct_error"].mean()
        max_err = sub["abs_error"].max()
        latencies = timing_records[p_name]
        mean_lat_us = float(np.mean(latencies))
        p95_lat_us = float(np.percentile(latencies, 95))

        overall_list.append({
            "predictor": p_name,
            "overall_mae": mae,
            "overall_rmse": rmse,
            "overall_mape_pct": mape,
            "overall_max_error": max_err,
            "mean_latency_us": mean_lat_us,
            "p95_latency_us": p95_lat_us
        })

    df_overall = pd.DataFrame(overall_list)
    df_overall.to_csv("workload_prediction_results.csv", index=False)
    print("Saved aggregated predictor results to: workload_prediction_results.csv")

    print("\n================== OVERALL PREDICTOR BENCHMARK RESULTS ==================")
    print(df_overall.to_string(index=False))

    # Horizon Degradation Table
    print("\n================== MAE BY PREDICTION HORIZON (h = 1 to 5) ==================")
    horizon_piv = df_horizons.pivot_table(index="predictor", columns="horizon", values="abs_error", aggfunc="mean")
    print(horizon_piv.to_string())

if __name__ == "__main__":
    run_prediction_benchmark()
