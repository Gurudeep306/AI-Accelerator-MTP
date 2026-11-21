#!/usr/bin/env python3
"""EXP-01: Multi-Step Workload Prediction Benchmarking.

Evaluates multi-step workload prediction accuracy across forecast horizons h in {1, 2, 3, 4, 5}:
1. Controller A: Historical EWMA (alpha = 0.45) + clipped recent trend
2. Controller A: Linear AR(p=3) [Verified: Unsupported in codebase, documented as limitation]
3. Controller B: Frame-preview workload prediction + bounded temporal slope forecaster
4. Persistence baseline: Last observed completed-frame workload
5. Ground-truth oracle: Acausal perfect foresight reference

Evaluates across official NuScenes 6/2/2 sequence splits (Train/Val/Test), prioritizing
held-out Test sequences (scene-1094, scene-1100) under both dynamic sparse and dense modes.
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frame_preview_mpc import FEATURE_NAMES, LinearWorkloadModel, TemporalWorkloadForecaster
from history_mpc_controller import HistoryWorkloadPredictor, WorkloadObservation


# ==============================================================================
# DATASET SPLIT DEFINITIONS (from CORRECTED_NUSCENES_SPLIT.md)
# ==============================================================================
SPLITS = {
    "train": ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757", "scene-0796"],
    "val": ["scene-0916", "scene-1077"],
    "test": ["scene-1094", "scene-1100"],
}

SCENE_TO_SPLIT = {}
for split_name, scenes in SPLITS.items():
    for s in scenes:
        SCENE_TO_SPLIT[s] = split_name


# ==============================================================================
# PREDICTOR WRAPPERS WITH STRICT CAUSAL BOUNDARIES
# ==============================================================================
class PersistencePredictorWrapper:
    """Persistence baseline: forecasts latest completed frame workload."""
    def __init__(self, cold_start_val: float = 1.0):
        self.cold_start_val = cold_start_val
        self.last_val = cold_start_val

    def update(self, completed_wl: float):
        self.last_val = float(completed_wl)

    def forecast(self, steps: int) -> List[float]:
        return [self.last_val] * steps


class ControllerAPredictorWrapper:
    """Controller A: Production EWMA (alpha=0.45) + clipped recent trend."""
    def __init__(self, history_size: int = 12, alpha: float = 0.45, max_relative_step: float = 0.50):
        self.predictor = HistoryWorkloadPredictor(
            history_size=history_size, alpha=alpha, max_relative_step=max_relative_step
        )
        self.has_history = False

    def update(self, frame_id: int, completed_wl: float):
        self.predictor.update(WorkloadObservation(frame_id, float(completed_wl)))
        self.has_history = True

    def forecast(self, steps: int) -> List[float]:
        if not self.has_history:
            return [1.0] * steps
        pred = self.predictor.predict_next()
        # In Controller A production, the 1-step forecast is replicated flat across future horizon
        return [pred.workload_scale] * steps


class ControllerBPipelineWrapper:
    """Controller B: Ridge preview for frame t + temporal slope forecaster for future frames."""
    def __init__(self, model_path: str, history_size: int = 5, max_relative_step: float = 0.25):
        self.model = LinearWorkloadModel(model_path)
        self.forecaster = TemporalWorkloadForecaster(
            history_size=history_size, max_relative_step=max_relative_step
        )

    def predict_current(self, features: Dict[str, float]) -> float:
        est = self.model.predict(features)
        return est.workload_scale

    def update_forecaster(self, current_pred: float):
        self.forecaster.update(current_pred)

    def forecast_future(self, steps: int) -> List[float]:
        return self.forecaster.forecast(steps, self.model.min_target, self.model.max_target)


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
def run_benchmark():
    results_dir = Path("results/EXP-01")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING EXP-01: MULTI-STEP WORKLOAD PREDICTION BENCHMARK")
    print("=" * 80)

    prediction_records = []
    timing_data = {
        "Persistence": [],
        "Controller_A_EWMA_Trend": [],
        "Controller_B_Preview_Inference": [],
        "Controller_B_Slope_Forecaster": [],
        "Controller_B_Total_Pipeline": [],
    }

    modes = ["sparse", "dense"]
    horizons = [1, 2, 3, 4, 5]

    for mode in modes:
        model_path = f"real_driving_workload_model_{mode}.json"
        if not Path(model_path).exists():
            print(f"Warning: {model_path} not found, falling back to synthetic_workload_model.json")
            model_path = "synthetic_workload_model.json"

        trace_dir = Path(f"traces/{mode}")
        scene_files = sorted(trace_dir.glob("scene-*.csv"))

        print(f"\nProcessing {len(scene_files)} scenes for {mode.upper()} mode using model: {model_path}")

        for sfile in scene_files:
            scene_id = sfile.stem
            split = SCENE_TO_SPLIT.get(scene_id, "unknown")
            df = pd.read_csv(sfile)
            n_frames = len(df)

            p_persist = PersistencePredictorWrapper(cold_start_val=1.0)
            p_ctrl_a = ControllerAPredictorWrapper(history_size=12, alpha=0.45)
            p_ctrl_b = ControllerBPipelineWrapper(model_path=model_path, history_size=5)

            actuals = df["actual_workload_scale"].to_numpy()

            for t in range(n_frames):
                row = df.iloc[t]
                feats = {k: float(row[k]) for k in FEATURE_NAMES}

                # 1. Update causal completed-frame observations (from frame t-1)
                if t > 0:
                    p_persist.update(actuals[t - 1])
                    p_ctrl_a.update(frame_id=t - 1, completed_wl=actuals[t - 1])

                # 2. Timing & execution of Controller A and Persistence
                t0 = time.perf_counter_ns()
                f_persist = p_persist.forecast(5)
                dt_persist = (time.perf_counter_ns() - t0) / 1000.0
                timing_data["Persistence"].append(dt_persist)

                t0 = time.perf_counter_ns()
                f_ctrl_a = p_ctrl_a.forecast(5)
                dt_ctrl_a = (time.perf_counter_ns() - t0) / 1000.0
                timing_data["Controller_A_EWMA_Trend"].append(dt_ctrl_a)

                # 3. Timing & execution of Controller B Preview & Forecaster
                t0 = time.perf_counter_ns()
                step0_b = p_ctrl_b.predict_current(feats)
                dt_preview = (time.perf_counter_ns() - t0) / 1000.0
                timing_data["Controller_B_Preview_Inference"].append(dt_preview)

                t0 = time.perf_counter_ns()
                p_ctrl_b.update_forecaster(step0_b)
                future_b = p_ctrl_b.forecast_future(5)
                dt_slope = (time.perf_counter_ns() - t0) / 1000.0
                timing_data["Controller_B_Slope_Forecaster"].append(dt_slope)
                timing_data["Controller_B_Total_Pipeline"].append(dt_preview + dt_slope)

                # Record Step-0 Preview (Current frame t, h=0)
                actual_now = actuals[t]
                err_step0 = step0_b - actual_now
                prediction_records.append({
                    "trace_mode": mode,
                    "split": split,
                    "scene_id": scene_id,
                    "frame_t": t,
                    "target_frame": t,
                    "horizon": 0,
                    "model_name": "Controller_B_Preview_Step0",
                    "y_true": actual_now,
                    "y_pred": step0_b,
                    "error": err_step0,
                    "abs_error": abs(err_step0),
                    "squared_error": err_step0 ** 2,
                })

                # Record Multi-Step Forecasts for h in {1, 2, 3, 4, 5}
                for h_idx, h in enumerate(horizons):
                    target_t = t + h
                    if target_t < n_frames:
                        y_target = actuals[target_t]

                        # Persistence
                        pred_p = f_persist[h_idx]
                        err_p = pred_p - y_target
                        prediction_records.append({
                            "trace_mode": mode,
                            "split": split,
                            "scene_id": scene_id,
                            "frame_t": t,
                            "target_frame": target_t,
                            "horizon": h,
                            "model_name": "Persistence",
                            "y_true": y_target,
                            "y_pred": pred_p,
                            "error": err_p,
                            "abs_error": abs(err_p),
                            "squared_error": err_p ** 2,
                        })

                        # Controller A (EWMA + Trend)
                        pred_a = f_ctrl_a[h_idx]
                        err_a = pred_a - y_target
                        prediction_records.append({
                            "trace_mode": mode,
                            "split": split,
                            "scene_id": scene_id,
                            "frame_t": t,
                            "target_frame": target_t,
                            "horizon": h,
                            "model_name": "Controller_A_EWMA_Trend",
                            "y_true": y_target,
                            "y_pred": pred_a,
                            "error": err_a,
                            "abs_error": abs(err_a),
                            "squared_error": err_a ** 2,
                        })

                        # Controller B (Forecaster)
                        pred_b = future_b[h_idx]
                        err_b = pred_b - y_target
                        prediction_records.append({
                            "trace_mode": mode,
                            "split": split,
                            "scene_id": scene_id,
                            "frame_t": t,
                            "target_frame": target_t,
                            "horizon": h,
                            "model_name": "Controller_B_Preview_Forecaster",
                            "y_true": y_target,
                            "y_pred": pred_b,
                            "error": err_b,
                            "abs_error": abs(err_b),
                            "squared_error": err_b ** 2,
                        })

                        # Ground Truth Oracle Reference
                        prediction_records.append({
                            "trace_mode": mode,
                            "split": split,
                            "scene_id": scene_id,
                            "frame_t": t,
                            "target_frame": target_t,
                            "horizon": h,
                            "model_name": "Ground_Truth_Oracle",
                            "y_true": y_target,
                            "y_pred": y_target,
                            "error": 0.0,
                            "abs_error": 0.0,
                            "squared_error": 0.0,
                        })

    df_preds = pd.DataFrame(prediction_records)
    preds_csv = results_dir / "exp01_predictions.csv"
    df_preds.to_csv(preds_csv, index=False)
    print(f"\nSaved {len(df_preds)} prediction records to {preds_csv}")

    # ==============================================================================
    # AGGREGATE METRICS CALCULATION
    # ==============================================================================
    print("\nComputing aggregate metrics...")
    agg_records = []

    subsets = {
        "Held_Out_Test": df_preds["split"] == "test",
        "All_Scenes": pd.Series(True, index=df_preds.index),
        "Validation": df_preds["split"] == "val",
        "Train": df_preds["split"] == "train",
    }

    for mode in modes:
        mode_mask = df_preds["trace_mode"] == mode
        for sub_name, sub_mask in subsets.items():
            df_sub = df_preds[mode_mask & sub_mask]
            if df_sub.empty:
                continue

            grouped = df_sub.groupby(["model_name", "horizon"])
            for (model_name, h), g in grouped:
                n = len(g)
                mae = g["abs_error"].mean()
                rmse = math.sqrt(g["squared_error"].mean())
                bias = g["error"].mean()
                mape = (g["abs_error"] / g["y_true"]).mean() * 100.0
                underpred_pct = (g["error"] < 0).mean() * 100.0
                p95_err = np.percentile(g["abs_error"], 95)
                max_err = g["abs_error"].max()

                agg_records.append({
                    "trace_mode": mode,
                    "eval_subset": sub_name,
                    "model_name": model_name,
                    "horizon": h,
                    "n_predictions": n,
                    "mae": mae,
                    "rmse": rmse,
                    "signed_bias": bias,
                    "mape_pct": mape,
                    "underprediction_rate_pct": underpred_pct,
                    "p95_abs_error": p95_err,
                    "max_abs_error": max_err,
                })

            # Add explicit documentation row for Controller A Linear AR(p=3)
            for h in horizons:
                agg_records.append({
                    "trace_mode": mode,
                    "eval_subset": sub_name,
                    "model_name": "Controller_A_Linear_AR3",
                    "horizon": h,
                    "n_predictions": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "signed_bias": np.nan,
                    "mape_pct": np.nan,
                    "underprediction_rate_pct": np.nan,
                    "p95_abs_error": np.nan,
                    "max_abs_error": np.nan,
                })

    df_agg = pd.DataFrame(agg_records)
    agg_csv = results_dir / "exp01_aggregate_metrics.csv"
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregate metrics to {agg_csv}")

    # ==============================================================================
    # PER-SCENE METRICS CALCULATION
    # ==============================================================================
    scene_records = []
    for mode in modes:
        mode_mask = df_preds["trace_mode"] == mode
        df_mode = df_preds[mode_mask]
        grouped = df_mode.groupby(["split", "scene_id", "model_name", "horizon"])
        for (split, scene_id, model_name, h), g in grouped:
            scene_records.append({
                "trace_mode": mode,
                "split": split,
                "scene_id": scene_id,
                "model_name": model_name,
                "horizon": h,
                "n_predictions": len(g),
                "mae": g["abs_error"].mean(),
                "rmse": math.sqrt(g["squared_error"].mean()),
                "signed_bias": g["error"].mean(),
            })

    df_scene = pd.DataFrame(scene_records)
    scene_csv = results_dir / "exp01_scene_metrics.csv"
    df_scene.to_csv(scene_csv, index=False)
    print(f"Saved per-scene metrics to {scene_csv}")

    # ==============================================================================
    # TIMING BENCHMARKING
    # ==============================================================================
    timing_records = []
    for op, latencies in timing_data.items():
        arr = np.array(latencies)
        timing_records.append({
            "operation": op,
            "sample_count": len(arr),
            "mean_us": float(np.mean(arr)),
            "std_us": float(np.std(arr)),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(np.max(arr)),
        })

    # Add documentation row for Controller A Linear AR3
    timing_records.append({
        "operation": "Controller_A_Linear_AR3",
        "sample_count": 0,
        "mean_us": np.nan,
        "std_us": np.nan,
        "p50_us": np.nan,
        "p95_us": np.nan,
        "p99_us": np.nan,
        "max_us": np.nan,
    })

    df_timing = pd.DataFrame(timing_records)
    timing_csv = results_dir / "exp01_timing_measurements.csv"
    df_timing.to_csv(timing_csv, index=False)
    print(f"Saved timing measurements to {timing_csv}")

    # ==============================================================================
    # GENERATE VISUALIZATIONS
    # ==============================================================================
    print("\nGenerating publication-quality visualization figures...")

    # Filter for Held-Out Test Set under Sparse Mode (the primary dynamic benchmark)
    test_sparse = df_agg[
        (df_agg["eval_subset"] == "Held_Out_Test") &
        (df_agg["trace_mode"] == "sparse") &
        (df_agg["horizon"].isin([1, 2, 3, 4, 5]))
    ]

    models_to_plot = [
        ("Controller_A_EWMA_Trend", "Controller A (EWMA, a=0.45)", "blue", "o-"),
        ("Controller_B_Preview_Forecaster", "Controller B (Preview Forecaster)", "green", "s-"),
        ("Persistence", "Persistence Baseline", "orange", "^--"),
        ("Ground_Truth_Oracle", "Oracle Reference", "black", "d:"),
    ]

    # Plot 1: MAE vs Horizon
    plt.figure(figsize=(8, 5))
    for m_id, label, color, style in models_to_plot:
        sub = test_sparse[test_sparse["model_name"] == m_id].sort_values("horizon")
        if not sub.empty:
            plt.plot(sub["horizon"], sub["mae"], style, label=label, color=color, linewidth=2, markersize=7)

    plt.title("EXP-01: Multi-Step Workload Prediction Error (MAE vs Horizon)\nHeld-Out NuScenes Test Sequences (Sparse Mode)", fontsize=12, fontweight="bold")
    plt.xlabel("Forecast Distance h (Frames Ahead)", fontsize=11)
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=11)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    mae_png = results_dir / "mae_vs_horizon.png"
    plt.savefig(mae_png, dpi=300)
    plt.close()
    print(f"Saved {mae_png}")

    # Plot 2: RMSE vs Horizon
    plt.figure(figsize=(8, 5))
    for m_id, label, color, style in models_to_plot:
        sub = test_sparse[test_sparse["model_name"] == m_id].sort_values("horizon")
        if not sub.empty:
            plt.plot(sub["horizon"], sub["rmse"], style, label=label, color=color, linewidth=2, markersize=7)

    plt.title("EXP-01: Multi-Step Workload Prediction RMSE vs Horizon\nHeld-Out NuScenes Test Sequences (Sparse Mode)", fontsize=12, fontweight="bold")
    plt.xlabel("Forecast Distance h (Frames Ahead)", fontsize=11)
    plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    rmse_png = results_dir / "rmse_vs_horizon.png"
    plt.savefig(rmse_png, dpi=300)
    plt.close()
    print(f"Saved {rmse_png}")

    # Plot 3: Signed Bias vs Horizon
    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, color="gray", linestyle="-", linewidth=1.5, alpha=0.7)
    for m_id, label, color, style in models_to_plot:
        sub = test_sparse[test_sparse["model_name"] == m_id].sort_values("horizon")
        if not sub.empty:
            plt.plot(sub["horizon"], sub["signed_bias"], style, label=label, color=color, linewidth=2, markersize=7)

    plt.title("EXP-01: Workload Prediction Signed Bias vs Horizon\nHeld-Out NuScenes Test Sequences (Sparse Mode)", fontsize=12, fontweight="bold")
    plt.xlabel("Forecast Distance h (Frames Ahead)", fontsize=11)
    plt.ylabel("Signed Bias (Prediction - Actual)", fontsize=11)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    bias_png = results_dir / "bias_vs_horizon.png"
    plt.savefig(bias_png, dpi=300)
    plt.close()
    print(f"Saved {bias_png}")

    # Plot 4: Actual vs Predicted Traces on Test Scene 1094
    df_scene_1094 = df_preds[
        (df_preds["scene_id"] == "scene-1094") &
        (df_preds["trace_mode"] == "sparse") &
        (df_preds["horizon"] == 1)
    ]

    plt.figure(figsize=(10, 5))
    ground_truth = df_scene_1094[df_scene_1094["model_name"] == "Ground_Truth_Oracle"].sort_values("frame_t")
    ctrl_a = df_scene_1094[df_scene_1094["model_name"] == "Controller_A_EWMA_Trend"].sort_values("frame_t")
    ctrl_b = df_scene_1094[df_scene_1094["model_name"] == "Controller_B_Preview_Forecaster"].sort_values("frame_t")
    persist = df_scene_1094[df_scene_1094["model_name"] == "Persistence"].sort_values("frame_t")

    if not ground_truth.empty:
        plt.plot(ground_truth["frame_t"], ground_truth["y_true"], "k-", linewidth=2.5, label="Ground Truth Actual (W_{t+1})")
    if not ctrl_b.empty:
        plt.plot(ctrl_b["frame_t"], ctrl_b["y_pred"], "g--", linewidth=1.8, label="Controller B (Preview Forecaster)")
    if not ctrl_a.empty:
        plt.plot(ctrl_a["frame_t"], ctrl_a["y_pred"], "b:", linewidth=1.8, label="Controller A (EWMA, a=0.45)")
    if not persist.empty:
        plt.plot(persist["frame_t"], persist["y_pred"], "m-.", linewidth=1.2, alpha=0.8, label="Persistence Baseline")

    plt.title("EXP-01: 1-Step-Ahead Workload Forecast Trace (h=1) on Test Scene 1094\n(Adverse Night Driving, After Rain)", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Index t", fontsize=11)
    plt.ylabel("Workload Scale", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    trace_png = results_dir / "actual_vs_predicted_traces.png"
    plt.savefig(trace_png, dpi=300)
    plt.close()
    print(f"Saved {trace_png}")

    print("\n" + "=" * 80)
    print("EXP-01 EXECUTION AND VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
