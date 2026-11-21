#!/usr/bin/env python3
"""Independent Audit and Metric Recalculation for Workload Prediction Benchmark."""

import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd

from frame_preview_mpc import LinearWorkloadModel, TemporalWorkloadForecaster
from history_mpc_controller import HistoryWorkloadPredictor, WorkloadObservation
from workload_prediction_benchmark import (
    PersistencePredictor, PureEWMAPredictor, ControllerAPredictor,
    ControllerBPreviewPipeline, ControllerCEstimator, generate_test_traces
)

def run_audit():
    print("=== STARTING INDEPENDENT WORKLOAD PREDICTION AUDIT ===")

    # 1. Reproduce predictions exactly with seed 42
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

    new_horizon_records = []
    timing_records = {p: [] for p in predictor_names}

    for pattern_name, trace_data in traces.items():
        n_frames = len(trace_data)
        actuals = [w for w, _ in trace_data]

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

            # 4. Controller B
            t0 = time.perf_counter_ns()
            est_b_now, future_b = p_ctrl_b.predict_current_and_forecast(feats_now, 5)
            dt_b_us = (time.perf_counter_ns() - t0) / 1000.0
            timing_records["Controller_B_Slope_Forecaster"].append(dt_b_us)
            timing_records["Controller_B_Preview_Step0"].append(dt_b_us)
            pred_ctrl_b = future_b

            # 5. Controller C
            t0 = time.perf_counter_ns()
            est_c_now, future_c = p_ctrl_c.estimate_current_and_forecast(5)
            timing_records["Controller_C_Static_MAC"].append((time.perf_counter_ns() - t0) / 1000.0)
            pred_ctrl_c = future_c

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

                        new_horizon_records.append({
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

    df_new = pd.DataFrame(new_horizon_records)
    print(f"Generated {len(df_new)} audit prediction records.")

    # Compare against published horizon results
    df_pub = pd.read_csv("workload_prediction_horizon_results.csv")
    print(f"Loaded {len(df_pub)} published prediction records.")

    assert len(df_new) == len(df_pub), "Record counts differ!"
    max_diff_true = (df_new["y_true"] - df_pub["y_true"]).abs().max()
    max_diff_pred = (df_new["y_pred"] - df_pub["y_pred"]).abs().max()
    max_diff_err = (df_new["error"] - df_pub["error"]).abs().max()
    print(f"Max y_true diff vs published: {max_diff_true:.2e}")
    print(f"Max y_pred diff vs published: {max_diff_pred:.2e}")
    print(f"Max error diff vs published: {max_diff_err:.2e}")

    # Load published overall results
    df_overall_pub = pd.read_csv("workload_prediction_results.csv").set_index("predictor")

    # 2. Compute Independent Recalculation Metrics (Overall)
    recalc_overall = []
    for p_name in predictor_names:
        sub = df_pub[df_pub["predictor"] == p_name]
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values
        err = sub["error"].values
        abs_err = sub["abs_error"].values
        pct_err = sub["pct_error"].values

        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(pct_err))
        max_abs_err = float(np.max(abs_err))
        p95_abs_err = float(np.percentile(abs_err, 95))

        # Mean Signed Error (Bias: y_hat - y_true)
        mean_bias = float(np.mean(err))

        # Underprediction metrics (y_hat < y_true <=> err < 0)
        underpred_mask = err < 0
        underpred_count = int(np.sum(underpred_mask))
        underpred_freq_pct = float((underpred_count / len(err)) * 100.0)
        
        underpred_mags = y_true[underpred_mask] - y_pred[underpred_mask] # positive magnitudes
        mean_underpred_mag = float(np.mean(underpred_mags)) if len(underpred_mags) > 0 else 0.0
        max_underpred_mag = float(np.max(underpred_mags)) if len(underpred_mags) > 0 else 0.0

        pub_mae = float(df_overall_pub.loc[p_name, "overall_mae"])
        pub_rmse = float(df_overall_pub.loc[p_name, "overall_rmse"])
        pub_mape = float(df_overall_pub.loc[p_name, "overall_mape_pct"])
        pub_max_err = float(df_overall_pub.loc[p_name, "overall_max_error"])
        pub_mean_lat = float(df_overall_pub.loc[p_name, "mean_latency_us"])
        pub_p95_lat = float(df_overall_pub.loc[p_name, "p95_latency_us"])

        recalc_overall.append({
            "predictor": p_name,
            "pub_mae": pub_mae,
            "recalc_mae": mae,
            "mae_match": math.isclose(pub_mae, mae, abs_tol=1e-6),
            "pub_rmse": pub_rmse,
            "recalc_rmse": rmse,
            "rmse_match": math.isclose(pub_rmse, rmse, abs_tol=1e-6),
            "pub_mape_pct": pub_mape,
            "recalc_mape_pct": mape,
            "mape_match": math.isclose(pub_mape, mape, abs_tol=1e-6),
            "pub_max_error": pub_max_err,
            "recalc_max_error": max_abs_err,
            "max_err_match": math.isclose(pub_max_err, max_abs_err, abs_tol=1e-6),
            "mean_signed_error_bias": mean_bias,
            "underpred_freq_pct": underpred_freq_pct,
            "mean_underpred_magnitude": mean_underpred_mag,
            "max_underpred_magnitude": max_underpred_mag,
            "p95_abs_error": p95_abs_err,
            "recalc_mean_latency_us": float(np.mean(timing_records[p_name])),
            "recalc_p95_latency_us": float(np.percentile(timing_records[p_name], 95)),
            "pub_mean_latency_us": pub_mean_lat,
            "pub_p95_latency_us": pub_p95_lat
        })

    df_recalc_overall = pd.DataFrame(recalc_overall)
    df_recalc_overall.to_csv("BENCHMARK_METRIC_RECALCULATION.csv", index=False)
    print("Saved BENCHMARK_METRIC_RECALCULATION.csv")
    print(df_recalc_overall.to_string())

    # 3. Compute Pattern-wise and Horizon-wise Audit Metrics
    recalc_detailed = []
    for (p_name, pat_name, h), grp in df_pub.groupby(["predictor", "pattern", "horizon"]):
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values
        err = grp["error"].values
        abs_err = grp["abs_error"].values
        pct_err = grp["pct_error"].values

        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(pct_err))
        max_abs_err = float(np.max(abs_err))
        p95_abs_err = float(np.percentile(abs_err, 95))
        mean_bias = float(np.mean(err))

        underpred_mask = err < 0
        underpred_count = int(np.sum(underpred_mask))
        underpred_freq_pct = float((underpred_count / len(err)) * 100.0)
        underpred_mags = y_true[underpred_mask] - y_pred[underpred_mask]
        mean_underpred_mag = float(np.mean(underpred_mags)) if len(underpred_mags) > 0 else 0.0
        max_underpred_mag = float(np.max(underpred_mags)) if len(underpred_mags) > 0 else 0.0

        recalc_detailed.append({
            "predictor": p_name,
            "pattern": pat_name,
            "horizon": h,
            "sample_count": len(grp),
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape,
            "max_abs_error": max_abs_err,
            "p95_abs_error": p95_abs_err,
            "mean_signed_error_bias": mean_bias,
            "underpred_freq_pct": underpred_freq_pct,
            "mean_underpred_magnitude": mean_underpred_mag,
            "max_underpred_magnitude": max_underpred_mag
        })

    df_detailed = pd.DataFrame(recalc_detailed)
    df_detailed.to_csv("PREDICTOR_PATTERN_HORIZON_AUDIT.csv", index=False)
    print("Saved PREDICTOR_PATTERN_HORIZON_AUDIT.csv")

    # 4. Special Audit of Underprediction During Sudden Increases & Congestion Spikes
    print("\n=== UNDERPREDICTION AUDIT: SUDDEN INCREASE & CONGESTION SPIKES ===")
    crit_patterns = ["Sudden_Workload_Increase", "Congestion_Spikes"]
    sub_crit = df_detailed[df_detailed["pattern"].isin(crit_patterns)]
    crit_summary = sub_crit.groupby(["predictor", "pattern"])[["mae", "max_underpred_magnitude", "underpred_freq_pct"]].mean()
    print(crit_summary.to_string())

if __name__ == "__main__":
    run_audit()
