#!/usr/bin/env python3
"""Causal workload variation & prediction analysis for Milestone B pilot data.

Evaluates what frame-to-frame execution properties (inference latency,
postprocessing latency, candidate count, detection count) are genuinely
predictable without future leakage under:
1. Scene-wise Phase 4 split (Train: 6 daytime scenes; Val: 2 mixed; Test: 2 night OOD).
2. 10-fold Leave-One-Scene-Out cross-validation across all 10 scenes.

Compares causal persistence, moving average (w=3), EWMA (alpha=0.45),
lagged history Ridge, current-frame preview Ridge, and combined Ridge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PREVIEW_FEATURE_COLS = [
    "preview_brightness_mean",
    "preview_brightness_std",
    "preview_edge_density",
    "preview_histogram_entropy",
    "preview_temporal_difference",
]

TARGETS = [
    "inference_ms",
    "inference_repeat_median_ms",
    "postprocess_ms",
    "candidates_before_nms",
    "final_detection_count",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute prediction error, bias, correlation, and R^2 metrics."""
    if len(y_true) == 0:
        return {
            "n": 0, "mae": None, "rmse": None, "mape_pct": None,
            "r2": None, "pearson_r": None, "signed_bias": None,
            "underprediction_rate_pct": None,
        }
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    signed_bias = float(np.mean(err))
    underpred_pct = float(np.mean(y_pred < y_true) * 100.0)

    nonzero = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero])) * 100.0) if np.any(nonzero) else None

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    if len(y_true) >= 3 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = None

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
        "pearson_r": pearson_r,
        "signed_bias": signed_bias,
        "underprediction_rate_pct": underpred_pct,
    }


def add_causal_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Construct causal per-scene lagged target & detection signals (t-1 only)."""
    out = df.sort_values(["scene_id", "frame_in_scene"]).copy()
    g = out.groupby("scene_id", sort=False)

    out["lag1_target"] = g[target].shift(1)
    out["lag1_ma3"] = g[target].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    out["lag1_ewma"] = g[target].transform(lambda s: s.shift(1).ewm(alpha=0.45, adjust=False).mean())
    out["lag1_candidates"] = g["candidates_before_nms"].shift(1)
    out["lag1_dets"] = g["final_detection_count"].shift(1)
    out["lag1_post_ms"] = g["postprocess_ms"].shift(1)
    out["lag1_inf_ms"] = g["inference_ms"].shift(1)
    return out


def _fit_predict_ridge(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    alpha: float = 10.0,
) -> np.ndarray:
    """Standardize strictly on train_df and predict on eval_df without leakage."""
    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    y_tr = train_df[target].to_numpy(dtype=float)
    X_ev = eval_df[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(X_tr)
    Z_ev = scaler.transform(X_ev)
    reg = Ridge(alpha=alpha, random_state=42)
    reg.fit(Z_tr, y_tr)
    return reg.predict(Z_ev)


def evaluate_split(
    df_lagged: pd.DataFrame,
    train_mask: pd.Series,
    eval_mask: pd.Series,
    target: str,
    ridge_alpha: float = 10.0,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Evaluate all causal predictors on rows with t >= 1 (valid lagged state)."""
    valid_lag = df_lagged["frame_in_scene"] >= 1
    tr = df_lagged[train_mask & valid_lag].copy()
    ev = df_lagged[eval_mask & valid_lag].copy()

    # For preview-only model (does not require lag), train on all train frames including t=0
    tr_all = df_lagged[train_mask].copy()
    y_true = ev[target].to_numpy(dtype=float)
    train_mean = float(tr_all[target].mean())

    preds = {
        "train_mean_constant": np.full_like(y_true, train_mean, dtype=float),
        "persistence_lag1": ev["lag1_target"].to_numpy(dtype=float),
        "moving_avg_w3": ev["lag1_ma3"].to_numpy(dtype=float),
        "ewma_alpha_045": ev["lag1_ewma"].to_numpy(dtype=float),
        "history_ridge_lag": _fit_predict_ridge(
            tr, ev, ["lag1_target", "lag1_ma3", "lag1_candidates", "lag1_dets"],
            target, alpha=ridge_alpha,
        ),
        "preview_ridge_frame_t": _fit_predict_ridge(
            tr_all, ev, PREVIEW_FEATURE_COLS, target, alpha=ridge_alpha,
        ),
        "combined_history_preview_ridge": _fit_predict_ridge(
            tr, ev, ["lag1_target", "lag1_ma3"] + PREVIEW_FEATURE_COLS,
            target, alpha=ridge_alpha,
        ),
    }
    return {name: compute_metrics(y_true, pred) for name, pred in preds.items()}


def run_loso_cv(
    df_lagged: pd.DataFrame,
    target: str,
    ridge_alpha: float = 10.0,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Run 10-fold Leave-One-Scene-Out cross-validation on causal frames (t >= 1)."""
    scenes = sorted(df_lagged["scene_id"].unique())
    valid_lag = df_lagged["frame_in_scene"] >= 1

    all_true: List[np.ndarray] = []
    model_preds: Dict[str, List[np.ndarray]] = {}

    for held_out in scenes:
        tr_mask = df_lagged["scene_id"] != held_out
        ev_mask = df_lagged["scene_id"] == held_out
        tr = df_lagged[tr_mask & valid_lag]
        tr_all = df_lagged[tr_mask]
        ev = df_lagged[ev_mask & valid_lag]

        y_ev = ev[target].to_numpy(dtype=float)
        all_true.append(y_ev)
        fold = {
            "train_mean_constant": np.full_like(y_ev, float(tr_all[target].mean()), dtype=float),
            "persistence_lag1": ev["lag1_target"].to_numpy(dtype=float),
            "moving_avg_w3": ev["lag1_ma3"].to_numpy(dtype=float),
            "ewma_alpha_045": ev["lag1_ewma"].to_numpy(dtype=float),
            "history_ridge_lag": _fit_predict_ridge(
                tr, ev, ["lag1_target", "lag1_ma3", "lag1_candidates", "lag1_dets"],
                target, alpha=ridge_alpha,
            ),
            "preview_ridge_frame_t": _fit_predict_ridge(
                tr_all, ev, PREVIEW_FEATURE_COLS, target, alpha=ridge_alpha,
            ),
            "combined_history_preview_ridge": _fit_predict_ridge(
                tr, ev, ["lag1_target", "lag1_ma3"] + PREVIEW_FEATURE_COLS,
                target, alpha=ridge_alpha,
            ),
        }
        for name, arr in fold.items():
            model_preds.setdefault(name, []).append(arr)

    y_concat = np.concatenate(all_true)
    return {
        name: compute_metrics(y_concat, np.concatenate(parts))
        for name, parts in model_preds.items()
    }


def analyze_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Measure bivariate Pearson correlations across numerical signals."""
    cols = [
        "inference_ms", "inference_repeat_median_ms", "postprocess_ms",
        "total_pipeline_ms", "candidates_before_nms", "final_detection_count",
        "preview_brightness_mean", "preview_brightness_std",
        "preview_edge_density", "preview_histogram_entropy", "preview_temporal_difference",
    ]
    present = [c for c in cols if c in df.columns]
    corr = df[present].corr(method="pearson")
    targets = ["inference_ms", "inference_repeat_median_ms", "postprocess_ms", "candidates_before_nms", "final_detection_count"]
    out: Dict[str, Dict[str, float]] = {}
    for tgt in [t for t in targets if t in corr.columns]:
        out[tgt] = {
            c: round(float(corr.loc[tgt, c]), 4)
            for c in present if c != tgt and pd.notna(corr.loc[tgt, c])
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="real_dnn_pilot_measurements.csv")
    parser.add_argument("--out-dir", default="results/milestone_b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    # Variance decomposition & signal-to-noise check
    summary: Dict[str, object] = {
        "n_frames": int(len(df)),
        "n_scenes": int(df["scene_id"].nunique()),
        "correlations_pearson": analyze_correlations(df),
        "signal_variance": {},
        "prediction_results": [],
    }

    for col in TARGETS:
        vals = df[col].to_numpy(dtype=float)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1))
        cv_pct = float(100.0 * std_v / abs(mean_v)) if abs(mean_v) > 1e-12 else 0.0
        between_scene_std = float(df.groupby("scene_id")[col].mean().std(ddof=1))
        entry = {
            "mean": round(mean_v, 4),
            "median": round(float(np.median(vals)), 4),
            "std": round(std_v, 4),
            "cv_pct": round(cv_pct, 2),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "between_scene_mean_std": round(between_scene_std, 4),
        }
        if col == "inference_ms" and "inference_repeat_std_ms" in df.columns:
            within_rms = float(np.sqrt(np.mean(df["inference_repeat_std_ms"].to_numpy(dtype=float) ** 2)))
            entry["within_frame_repeat_rms_std_ms"] = round(within_rms, 4)
        summary["signal_variance"][col] = entry

    rows = []
    for target in TARGETS:
        df_lag = add_causal_lag_features(df, target)
        train_m = df_lag["scene_split"] == "train"
        val_m = df_lag["scene_split"] == "val"
        test_m = df_lag["scene_split"] == "test"

        eval_sets = {
            "phase4_val_mixed": evaluate_split(df_lag, train_m, val_m, target),
            "phase4_test_night_ood": evaluate_split(df_lag, train_m, test_m, target),
            "loso_10fold_cv": run_loso_cv(df_lag, target),
        }
        for eval_name, res_map in eval_sets.items():
            for model_name, metrics in res_map.items():
                row = {"target": target, "evaluation": eval_name, "predictor": model_name}
                row.update(metrics)
                rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_csv = out_dir / "workload_prediction_analysis.csv"
    pred_df.to_csv(pred_csv, index=False)

    summary["prediction_results"] = rows
    summary_json = out_dir / "workload_prediction_analysis.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved workload analysis to {pred_csv} and {summary_json}")
    for target in ["inference_ms", "inference_repeat_median_ms", "postprocess_ms", "candidates_before_nms"]:
        sub = pred_df[(pred_df["target"] == target) & (pred_df["evaluation"] == "loso_10fold_cv")]
        print(f"\n--- LOSO 10-Fold Causal Prediction: {target} ---")
        print(sub[["predictor", "mae", "rmse", "r2", "pearson_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
