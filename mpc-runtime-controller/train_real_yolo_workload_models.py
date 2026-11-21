#!/usr/bin/env python3
"""Train and validate Controller B's LinearWorkloadModel on real YOLOv8n nuScenes data.

Uses:
- Train split: traces/real_yolo/calibration/train_real_yolo.csv (6 scenes, 242 frames)
- Val split:   traces/real_yolo/calibration/val_real_yolo.csv   (2 scenes, 82 frames)
- Test split:  traces/real_yolo/calibration/test_real_yolo.csv  (2 scenes, 80 frames)
Outputs:
- real_driving_workload_model_yolo.json
- yolo_workload_model_summary.json
"""

import json
import csv
from pathlib import Path
import numpy as np

from frame_preview_mpc import fit_workload_model, LinearWorkloadModel, FEATURE_NAMES


def evaluate_model(model: LinearWorkloadModel, csv_path: Path) -> dict:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    y_true = np.array([float(r["measured_workload_scale"]) for r in rows])
    y_pred = []
    in_ranges = []

    for r in rows:
        feats = {k: float(r[k]) for k in FEATURE_NAMES}
        est = model.predict(feats)
        y_pred.append(est.workload_scale)
        in_ranges.append(est.in_calibration_range)

    y_pred = np.array(y_pred)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    signed_bias = float(np.mean(errors))
    underpred_rate = float(np.mean(y_pred < y_true) * 100.0)
    in_range_pct = float(np.mean(in_ranges) * 100.0)

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    return {
        "n_samples": len(rows),
        "mae": mae,
        "rmse": rmse,
        "signed_bias": signed_bias,
        "underprediction_rate_pct": underpred_rate,
        "in_range_pct": in_range_pct,
        "r2": r2,
        "mean_true": float(np.mean(y_true)),
        "std_true": float(np.std(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_pred": float(np.std(y_pred)),
    }


def main():
    print("=" * 70)
    print("TRAINING CONTROLLER B LINEAR WORKLOAD MODEL ON REAL YOLO DATA")
    print("=" * 70)

    calib_dir = Path("traces/real_yolo/calibration")
    train_csv = calib_dir / "train_real_yolo.csv"
    val_csv = calib_dir / "val_real_yolo.csv"
    test_csv = calib_dir / "test_real_yolo.csv"

    alphas = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

    best_alpha = 1.0
    best_val_mae = float("inf")
    alpha_search_results = {}

    for alpha in alphas:
        tmp_json = Path(f"tmp_model_yolo_alpha_{alpha}.json")
        fit_workload_model(train_csv, tmp_json, ridge_alpha=alpha)
        model = LinearWorkloadModel(tmp_json)
        val_metrics = evaluate_model(model, val_csv)
        alpha_search_results[str(alpha)] = val_metrics
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_alpha = alpha
        tmp_json.unlink()

    print(f"Optimal alpha on Validation split: {best_alpha} (Val MAE: {best_val_mae:.5f})")

    final_json = Path("real_driving_workload_model_yolo.json")
    fit_workload_model(train_csv, final_json, ridge_alpha=best_alpha)
    final_model = LinearWorkloadModel(final_json)

    train_eval = evaluate_model(final_model, train_csv)
    val_eval = evaluate_model(final_model, val_csv)
    test_eval = evaluate_model(final_model, test_csv)

    results = {
        "best_alpha": best_alpha,
        "alpha_grid_search": alpha_search_results,
        "train": train_eval,
        "val": val_eval,
        "test": test_eval,
    }

    print("\n[REAL YOLO WORKLOAD MODEL EVALUATION]")
    print(f"  Train ({train_eval['n_samples']} frames): MAE={train_eval['mae']:.4f}, RMSE={train_eval['rmse']:.4f}, Bias={train_eval['signed_bias']:+.4f}, R2={train_eval['r2']:.4f}")
    print(f"  Val   ({val_eval['n_samples']} frames):   MAE={val_eval['mae']:.4f}, RMSE={val_eval['rmse']:.4f}, Bias={val_eval['signed_bias']:+.4f}, InRange={val_eval['in_range_pct']:.1f}%")
    print(f"  Test  ({test_eval['n_samples']} frames):  MAE={test_eval['mae']:.4f}, RMSE={test_eval['rmse']:.4f}, Bias={test_eval['signed_bias']:+.4f}, InRange={test_eval['in_range_pct']:.1f}%")

    with open("yolo_workload_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved trained model to real_driving_workload_model_yolo.json")
    print("Saved summary metrics to yolo_workload_model_summary.json")


if __name__ == "__main__":
    main()
