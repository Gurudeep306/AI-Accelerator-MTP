#!/usr/bin/env python3
"""Train and validate Controller B's LinearWorkloadModel on real nuScenes CAM_FRONT data.

Strictly preserves:
- Train split: 6 scenes (scene-0061, 0103, 0553, 0655, 0757, 0796)
- Validation split: 2 scenes (scene-0916, 1077)
- Test split: 2 scenes (scene-1094, 1100)
- Outputs:
    real_driving_workload_model_dense.json
    real_driving_workload_model_sparse.json
    workload_model_performance_summary.json
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
    
    # R2
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
        "std_pred": float(np.std(y_pred))
    }


def main():
    calib_dir = Path("traces/calibration")
    alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    
    results = {}
    
    for mode in ["dense", "sparse"]:
        train_csv = calib_dir / f"train_{mode}.csv"
        val_csv = calib_dir / f"val_{mode}.csv"
        test_csv = calib_dir / f"test_{mode}.csv"
        
        best_alpha = 1.0
        best_val_mae = float("inf")
        alpha_search_results = {}
        
        # Grid search over alpha on Validation split
        for alpha in alphas:
            tmp_json = Path(f"tmp_model_{mode}_alpha_{alpha}.json")
            fit_workload_model(train_csv, tmp_json, ridge_alpha=alpha)
            model = LinearWorkloadModel(tmp_json)
            val_metrics = evaluate_model(model, val_csv)
            alpha_search_results[str(alpha)] = val_metrics
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_alpha = alpha
            tmp_json.unlink()
            
        print(f"[{mode.upper()} MODE] Optimal alpha on Validation: {best_alpha} (Val MAE: {best_val_mae:.5f})")
        
        # Final training with best_alpha
        final_json = Path(f"real_driving_workload_model_{mode}.json")
        fit_workload_model(train_csv, final_json, ridge_alpha=best_alpha)
        final_model = LinearWorkloadModel(final_json)
        
        train_eval = evaluate_model(final_model, train_csv)
        val_eval = evaluate_model(final_model, val_csv)
        test_eval = evaluate_model(final_model, test_csv)
        
        results[mode] = {
            "best_alpha": best_alpha,
            "alpha_grid_search": alpha_search_results,
            "train": train_eval,
            "val": val_eval,
            "test": test_eval
        }
        
        print(f"[{mode.upper()} MODE RESULTS]")
        print(f"  Train: MAE={train_eval['mae']:.5f}, Bias={train_eval['signed_bias']:+.5f}, R2={train_eval['r2']:.4f}")
        print(f"  Val:   MAE={val_eval['mae']:.5f}, Bias={val_eval['signed_bias']:+.5f}, InRange={val_eval['in_range_pct']:.1f}%")
        print(f"  Test:  MAE={test_eval['mae']:.5f}, Bias={test_eval['signed_bias']:+.5f}, InRange={test_eval['in_range_pct']:.1f}%")

    with open("workload_model_performance_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSummary saved to workload_model_performance_summary.json")


if __name__ == "__main__":
    main()
