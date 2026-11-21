#!/usr/bin/env python3
"""Train Controller B Three-Class Computational Budget / Resolution Tier Classifier.

Inputs:
  traces/yolov8n_resolution_tier_labels.csv

Features:
  7 visual preview statistics extracted from camera frames:
  - brightness_mean
  - brightness_std
  - edge_density
  - histogram_entropy
  - temporal_difference
  - aspect_ratio
  - normalized_pixel_count

Target:
  3-class computational budget tier:
  - Tier 0: R1_320x192 (0.6558 GMAC)
  - Tier 1: R2_480x288 (1.4755 GMAC)
  - Tier 2: R3_640x384 (2.6232 GMAC)

Splits:
  Strict scene-level splitting (no temporal frame leakage):
  - Train: scene-0061, scene-0553, scene-0655, scene-0757, scene-0796, scene-0916 (242 frames)
  - Val:   scene-0103, scene-1077 (82 frames)
  - Test:  scene-1094, scene-1100 (80 frames)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "brightness_mean",
    "brightness_std",
    "edge_density",
    "histogram_entropy",
    "temporal_difference",
    "aspect_ratio",
    "normalized_pixel_count",
]

TIER_GMAC_MAP = {
    0: 0.655799040,
    1: 1.475547840,
    2: 2.623196160,
}

TIER_NAMES = {
    0: "R1_320x192",
    1: "R2_480x288",
    2: "R3_640x384",
}


def train_and_evaluate(csv_path: Path, output_json_path: Path, output_pred_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    # Train / Val / Test masks based on scene_split
    train_mask = df["scene_split"] == "train"
    val_mask = df["scene_split"] == "val"
    test_mask = df["scene_split"] == "test"

    X_train = df.loc[train_mask, FEATURE_NAMES].values
    y_train = df.loc[train_mask, "proxy_tier_label"].values

    X_val = df.loc[val_mask, FEATURE_NAMES].values
    y_val = df.loc[val_mask, "proxy_tier_label"].values

    X_test = df.loc[test_mask, FEATURE_NAMES].values
    y_test = df.loc[test_mask, "proxy_tier_label"].values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train multinomial logistic regression
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = clf.predict(X_train_scaled)
    y_val_pred = clf.predict(X_val_scaled)
    y_test_pred = clf.predict(X_test_scaled)

    train_acc = float(accuracy_score(y_train, y_train_pred))
    val_acc = float(accuracy_score(y_val, y_val_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))

    val_f1 = float(f1_score(y_val, y_val_pred, average="weighted"))
    test_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))

    print("=" * 70)
    print("CONTROLLER B 3-CLASS TIER CLASSIFIER EVALUATION")
    print("=" * 70)
    print(f"Train samples: {len(X_train)} across 6 scenes")
    print(f"Val samples:   {len(X_val)} across 2 scenes")
    print(f"Test samples:  {len(X_test)} across 2 scenes (held-out: scene-1094, scene-1100)")
    print("-" * 70)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f} (Weighted F1: {val_f1:.4f})")
    print(f"Test Accuracy:  {test_acc:.4f} (Weighted F1: {test_f1:.4f})")
    print("-" * 70)
    print("Classification Report on Held-Out Test Split:")
    print(classification_report(y_test, y_test_pred, target_names=["R1 (0.66G)", "R2 (1.48G)", "R3 (2.62G)"]))

    # Save predictions dataframe
    df_all_scaled = scaler.transform(df[FEATURE_NAMES].values)
    probs = clf.predict_proba(df_all_scaled)
    preds = clf.predict(df_all_scaled)

    df["predicted_tier_label"] = preds
    df["predicted_tier_name"] = [TIER_NAMES[p] for p in preds]
    df["predicted_tier_gmac"] = [TIER_GMAC_MAP[p] for p in preds]
    df["prob_tier_0"] = np.round(probs[:, 0], 4)
    df["prob_tier_1"] = np.round(probs[:, 1], 4)
    df["prob_tier_2"] = np.round(probs[:, 2], 4)
    df["prediction_confidence"] = np.round(probs.max(axis=1), 4)

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_pred_path, index=False)

    # Save model metadata to JSON
    model_data = {
        "model_type": "MultinomialLogisticRegression",
        "target_type": "three_class_computational_budget_tier",
        "feature_names": FEATURE_NAMES,
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "classes": [int(c) for c in clf.classes_],
        "tier_gmac_map": {str(k): v for k, v in TIER_GMAC_MAP.items()},
        "coefficients": clf.coef_.tolist(),
        "intercepts": clf.intercept_.tolist(),
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "val_f1_weighted": round(val_f1, 4),
        "test_f1_weighted": round(test_f1, 4),
        "split_method": "scene_based",
        "train_scenes": sorted(df.loc[train_mask, "scene_id"].unique().tolist()),
        "val_scenes": sorted(df.loc[val_mask, "scene_id"].unique().tolist()),
        "test_scenes": sorted(df.loc[test_mask, "scene_id"].unique().tolist()),
        "provenance_note": (
            "Target labels are proxy detection-retention tiers relative to R3; "
            "does not represent ground-truth 2D bounding box mAP against nuScenes annotations."
        ),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    print(f"Model saved -> {output_json_path}")
    print(f"Predictions saved -> {output_pred_path}")
    return model_data


if __name__ == "__main__":
    csv_file = Path("C:/AI accelerator project/traces/yolov8n_resolution_tier_labels.csv")
    model_json = Path("C:/AI accelerator project/workload_tier_model_yolo.json")
    pred_csv = Path("C:/AI accelerator project/results/controller_b_tier_predictions.csv")
    train_and_evaluate(csv_file, model_json, pred_csv)
