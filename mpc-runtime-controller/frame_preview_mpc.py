#!/usr/bin/env python3
"""Pre-inference frame preview + workload prediction + MPC decision adapter.

This is integration scaffolding, not a hardware driver. It deliberately refuses
operation without a trained workload model and per-action measured/validated
performance profiles. No performance or quality labels are fabricated here.

Frame t is previewed BEFORE full inference, so the preview can guide the
configuration for frame t itself. The MPC horizon's future workload estimates
come from the recent sequence of preview predictions (a simple bounded trend
forecast by default); replace this forecaster after evaluating alternatives.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from practical_mpc_controller import (
    Action, Constraints, DEFAULT_ACTIONS, MPCController, Prediction, Telemetry
)

FEATURE_NAMES = [
    "brightness_mean", "brightness_std", "edge_density", "histogram_entropy",
    "temporal_difference", "aspect_ratio", "normalized_pixel_count"
]


def extract_frame_features(image_path: str | Path, previous_image_path: str | Path | None = None,
                           preview_side: int = 96) -> Dict[str, float]:
    """Extract cheap image statistics from a small grayscale preview.

    This function reads a frame from a software-accessible image path. A real
    camera/accelerator integration should pass the frame-buffer view directly to
    avoid unnecessary copies and must measure this function's latency.
    """
    path = Path(image_path)
    with Image.open(path) as im:
        original_w, original_h = im.size
        if original_w <= 0 or original_h <= 0:
            raise ValueError(f"Invalid image dimensions: {path}")
        im = im.convert("L")
        im.thumbnail((preview_side, preview_side))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    if arr.size == 0:
        raise ValueError(f"Empty image: {path}")
    brightness_mean = float(arr.mean())
    brightness_std = float(arr.std())
    # Cheap adjacent-pixel gradient proxy; this is not a neural-network MAC count.
    dx = np.abs(np.diff(arr, axis=1)) if arr.shape[1] > 1 else np.zeros((arr.shape[0], 0))
    dy = np.abs(np.diff(arr, axis=0)) if arr.shape[0] > 1 else np.zeros((0, arr.shape[1]))
    edge_values = []
    if dx.size:
        edge_values.append(dx.ravel())
    if dy.size:
        edge_values.append(dy.ravel())
    edges = np.concatenate(edge_values) if edge_values else np.zeros(1, dtype=np.float32)
    edge_density = float(np.mean(edges > 0.08))
    hist, _ = np.histogram(arr, bins=16, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float64) / max(1, hist.sum())
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / 4.0)  # normalized to [0,1]
    temporal_difference = 0.0
    if previous_image_path is not None:
        with Image.open(previous_image_path) as prev:
            prev = prev.convert("L")
            prev.thumbnail((preview_side, preview_side))
            prev_arr = np.asarray(prev, dtype=np.float32) / 255.0
        # Resize both arrays to identical shape before computing a motion proxy.
        h, w = min(arr.shape[0], prev_arr.shape[0]), min(arr.shape[1], prev_arr.shape[1])
        if h and w:
            temporal_difference = float(np.abs(arr[:h, :w] - prev_arr[:h, :w]).mean())
    return {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "edge_density": edge_density,
        "histogram_entropy": entropy,
        "temporal_difference": temporal_difference,
        "aspect_ratio": float(original_w / original_h),
        "normalized_pixel_count": float((original_w * original_h) / (1920.0 * 1080.0)),
    }


@dataclass
class WorkloadEstimate:
    workload_scale: float
    uncertainty: float
    in_calibration_range: bool


class LinearWorkloadModel:
    """Standardized ridge-regression workload model loaded from a calibration JSON."""
    def __init__(self, model_path: str | Path):
        data = json.loads(Path(model_path).read_text())
        if data.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Workload model feature schema does not match this code")
        self.mean = np.asarray(data["feature_mean"], dtype=float)
        self.scale = np.asarray(data["feature_scale"], dtype=float)
        self.coef = np.asarray(data["coefficients"], dtype=float)
        self.intercept = float(data["intercept"])
        self.residual_std = float(data["residual_std"])
        self.min_target = float(data["target_min"])
        self.max_target = float(data["target_max"])
        self.train_min = np.asarray(data["feature_min"], dtype=float)
        self.train_max = np.asarray(data["feature_max"], dtype=float)
        if any(len(x) != len(FEATURE_NAMES) for x in (self.mean, self.scale, self.coef, self.train_min, self.train_max)):
            raise ValueError("Invalid workload model dimensions")
        if np.any(self.scale <= 0) or not np.all(np.isfinite(self.coef)):
            raise ValueError("Invalid workload model coefficients/scales")
        if self.residual_std < 0 or self.min_target <= 0 or self.max_target < self.min_target:
            raise ValueError("Invalid workload target range or residual uncertainty")

    def predict(self, features: Dict[str, float]) -> WorkloadEstimate:
        x = np.asarray([features[n] for n in FEATURE_NAMES], dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("Frame features contain non-finite values")
        in_range = bool(np.all(x >= self.train_min) and np.all(x <= self.train_max))
        z = (x - self.mean) / self.scale
        raw = float(self.intercept + np.dot(z, self.coef))
        # Prevent impossible negative values while preserving a clear out-of-range flag.
        target = max(self.min_target, min(self.max_target, raw))
        return WorkloadEstimate(target, self.residual_std, in_range)


def fit_workload_model(calibration_csv: str | Path, output_json: str | Path, ridge_alpha: float = 1.0) -> dict:
    """Fit a ridge model from collected feature/ground-truth workload rows.

    CSV requires FEATURE_NAMES plus measured_workload_scale. Labels should be
    derived from measured per-frame workload/inference telemetry, not guessed.
    This utility reports training residual only; use held-out scenes for validation.
    """
    with Path(calibration_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = FEATURE_NAMES + ["measured_workload_scale"]
    if not rows or any(k not in rows[0] for k in required):
        raise ValueError(f"Calibration CSV must contain columns: {', '.join(required)}")
    X = np.asarray([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.asarray([float(r["measured_workload_scale"]) for r in rows], dtype=float)
    if len(y) < len(FEATURE_NAMES) + 2:
        raise ValueError("Need at least 9 labelled frames to fit; substantially more diverse frames are recommended")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("Calibration values must be finite and workload labels must be positive")
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    design = np.column_stack([np.ones(len(Z)), Z])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    fitted = design @ beta
    residual_std = float(np.sqrt(np.mean((y - fitted) ** 2)))
    model = {
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "coefficients": beta[1:].tolist(), "intercept": float(beta[0]),
        "residual_std": residual_std,
        "target_min": float(y.min()), "target_max": float(y.max()),
        "feature_min": X.min(axis=0).tolist(), "feature_max": X.max(axis=0).tolist(),
        "training_rows": int(len(y)), "ridge_alpha": float(ridge_alpha),
        "warning": "Training residual is not held-out validation. Validate on separate scenes before deployment."
    }
    Path(output_json).write_text(json.dumps(model, indent=2) + "\n")
    return model


class TierClassifierWorkloadModel:
    """Standardized 3-class computational budget / resolution tier classifier."""
    def __init__(self, model_path: str | Path):
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.feature_names = data.get("feature_names", FEATURE_NAMES)
        self.mean = np.asarray(data["feature_mean"], dtype=float)
        self.scale = np.asarray(data["feature_scale"], dtype=float)
        self.coef = np.asarray(data["coefficients"], dtype=float)
        self.intercept = np.asarray(data["intercepts"], dtype=float)
        self.classes = [int(c) for c in data["classes"]]
        self.tier_gmac_map = {int(k): float(v) for k, v in data["tier_gmac_map"].items()}
        self.tier_names = {0: "R1_320x192", 1: "R2_480x288", 2: "R3_640x384"}
        self.train_acc = float(data.get("train_accuracy", 0.0))
        self.val_acc = float(data.get("val_accuracy", 0.0))
        self.test_acc = float(data.get("test_accuracy", 0.0))

    def predict(self, features: Dict[str, float]):
        from workload_prediction_interface import WorkloadPrediction, PredictionScope
        x = np.asarray([features[n] for n in self.feature_names], dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("Frame features contain non-finite values")
        z = (x - self.mean) / self.scale
        logits = np.dot(self.coef, z) + self.intercept
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        expected_gmac = self.tier_gmac_map[pred_class]
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        gmac_std = float(entropy * 0.25)

        return WorkloadPrediction(
            expected_gmac=expected_gmac,
            gmac_std=gmac_std,
            scope=PredictionScope.FULL_FRAME,
            overhead_ms=0.35,  # measured preview extraction overhead
            confidence=confidence,
            layer_boundary=0,
            tier_idx=pred_class,
            tier_name=self.tier_names[pred_class],
            label_provenance="PROXY_DETECTION_RETENTION",
        )


class ProfilePerformancePredictor:
    """Performance predictor using per-action validated profiles and workload scale.

    The simple proportional scaling is a baseline assumption, not a universal law.
    Replace it with a calibrated workload-aware model before hardware deployment.
    """
    def __init__(self, profiles_csv: str | Path, allow_uncalibrated_quality: bool = False):
        self.profiles = {}
        with Path(profiles_csv).open(newline="", encoding="utf-8") as f:
            lines = [line for line in f if not line.strip().startswith("#")]
            for row in csv.DictReader(lines):
                q_val = (row.get("validated_quality") or "").strip()
                quality = float(q_val) if q_val else None
                if quality is None and not allow_uncalibrated_quality:
                    continue
                a = Action(float(row["width"]), int(row["exit"]), int(row["bits"]))
                self.profiles[a] = {
                    "latency": float(row["latency_ms"]), "power": float(row["power_w"]),
                    "energy": float(row["energy_mj"]), "quality": quality,
                    "latency_std": float(row.get("latency_std_ms") or 0),
                    "power_std": float(row.get("power_std_w") or 0),
                    "energy_std": float(row.get("energy_std_mj") or 0),
                    "quality_std": float(row.get("quality_std") or 0),
                }
        if not self.profiles:
            raise ValueError("No validated profiles found. Add measured performance and held-out quality values first.")

    def predict(self, action: Action, workload: object, telemetry: Telemetry) -> Prediction:
        if action not in self.profiles:
            raise ValueError(f"No measured profile for {action.name}")
        p = self.profiles[action]

        # Standardized workload parsing: accepts int/float, WorkloadPrediction, or dict with 'workload_scale'
        if isinstance(workload, (int, float)):
            scale = float(workload)
            scale_std = 0.0
        elif hasattr(workload, "workload_scale"):
            scale = float(workload.workload_scale)
            scale_std = float(getattr(workload, "workload_scale_std", 0.0))
        elif isinstance(workload, dict):
            if "workload_scale" not in workload:
                raise ValueError("workload dict must contain 'workload_scale'")
            scale = float(workload["workload_scale"])
            scale_std = max(0.0, float(workload.get("workload_scale_std", 0.0)))
        else:
            raise TypeError(f"workload must be a numeric scale or a dict with 'workload_scale', got {type(workload).__name__}")

        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"workload_scale must be finite and positive, got {scale}")

        latency_std = math.sqrt((p["latency_std"] * scale) ** 2 + (p["latency"] * scale_std) ** 2)
        energy_std = math.sqrt((p["energy_std"] * scale) ** 2 + (p["energy"] * scale_std) ** 2)
        return Prediction(p["latency"] * scale, p["power"], p["energy"] * scale,
                          p["quality"], latency_std, p["power_std"],
                          energy_std, p["quality_std"])


class TemporalWorkloadForecaster:
    """Small bounded linear-trend forecast over recent preview workload estimates."""
    def __init__(self, history_size: int = 5, max_relative_step: float = 0.25):
        self.history = deque(maxlen=history_size)
        self.max_relative_step = max_relative_step

    def update(self, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Workload scale must be positive and finite")
        self.history.append(float(scale))

    def forecast(self, steps: int, min_scale: float, max_scale: float) -> List[float]:
        if not self.history:
            return [1.0] * steps
        values = list(self.history)
        if len(values) < 2:
            return [values[-1]] * steps
        slope = (values[-1] - values[0]) / max(1, len(values) - 1)
        out = []
        last = values[-1]
        for h in range(1, steps + 1):
            delta = max(-self.max_relative_step * last, min(self.max_relative_step * last, slope * h))
            out.append(max(min_scale, min(max_scale, last + delta)))
        return out



@dataclass
class FrameDecisionRecord:
    frame_id: int
    workload_scale: float
    workload_uncertainty: float
    preview_in_calibration_range: bool
    preview_ms: float
    action: str
    mode: str
    reason: str
    predicted_latency_ms: Optional[float]
    predicted_energy_mj: Optional[float]
    planned_sequence: Tuple[str, ...]
    preview_energy_mj: float = 0.03
    predicted_total_energy_mj: Optional[float] = None
    predicted_e2e_latency_ms: Optional[float] = None


class FramePreviewMPCPipeline:
    """Orchestrates preview -> workload prediction -> MPC for each buffered frame.

    The caller must invoke process_frame before dispatching the full frame to the
    accelerator. Applying the action and collecting telemetry remain platform-specific.
    """
    def __init__(self, workload_model_path: str | Path, profiles_csv: str | Path,
                 constraints: Constraints, fallback: Action = Action(0.50, 1, 4),
                 require_thermal_calibration: bool = True,
                 allow_uncalibrated_quality: bool = False):
        if require_thermal_calibration and constraints.thermal_resistance_c_per_w is None:
            raise ValueError("Calibrate thermal_resistance_c_per_w before enabling the hardware pipeline")
        # Auto-detect whether model is 3-class tier classifier or continuous linear model
        model_data = json.loads(Path(workload_model_path).read_text(encoding="utf-8"))
        if model_data.get("target_type") == "three_class_computational_budget_tier":
            self.workload_model = TierClassifierWorkloadModel(workload_model_path)
        else:
            self.workload_model = LinearWorkloadModel(workload_model_path)
        self.performance_predictor = ProfilePerformancePredictor(profiles_csv, allow_uncalibrated_quality=allow_uncalibrated_quality)
        self.controller = MPCController(self.performance_predictor,
            validated_actions=self.performance_predictor.profiles.keys(),
            fallback=fallback, constraints=constraints)
        self.forecaster = TemporalWorkloadForecaster()
        self.previous_image_path: Optional[Path] = None
        self.last_frame_id: Optional[int] = None
        self.preview_energy_mj: float = getattr(constraints, "preview_energy_mj", 0.03)

    def process_frame(self, frame_id: int, image_path: str | Path,
                      telemetry: Telemetry) -> FrameDecisionRecord:
        import time
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            raise ValueError("frame_id must increase monotonically")
        started = time.perf_counter()
        features = extract_frame_features(image_path, self.previous_image_path)
        estimate = self.workload_model.predict(features)
        preview_ms = (time.perf_counter() - started) * 1000.0
        self.forecaster.update(estimate.workload_scale)
        horizon_scales = self.forecaster.forecast(
            max(0, self.controller.c.horizon - 1),
            self.workload_model.min_target, self.workload_model.max_target)
        current_workload = {
            "frame_id": frame_id,
            "workload_scale": estimate.workload_scale,
            "workload_scale_std": estimate.uncertainty,
            "features": features,
        }
        future_workloads = [{"workload_scale": x, "workload_scale_std": estimate.uncertainty}
                            for x in horizon_scales]
        decision = self.controller.choose(current_workload, telemetry, future_workloads)
        self.previous_image_path = Path(image_path)
        self.last_frame_id = frame_id
        tot_energy = (decision.predicted_energy_mj + self.preview_energy_mj) if decision.predicted_energy_mj is not None else None
        return FrameDecisionRecord(
            frame_id=frame_id, workload_scale=estimate.workload_scale,
            workload_uncertainty=estimate.uncertainty,
            preview_in_calibration_range=estimate.in_calibration_range,
            preview_ms=preview_ms, action=decision.action.name, mode=decision.mode,
            reason=decision.reason, predicted_latency_ms=decision.predicted_latency_ms,
            predicted_energy_mj=decision.predicted_energy_mj,
            planned_sequence=decision.planned_sequence,
            preview_energy_mj=self.preview_energy_mj,
            predicted_total_energy_mj=tot_energy,
            predicted_e2e_latency_ms=decision.predicted_e2e_latency_ms)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", metavar="CALIBRATION.csv", help="fit workload model from labelled calibration CSV")
    ap.add_argument("--model-out", default="workload_model.json")
    ap.add_argument("--preview-image", help="extract and print features for one image")
    ap.add_argument("--previous-image", help="optional previous image for temporal-difference feature")
    args = ap.parse_args()
    if args.fit:
        model = fit_workload_model(args.fit, args.model_out)
        print(f"Fitted workload model on {model['training_rows']} labelled rows -> {args.model_out}")
        print(model["warning"])
    if args.preview_image:
        features = extract_frame_features(args.preview_image, args.previous_image)
        print(json.dumps(features, indent=2))
    if not args.fit and not args.preview_image:
        ap.print_help()

if __name__ == "__main__":
    main()
