#!/usr/bin/env python3
"""Reproducible CPU inference and measurement pipeline for pretrained YOLOv8n.

Supports both a small smoke test (--mode smoke) and the 100-frame multi-scene
nuScenes CAM_FRONT pilot (--mode pilot). Preserves real ~2 Hz timestamps and
keeps host CPU timing strictly separated from target-accelerator claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from frame_preview_mpc import extract_frame_features
from milestone_b_pilot_dataset import (
    PilotFrameRecord,
    load_cam_front_keyframes,
    select_smoke_test_frames,
)
from milestone_b_static_profiler import (
    DEFAULT_WEIGHTS_PATH,
    compute_sha256,
    load_yolo_weights,
    preprocess_bgr_image,
    profile_static_layers,
)


def configure_cpu_environment(num_threads: int = 4) -> None:
    """Fix PyTorch and OpenCV thread counts for reproducible CPU measurement."""
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(min(1, num_threads))


def compute_distribution_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute summary statistics including p90, p95, and p99 percentiles."""
    if not values:
        return {
            "count": 0, "mean": None, "median": None, "std": None,
            "min": None, "max": None, "p90": None, "p95": None, "p99": None, "cv_pct": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv_pct = float(100.0 * std_v / abs(mean_v)) if abs(mean_v) > 1e-12 else 0.0
    return {
        "count": int(len(arr)),
        "mean": round(mean_v, 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(std_v, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "cv_pct": round(cv_pct, 2),
    }


def run_single_postprocess(
    raw_preds: Any,
    input_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    names: Dict[int, str],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> Dict[str, Any]:
    """Count pre-NMS thresholded candidates, apply NMS, and summarize boxes."""
    pred_tensor = raw_preds[0] if isinstance(raw_preds, (list, tuple)) else raw_preds
    nc = len(names) if names else (pred_tensor.shape[1] - 4)
    raw_anchor_count = int(pred_tensor.shape[-1])
    candidates_before_nms = int((pred_tensor[:, 4 : 4 + nc, :].amax(dim=1) > conf_thres).sum().item())

    dets_list = non_max_suppression(
        raw_preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )
    det = dets_list[0]
    if len(det):
        det = det.clone()
        det[:, :4] = scale_boxes(input_hw, det[:, :4], orig_hw).round()
        confs = det[:, 4].cpu().numpy()
        cls_ids = det[:, 5].cpu().numpy().astype(int)
        class_counts = Counter(names.get(int(cid), str(int(cid))) for cid in cls_ids)
        mean_conf = round(float(np.mean(confs)), 4)
        max_conf = round(float(np.max(confs)), 4)
        final_count = int(len(det))
    else:
        class_counts = Counter()
        mean_conf = None
        max_conf = None
        final_count = 0

    return {
        "raw_anchor_count": raw_anchor_count,
        "candidates_before_nms": candidates_before_nms,
        "final_detection_count": final_count,
        "mean_detection_conf": mean_conf,
        "max_detection_conf": max_conf,
        "detected_class_counts": dict(sorted(class_counts.items())),
    }


def execute_single_frame_stages(
    net: torch.nn.Module,
    image_path: str | Path,
    names: Dict[int, str],
    target_hw: Tuple[int, int] = (384, 640),
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> Tuple[Dict[str, Any], torch.Tensor, Tuple[int, int]]:
    """Run and separately time image load, preprocess, inference, and postprocess."""
    t_total_0 = time.perf_counter_ns()

    t_load_0 = time.perf_counter_ns()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    t_load_1 = time.perf_counter_ns()

    t_prep_0 = time.perf_counter_ns()
    tensor, prep_meta = preprocess_bgr_image(img_bgr, target_hw=target_hw)
    t_prep_1 = time.perf_counter_ns()

    t_inf_0 = time.perf_counter_ns()
    with torch.inference_mode():
        raw_preds = net(tensor)
    t_inf_1 = time.perf_counter_ns()

    t_post_0 = time.perf_counter_ns()
    orig_hw = (prep_meta["orig_height"], prep_meta["orig_width"])
    post_info = run_single_postprocess(
        raw_preds, target_hw, orig_hw, names, conf_thres, iou_thres, max_det
    )
    t_post_1 = time.perf_counter_ns()

    t_total_1 = time.perf_counter_ns()

    record: Dict[str, Any] = {}
    record.update(prep_meta)
    record.update(post_info)
    record.update(
        {
            "image_load_ms": (t_load_1 - t_load_0) / 1e6,
            "preprocess_ms": (t_prep_1 - t_prep_0) / 1e6,
            "inference_ms": (t_inf_1 - t_inf_0) / 1e6,
            "postprocess_ms": (t_post_1 - t_post_0) / 1e6,
            "total_pipeline_ms": (t_total_1 - t_total_0) / 1e6,
        }
    )
    return record, tensor, orig_hw


def run_benchmark_pipeline(
    frames: List[PilotFrameRecord],
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
    num_threads: int = 4,
    warmup_iters: int = 5,
    repeat_count: int = 3,
    target_hw: Tuple[int, int] = (384, 640),
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run load -> warmup -> chronological steady-state pass -> repeat reliability pass."""
    configure_cpu_environment(num_threads=num_threads)
    proc = psutil.Process(os.getpid())
    rss_before_load_mb = proc.memory_info().rss / (1024.0 ** 2)

    t_model_0 = time.perf_counter_ns()
    yolo = load_yolo_weights(weights_path, fuse=True)
    net = yolo.model
    names = yolo.names
    t_model_1 = time.perf_counter_ns()
    model_load_ms = (t_model_1 - t_model_0) / 1e6
    rss_after_load_mb = proc.memory_info().rss / (1024.0 ** 2)

    # Static profile at the configured target shape
    _, static_summary = profile_static_layers(
        net, input_shape=(1, 3, target_hw[0], target_hw[1]), weights_path=weights_path
    )
    static_params = int(static_summary["total_parameters_fused"])
    static_macs = int(static_summary["hook_dense_macs"])
    static_flops = int(static_summary["hook_dense_flops_2x"])

    # Warm-up passes excluded from steady-state measurements
    warmup_records = []
    if frames:
        w_frame = frames[0]
        for wi in range(warmup_iters):
            w_rec, _, _ = execute_single_frame_stages(
                net, w_frame.image_path, names, target_hw, conf_thres, iou_thres, max_det
            )
            warmup_records.append(
                {
                    "warmup_idx": wi,
                    "inference_ms": round(w_rec["inference_ms"], 4),
                    "total_pipeline_ms": round(w_rec["total_pipeline_ms"], 4),
                }
            )

    # Reset CPU percent counter right before chronological pass
    _ = proc.cpu_percent(interval=None)

    rows: List[Dict[str, Any]] = []
    cached_tensors: List[Tuple[torch.Tensor, Tuple[int, int]]] = []
    prev_scene_img: Dict[str, Optional[str]] = {}

    for rec_meta in frames:
        rss_pre_mb = proc.memory_info().rss / (1024.0 ** 2)
        _ = proc.cpu_percent(interval=None)
        stage_res, tensor, orig_hw = execute_single_frame_stages(
            net, rec_meta.image_path, names, target_hw, conf_thres, iou_thres, max_det
        )
        cpu_pct = proc.cpu_percent(interval=None)
        rss_post_mb = proc.memory_info().rss / (1024.0 ** 2)
        cached_tensors.append((tensor, orig_hw))

        # Causal pre-inference preview features (timed separately so PIL disk I/O
        # never alters detector stage latencies).
        prev_path = prev_scene_img.get(rec_meta.scene_id)
        t_pv_0 = time.perf_counter_ns()
        pv = extract_frame_features(rec_meta.image_path, previous_image_path=prev_path)
        t_pv_1 = time.perf_counter_ns()
        prev_scene_img[rec_meta.scene_id] = rec_meta.image_path

        compute_only_ms = (
            stage_res["preprocess_ms"] + stage_res["inference_ms"] + stage_res["postprocess_ms"]
        )

        row = {
            "frame_id": rec_meta.frame_id,
            "scene_id": rec_meta.scene_id,
            "scene_split": rec_meta.scene_split,
            "environment_tag": rec_meta.environment_tag,
            "frame_in_scene": rec_meta.frame_in_scene,
            "sample_token": rec_meta.sample_token,
            "sample_data_token": rec_meta.sample_data_token,
            "timestamp_us": rec_meta.timestamp_us,
            "scene_elapsed_ms": rec_meta.scene_elapsed_ms,
            "dt_from_prev_ms": rec_meta.dt_from_prev_ms if rec_meta.dt_from_prev_ms is not None else "",
            "image_path": rec_meta.image_path,
            "orig_width": stage_res["orig_width"],
            "orig_height": stage_res["orig_height"],
            "input_width": stage_res["input_width"],
            "input_height": stage_res["input_height"],
            "letterbox_scale": stage_res["letterbox_scale"],
            "pad_top_px": stage_res["pad_top_px"],
            "pad_bottom_px": stage_res["pad_bottom_px"],
            "model_name": "yolov8n",
            "model_version": ultralytics.__version__,
            "device": "cpu",
            "precision": "fp32",
            "num_threads": num_threads,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "image_load_ms": round(stage_res["image_load_ms"], 4),
            "preprocess_ms": round(stage_res["preprocess_ms"], 4),
            "inference_ms": round(stage_res["inference_ms"], 4),
            "postprocess_ms": round(stage_res["postprocess_ms"], 4),
            "compute_only_ms": round(compute_only_ms, 4),
            "total_pipeline_ms": round(stage_res["total_pipeline_ms"], 4),
            "repeat_count": repeat_count,
            "inference_repeat_mean_ms": "",
            "inference_repeat_median_ms": "",
            "inference_repeat_std_ms": "",
            "postprocess_repeat_median_ms": "",
            "static_params": static_params,
            "static_macs": static_macs,
            "static_flops_2x": static_flops,
            "measured_activation_sparsity": "",
            "raw_anchor_count": stage_res["raw_anchor_count"],
            "candidates_before_nms": stage_res["candidates_before_nms"],
            "final_detection_count": stage_res["final_detection_count"],
            "mean_detection_conf": stage_res["mean_detection_conf"] if stage_res["mean_detection_conf"] is not None else "",
            "max_detection_conf": stage_res["max_detection_conf"] if stage_res["max_detection_conf"] is not None else "",
            "detected_class_counts": json.dumps(stage_res["detected_class_counts"], sort_keys=True),
            "cpu_percent_proc": round(float(cpu_pct), 2),
            "rss_memory_mb": round(float(rss_post_mb), 3),
            "rss_delta_mb": round(float(rss_post_mb - rss_pre_mb), 4),
            "preview_extract_ms": round((t_pv_1 - t_pv_0) / 1e6, 4),
            "preview_brightness_mean": round(float(pv["brightness_mean"]), 6),
            "preview_brightness_std": round(float(pv["brightness_std"]), 6),
            "preview_edge_density": round(float(pv["edge_density"]), 6),
            "preview_histogram_entropy": round(float(pv["histogram_entropy"]), 6),
            "preview_temporal_difference": round(float(pv["temporal_difference"]), 6),
        }
        rows.append(row)

    # Pass 2: Repeat steady-state timing on cached tensors for repeatability noise floor
    if repeat_count > 0:
        for idx, (tensor, orig_hw) in enumerate(cached_tensors):
            inf_samples: List[float] = []
            post_samples: List[float] = []
            for _ in range(repeat_count):
                t0 = time.perf_counter_ns()
                with torch.inference_mode():
                    raw_p = net(tensor)
                t1 = time.perf_counter_ns()
                _ = run_single_postprocess(
                    raw_p, target_hw, orig_hw, names, conf_thres, iou_thres, max_det
                )
                t2 = time.perf_counter_ns()
                inf_samples.append((t1 - t0) / 1e6)
                post_samples.append((t2 - t1) / 1e6)
            rows[idx]["inference_repeat_mean_ms"] = round(float(np.mean(inf_samples)), 4)
            rows[idx]["inference_repeat_median_ms"] = round(float(np.median(inf_samples)), 4)
            rows[idx]["inference_repeat_std_ms"] = round(
                float(np.std(inf_samples, ddof=1)) if len(inf_samples) > 1 else 0.0, 4
            )
            rows[idx]["postprocess_repeat_median_ms"] = round(float(np.median(post_samples)), 4)

    # Aggregate class distribution & latency summaries
    agg_classes: Counter = Counter()
    for r in rows:
        agg_classes.update(json.loads(r["detected_class_counts"]))

    wpath = Path(weights_path)
    summary = {
        "model": {
            "name": "yolov8n",
            "weights_path": wpath.as_posix(),
            "weights_sha256": compute_sha256(wpath) if wpath.is_file() else None,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "device": "cpu",
            "precision": "fp32",
            "num_threads": num_threads,
            "input_hw": list(target_hw),
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "max_det": max_det,
            "static_params_fused": static_params,
            "static_dense_macs": static_macs,
            "static_dense_flops_2x": static_flops,
        },
        "cold_start": {
            "model_load_ms": round(model_load_ms, 4),
            "rss_before_load_mb": round(rss_before_load_mb, 3),
            "rss_after_load_mb": round(rss_after_load_mb, 3),
            "model_load_rss_delta_mb": round(rss_after_load_mb - rss_before_load_mb, 3),
            "warmup_iterations": warmup_iters,
            "warmup_samples": warmup_records,
        },
        "dataset": {
            "n_frames": len(rows),
            "n_scenes": len(set(r["scene_id"] for r in rows)),
            "scenes": sorted(set(r["scene_id"] for r in rows)),
            "dt_ms_stats": compute_distribution_stats(
                [float(r["dt_from_prev_ms"]) for r in rows if r["dt_from_prev_ms"] != ""]
            ),
        },
        "latency_stats_ms": {
            "image_load_ms": compute_distribution_stats([float(r["image_load_ms"]) for r in rows]),
            "preprocess_ms": compute_distribution_stats([float(r["preprocess_ms"]) for r in rows]),
            "inference_ms": compute_distribution_stats([float(r["inference_ms"]) for r in rows]),
            "inference_repeat_median_ms": compute_distribution_stats(
                [float(r["inference_repeat_median_ms"]) for r in rows if r["inference_repeat_median_ms"] != ""]
            ),
            "postprocess_ms": compute_distribution_stats([float(r["postprocess_ms"]) for r in rows]),
            "compute_only_ms": compute_distribution_stats([float(r["compute_only_ms"]) for r in rows]),
            "total_pipeline_ms": compute_distribution_stats([float(r["total_pipeline_ms"]) for r in rows]),
            "preview_extract_ms": compute_distribution_stats([float(r["preview_extract_ms"]) for r in rows]),
        },
        "workload_counts": {
            "candidates_before_nms": compute_distribution_stats([float(r["candidates_before_nms"]) for r in rows]),
            "final_detection_count": compute_distribution_stats([float(r["final_detection_count"]) for r in rows]),
            "total_final_detections": int(sum(int(r["final_detection_count"]) for r in rows)),
            "detected_class_totals": dict(agg_classes.most_common()),
        },
        "system_telemetry": {
            "cpu_percent_proc": compute_distribution_stats([float(r["cpu_percent_proc"]) for r in rows]),
            "rss_memory_mb": compute_distribution_stats([float(r["rss_memory_mb"]) for r in rows]),
        },
    }
    return rows, summary


def write_measurements_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No measurement rows to write")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--frames-per-scene", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-dir", default="results/milestone_b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pilot = load_cam_front_keyframes(frames_per_scene=args.frames_per_scene)
    if args.mode == "smoke":
        frames = select_smoke_test_frames(all_pilot, n_frames=5)
        out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "smoke_test_measurements.csv")
        out_json = out_dir / "smoke_test_summary.json"
    else:
        frames = all_pilot
        out_csv = Path(args.out_csv) if args.out_csv else Path("real_dnn_pilot_measurements.csv")
        out_json = out_dir / "pilot_summary.json"

    rows, summary = run_benchmark_pipeline(
        frames=frames,
        weights_path=args.weights,
        num_threads=args.threads,
        warmup_iters=args.warmup,
        repeat_count=args.repeats,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )
    write_measurements_csv(rows, out_csv)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[{args.mode.upper()}] Processed {len(rows)} frames -> {out_csv}")
    print(f"  Model load: {summary['cold_start']['model_load_ms']:.2f} ms (RSS +{summary['cold_start']['model_load_rss_delta_mb']:.1f} MB)")
    inf_s = summary["latency_stats_ms"]["inference_ms"]
    post_s = summary["latency_stats_ms"]["postprocess_ms"]
    tot_s = summary["latency_stats_ms"]["total_pipeline_ms"]
    print(f"  Inference ms: mean={inf_s['mean']:.2f}, med={inf_s['median']:.2f}, std={inf_s['std']:.2f}, p95={inf_s['p95']:.2f}")
    print(f"  Postprocess ms: mean={post_s['mean']:.2f}, std={post_s['std']:.2f}, p95={post_s['p95']:.2f}")
    print(f"  Total pipeline ms: mean={tot_s['mean']:.2f}, p95={tot_s['p95']:.2f}, p99={tot_s['p99']:.2f}")
    print(f"  Detections: total={summary['workload_counts']['total_final_detections']}, summary -> {out_json}")


if __name__ == "__main__":
    main()
