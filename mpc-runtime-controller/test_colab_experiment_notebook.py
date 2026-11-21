#!/usr/bin/env python3
"""Tests verifying YOLOv8n_Colab_Latency_Experiment.ipynb structure, GPU guard,
404-frame manifest integrity, preprocessing parity, and report/plot helpers.
"""
from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

from milestone_b_pilot_dataset import load_cam_front_keyframes
from milestone_b_static_profiler import DEFAULT_WEIGHTS_PATH, compute_sha256, load_yolo_weights, preprocess_bgr_image as mb_preprocess


NB_PATH = Path("YOLOv8n_Colab_Latency_Experiment.ipynb")
MANIFEST_CSV = Path("results/colab_experiment/nuscenes_cam_front_404_manifest.csv")
BUNDLE_ZIP = Path("results/colab_experiment/colab_upload_bundle.zip")


def _load_nb() -> Dict[str, Any]:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _get_cell_source(nb: Dict[str, Any], cell_id: str) -> str:
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise KeyError(f"Missing cell id {cell_id!r}")


def _exec_defs_from_cell(source: str, ns: Dict[str, Any]) -> None:
    """Execute imports, pure static constants/classes, and def definitions from a cell."""
    tree = ast.parse(source)
    allowed_const_names = {
        "USE_GOOGLE_DRIVE", "DRIVE_BASE_DIR", "BUNDLE_ZIP_PATH", "LOCAL_STAGE_DIR",
        "OUTPUT_DIR", "COPY_OUTPUTS_TO_DRIVE", "MODEL_NAME", "EXPECTED_WEIGHTS_SHA256",
        "STRICT_WEIGHTS_SHA256", "PRECISION", "ENFORCE_STRICT_FP32", "RESOLUTIONS",
        "CONF_THRES", "IOU_THRES", "MAX_DET", "WARMUP_ITERS", "REPEAT_COUNT",
        "CPU_PILOT_REF_640x384", "SCENE_SPLITS",
    }
    keep_nodes = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in allowed_const_names for t in node.targets)
        )
    ]
    mod = ast.Module(body=keep_nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, filename="<colab_cell>", mode="exec"), ns)


def test_notebook_schema_and_ast_syntax() -> None:
    nb = _load_nb()
    assert nb["nbformat"] == 4
    assert nb["metadata"].get("accelerator") == "GPU"
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 9

    all_code = "\n".join("".join(c["source"]) for c in code_cells)
    for cell in code_cells:
        ast.parse("".join(cell["source"]))

    for marker in [
        "torch.cuda.synchronize",
        "torch.cuda.Event(enable_timing=True)",
        "allow_tf32 = False",
        "R1_320x192",
        "R2_480x288",
        "R3_640x384",
        "CONF_THRES = 0.25",
        "IOU_THRES = 0.45",
        "MAX_DET = 300",
        "WARMUP_ITERS = 15",
        "REPEAT_COUNT = 3",
        "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    ]:
        assert marker in all_code


def test_gpu_guard_halts_loudly_without_cpu_fallback() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this runtime; CPU-halt test applies on CPU-only host.")
    nb = _load_nb()
    ns: Dict[str, Any] = {}
    _exec_defs_from_cell(_get_cell_source(nb, "cell_01_config"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_02_env_gpu_check"), ns)

    with pytest.raises(RuntimeError, match="GPU_UNAVAILABLE"):
        ns["fail_fast_if_no_gpu"]()

    with pytest.raises(RuntimeError, match="Runtime -> Change runtime type"):
        ns["enforce_and_inspect_gpu"](enforce_strict_fp32=True)

    # Also execute full cell 1 + cell 2 verbatim to prove immediate fail-fast halt
    full_ns: Dict[str, Any] = {}
    exec(_get_cell_source(nb, "cell_01_config"), full_ns)
    with pytest.raises(RuntimeError, match="GPU_UNAVAILABLE"):
        exec(_get_cell_source(nb, "cell_02_env_gpu_check"), full_ns)


def test_manifest_and_bundle_cover_all_404_real_frames(tmp_path: Path) -> None:
    assert MANIFEST_CSV.is_file()
    df = pd.read_csv(MANIFEST_CSV)
    assert len(df) == 404
    assert df["sample_data_token"].nunique() == 404
    assert df["scene_id"].nunique() == 10
    assert int(df["is_milestone_b_pilot_frame"].sum()) == 100

    p100 = load_cam_front_keyframes(frames_per_scene=10)
    sub100 = df[df["is_milestone_b_pilot_frame"] == True]  # noqa: E712
    assert list(sub100["sample_data_token"]) == [r.sample_data_token for r in p100]

    nb = _load_nb()
    ns: Dict[str, Any] = {}
    _exec_defs_from_cell(_get_cell_source(nb, "cell_01_config"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_04_manifest_verify"), ns)

    recs, issues = ns["build_cam_front_manifest"](Path("data/nuscenes_mini"))
    assert len(recs) == 404
    assert len(issues) == 0
    assert [r.sample_data_token for r in recs] == list(df["sample_data_token"])
    assert {(r.meta_width, r.meta_height) for r in recs} == {(1600, 900)}

    # Missing image test: does not invent or duplicate frames
    fake_nusc = tmp_path / "nusc_partial"
    (fake_nusc / "v1.0-mini").mkdir(parents=True)
    for fn in ["scene.json", "sample.json", "sample_data.json"]:
        (fake_nusc / "v1.0-mini" / fn).write_text(
            (Path("data/nuscenes_mini/v1.0-mini") / fn).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    # Create only the first image file out of 404
    first_img = recs[0].rel_image_path
    dst = fake_nusc / first_img
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Path(recs[0].image_path).read_bytes())
    partial_recs, partial_issues = ns["build_cam_front_manifest"](fake_nusc)
    assert len(partial_recs) == 1
    assert len(partial_issues) == 403

    # Verify compact zip bundle contents and weights SHA-256
    assert BUNDLE_ZIP.is_file()
    with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
        names = set(zf.namelist())
    assert "results/milestone_b/weights/yolov8n.pt" in names
    assert "results/colab_experiment/nuscenes_cam_front_404_manifest.csv" in names
    assert "data/nuscenes_mini/v1.0-mini/scene.json" in names
    assert "data/nuscenes_mini/v1.0-mini/sample.json" in names
    assert "data/nuscenes_mini/v1.0-mini/sample_data.json" in names
    assert sum(1 for n in names if n.startswith("data/nuscenes_mini/samples/CAM_FRONT/")) == 404
    assert compute_sha256(DEFAULT_WEIGHTS_PATH) == "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def test_notebook_preprocessing_and_postprocess_parity_across_resolutions() -> None:
    nb = _load_nb()
    ns: Dict[str, Any] = {}
    _exec_defs_from_cell(_get_cell_source(nb, "cell_01_config"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_05_helpers"), ns)

    df = pd.read_csv(MANIFEST_CSV)
    img_path = df.iloc[0]["image_path"]
    bgr = cv2.imread(img_path)
    assert bgr is not None

    yolo = load_yolo_weights(DEFAULT_WEIGHTS_PATH, fuse=True)
    expected_anchors = {
        (192, 320): 1260,
        (288, 480): 2835,
        (384, 640): 5040,
    }
    expected_pads = {
        (192, 320): 6,
        (288, 480): 9,
        (384, 640): 12,
    }

    for _, hw in ns["RESOLUTIONS"]:
        t_nb, meta_nb = ns["preprocess_bgr_image"](bgr, hw)
        t_mb, meta_mb = mb_preprocess(bgr, target_hw=hw)
        assert tuple(t_nb.shape) == (1, 3, hw[0], hw[1])
        assert t_nb.dtype == torch.float32
        assert torch.equal(t_nb, t_mb)
        assert meta_nb["pad_top_px"] == expected_pads[hw]
        assert meta_nb["pad_bottom_px"] == expected_pads[hw]
        assert meta_mb["pad_top_px"] == expected_pads[hw]

        with torch.inference_mode():
            raw = yolo.model(t_nb)
        post = ns["run_single_postprocess"](raw, hw, (900, 1600), yolo.names, 0.25, 0.45, 300)
        assert post["raw_anchor_count"] == expected_anchors[hw]
        assert post["candidates_before_nms"] >= post["final_detection_count"] >= 0


def test_summary_plots_and_report_pipeline_end_to_end_in_tmp(tmp_path: Path) -> None:
    matplotlib.use("Agg", force=True)
    nb = _load_nb()
    ns: Dict[str, Any] = {}
    _exec_defs_from_cell(_get_cell_source(nb, "cell_01_config"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_04_manifest_verify"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_05_helpers"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_07_run_summarize"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_08_plots"), ns)
    _exec_defs_from_cell(_get_cell_source(nb, "cell_09_report_bundle"), ns)

    frames, issues = ns["build_cam_front_manifest"](Path("data/nuscenes_mini"))
    assert len(issues) == 0

    rng = np.random.default_rng(7)
    per_frame_rows = []
    repeat_rows = []
    run_metas = []
    env_info = {
        "python_version": "3.11.9",
        "platform": "Linux-x86_64",
        "torch_version": "2.6.0+cu124",
        "torchvision_version": "0.21.0+cu124",
        "ultralytics_version": "8.4.210",
        "opencv_version": "4.10.0",
        "cuda_version": "12.4",
        "cudnn_version": 90100,
        "gpu_name": "Tesla T4 (PYTEST_FIXTURE)",
        "gpu_compute_capability": "7.5",
        "gpu_multiprocessors": 40,
        "gpu_total_vram_mb": 15360.0,
        "gpu_driver_version": "550.54.15",
        "cudnn_benchmark": True,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
    }
    dataset_audit = {
        "expected_keyframes_in_nuscenes_mini": 404,
        "accessible_valid_frames": 404,
        "missing_or_corrupted_count": 0,
        "n_scenes": 10,
    }

    for r_idx, (lbl, hw) in enumerate(ns["RESOLUTIONS"]):
        base_fw = 2.0 + 1.5 * r_idx
        run_metas.append(
            {
                "resolution_label": lbl,
                "input_hw": list(hw),
                "raw_prediction_shape": "1x84x5040",
                "gpu_name": env_info["gpu_name"],
                "n_frames": len(frames),
                "static_params_fused": 3151904,
                "static_conv2d_macs": 655799040 * (r_idx + 1),
                "model_load_ms": 120.5 + r_idx,
            }
        )
        for rec in frames:
            reps = (base_fw + rng.normal(0.0, 0.2, size=3)).clip(min=0.5)
            fw = float(base_fw + rng.uniform(0.0, 0.5))
            post = float(0.8 + 0.1 * r_idx)
            per_frame_rows.append(
                {
                    "frame_id": rec.frame_id,
                    "scene_id": rec.scene_id,
                    "scene_split": rec.scene_split,
                    "environment_tag": rec.environment_tag,
                    "frame_in_scene": rec.frame_in_scene,
                    "is_milestone_b_pilot_frame": rec.is_milestone_b_pilot_frame,
                    "sample_token": rec.sample_token,
                    "sample_data_token": rec.sample_data_token,
                    "timestamp_us": rec.timestamp_us,
                    "dt_from_prev_ms": rec.dt_from_prev_ms if rec.dt_from_prev_ms is not None else "",
                    "resolution_label": lbl,
                    "input_width": hw[1],
                    "input_height": hw[0],
                    "orig_width": 1600,
                    "orig_height": 900,
                    "model_name": "yolov8n",
                    "precision": "fp32",
                    "gpu_name": env_info["gpu_name"],
                    "static_params": 3151904,
                    "static_macs": 655799040 * (r_idx + 1),
                    "image_load_ms": 5.0,
                    "preprocess_ms": 1.2,
                    "h2d_transfer_ms": 0.4,
                    "forward_pass_ms": fw,
                    "forward_wall_ms": fw + 0.05,
                    "postprocess_ms": post,
                    "compute_only_ms": 1.6 + fw + 0.05 + post,
                    "total_pipeline_ms": 6.6 + fw + 0.05 + post,
                    "repeat_count": 3,
                    "forward_repeat_mean_ms": float(np.mean(reps)),
                    "forward_repeat_median_ms": float(np.median(reps)),
                    "forward_repeat_std_ms": float(np.std(reps, ddof=1)),
                    "forward_repeat_p95_ms": float(np.percentile(reps, 95)),
                    "forward_repeat_max_ms": float(np.max(reps)),
                    "postprocess_repeat_median_ms": post,
                    "raw_anchor_count": 5040,
                    "candidates_before_nms": 20 + 5 * r_idx,
                    "final_detection_count": 4 + r_idx,
                    "mean_detection_conf": 0.65,
                    "max_detection_conf": 0.85,
                    "detected_class_counts": '{"car": 4}',
                    "gpu_mem_allocated_mb": 42.0,
                    "gpu_mem_peak_allocated_mb": 95.0 + 25.0 * r_idx,
                    "gpu_mem_reserved_mb": 128.0,
                }
            )
            for rep_i, rv in enumerate(reps):
                repeat_rows.append(
                    {
                        "frame_id": rec.frame_id,
                        "sample_data_token": rec.sample_data_token,
                        "scene_id": rec.scene_id,
                        "resolution_label": lbl,
                        "repeat_idx": rep_i,
                        "forward_pass_ms": float(rv),
                        "forward_wall_ms": float(rv + 0.03),
                        "postprocess_ms": post,
                    }
                )

    pf_df, rep_df, sum_df, exp_sum = ns["summarize_and_export"](
        per_frame_rows=per_frame_rows,
        raw_repeat_rows=repeat_rows,
        run_metas=run_metas,
        verified_frames=frames,
        env_info=env_info,
        weights_sha256="f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        dataset_audit=dataset_audit,
        output_dir=tmp_path,
    )
    assert len(pf_df) == 1212
    assert len(rep_df) == 3636
    assert len(sum_df) == 3
    plots = ns["generate_latency_plots"](pf_df, sum_df, tmp_path, show_inline=False)
    assert len(plots) == 4 and all(p.is_file() and p.stat().st_size > 1000 for p in plots)
    report_path = ns["render_colab_latency_report"](sum_df, exp_sum, tmp_path)
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Tesla T4 (PYTEST_FIXTURE)" in report_text
    assert "404" in report_text
