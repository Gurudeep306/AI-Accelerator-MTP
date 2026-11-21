#!/usr/bin/env python3
"""Generate YOLOv8n_Colab_Latency_Experiment.ipynb, locked 404-frame manifest, and compact upload bundle."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from milestone_b_pilot_dataset import load_cam_front_keyframes
from milestone_b_static_profiler import DEFAULT_WEIGHTS_PATH, compute_sha256


OUT_NOTEBOOK = Path("YOLOv8n_Colab_Latency_Experiment.ipynb")
OUT_DIR = Path("results/colab_experiment")
MANIFEST_CSV = OUT_DIR / "nuscenes_cam_front_404_manifest.csv"
UPLOAD_BUNDLE_ZIP = OUT_DIR / "colab_upload_bundle.zip"


def _md_cell(text: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in text.rstrip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def _code_cell(code: str, cell_id: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in code.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


CELL_01_MD = r"""# YOLOv8n Google Colab GPU Latency Experiment (R1 / R2 / R3)

**Project:** AI Accelerator MPC Runtime Controller (`C:\AI accelerator project`)  
**Experiment Scope:** Measure **steady-state inference & pipeline latency** of pretrained `YOLOv8n` (`yolov8n.pt`, **FP32**) on a **Google Colab NVIDIA GPU** across three native `16:9` stride-32 resolutions on `nuScenes v1.0-mini` `CAM_FRONT` keyframes.

| Config ID | Target Input (`W×H`) | Tensor Shape (`N,C,H,W`) | Scale vs `1600×900` | LetterBox Pad (Top/Bottom) |
|---|---:|---|---:|---:|
| `R1_320x192` | `320×192` | `(1, 3, 192, 320)` | `0.20` | `6 px` / `6 px` |
| `R2_480x288` | `480×288` | `(1, 3, 288, 480)` | `0.30` | `9 px` / `9 px` |
| `R3_640x384` | `640×384` | `(1, 3, 384, 640)` | `0.40` | `12 px` / `12 px` |

### Strict Methodology & Restrictions
- **GPU Required:** Halts immediately if CUDA GPU is unavailable (`Runtime -> Change runtime type -> T4 GPU`). **Zero silent CPU fallback.**
- **Identical Setup Across Resolutions:** Same checkpoint (`yolov8n.pt`), **FP32** only (`allow_tf32=False`), `conf_thres=0.25`, `iou_thres=0.45`, `max_det=300`, `LetterBox(stride=32, auto=False, center=True)`, identical frame order.
- **Strict Scope Boundaries:** No INT8/FP16, no extra models, no nuScenes 3D→2D quality projection, no MPC edits, and no extrapolation of Colab datacenter GPU latency/power/energy to our target embedded accelerator."""


CELL_02_CONFIG = r'''# @title 1. Experiment Configuration & Milestone B CPU Reference
from pathlib import Path

# Dataset staging mode:
# - Set USE_GOOGLE_DRIVE = True if your bundle or nuScenes folder is stored in Google Drive.
# - Keep False to use /content/colab_upload_bundle.zip (direct Colab upload) or local workspace paths.
USE_GOOGLE_DRIVE = False
DRIVE_BASE_DIR = Path("/content/drive/MyDrive/AI_accelerator_project")
BUNDLE_ZIP_PATH = Path("/content/colab_upload_bundle.zip")
LOCAL_STAGE_DIR = Path("/content/colab_data")
OUTPUT_DIR = Path("/content/colab_outputs")
COPY_OUTPUTS_TO_DRIVE = True

# Fixed model & inference parameters (must remain identical across R1, R2, R3)
MODEL_NAME = "yolov8n"
EXPECTED_WEIGHTS_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
STRICT_WEIGHTS_SHA256 = True
PRECISION = "fp32"
ENFORCE_STRICT_FP32 = True  # Disables TF32 on Ampere+ GPUs so compute remains true FP32

RESOLUTIONS = [
    ("R1_320x192", (192, 320)),
    ("R2_480x288", (288, 480)),
    ("R3_640x384", (384, 640)),
]

CONF_THRES = 0.25
IOU_THRES = 0.45
MAX_DET = 300
WARMUP_ITERS = 15   # >= 10 required warm-up runs before frame 0 at each resolution
REPEAT_COUNT = 3    # >= 3 steady-state repeats per frame (cached tensor on GPU)

# Verified Milestone B host CPU baseline at R3 (640x384), N=100 pilot frames (Intel Core i5-10400H, 4 threads)
CPU_PILOT_REF_640x384 = {
    "hardware": "Intel(R) Core(TM) i5-10400H CPU @ 2.60GHz (4 threads)",
    "n_frames": 100,
    "resolution": "R3_640x384",
    "image_load_ms": {"mean": 6.17, "median": 5.64, "p95": 9.46, "std": 1.56, "max": 12.92},
    "preprocess_ms": {"mean": 1.92, "median": 1.75, "p95": 2.57, "std": 0.39, "max": 3.69},
    "forward_pass_ms": {"mean": 44.34, "median": 40.76, "p95": 57.37, "std": 11.55, "max": 138.57},
    "forward_repeat_median_ms": {"mean": 47.52, "median": 44.66, "p95": 62.55, "std": 6.71, "max": 66.77},
    "postprocess_ms": {"mean": 2.21, "median": 2.08, "p95": 3.54, "std": 0.68, "max": 4.41},
    "total_pipeline_ms": {"mean": 54.64, "median": 51.11, "p95": 72.35, "std": 12.03, "max": 148.49},
    "candidates_before_nms": {"mean": 40.53, "median": 37.0, "p95": 81.20},
    "final_detection_count": {"mean": 7.37, "median": 6.5, "p95": 15.00, "total": 737},
}

print("Configuration loaded. Resolutions:", [r[0] for r in RESOLUTIONS])'''


CELL_03_ENV = r'''# @title 2. Check GPU Runtime Immediately, Install Compatible Packages & Enforce FP32 Telemetry
import importlib
import platform
import subprocess
import sys


def fail_fast_if_no_gpu() -> None:
    """Stop immediately before package installation if CUDA GPU is unavailable."""
    import torch

    if (not torch.cuda.is_available()) or (torch.cuda.device_count() < 1):
        msg = (
            "GPU_UNAVAILABLE: torch.cuda.is_available() is False or no CUDA devices were detected.\n"
            "How to enable GPU in Google Colab:\n"
            "  1. Click Runtime -> Change runtime type\n"
            "  2. Select Hardware accelerator: T4 GPU (or L4 / A100 GPU)\n"
            "  3. Click Save, wait for runtime reconnect, and Run All again.\n"
            "Per experiment rules, this notebook will NOT silently fall back to CPU."
        )
        print(msg)
        raise RuntimeError(msg)


def install_colab_packages() -> None:
    """Install Ultralytics + profiling deps without overwriting CUDA-enabled PyTorch."""
    needed = {
        "ultralytics": "ultralytics==8.4.210",
        "thop": "thop==0.1.1.post2209072238",
        "pynvml": "nvidia-ml-py",
    }
    to_install = [pkg for mod, pkg in needed.items() if importlib.util.find_spec(mod) is None]
    if to_install:
        print("Installing packages:", to_install)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    else:
        print("All required packages already installed.")


def enforce_and_inspect_gpu(enforce_strict_fp32: bool = True) -> dict:
    """Verify CUDA GPU availability; halt loudly (no CPU fallback) if absent."""
    import cv2
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    import torchvision
    import ultralytics

    if (not torch.cuda.is_available()) or (torch.cuda.device_count() < 1):
        msg = (
            "GPU_UNAVAILABLE: torch.cuda.is_available() is False or no CUDA devices were detected.\n"
            "How to enable GPU in Google Colab:\n"
            "  1. Click Runtime -> Change runtime type\n"
            "  2. Select Hardware accelerator: T4 GPU (or L4 / A100 GPU)\n"
            "  3. Click Save, wait for runtime reconnect, and Run All again.\n"
            "Per experiment rules, this notebook will NOT silently fall back to CPU."
        )
        print(msg)
        raise RuntimeError(msg)

    # Explicit precision policy: keep IEEE FP32 strictly enforced (disable TF32 matmul/cuDNN shortcuts)
    if enforce_strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    dev_idx = 0
    torch.cuda.set_device(dev_idx)
    props = torch.cuda.get_device_properties(dev_idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)

    nvml_info = {"available": False, "driver_version": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
        drv = pynvml.nvmlSystemGetDriverVersion()
        nvml_info["driver_version"] = drv.decode("utf-8") if isinstance(drv, bytes) else str(drv)
        nvml_info["available"] = True
    except Exception as exc:
        nvml_info["error"] = str(exc)

    env_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_processor": platform.processor() or platform.machine(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_device_index": dev_idx,
        "gpu_name": torch.cuda.get_device_name(dev_idx),
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "gpu_multiprocessors": int(props.multi_processor_count),
        "gpu_total_vram_mb": round(props.total_memory / (1024.0 ** 2), 2),
        "gpu_free_vram_at_start_mb": round(free_bytes / (1024.0 ** 2), 2),
        "gpu_driver_version": nvml_info.get("driver_version"),
        "nvml_telemetry_available": nvml_info["available"],
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "precision": "fp32",
    }
    return env_info


fail_fast_if_no_gpu()
install_colab_packages()
ENV_INFO = enforce_and_inspect_gpu(enforce_strict_fp32=ENFORCE_STRICT_FP32)
INITIAL_GPU_NAME = ENV_INFO["gpu_name"]

print("=== Verified Colab GPU & Software Environment ===")
for k, v in ENV_INFO.items():
    print(f"  {k}: {v}")'''


CELL_04_DATA = r'''# @title 3. Access Dataset & Existing YOLOv8n Checkpoint (With Exact SHA-256 Verification)
import hashlib
import shutil
from typing import Optional, Tuple
import zipfile
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_dataset_and_weights(
    use_google_drive: bool,
    drive_base_dir: Path,
    bundle_zip_path: Path,
    local_stage_dir: Path,
) -> Tuple[Path, Path, Optional[Path]]:
    """Locate or extract nuScenes CAM_FRONT data, yolov8n.pt, and optional manifest CSV."""
    if use_google_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except ImportError:
            print("Note: google.colab.drive unavailable outside Colab; reading paths directly.")

        # Case 1A: Full repo layout copied to Drive
        drive_nusc = drive_base_dir / "data" / "nuscenes_mini"
        drive_weights = drive_base_dir / "results" / "milestone_b" / "weights" / "yolov8n.pt"
        drive_manifest = drive_base_dir / "results" / "colab_experiment" / "nuscenes_cam_front_404_manifest.csv"
        if drive_nusc.is_dir() and drive_weights.is_file():
            return drive_nusc, drive_weights, (drive_manifest if drive_manifest.is_file() else None)

        # Case 1B: Compact bundle zip stored in Drive
        drive_zip = drive_base_dir / "colab_upload_bundle.zip"
        if drive_zip.is_file():
            bundle_zip_path = drive_zip

    # Case 2: Local project checkout (for testing) or extracted upload bundle
    local_repo_nusc = Path("data/nuscenes_mini")
    local_repo_weights = Path("results/milestone_b/weights/yolov8n.pt")
    local_repo_manifest = Path("results/colab_experiment/nuscenes_cam_front_404_manifest.csv")
    if local_repo_nusc.is_dir() and local_repo_weights.is_file():
        return (
            local_repo_nusc,
            local_repo_weights,
            (local_repo_manifest if local_repo_manifest.is_file() else None),
        )

    if not bundle_zip_path.is_file():
        # Offer interactive Colab file upload for convenience when bundle zip isn't present yet
        try:
            from google.colab import files
            print(f"Bundle {bundle_zip_path} not found. Prompting upload of colab_upload_bundle.zip...")
            uploaded = files.upload()
            if not uploaded:
                raise FileNotFoundError("No file uploaded.")
            first_name = next(iter(uploaded.keys()))
            bundle_zip_path = Path("/content") / first_name
        except ImportError:
            pass

    if not bundle_zip_path.is_file():
        raise FileNotFoundError(
            f"Could not find dataset/weights bundle at {bundle_zip_path} or Drive paths under {drive_base_dir}."
        )

    extract_root = local_stage_dir
    if not (extract_root / "data" / "nuscenes_mini" / "v1.0-mini").is_dir():
        print(f"Extracting {bundle_zip_path} -> {extract_root} ...")
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip_path, "r") as zf:
            zf.extractall(extract_root)

    nusc_root = extract_root / "data" / "nuscenes_mini"
    weights_path = extract_root / "results" / "milestone_b" / "weights" / "yolov8n.pt"
    manifest_csv = extract_root / "results" / "colab_experiment" / "nuscenes_cam_front_404_manifest.csv"
    return nusc_root, weights_path, (manifest_csv if manifest_csv.is_file() else None)


NUSCENES_ROOT, WEIGHTS_PATH, EXPECTED_MANIFEST_CSV = stage_dataset_and_weights(
    USE_GOOGLE_DRIVE, DRIVE_BASE_DIR, BUNDLE_ZIP_PATH, LOCAL_STAGE_DIR
)

if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH}")

WEIGHTS_BYTES = WEIGHTS_PATH.stat().st_size
WEIGHTS_SHA256 = compute_sha256(WEIGHTS_PATH)
print(f"Weights path  : {WEIGHTS_PATH} ({WEIGHTS_BYTES:,} bytes)")
print(f"Weights SHA256: {WEIGHTS_SHA256}")
print(f"Expected SHA  : {EXPECTED_WEIGHTS_SHA256}")

if STRICT_WEIGHTS_SHA256 and WEIGHTS_SHA256 != EXPECTED_WEIGHTS_SHA256:
    raise RuntimeError(
        f"Checkpoint SHA256 mismatch! Got {WEIGHTS_SHA256}, expected {EXPECTED_WEIGHTS_SHA256}."
    )
print("Checkpoint identity verified: EXACT MATCH.")'''


CELL_05_MANIFEST = r'''# @title 4. Deterministic 404-Frame Dataset Manifest & Image Integrity Verification
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import cv2
import pandas as pd

SCENE_SPLITS = {
    "scene-0061": ("train", "day_rain"),
    "scene-0103": ("train", "day_clear"),
    "scene-0553": ("train", "day_clear"),
    "scene-0655": ("train", "day_clear"),
    "scene-0757": ("train", "day_clear"),
    "scene-0796": ("train", "day_clear"),
    "scene-0916": ("val", "day_clear"),
    "scene-1077": ("val", "night_urban"),
    "scene-1094": ("test", "night_rain"),
    "scene-1100": ("test", "night_glare"),
}


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    scene_id: str
    scene_split: str
    environment_tag: str
    frame_in_scene: int
    is_milestone_b_pilot_frame: bool
    sample_token: str
    sample_data_token: str
    timestamp_us: int
    scene_elapsed_ms: float
    dt_from_prev_ms: Optional[float]
    rel_image_path: str
    image_path: str
    meta_width: int
    meta_height: int


def build_cam_front_manifest(nuscenes_root: Path) -> Tuple[List[FrameRecord], List[dict]]:
    """Load chronological CAM_FRONT keyframes directly from nuScenes v1.0-mini JSONs.

    Never invents or duplicates frames; logs any missing file explicitly.
    """
    meta_dir = nuscenes_root / "v1.0-mini"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"nuScenes metadata dir not found: {meta_dir}")

    scenes = json.loads((meta_dir / "scene.json").read_text(encoding="utf-8"))
    samples = json.loads((meta_dir / "sample.json").read_text(encoding="utf-8"))
    sample_data = json.loads((meta_dir / "sample_data.json").read_text(encoding="utf-8"))

    scene_by_token = {s["token"]: s["name"] for s in scenes}
    sample_to_scene = {s["token"]: scene_by_token[s["scene_token"]] for s in samples}

    per_scene_raw: Dict[str, List[dict]] = {}
    for sd in sample_data:
        filename = sd.get("filename", "")
        if filename.startswith("samples/CAM_FRONT/") and sd.get("is_key_frame"):
            s_name = sample_to_scene.get(sd["sample_token"])
            if s_name is not None:
                per_scene_raw.setdefault(s_name, []).append(sd)

    records: List[FrameRecord] = []
    missing_or_corrupt: List[dict] = []
    global_fid = 0

    for scene_name in sorted(per_scene_raw.keys()):
        items = sorted(per_scene_raw[scene_name], key=lambda item: int(item["timestamp"]))
        split, env_tag = SCENE_SPLITS.get(scene_name, ("unknown", "unknown"))
        first_ts = int(items[0]["timestamp"])
        prev_ts: Optional[int] = None

        for idx, item in enumerate(items):
            ts_us = int(item["timestamp"])
            rel_path = item["filename"]
            img_path = nuscenes_root / rel_path
            if not img_path.is_file():
                missing_or_corrupt.append(
                    {
                        "scene_id": scene_name,
                        "frame_in_scene": idx,
                        "sample_data_token": str(item["token"]),
                        "rel_image_path": rel_path,
                        "reason": "file_missing",
                    }
                )
                prev_ts = ts_us
                continue

            img_check = cv2.imread(str(img_path))
            if img_check is None:
                missing_or_corrupt.append(
                    {
                        "scene_id": scene_name,
                        "frame_in_scene": idx,
                        "sample_data_token": str(item["token"]),
                        "rel_image_path": rel_path,
                        "reason": "cv2_imread_corrupted",
                    }
                )
                prev_ts = ts_us
                continue

            h_dec, w_dec = img_check.shape[:2]
            elapsed_ms = (ts_us - first_ts) / 1000.0
            dt_ms = None if prev_ts is None else (ts_us - prev_ts) / 1000.0
            records.append(
                FrameRecord(
                    frame_id=global_fid,
                    scene_id=scene_name,
                    scene_split=split,
                    environment_tag=env_tag,
                    frame_in_scene=idx,
                    is_milestone_b_pilot_frame=(idx < 10),
                    sample_token=str(item["sample_token"]),
                    sample_data_token=str(item["token"]),
                    timestamp_us=ts_us,
                    scene_elapsed_ms=round(elapsed_ms, 3),
                    dt_from_prev_ms=None if dt_ms is None else round(dt_ms, 3),
                    rel_image_path=rel_path,
                    image_path=img_path.as_posix(),
                    meta_width=int(w_dec),
                    meta_height=int(h_dec),
                )
            )
            prev_ts = ts_us
            global_fid += 1

    return records, missing_or_corrupt


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VERIFIED_FRAMES, DATASET_ISSUES = build_cam_front_manifest(NUSCENES_ROOT)

if len(VERIFIED_FRAMES) == 0:
    raise FileNotFoundError("Zero valid CAM_FRONT images decoded from dataset.")

if EXPECTED_MANIFEST_CSV is not None and EXPECTED_MANIFEST_CSV.is_file():
    exp_df = pd.read_csv(EXPECTED_MANIFEST_CSV)
    exp_tokens = list(exp_df["sample_data_token"])
    got_tokens = [r.sample_data_token for r in VERIFIED_FRAMES]
    if len(VERIFIED_FRAMES) == len(exp_tokens) and got_tokens != exp_tokens:
        raise RuntimeError("Decoded frame order/tokens do not match locked 404-frame manifest!")
    elif len(VERIFIED_FRAMES) < len(exp_tokens):
        missing_set = set(exp_tokens) - set(got_tokens)
        print(f"WARNING: Only {len(VERIFIED_FRAMES)}/{len(exp_tokens)} manifest frames accessible ({len(missing_set)} missing).")
    else:
        print("Manifest cross-check PASSED: exact 404 sample_data_token sequence matches local repository.")

unique_fids = len({r.frame_id for r in VERIFIED_FRAMES})
unique_tokens = len({r.sample_data_token for r in VERIFIED_FRAMES})
unique_scenes = sorted({r.scene_id for r in VERIFIED_FRAMES})
unique_dims = sorted({(r.meta_width, r.meta_height) for r in VERIFIED_FRAMES})

DATASET_AUDIT = {
    "expected_keyframes_in_nuscenes_mini": 404,
    "accessible_valid_frames": len(VERIFIED_FRAMES),
    "missing_or_corrupted_count": len(DATASET_ISSUES),
    "missing_or_corrupted_details": DATASET_ISSUES,
    "unique_frame_ids": unique_fids,
    "unique_sample_data_tokens": unique_tokens,
    "n_scenes": len(unique_scenes),
    "scenes": unique_scenes,
    "decoded_dimensions_wh": [list(d) for d in unique_dims],
    "timestamp_us_min": min(r.timestamp_us for r in VERIFIED_FRAMES),
    "timestamp_us_max": max(r.timestamp_us for r in VERIFIED_FRAMES),
}

manifest_df = pd.DataFrame([asdict(r) for r in VERIFIED_FRAMES])
manifest_df.to_csv(OUTPUT_DIR / "verified_frame_manifest.csv", index=False)

print("=== Dataset Verification Summary ===")
for k, v in DATASET_AUDIT.items():
    if k != "missing_or_corrupted_details":
        print(f"  {k}: {v}")
if DATASET_ISSUES:
    print("  Issues (first 5):", DATASET_ISSUES[:5])'''


CELL_06_HELPERS = r'''# @title 5. Model Load, Preprocessing, Static Conv2d MACs & Synchronized GPU Postprocessing
from collections import Counter
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


def preprocess_bgr_image(img_bgr: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, dict]:
    """Identical LetterBox + BGR->RGB + CHW FP32 [0,1] normalization to Milestone B CPU."""
    orig_h, orig_w = img_bgr.shape[:2]
    lb = LetterBox(new_shape=target_hw, auto=False, scale_fill=False, scaleup=True, center=True, stride=32)
    padded = lb(image=img_bgr)
    in_h, in_w = padded.shape[:2]

    scale = min(target_hw[0] / orig_h, target_hw[1] / orig_w)
    unpad_h, unpad_w = round(orig_h * scale), round(orig_w * scale)
    dw = (in_w - unpad_w) / 2.0
    dh = (in_h - unpad_h) / 2.0

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(chw).unsqueeze(0)  # float32 CPU tensor
    meta = {
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "input_width": int(in_w),
        "input_height": int(in_h),
        "letterbox_scale": round(float(scale), 6),
        "pad_top_px": int(round(dh - 0.1)),
        "pad_bottom_px": int(round(dh + 0.1)),
    }
    return tensor, meta


def load_yolo_on_gpu(weights_path: Path, device: torch.device) -> Tuple[YOLO, nn.Module, dict, float, float]:
    """Load, fuse, move to GPU in strict FP32, and record separate model load time & VRAM."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    mem_before_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)

    t0 = time.perf_counter_ns()
    yolo = YOLO(str(weights_path))
    yolo.fuse()
    net = yolo.model.to(device=device, dtype=torch.float32)
    net.eval()
    torch.cuda.synchronize(device)
    t1 = time.perf_counter_ns()

    mem_after_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
    params = list(net.parameters())
    assert params and all(p.is_cuda for p in params), "Model parameters are not on CUDA!"
    assert all(p.dtype == torch.float32 for p in params), "Model parameters are not FP32!"
    return yolo, net, yolo.names, (t1 - t0) / 1e6, (mem_after_mb - mem_before_mb)


def compute_conv2d_static_macs(net: nn.Module, target_hw: Tuple[int, int], device: torch.device) -> Tuple[int, int, str]:
    """Count exact hooked Conv2d MACs and fused params at target_hw on GPU."""
    total_macs = 0
    hooks = []

    def hook_fn(mod: nn.Module, _inp, out) -> None:
        nonlocal total_macs
        if isinstance(mod, nn.Conv2d) and isinstance(out, torch.Tensor):
            b, c_out, h_out, w_out = out.shape
            k_h, k_w = mod.kernel_size
            c_in_per_group = mod.in_channels // int(mod.groups)
            total_macs += int(b * h_out * w_out * c_out * c_in_per_group * k_h * k_w)

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_fn))

    dummy = torch.zeros((1, 3, target_hw[0], target_hw[1]), dtype=torch.float32, device=device)
    try:
        with torch.inference_mode():
            out = net(dummy)
            torch.cuda.synchronize(device)
    finally:
        for h in hooks:
            h.remove()

    pred0 = out[0] if isinstance(out, (list, tuple)) else out
    raw_shape = "x".join(str(d) for d in pred0.shape)
    static_params = sum(int(p.numel()) for p in net.parameters())
    return int(static_params), int(total_macs), raw_shape


def run_single_postprocess(
    raw_preds,
    input_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    names: dict,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> dict:
    """Count pre-NMS candidates (>conf_thres), apply NMS, scale_boxes, and extract counts."""
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
        confs = det[:, 4].detach().cpu().numpy()
        cls_ids = det[:, 5].detach().cpu().numpy().astype(int)
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


def sample_nvml_telemetry(dev_idx: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Query GPU utilization %, memory utilization %, and temperature C via NVML if permitted."""
    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return float(util.gpu), float(util.memory), float(temp)
    except Exception:
        return None, None, None


def compute_stats(values: List[float]) -> dict:
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
    }'''


CELL_07_BENCHMARK = r'''# @title 6. Synchronized GPU Benchmark Engine (CUDA Events + Pass-1 Chronological + Pass-2 Repeats)
import gc
from typing import Dict, List, Tuple


def benchmark_single_resolution(
    res_label: str,
    target_hw: Tuple[int, int],
    frames: List[FrameRecord],
    weights_path: Path,
    initial_gpu_name: str,
    env_info: dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    warmup_iters: int = 15,
    repeat_count: int = 3,
) -> Tuple[List[dict], List[dict], dict]:
    """Execute load -> warmup -> chronological pipeline -> R repeats on GPU for one resolution."""
    device = torch.device("cuda:0")
    current_gpu = torch.cuda.get_device_name(0)
    if current_gpu != initial_gpu_name:
        raise RuntimeError(f"GPU changed between runs! Initial={initial_gpu_name}, Current={current_gpu}")

    yolo, net, names, model_load_ms, model_vram_delta_mb = load_yolo_on_gpu(weights_path, device)
    static_params, static_macs, raw_pred_shape = compute_conv2d_static_macs(net, target_hw, device)

    # Warm-up runs on GPU using first real frame (excluded cleanly from steady-state stats)
    w_bgr = cv2.imread(frames[0].image_path)
    w_cpu, _ = preprocess_bgr_image(w_bgr, target_hw)
    w_gpu = w_cpu.to(device=device, dtype=torch.float32)
    assert w_gpu.is_cuda and w_gpu.dtype == torch.float32

    warmup_forward_ms = []
    torch.cuda.synchronize(device)
    for _ in range(warmup_iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        with torch.inference_mode():
            w_out = net(w_gpu)
        ev1.record()
        _ = run_single_postprocess(
            w_out, target_hw, (frames[0].meta_height, frames[0].meta_width),
            names, conf_thres, iou_thres, max_det,
        )
        torch.cuda.synchronize(device)
        warmup_forward_ms.append(round(float(ev0.elapsed_time(ev1)), 4))

    del w_bgr, w_cpu, w_gpu, w_out
    torch.cuda.synchronize(device)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    per_frame_rows: List[dict] = []
    cached_cpu_tensors: List[Tuple[torch.Tensor, Tuple[int, int]]] = []

    # Pass 1: Chronological per-frame full pipeline measurement in locked frame manifest order
    for rec in frames:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t_tot_0 = time.perf_counter_ns()

        t_load_0 = time.perf_counter_ns()
        img_bgr = cv2.imread(rec.image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read {rec.image_path}")
        t_load_1 = time.perf_counter_ns()

        t_prep_0 = time.perf_counter_ns()
        tensor_cpu, prep_meta = preprocess_bgr_image(img_bgr, target_hw=target_hw)
        t_prep_1 = time.perf_counter_ns()

        torch.cuda.synchronize(device)
        t_h2d_0 = time.perf_counter_ns()
        tensor_gpu = tensor_cpu.to(device=device, dtype=torch.float32)
        torch.cuda.synchronize(device)
        t_h2d_1 = time.perf_counter_ns()

        # Verify GPU execution & FP32 explicitly on frame 0
        if rec.frame_id == 0:
            assert tensor_gpu.is_cuda and tensor_gpu.dtype == torch.float32

        t_fw_0 = time.perf_counter_ns()
        ev_start.record()
        with torch.inference_mode():
            raw_preds = net(tensor_gpu)
        ev_end.record()
        torch.cuda.synchronize(device)
        t_fw_1 = time.perf_counter_ns()
        forward_cuda_event_ms = float(ev_start.elapsed_time(ev_end))
        forward_wall_ms = (t_fw_1 - t_fw_0) / 1e6

        torch.cuda.synchronize(device)
        t_post_0 = time.perf_counter_ns()
        orig_hw = (prep_meta["orig_height"], prep_meta["orig_width"])
        post_info = run_single_postprocess(
            raw_preds, target_hw, orig_hw, names, conf_thres, iou_thres, max_det
        )
        torch.cuda.synchronize(device)
        t_post_1 = time.perf_counter_ns()
        t_tot_1 = time.perf_counter_ns()

        gpu_alloc_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
        gpu_peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        gpu_reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
        gpu_util_pct, gpu_mem_util_pct, gpu_temp_c = sample_nvml_telemetry(0)

        image_load_ms = (t_load_1 - t_load_0) / 1e6
        preprocess_ms = (t_prep_1 - t_prep_0) / 1e6
        h2d_transfer_ms = (t_h2d_1 - t_h2d_0) / 1e6
        postprocess_ms = (t_post_1 - t_post_0) / 1e6
        compute_only_ms = preprocess_ms + h2d_transfer_ms + forward_wall_ms + postprocess_ms
        total_pipeline_ms = (t_tot_1 - t_tot_0) / 1e6

        cached_cpu_tensors.append((tensor_cpu, orig_hw))

        del tensor_gpu, raw_preds

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
                "resolution_label": res_label,
                "input_width": prep_meta["input_width"],
                "input_height": prep_meta["input_height"],
                "orig_width": prep_meta["orig_width"],
                "orig_height": prep_meta["orig_height"],
                "model_name": "yolov8n",
                "precision": "fp32",
                "gpu_name": current_gpu,
                "static_params": static_params,
                "static_macs": static_macs,
                "image_load_ms": round(image_load_ms, 4),
                "preprocess_ms": round(preprocess_ms, 4),
                "h2d_transfer_ms": round(h2d_transfer_ms, 4),
                "forward_pass_ms": round(forward_cuda_event_ms, 4),
                "forward_wall_ms": round(forward_wall_ms, 4),
                "postprocess_ms": round(postprocess_ms, 4),
                "compute_only_ms": round(compute_only_ms, 4),
                "total_pipeline_ms": round(total_pipeline_ms, 4),
                "repeat_count": repeat_count,
                "forward_repeat_mean_ms": "",
                "forward_repeat_median_ms": "",
                "forward_repeat_std_ms": "",
                "forward_repeat_p95_ms": "",
                "forward_repeat_max_ms": "",
                "postprocess_repeat_median_ms": "",
                "raw_anchor_count": post_info["raw_anchor_count"],
                "candidates_before_nms": post_info["candidates_before_nms"],
                "final_detection_count": post_info["final_detection_count"],
                "mean_detection_conf": post_info["mean_detection_conf"] if post_info["mean_detection_conf"] is not None else "",
                "max_detection_conf": post_info["max_detection_conf"] if post_info["max_detection_conf"] is not None else "",
                "detected_class_counts": json.dumps(post_info["detected_class_counts"], sort_keys=True),
                "gpu_mem_allocated_mb": round(gpu_alloc_mb, 3),
                "gpu_mem_peak_allocated_mb": round(gpu_peak_alloc_mb, 3),
                "gpu_mem_reserved_mb": round(gpu_reserved_mb, 3),
                "gpu_util_pct": gpu_util_pct if gpu_util_pct is not None else "",
                "gpu_mem_util_pct": gpu_mem_util_pct if gpu_mem_util_pct is not None else "",
                "gpu_temp_c": gpu_temp_c if gpu_temp_c is not None else "",
            }
        )

    # Pass 2: R steady-state repeats per frame (cached tensor transferred once per frame to GPU)
    raw_repeat_rows: List[dict] = []
    for idx, (t_cpu, orig_hw) in enumerate(cached_cpu_tensors):
        t_gpu = t_cpu.to(device=device, dtype=torch.float32)
        torch.cuda.synchronize(device)
        fw_samples = []
        post_samples = []
        for rep_i in range(repeat_count):
            t_fw0 = time.perf_counter_ns()
            ev_start.record()
            with torch.inference_mode():
                rp = net(t_gpu)
            ev_end.record()
            torch.cuda.synchronize(device)
            t_fw1 = time.perf_counter_ns()
            fw_ev_ms = float(ev_start.elapsed_time(ev_end))
            fw_wl_ms = (t_fw1 - t_fw0) / 1e6

            t_p0 = time.perf_counter_ns()
            _ = run_single_postprocess(rp, target_hw, orig_hw, names, conf_thres, iou_thres, max_det)
            torch.cuda.synchronize(device)
            t_p1 = time.perf_counter_ns()
            post_ms = (t_p1 - t_p0) / 1e6

            fw_samples.append(fw_ev_ms)
            post_samples.append(post_ms)
            raw_repeat_rows.append(
                {
                    "frame_id": per_frame_rows[idx]["frame_id"],
                    "sample_data_token": per_frame_rows[idx]["sample_data_token"],
                    "scene_id": per_frame_rows[idx]["scene_id"],
                    "resolution_label": res_label,
                    "repeat_idx": rep_i,
                    "forward_pass_ms": round(fw_ev_ms, 4),
                    "forward_wall_ms": round(fw_wl_ms, 4),
                    "postprocess_ms": round(post_ms, 4),
                }
            )
            del rp

        del t_gpu
        per_frame_rows[idx]["forward_repeat_mean_ms"] = round(float(np.mean(fw_samples)), 4)
        per_frame_rows[idx]["forward_repeat_median_ms"] = round(float(np.median(fw_samples)), 4)
        per_frame_rows[idx]["forward_repeat_std_ms"] = round(
            float(np.std(fw_samples, ddof=1)) if len(fw_samples) > 1 else 0.0, 4
        )
        per_frame_rows[idx]["forward_repeat_p95_ms"] = round(float(np.percentile(fw_samples, 95)), 4)
        per_frame_rows[idx]["forward_repeat_max_ms"] = round(float(np.max(fw_samples)), 4)
        per_frame_rows[idx]["postprocess_repeat_median_ms"] = round(float(np.median(post_samples)), 4)

    end_gpu = torch.cuda.get_device_name(0)
    if end_gpu != initial_gpu_name:
        raise RuntimeError(f"GPU changed mid-run! Initial={initial_gpu_name}, End={end_gpu}")

    meta = {
        "resolution_label": res_label,
        "input_hw": list(target_hw),
        "raw_prediction_shape": raw_pred_shape,
        "gpu_name": end_gpu,
        "n_frames": len(per_frame_rows),
        "static_params_fused": static_params,
        "static_conv2d_macs": static_macs,
        "model_load_ms": round(model_load_ms, 4),
        "model_vram_delta_mb": round(model_vram_delta_mb, 3),
        "warmup_iters": warmup_iters,
        "warmup_forward_ms": warmup_forward_ms,
        "repeat_count": repeat_count,
    }

    del yolo, net, cached_cpu_tensors
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return per_frame_rows, raw_repeat_rows, meta'''


CELL_08_RUN_AND_SUMMARIZE = r'''# @title 7. Execute Across R1, R2, R3, Build Summary & Verify Fairness Checks
from typing import List, Tuple


def summarize_and_export(
    per_frame_rows: List[dict],
    raw_repeat_rows: List[dict],
    run_metas: List[dict],
    verified_frames: List[FrameRecord],
    env_info: dict,
    weights_sha256: str,
    dataset_audit: dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Verify fairness checks, build summary CSV/JSON, and export measured CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_df = pd.DataFrame(per_frame_rows)
    repeat_df = pd.DataFrame(raw_repeat_rows)

    # Strict fairness assertions
    expected_tokens = [r.sample_data_token for r in verified_frames]
    gpu_names = sorted(per_frame_df["gpu_name"].unique().tolist())
    assert len(gpu_names) == 1, f"Multiple GPU models mixed in single run: {gpu_names}"
    assert set(per_frame_df["precision"].unique()) == {"fp32"}

    for label, _ in RESOLUTIONS:
        sub = per_frame_df[per_frame_df["resolution_label"] == label]
        assert len(sub) == len(expected_tokens), f"Frame count mismatch for {label}"
        assert list(sub["sample_data_token"]) == expected_tokens, f"Frame order mismatch for {label}"
        for col in [
            "image_load_ms", "preprocess_ms", "h2d_transfer_ms", "forward_pass_ms",
            "forward_wall_ms", "postprocess_ms", "total_pipeline_ms",
            "forward_repeat_mean_ms", "forward_repeat_median_ms",
            "candidates_before_nms", "final_detection_count",
        ]:
            assert not sub[col].isna().any(), f"Unexpected NaN in {label}.{col}"

    meta_by_res = {m["resolution_label"]: m for m in run_metas}
    summary_rows = []
    detailed_json = {
        "environment": env_info,
        "weights_sha256": weights_sha256,
        "dataset_audit": dataset_audit,
        "resolution_runs": meta_by_res,
        "metrics_by_resolution": {},
        "cpu_comparison_reference_n100": CPU_PILOT_REF_640x384,
    }

    for label, _ in RESOLUTIONS:
        sub = per_frame_df[per_frame_df["resolution_label"] == label]
        rep_sub = repeat_df[repeat_df["resolution_label"] == label]
        m = meta_by_res[label]

        fw_s = compute_stats(sub["forward_pass_ms"].astype(float).tolist())
        fw_rep_raw_s = compute_stats(rep_sub["forward_pass_ms"].astype(float).tolist())
        fw_rep_med_s = compute_stats(sub["forward_repeat_median_ms"].astype(float).tolist())
        load_s = compute_stats(sub["image_load_ms"].astype(float).tolist())
        prep_s = compute_stats(sub["preprocess_ms"].astype(float).tolist())
        h2d_s = compute_stats(sub["h2d_transfer_ms"].astype(float).tolist())
        post_s = compute_stats(sub["postprocess_ms"].astype(float).tolist())
        comp_s = compute_stats(sub["compute_only_ms"].astype(float).tolist())
        tot_s = compute_stats(sub["total_pipeline_ms"].astype(float).tolist())
        cand_s = compute_stats(sub["candidates_before_nms"].astype(float).tolist())
        det_s = compute_stats(sub["final_detection_count"].astype(float).tolist())
        vram_s = compute_stats(sub["gpu_mem_peak_allocated_mb"].astype(float).tolist())

        # Exact N=100 matched subset for apples-to-apples check vs Milestone B CPU pilot
        sub100 = sub[sub["is_milestone_b_pilot_frame"] == True]  # noqa: E712
        sub100_fw = compute_stats(sub100["forward_pass_ms"].astype(float).tolist())
        sub100_tot = compute_stats(sub100["total_pipeline_ms"].astype(float).tolist())
        sub100_post = compute_stats(sub100["postprocess_ms"].astype(float).tolist())

        detailed_json["metrics_by_resolution"][label] = {
            "forward_pass_ms": fw_s,
            "forward_pass_individual_repeats_ms": fw_rep_raw_s,
            "forward_pass_per_frame_repeat_median_ms": fw_rep_med_s,
            "image_load_ms": load_s,
            "preprocess_ms": prep_s,
            "h2d_transfer_ms": h2d_s,
            "postprocess_ms": post_s,
            "compute_only_ms": comp_s,
            "total_pipeline_ms": tot_s,
            "candidates_before_nms": cand_s,
            "final_detection_count": det_s,
            "total_final_detections": int(sub["final_detection_count"].sum()),
            "gpu_mem_peak_allocated_mb": vram_s,
            "matched_pilot_100_subset": {
                "n_frames": int(len(sub100)),
                "forward_pass_ms": sub100_fw,
                "postprocess_ms": sub100_post,
                "total_pipeline_ms": sub100_tot,
            },
        }

        summary_rows.append(
            {
                "resolution_label": label,
                "input_width": int(m["input_hw"][1]),
                "input_height": int(m["input_hw"][0]),
                "gpu_name": m["gpu_name"],
                "precision": "fp32",
                "n_frames": int(m["n_frames"]),
                "static_macs": int(m["static_conv2d_macs"]),
                "model_load_ms": m["model_load_ms"],
                "forward_mean_ms": fw_s["mean"],
                "forward_median_ms": fw_s["median"],
                "forward_p95_ms": fw_s["p95"],
                "forward_std_ms": fw_s["std"],
                "forward_max_ms": fw_s["max"],
                "forward_repeat_raw_mean_ms": fw_rep_raw_s["mean"],
                "forward_repeat_raw_median_ms": fw_rep_raw_s["median"],
                "forward_repeat_raw_p95_ms": fw_rep_raw_s["p95"],
                "forward_repeat_raw_std_ms": fw_rep_raw_s["std"],
                "forward_repeat_raw_max_ms": fw_rep_raw_s["max"],
                "image_load_mean_ms": load_s["mean"],
                "preprocess_mean_ms": prep_s["mean"],
                "h2d_transfer_mean_ms": h2d_s["mean"],
                "postprocess_mean_ms": post_s["mean"],
                "postprocess_p95_ms": post_s["p95"],
                "compute_only_mean_ms": comp_s["mean"],
                "total_pipeline_mean_ms": tot_s["mean"],
                "total_pipeline_median_ms": tot_s["median"],
                "total_pipeline_p95_ms": tot_s["p95"],
                "total_pipeline_std_ms": tot_s["std"],
                "total_pipeline_max_ms": tot_s["max"],
                "forward_share_of_total_pct": round(100.0 * fw_s["mean"] / max(1e-9, tot_s["mean"]), 2),
                "candidates_mean": cand_s["mean"],
                "candidates_median": cand_s["median"],
                "candidates_p95": cand_s["p95"],
                "candidates_max": cand_s["max"],
                "detections_mean": det_s["mean"],
                "detections_median": det_s["median"],
                "detections_p95": det_s["p95"],
                "detections_max": det_s["max"],
                "detections_total": int(sub["final_detection_count"].sum()),
                "gpu_peak_vram_max_mb": vram_s["max"],
                "matched100_forward_mean_ms": sub100_fw["mean"],
                "matched100_total_mean_ms": sub100_tot["mean"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    per_frame_csv = output_dir / "colab_yolov8n_per_frame_latency.csv"
    repeat_csv = output_dir / "colab_yolov8n_repeat_measurements.csv"
    summary_csv = output_dir / "colab_yolov8n_resolution_summary.csv"
    summary_json = output_dir / "colab_experiment_summary.json"

    per_frame_df.to_csv(per_frame_csv, index=False)
    repeat_df.to_csv(repeat_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(detailed_json, indent=2) + "\n", encoding="utf-8")
    return per_frame_df, repeat_df, summary_df, detailed_json


ALL_FRAME_ROWS = []
ALL_REPEAT_ROWS = []
ALL_RUN_METAS = []

for res_label, res_hw in RESOLUTIONS:
    print(f"\n>>> Benchmarking {res_label} ({res_hw[1]}x{res_hw[0]}) on {INITIAL_GPU_NAME} ...")
    f_rows, r_rows, meta = benchmark_single_resolution(
        res_label=res_label,
        target_hw=res_hw,
        frames=VERIFIED_FRAMES,
        weights_path=WEIGHTS_PATH,
        initial_gpu_name=INITIAL_GPU_NAME,
        env_info=ENV_INFO,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        max_det=MAX_DET,
        warmup_iters=WARMUP_ITERS,
        repeat_count=REPEAT_COUNT,
    )
    ALL_FRAME_ROWS.extend(f_rows)
    ALL_REPEAT_ROWS.extend(r_rows)
    ALL_RUN_METAS.append(meta)

PER_FRAME_DF, REPEAT_DF, SUMMARY_DF, EXPERIMENT_SUMMARY = summarize_and_export(
    per_frame_rows=ALL_FRAME_ROWS,
    raw_repeat_rows=ALL_REPEAT_ROWS,
    run_metas=ALL_RUN_METAS,
    verified_frames=VERIFIED_FRAMES,
    env_info=ENV_INFO,
    weights_sha256=WEIGHTS_SHA256,
    dataset_audit=DATASET_AUDIT,
    output_dir=OUTPUT_DIR,
)

print("\n=== Resolution-Wise Summary ===")
print(SUMMARY_DF[[
    "resolution_label", "n_frames", "model_load_ms",
    "forward_mean_ms", "forward_median_ms", "forward_p95_ms",
    "total_pipeline_mean_ms", "total_pipeline_median_ms", "total_pipeline_p95_ms",
    "detections_mean", "detections_total", "gpu_peak_vram_max_mb",
]].to_string(index=False))'''


CELL_09_PLOTS = r'''# @title 8. Generate Required Latency & Detection Comparison Plots
from typing import List
import matplotlib
import matplotlib.pyplot as plt


def generate_latency_plots(
    per_frame_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    show_inline: bool = True,
) -> List[Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    order = [r[0] for r in RESOLUTIONS]
    summary_ord = summary_df.set_index("resolution_label").loc[order].reset_index()
    gpu_label = summary_ord["gpu_name"].iloc[0]
    n_fr = int(summary_ord["n_frames"].iloc[0])

    saved: List[Path] = []

    # Plot 1: Forward-pass latency vs resolution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    x = np.arange(len(order))
    ax.plot(x, summary_ord["forward_mean_ms"], marker="o", linewidth=2.2, label="Pass-1 Mean (CUDA Event)")
    ax.plot(x, summary_ord["forward_median_ms"], marker="s", linewidth=2.0, label="Pass-1 Median")
    ax.plot(x, summary_ord["forward_p95_ms"], marker="^", linestyle="--", linewidth=1.8, label="Pass-1 p95")
    ax.plot(x, summary_ord["forward_repeat_raw_median_ms"], marker="D", linestyle=":", linewidth=1.8, label="Pass-2 Repeat Median")
    for idx, val in enumerate(summary_ord["forward_mean_ms"]):
        ax.annotate(f"{val:.2f} ms", (x[idx], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Forward-Pass Latency (ms)")
    ax.set_title(f"YOLOv8n FP32 GPU Forward-Pass Latency vs Resolution\n({gpu_label}, N={n_fr} nuScenes CAM_FRONT frames)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p1 = plots_dir / "forward_pass_vs_resolution.png"
    fig.tight_layout()
    fig.savefig(p1)
    if show_inline:
        plt.show()
    plt.close(fig)
    saved.append(p1)

    # Plot 2: Total pipeline latency vs resolution (stacked mean stages + p95 marker)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)
    b1 = summary_ord["image_load_mean_ms"].to_numpy()
    b2 = summary_ord["preprocess_mean_ms"].to_numpy()
    b3 = summary_ord["h2d_transfer_mean_ms"].to_numpy()
    b4 = summary_ord["forward_mean_ms"].to_numpy()
    b5 = summary_ord["postprocess_mean_ms"].to_numpy()
    ax.bar(x, b1, width=0.52, label="Image Load (cv2.imread)")
    ax.bar(x, b2, width=0.52, bottom=b1, label="CPU Preprocess (LetterBox)")
    ax.bar(x, b3, width=0.52, bottom=b1 + b2, label="Host->Device Transfer")
    ax.bar(x, b4, width=0.52, bottom=b1 + b2 + b3, label="GPU Forward Pass (CUDA)")
    ax.bar(x, b5, width=0.52, bottom=b1 + b2 + b3 + b4, label="Postprocess & NMS")
    ax.plot(x, summary_ord["total_pipeline_p95_ms"], color="black", marker="_", markersize=18, linestyle="none", markeredgewidth=2.5, label="Total Pipeline p95")
    for idx, tot_v in enumerate(summary_ord["total_pipeline_mean_ms"]):
        ax.annotate(f"Mean {tot_v:.2f} ms", (x[idx], tot_v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Pipeline Latency (ms)")
    ax.set_title(f"YOLOv8n Stage-Separated Total Pipeline Latency vs Resolution\n({gpu_label}, N={n_fr} frames)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    p2 = plots_dir / "total_pipeline_vs_resolution.png"
    fig.tight_layout()
    fig.savefig(p2)
    if show_inline:
        plt.show()
    plt.close(fig)
    saved.append(p2)

    # Plot 3: Latency distribution for each resolution (outliers retained!)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    fw_groups = [per_frame_df[per_frame_df["resolution_label"] == lbl]["forward_pass_ms"].astype(float).tolist() for lbl in order]
    tot_groups = [per_frame_df[per_frame_df["resolution_label"] == lbl]["total_pipeline_ms"].astype(float).tolist() for lbl in order]
    axes[0].boxplot(fw_groups, showfliers=True)
    axes[0].set_xticks(np.arange(1, len(order) + 1))
    axes[0].set_xticklabels(order)
    axes[0].set_title("GPU Forward-Pass Distribution (All Outliers Retained)")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].boxplot(tot_groups, showfliers=True)
    axes[1].set_xticks(np.arange(1, len(order) + 1))
    axes[1].set_xticklabels(order)
    axes[1].set_title("Total Pipeline Distribution (All Outliers Retained)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"YOLOv8n FP32 Latency Distributions ({gpu_label}, N={n_fr})", y=1.02)
    p3 = plots_dir / "latency_distribution_by_resolution.png"
    fig.tight_layout()
    fig.savefig(p3, bbox_inches="tight")
    if show_inline:
        plt.show()
    plt.close(fig)
    saved.append(p3)

    # Plot 4: Detection count vs resolution
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=140)
    cand_groups = [per_frame_df[per_frame_df["resolution_label"] == lbl]["candidates_before_nms"].astype(float).tolist() for lbl in order]
    det_groups = [per_frame_df[per_frame_df["resolution_label"] == lbl]["final_detection_count"].astype(float).tolist() for lbl in order]
    axes[0].boxplot(cand_groups, showfliers=True)
    axes[0].set_xticks(np.arange(1, len(order) + 1))
    axes[0].set_xticklabels(order)
    axes[0].set_title("Pre-NMS Candidates (>0.25) vs Resolution")
    axes[0].set_ylabel("Candidate Count / Frame")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].boxplot(det_groups, showfliers=True)
    axes[1].set_xticks(np.arange(1, len(order) + 1))
    axes[1].set_xticklabels(order)
    axes[1].plot(np.arange(1, len(order) + 1), summary_ord["detections_mean"], marker="o", color="tab:red", label="Mean Final Dets")
    axes[1].set_title("Final Detections (Post-NMS) vs Resolution")
    axes[1].set_ylabel("Detection Count / Frame")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"YOLOv8n Detection Counts Across Identical Frames (N={n_fr})", y=1.02)
    p4 = plots_dir / "detection_count_vs_resolution.png"
    fig.tight_layout()
    fig.savefig(p4, bbox_inches="tight")
    if show_inline:
        plt.show()
    plt.close(fig)
    saved.append(p4)
    return saved


PLOT_PATHS = generate_latency_plots(PER_FRAME_DF, SUMMARY_DF, OUTPUT_DIR, show_inline=True)
print("Saved plots:", [str(p) for p in PLOT_PATHS])'''


CELL_10_REPORT = r'''# @title 9. Generate Measured COLAB_LATENCY_REPORT.md & Downloadable Results Bundle
import datetime
import shutil
import zipfile


def render_colab_latency_report(
    summary_df: pd.DataFrame,
    experiment_summary: dict,
    output_dir: Path,
) -> Path:
    """Write COLAB_LATENCY_REPORT.md strictly from measured Colab GPU run values."""
    env = experiment_summary["environment"]
    ds = experiment_summary["dataset_audit"]
    cpu = experiment_summary["cpu_comparison_reference_n100"]
    m_res = experiment_summary["metrics_by_resolution"]
    order = [r[0] for r in RESOLUTIONS]
    s_ord = summary_df.set_index("resolution_label").loc[order]

    r3 = m_res["R3_640x384"]
    r3_100 = r3["matched_pilot_100_subset"]
    speedup_fw_100 = cpu["forward_pass_ms"]["mean"] / max(1e-9, r3_100["forward_pass_ms"]["mean"])
    speedup_tot_100 = cpu["total_pipeline_ms"]["mean"] / max(1e-9, r3_100["total_pipeline_ms"]["mean"])
    speedup_fw_404 = cpu["forward_pass_ms"]["mean"] / max(1e-9, r3["forward_pass_ms"]["mean"])
    speedup_tot_404 = cpu["total_pipeline_ms"]["mean"] / max(1e-9, r3["total_pipeline_ms"]["mean"])

    lines = [
        "# Google Colab GPU Latency Report (`YOLOv8n` FP32)",
        "",
        f"**Generated UTC:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Model / Weights:** `yolov8n.pt` (SHA-256 `{experiment_summary['weights_sha256']}`)  ",
        f"**GPU Model:** `{env['gpu_name']}` (VRAM `{env['gpu_total_vram_mb']} MiB`, Compute `{env['gpu_compute_capability']}`, Driver `{env.get('gpu_driver_version')}`)  ",
        f"**Dataset:** `nuScenes v1.0-mini` `CAM_FRONT` (`{ds['accessible_valid_frames']}` / `{ds['expected_keyframes_in_nuscenes_mini']}` frames across `{ds['n_scenes']}` scenes)",
        "",
        "---",
        "",
        "## 1. Software & Hardware Environment",
        "",
        "| Field | Verified Runtime Value |",
        "|---|---|",
        f"| GPU Device | `{env['gpu_name']}` (`{env['gpu_multiprocessors']}` SMs, `{env['gpu_total_vram_mb']} MiB` VRAM) |",
        f"| CUDA / cuDNN | CUDA `{env['cuda_version']}` / cuDNN `{env['cudnn_version']}` |",
        f"| PyTorch / torchvision | `{env['torch_version']}` / `{env['torchvision_version']}` |",
        f"| Ultralytics / OpenCV / Python | `{env['ultralytics_version']}` / `{env['opencv_version']}` / `{env['python_version']}` |",
        f"| Precision Policy | **Strict FP32** (`matmul.allow_tf32={env['matmul_allow_tf32']}`, `cudnn.allow_tf32={env['cudnn_allow_tf32']}`, `cudnn.benchmark={env['cudnn_benchmark']}`) |",
        f"| Detection Config | `conf={CONF_THRES}`, `iou={IOU_THRES}`, `max_det={MAX_DET}`, `warmup={WARMUP_ITERS}`, `repeats={REPEAT_COUNT}` |",
        "",
        "---",
        "",
        "## 2. Resolution-Wise Measured Latency (`Pass-1 Chronological` & `Pass-2 CUDA Event Repeats`)",
        "",
        "| Resolution | Dense MACs | Load (ms) | Forward Mean (ms) | Forward Med (ms) | Forward p95 (ms) | Forward Std / Max (ms) | Pass-2 Rep Med (ms) | Total Pipe Mean (ms) | Total Pipe Med (ms) | Total Pipe p95 / Max (ms) | Peak VRAM (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for lbl in order:
        row = s_ord.loc[lbl]
        lines.append(
            f"| `{lbl}` | `{int(row['static_macs']):,}` | {row['model_load_ms']:.2f} | "
            f"**{row['forward_mean_ms']:.3f}** | {row['forward_median_ms']:.3f} | {row['forward_p95_ms']:.3f} | "
            f"{row['forward_std_ms']:.3f} / {row['forward_max_ms']:.3f} | {row['forward_repeat_raw_median_ms']:.3f} | "
            f"**{row['total_pipeline_mean_ms']:.3f}** | {row['total_pipeline_median_ms']:.3f} | "
            f"{row['total_pipeline_p95_ms']:.3f} / {row['total_pipeline_max_ms']:.3f} | {row['gpu_peak_vram_max_mb']:.1f} |"
        )

    lines.extend([
        "",
        "### Stage Breakdown (`Mean ms` Across Verified Frames)",
        "",
        "| Resolution | Image Load (`cv2.imread`) | CPU Preprocess (`LetterBox`) | Host->GPU H2D | GPU Forward (`CUDA Event`) | Postprocess & NMS | Compute-Only | Total Pipeline | Forward Share of Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for lbl in order:
        row = s_ord.loc[lbl]
        lines.append(
            f"| `{lbl}` | {row['image_load_mean_ms']:.3f} | {row['preprocess_mean_ms']:.3f} | "
            f"{row['h2d_transfer_mean_ms']:.3f} | {row['forward_mean_ms']:.3f} | {row['postprocess_mean_ms']:.3f} | "
            f"{row['compute_only_mean_ms']:.3f} | {row['total_pipeline_mean_ms']:.3f} | {row['forward_share_of_total_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detection Counts Across Identical Frames",
        "",
        "| Resolution | Mean Pre-NMS Candidates | Median Candidates | p95 Candidates | Max Candidates | Mean Final Dets | Median Dets | p95 / Max Dets | Total Detections |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for lbl in order:
        row = s_ord.loc[lbl]
        lines.append(
            f"| `{lbl}` | {row['candidates_mean']:.2f} | {row['candidates_median']:.1f} | {row['candidates_p95']:.1f} | {int(row['candidates_max'])} | "
            f"{row['detections_mean']:.2f} | {row['detections_median']:.1f} | {row['detections_p95']:.1f} / {int(row['detections_max'])} | {int(row['detections_total'])} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Comparison Against Milestone B CPU Pilot (`R3: 640×384`)",
        "",
        "| Execution Platform | Matched Scope (`N`) | Forward Mean / Med / p95 (ms) | Postprocess Mean (ms) | Total Pipeline Mean / Med / p95 (ms) | GPU Speedup vs CPU (Forward / Total) |",
        "|---|---:|---|---|---|---:|",
        f"| Milestone B CPU (`{cpu['hardware']}`) | `{cpu['n_frames']}` | {cpu['forward_pass_ms']['mean']:.2f} / {cpu['forward_pass_ms']['median']:.2f} / {cpu['forward_pass_ms']['p95']:.2f} | {cpu['postprocess_ms']['mean']:.2f} | {cpu['total_pipeline_ms']['mean']:.2f} / {cpu['total_pipeline_ms']['median']:.2f} / {cpu['total_pipeline_ms']['p95']:.2f} | `1.00× / 1.00×` |",
        f"| Colab GPU (`{env['gpu_name']}`) — Matched 100 | `{r3_100['n_frames']}` | {r3_100['forward_pass_ms']['mean']:.3f} / {r3_100['forward_pass_ms']['median']:.3f} / {r3_100['forward_pass_ms']['p95']:.3f} | {r3_100['postprocess_ms']['mean']:.3f} | {r3_100['total_pipeline_ms']['mean']:.3f} / {r3_100['total_pipeline_ms']['median']:.3f} / {r3_100['total_pipeline_ms']['p95']:.3f} | `{speedup_fw_100:.2f}× / {speedup_tot_100:.2f}×` |",
        f"| Colab GPU (`{env['gpu_name']}`) — Full Set | `{ds['accessible_valid_frames']}` | {r3['forward_pass_ms']['mean']:.3f} / {r3['forward_pass_ms']['median']:.3f} / {r3['forward_pass_ms']['p95']:.3f} | {r3['postprocess_ms']['mean']:.3f} | {r3['total_pipeline_ms']['mean']:.3f} / {r3['total_pipeline_ms']['median']:.3f} / {r3['total_pipeline_ms']['p95']:.3f} | `{speedup_fw_404:.2f}× / {speedup_tot_404:.2f}×` |",
        "",
        "---",
        "",
        "## 5. Experimental Limitations & Scope Boundaries",
        "",
        "1. **Colab Datacenter GPU != Target Edge ASIC/NPU:** Measured latencies reflect shared Google Colab virtualized host + datacenter GPU hardware, including host disk JPEG decode (`cv2.imread`), Python/PCIe H2D launch overhead, and host/GPU NMS.",
        "2. **No Target Energy/Power Conclusions:** No target embedded accelerator energy, thermal trajectory, or power feasibility conclusions are drawn from Colab GPU execution.",
        "3. **No Quality / Ground-Truth Claims:** COCO-80 detection counts illustrate output sensitivity across resolutions on identical images and do **not** measure nuScenes detection accuracy (`mAP` / `NDS`).",
        f"4. **Dataset Access Check:** `{ds['accessible_valid_frames']}` of `{ds['expected_keyframes_in_nuscenes_mini']}` keyframes accessed; `{ds['missing_or_corrupted_count']}` missing/corrupted frames.",
    ])

    report_path = output_dir / "COLAB_LATENCY_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


REPORT_PATH = render_colab_latency_report(SUMMARY_DF, EXPERIMENT_SUMMARY, OUTPUT_DIR)

# Package complete deliverables zip for one-click download or Drive sync
BUNDLE_OUT = OUTPUT_DIR / "colab_latency_results_bundle.zip"
with zipfile.ZipFile(BUNDLE_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for item in [
        OUTPUT_DIR / "verified_frame_manifest.csv",
        OUTPUT_DIR / "colab_yolov8n_per_frame_latency.csv",
        OUTPUT_DIR / "colab_yolov8n_repeat_measurements.csv",
        OUTPUT_DIR / "colab_yolov8n_resolution_summary.csv",
        OUTPUT_DIR / "colab_experiment_summary.json",
        REPORT_PATH,
        *PLOT_PATHS,
    ]:
        if item.is_file():
            zf.write(item, arcname=str(item.relative_to(OUTPUT_DIR)))

print(f"Report generated: {REPORT_PATH}")
print(f"Output bundle   : {BUNDLE_OUT} ({BUNDLE_OUT.stat().st_size / 1024.0:.1f} KB)")

if USE_GOOGLE_DRIVE and COPY_OUTPUTS_TO_DRIVE and DRIVE_BASE_DIR.is_dir():
    drive_out = DRIVE_BASE_DIR / "colab_outputs"
    drive_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLE_OUT, drive_out / BUNDLE_OUT.name)
    print(f"Copied output bundle to Drive: {drive_out / BUNDLE_OUT.name}")

try:
    from google.colab import files
    files.download(str(BUNDLE_OUT))
except ImportError:
    pass'''


def generate_notebook_and_bundle() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build 404-frame manifest CSV
    recs = load_cam_front_keyframes(frames_per_scene=None)
    assert len(recs) == 404, f"Expected 404 frames locally, found {len(recs)}"
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_id",
            "scene_id",
            "scene_split",
            "environment_tag",
            "frame_in_scene",
            "is_milestone_b_pilot_frame",
            "sample_token",
            "sample_data_token",
            "timestamp_us",
            "scene_elapsed_ms",
            "dt_from_prev_ms",
            "image_path",
            "meta_width",
            "meta_height",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in recs:
            writer.writerow(
                {
                    "frame_id": r.frame_id,
                    "scene_id": r.scene_id,
                    "scene_split": r.scene_split,
                    "environment_tag": r.environment_tag,
                    "frame_in_scene": r.frame_in_scene,
                    "is_milestone_b_pilot_frame": r.frame_in_scene < 10,
                    "sample_token": r.sample_token,
                    "sample_data_token": r.sample_data_token,
                    "timestamp_us": r.timestamp_us,
                    "scene_elapsed_ms": r.scene_elapsed_ms,
                    "dt_from_prev_ms": r.dt_from_prev_ms if r.dt_from_prev_ms is not None else "",
                    "image_path": r.image_path,
                    "meta_width": r.meta_width,
                    "meta_height": r.meta_height,
                }
            )

    # 2. Build compact ~75 MB Colab upload zip (only needed nuScenes JSONs, 404 JPEGs, weights, manifest)
    weights_path = DEFAULT_WEIGHTS_PATH
    assert weights_path.is_file(), f"Missing {weights_path}"
    with zipfile.ZipFile(UPLOAD_BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for jname in ["scene.json", "sample.json", "sample_data.json"]:
            jpath = Path("data/nuscenes_mini/v1.0-mini") / jname
            zf.write(jpath, arcname=jpath.as_posix())
        for r in recs:
            ipath = Path(r.image_path)
            zf.write(ipath, arcname=ipath.as_posix())
        zf.write(weights_path, arcname=weights_path.as_posix())
        zf.write(MANIFEST_CSV, arcname=MANIFEST_CSV.as_posix())

    # 3. Assemble valid Jupyter notebook (nbformat 4.5)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "accelerator": "GPU",
        },
        "cells": [
            _md_cell(CELL_01_MD),
            _code_cell(CELL_02_CONFIG, "cell_01_config"),
            _code_cell(CELL_03_ENV, "cell_02_env_gpu_check"),
            _code_cell(CELL_04_DATA, "cell_03_data_weights_stage"),
            _code_cell(CELL_05_MANIFEST, "cell_04_manifest_verify"),
            _code_cell(CELL_06_HELPERS, "cell_05_helpers"),
            _code_cell(CELL_07_BENCHMARK, "cell_06_benchmark_engine"),
            _code_cell(CELL_08_RUN_AND_SUMMARIZE, "cell_07_run_summarize"),
            _code_cell(CELL_09_PLOTS, "cell_08_plots"),
            _code_cell(CELL_10_REPORT, "cell_09_report_bundle"),
        ],
    }
    OUT_NOTEBOOK.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    print(f"Created notebook : {OUT_NOTEBOOK} ({OUT_NOTEBOOK.stat().st_size:,} bytes)")
    print(f"Created manifest : {MANIFEST_CSV} ({len(recs)} rows, {MANIFEST_CSV.stat().st_size:,} bytes)")
    print(
        f"Created zip      : {UPLOAD_BUNDLE_ZIP} ({UPLOAD_BUNDLE_ZIP.stat().st_size / (1024.0 ** 2):.2f} MB)"
        f", weights sha256={compute_sha256(weights_path)}"
    )


if __name__ == "__main__":
    generate_notebook_and_bundle()
