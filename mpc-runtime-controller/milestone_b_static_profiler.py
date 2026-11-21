#!/usr/bin/env python3
"""Static and single-frame architectural profiler for pretrained YOLOv8n.

Collects:
- Weight file size and SHA-256 digest
- Unfused vs fused parameter counts and module counts
- Layer-wise dense MACs, FLOPs, parameter counts, and input/output activation
  tensor dimensions at fixed input shape (1, 3, 384, 640)
- Cross-check against thop.profile() MAC/FLOP totals
- Optional empirical measurement of exact-zero (== 0.0) elements across
  post-SiLU activation tensors on real pilot images (to verify rather than
  assume activation sparsity behavior).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import thop
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


DEFAULT_WEIGHTS_PATH = Path("results/milestone_b/weights/yolov8n.pt")


def load_yolo_weights(weights_path: Path | str = DEFAULT_WEIGHTS_PATH, fuse: bool = True) -> YOLO:
    """Load pretrained YOLOv8n into an explicit path inside results/milestone_b."""
    wpath = Path(weights_path)
    wpath.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(wpath))
    if fuse:
        model.fuse()
    model.model.eval()
    return model


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_bgr_image(img_bgr: np.ndarray, target_hw: Tuple[int, int] = (384, 640)) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply Ultralytics LetterBox + BGR->RGB + CHW FP32 [0,1] normalization."""
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
    tensor = torch.from_numpy(chw).unsqueeze(0)
    meta = {
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "input_width": int(in_w),
        "input_height": int(in_h),
        "letterbox_scale": round(float(scale), 6),
        "pad_top_px": int(round(dh - 0.1)),
        "pad_bottom_px": int(round(dh + 0.1)),
        "pad_left_px": int(round(dw - 0.1)),
        "pad_right_px": int(round(dw + 0.1)),
    }
    return tensor, meta


def _tensor_shape_str(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return "x".join(str(d) for d in obj.shape)
    if isinstance(obj, (list, tuple)):
        parts = [_tensor_shape_str(x) for x in obj]
        return "; ".join(p for p in parts if p)
    return ""


def _tensor_numel(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return int(obj.numel())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_numel(x) for x in obj)
    return 0


def profile_static_layers(
    net: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 384, 640),
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Hook leaf modules at input_shape to record dense MACs, params, and shapes."""
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    records: List[Dict[str, Any]] = []
    hooks = []
    exec_order = 0

    leaf_types = (nn.Conv2d, nn.MaxPool2d, nn.Upsample, nn.SiLU)

    def make_hook(name: str, mod: nn.Module):
        def hook_fn(_module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            nonlocal exec_order
            in_shape = _tensor_shape_str(inputs[0] if inputs else None)
            out_shape = _tensor_shape_str(output)
            in_elems = _tensor_numel(inputs[0] if inputs else None)
            out_elems = _tensor_numel(output)
            params = sum(int(p.numel()) for p in mod.parameters(recurse=False))
            macs = 0
            kernel_str = ""
            stride_str = ""
            groups_val = 1

            if isinstance(mod, nn.Conv2d) and isinstance(output, torch.Tensor):
                # MACs = B * H_out * W_out * C_out * (C_in / groups) * K_h * K_w
                b, c_out, h_out, w_out = output.shape
                k_h, k_w = mod.kernel_size
                groups_val = int(mod.groups)
                c_in_per_group = mod.in_channels // groups_val
                macs = int(b * h_out * w_out * c_out * c_in_per_group * k_h * k_w)
                kernel_str = f"{k_h}x{k_w}"
                stride_str = f"{mod.stride[0]}x{mod.stride[1]}"
            elif isinstance(mod, nn.MaxPool2d):
                k = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
                s = mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride)
                kernel_str = f"{k[0]}x{k[1]}"
                stride_str = f"{s[0]}x{s[1]}"

            records.append(
                {
                    "exec_idx": exec_order,
                    "module_name": name,
                    "module_type": mod.__class__.__name__,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                    "input_elements": in_elems,
                    "output_elements": out_elems,
                    "params": params,
                    "kernel_size": kernel_str,
                    "stride": stride_str,
                    "groups": groups_val,
                    "macs": macs,
                    "flops_2x_mac": 2 * macs,
                }
            )
            exec_order += 1

        return hook_fn

    for name, mod in net.named_modules():
        if isinstance(mod, leaf_types):
            hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    try:
        with torch.inference_mode():
            out = net(dummy)
    finally:
        for h in hooks:
            h.remove()

    # Cross-check with thop.profile on a fresh instance (thop attaches temp buffers)
    net_for_thop = load_yolo_weights(weights_path, fuse=True)
    thop_macs, thop_params = thop.profile(net_for_thop.model, inputs=(dummy,), verbose=False)

    hook_total_macs = sum(r["macs"] for r in records)
    hook_total_act_elems = sum(r["output_elements"] for r in records if r["module_type"] == "Conv2d")
    total_params = sum(int(p.numel()) for p in net.parameters())

    raw_pred_shape = _tensor_shape_str(out[0] if isinstance(out, (list, tuple)) else out)

    summary = {
        "input_shape": list(input_shape),
        "raw_prediction_shape": raw_pred_shape,
        "total_parameters_fused": int(total_params),
        "hook_conv2d_count": sum(1 for r in records if r["module_type"] == "Conv2d"),
        "hook_silu_count": sum(1 for r in records if r["module_type"] == "SiLU"),
        "hook_dense_macs": int(hook_total_macs),
        "hook_dense_flops_2x": int(2 * hook_total_macs),
        "thop_macs": float(thop_macs),
        "thop_flops_2x": float(2.0 * thop_macs),
        "mac_hook_vs_thop_diff_pct": round(
            100.0 * abs(hook_total_macs - float(thop_macs)) / max(1.0, float(thop_macs)), 4
        ),
        "conv2d_output_activation_elements": int(hook_total_act_elems),
        "conv2d_output_activation_fp32_mb": round(hook_total_act_elems * 4.0 / (1024.0 ** 2), 4),
        "weight_fp32_mb": round(total_params * 4.0 / (1024.0 ** 2), 4),
    }
    return records, summary


def measure_empirical_silu_zero_fraction(
    net: nn.Module,
    image_paths: List[str],
) -> Dict[str, Any]:
    """Measure exact == 0.0 post-SiLU elements directly (no synthetic sparsity)."""
    per_image_exact_zero_frac: List[float] = []
    per_image_near_zero_1e3_frac: List[float] = []

    for img_path in image_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)
        tensor, _ = preprocess_bgr_image(bgr)

        total_elems = 0
        exact_zeros = 0
        near_zeros = 0
        hooks = []

        def act_hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            nonlocal total_elems, exact_zeros, near_zeros
            if isinstance(out, torch.Tensor):
                total_elems += int(out.numel())
                exact_zeros += int((out == 0.0).sum().item())
                near_zeros += int((out.abs() < 1e-3).sum().item())

        for mod in net.modules():
            if isinstance(mod, nn.SiLU):
                hooks.append(mod.register_forward_hook(act_hook))
        try:
            with torch.inference_mode():
                _ = net(tensor)
        finally:
            for h in hooks:
                h.remove()

        per_image_exact_zero_frac.append(exact_zeros / max(1, total_elems))
        per_image_near_zero_1e3_frac.append(near_zeros / max(1, total_elems))

    return {
        "n_images_checked": len(image_paths),
        "mean_exact_zero_fraction_silu": float(np.mean(per_image_exact_zero_frac)),
        "max_exact_zero_fraction_silu": float(np.max(per_image_exact_zero_frac)),
        "mean_abs_lt_1e3_fraction_silu": float(np.mean(per_image_near_zero_1e3_frac)),
        "min_abs_lt_1e3_fraction_silu": float(np.min(per_image_near_zero_1e3_frac)),
        "max_abs_lt_1e3_fraction_silu": float(np.max(per_image_near_zero_1e3_frac)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    parser.add_argument("--out-dir", default="results/milestone_b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = Path(args.weights)

    # First inspect unfused param count cleanly
    yolo_unfused = load_yolo_weights(wpath, fuse=False)
    unfused_params = sum(int(p.numel()) for p in yolo_unfused.model.parameters())
    unfused_modules = sum(1 for _ in yolo_unfused.model.modules())

    # Now inspect fused inference model
    yolo_fused = load_yolo_weights(wpath, fuse=True)
    records, summary = profile_static_layers(yolo_fused.model, input_shape=(1, 3, 384, 640), weights_path=wpath)

    summary["weights_path"] = wpath.as_posix()
    summary["weights_file_bytes"] = int(wpath.stat().st_size)
    summary["weights_sha256"] = compute_sha256(wpath)
    summary["total_parameters_unfused"] = int(unfused_params)
    summary["total_modules_unfused"] = int(unfused_modules)
    summary["total_modules_fused"] = sum(1 for _ in yolo_fused.model.modules())
    summary["class_count"] = len(yolo_fused.names)

    # Empirically verify SiLU zero fraction on 5 diverse smoke images
    from milestone_b_pilot_dataset import load_cam_front_keyframes, select_smoke_test_frames

    pilot = load_cam_front_keyframes(frames_per_scene=1)
    smoke = select_smoke_test_frames(pilot, n_frames=5)
    summary["silu_activation_check"] = measure_empirical_silu_zero_fraction(
        yolo_fused.model, [r.image_path for r in smoke]
    )

    csv_path = out_dir / "layer_wise_macs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    json_path = out_dir / "static_model_profile.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Weights: {wpath} ({summary['weights_file_bytes']:,} bytes, sha256={summary['weights_sha256'][:16]}...)")
    print(f"Params unfused={summary['total_parameters_unfused']:,}, fused={summary['total_parameters_fused']:,}")
    print(f"Dense MACs (640x384) hook={summary['hook_dense_macs']:,}, thop={summary['thop_macs']:,.0f} (diff={summary['mac_hook_vs_thop_diff_pct']}%)")
    print(f"SiLU exact-zero fraction (5 frames): {summary['silu_activation_check']['mean_exact_zero_fraction_silu']:.6f}")
    print(f"Saved {csv_path} ({len(records)} rows) and {json_path}")


if __name__ == "__main__":
    main()
