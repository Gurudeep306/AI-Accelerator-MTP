"""
models/postprocess.py
------------------------
Turns one tap's raw (cls_logits, obj_logit, reg_raw) per scale into final,
NMS'd, per-image detections in MODEL (letterboxed) pixel coordinates.
Coordinate remap to the original image is a separate step
(data.yolo_dataset.unletterbox_boxes) done by the caller, since that
step needs per-image letterbox metadata this function doesn't have.

Speed: decoding runs on the model's device; then the top `max_candidates`
cells are copied to the CPU in ONE transfer and NMS runs in numpy. The
earlier version ran a per-class, per-box torch loop on the GPU, where every
iteration forced a GPU sync -- ~13 ms per head on a GTX 1650 under Windows
even with only ~30 candidates. Results are identical (same greedy NMS).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from .detect_head import decode_scale


def _greedy_nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Standard greedy NMS. boxes (N,4) xyxy, scores (N,). Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        inter = (np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
                 * np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None))
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess_head(
    scale_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    strides: Sequence[int],
    conf_thres: float = 0.3,
    iou_thres: float = 0.5,
    max_detections: int = 100,
    max_candidates: int = 1000,
) -> List[dict]:
    """Returns a list (length = batch size) of dicts:
        {"boxes": (M,4) tensor, "scores": (M,) tensor, "cls_ids": (M,) tensor}
    on the CPU, in MODEL/letterboxed pixel coordinates, highest score first.
    """
    all_boxes, all_scores, all_cls = [], [], []
    for (cls_logits, obj_logit, reg_raw), stride in zip(scale_outputs, strides):
        boxes, scores, cls_ids = decode_scale(cls_logits, obj_logit, reg_raw, stride)
        b = scores.shape[0]
        all_boxes.append(boxes.reshape(b, -1, 4))
        all_scores.append(scores.reshape(b, -1))
        all_cls.append(cls_ids.reshape(b, -1))
    boxes = torch.cat(all_boxes, dim=1)    # (B, 8400, 4)
    scores = torch.cat(all_scores, dim=1)  # (B, 8400)
    cls_ids = torch.cat(all_cls, dim=1)    # (B, 8400)

    # Top-k on the device, then a single device->host copy for the whole batch.
    k = min(max_candidates, scores.shape[1])
    top_scores, top_idx = scores.topk(k, dim=1)
    top_boxes = torch.gather(boxes, 1, top_idx.unsqueeze(-1).expand(-1, -1, 4))
    top_cls = torch.gather(cls_ids, 1, top_idx)
    packed = torch.cat([top_boxes, top_scores.unsqueeze(-1), top_cls.unsqueeze(-1).to(top_boxes.dtype)], dim=-1)
    packed = packed.float().cpu().numpy()  # (B, k, 6)

    results = []
    for det in packed:
        det = det[det[:, 4] >= conf_thres]
        if det.shape[0]:
            # Per-class NMS in one pass: shift each class's boxes so classes never overlap.
            offset = det[:, 5:6] * (np.abs(det[:, :4]).max() + 1.0)
            keep = _greedy_nms_numpy(det[:, :4] + offset, det[:, 4], iou_thres)[:max_detections]
            det = det[keep]
        results.append({
            "boxes": torch.from_numpy(np.ascontiguousarray(det[:, :4])),
            "scores": torch.from_numpy(np.ascontiguousarray(det[:, 4])),
            "cls_ids": torch.from_numpy(det[:, 5].astype(np.int64)),
        })
    return results
