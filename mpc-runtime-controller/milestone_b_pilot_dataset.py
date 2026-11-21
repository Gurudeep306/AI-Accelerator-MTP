#!/usr/bin/env python3
"""Deterministic nuScenes v1.0-mini CAM_FRONT frame selection helper for Milestone B.

Reads relational metadata directly from data/nuscenes_mini/v1.0-mini without
modifying the original dataset and preserves real ~2 Hz keyframe timestamps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_NUSCENES_ROOT = Path("data/nuscenes_mini")

# Matches the scene-level split established in extract_phase4_traces.py
SCENE_SPLITS: Dict[str, Tuple[str, str]] = {
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
class PilotFrameRecord:
    """Single chronological CAM_FRONT keyframe record with real nuScenes timing."""

    frame_id: int
    scene_id: str
    scene_split: str
    environment_tag: str
    frame_in_scene: int
    sample_token: str
    sample_data_token: str
    timestamp_us: int
    scene_elapsed_ms: float
    dt_from_prev_ms: Optional[float]
    image_path: str
    meta_width: int
    meta_height: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_cam_front_keyframes(
    nuscenes_root: Path | str = DEFAULT_NUSCENES_ROOT,
    frames_per_scene: Optional[int] = 10,
    scene_ids: Optional[List[str]] = None,
) -> List[PilotFrameRecord]:
    """Load chronological CAM_FRONT keyframes with real timestamps across scenes.

    Args:
        nuscenes_root: Root directory containing v1.0-mini/ and samples/CAM_FRONT/.
        frames_per_scene: Consecutive keyframes from the start of each scene, or None
            for all available keyframes.
        scene_ids: Optional ordered subset of scene names; defaults to all 10 sorted scenes.
    """
    root = Path(nuscenes_root)
    meta_dir = root / "v1.0-mini"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"nuScenes metadata directory not found: {meta_dir}")

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

    selected_scenes = (
        list(scene_ids)
        if scene_ids is not None
        else sorted(per_scene_raw.keys())
    )

    records: List[PilotFrameRecord] = []
    global_fid = 0
    for scene_name in selected_scenes:
        if scene_name not in per_scene_raw:
            raise ValueError(f"Scene {scene_name!r} has no CAM_FRONT keyframes")
        items = sorted(per_scene_raw[scene_name], key=lambda item: int(item["timestamp"]))
        if frames_per_scene is not None:
            items = items[:frames_per_scene]

        split, env_tag = SCENE_SPLITS.get(scene_name, ("unknown", "unknown"))
        first_ts = int(items[0]["timestamp"])
        prev_ts: Optional[int] = None

        for idx, item in enumerate(items):
            ts_us = int(item["timestamp"])
            img_path = root / item["filename"]
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing CAM_FRONT image file: {img_path}")
            elapsed_ms = (ts_us - first_ts) / 1000.0
            dt_ms = None if prev_ts is None else (ts_us - prev_ts) / 1000.0
            records.append(
                PilotFrameRecord(
                    frame_id=global_fid,
                    scene_id=scene_name,
                    scene_split=split,
                    environment_tag=env_tag,
                    frame_in_scene=idx,
                    sample_token=str(item["sample_token"]),
                    sample_data_token=str(item["token"]),
                    timestamp_us=ts_us,
                    scene_elapsed_ms=round(elapsed_ms, 3),
                    dt_from_prev_ms=None if dt_ms is None else round(dt_ms, 3),
                    image_path=img_path.as_posix(),
                    meta_width=int(item["width"]),
                    meta_height=int(item["height"]),
                )
            )
            prev_ts = ts_us
            global_fid += 1

    return records


def select_smoke_test_frames(
    records: List[PilotFrameRecord],
    n_frames: int = 5,
) -> List[PilotFrameRecord]:
    """Select n_frames spanning diverse scenes (day, rain, night) for smoke checks."""
    preferred_scenes = ["scene-0061", "scene-0103", "scene-0916", "scene-1077", "scene-1094"]
    chosen: List[PilotFrameRecord] = []
    seen = set()
    for sc in preferred_scenes:
        match = next((r for r in records if r.scene_id == sc and r.frame_in_scene == 0), None)
        if match is not None and match.frame_id not in seen:
            chosen.append(match)
            seen.add(match.frame_id)
            if len(chosen) >= n_frames:
                return chosen
    for r in records:
        if r.frame_id not in seen:
            chosen.append(r)
            seen.add(r.frame_id)
            if len(chosen) >= n_frames:
                break
    return chosen


if __name__ == "__main__":
    pilot = load_cam_front_keyframes(frames_per_scene=10)
    print(f"Loaded {len(pilot)} pilot frames across {len(set(r.scene_id for r in pilot))} scenes")
