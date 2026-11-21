import json
from pathlib import Path
from collections import defaultdict

meta_dir = Path("data/nuscenes_mini/v1.0-mini")
scenes = json.load(open(meta_dir / "scene.json"))
samples = json.load(open(meta_dir / "sample.json"))
sample_data = json.load(open(meta_dir / "sample_data.json"))

sample_to_scene = {s["token"]: s["scene_token"] for s in samples}

# Group CAM_FRONT keyframes by scene
scene_cam_front = defaultdict(list)
for sd in sample_data:
    if "CAM_FRONT" in sd["filename"] and sd["is_key_frame"]:
        st = sd["sample_token"]
        if st in sample_to_scene:
            scene_tok = sample_to_scene[st]
            scene_cam_front[scene_tok].append(sd)

print(f"{'Scene Name':<12} | {'Frames':<6} | {'Duration(s)':<11} | {'Rate(Hz)':<8} | {'Description'}")
print("-" * 80)
total_kf = 0
scene_map = {s["token"]: s for s in scenes}
for sc_tok in sorted(scene_cam_front.keys(), key=lambda t: scene_map[t]["name"]):
    sc = scene_map[sc_tok]
    items = sorted(scene_cam_front[sc_tok], key=lambda x: x["timestamp"])
    total_kf += len(items)
    t0 = items[0]["timestamp"]
    t1 = items[-1]["timestamp"]
    dt_s = (t1 - t0) / 1e6
    rate = (len(items) - 1) / dt_s if dt_s > 0 else 0
    print(f"{sc['name']:<12} | {len(items):<6} | {dt_s:<11.2f} | {rate:<8.2f} | {sc['description'][:40]}")

print("-" * 80)
print(f"Total CAM_FRONT Keyframes across 10 scenes: {total_kf}")
