"""Scenario sources -> one normalised `Scene` (pedestrian trajectories in metres).

Sources
  * dataset-based : ETH/UCY (BIWI ETH, BIWI Hotel, UCY Univ, Zara01/02/03) or any
                    trajectory CSV with columns  frame, ped_id, x, y
  * equation-based: a parametric multi-phase crowd generator (synthetic)

A robot is then replayed through the *real* recorded crowd (CrowdNav-style
replay). Pedestrians do not react to the robot; see README "Limitations".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eth_ucy"

# Standard ETH/UCY leave-one-scene-out groups. zara03 & uni_examples are
# training-only in the common protocol (Social-GAN / Trajectron++).
SCENE_GROUPS = {
    "eth": ["biwi_eth"],
    "hotel": ["biwi_hotel"],
    "univ": ["students001", "students003"],
    "zara1": ["crowds_zara01"],
    "zara2": ["crowds_zara02"],
}
TRAIN_ONLY = ["crowds_zara03", "uni_examples"]
ALL_SEQUENCES = [s for g in SCENE_GROUPS.values() for s in g] + TRAIN_ONLY


@dataclass
class Scene:
    name: str
    source: str
    dt: float
    frames: list            # list of (ids[n], pos[n,2], vel[n,2]) per time step
    path_start: np.ndarray = field(default=None)
    path_end: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.frames)

    @property
    def n_pedestrians(self):
        return len({int(i) for ids, _, _ in self.frames for i in ids if i >= 0})


# --------------------------------------------------------------------- loaders
def _from_rows(rows: np.ndarray, name: str, source: str, dt: float) -> Scene:
    """rows: [frame, ped_id, x, y] -> Scene with finite-difference velocities."""
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    frame_ids = np.unique(rows[:, 0])
    step = np.median(np.diff(frame_ids)) if len(frame_ids) > 1 else 1.0
    full = np.arange(frame_ids[0], frame_ids[-1] + step / 2, step)   # keep empty frames
    by_frame = {f: rows[rows[:, 0] == f] for f in frame_ids}
    pos_of = [{int(r[1]): r[2:4] for r in by_frame.get(f, np.empty((0, 4)))} for f in full]
    frames = []
    for k, cur in enumerate(pos_of):
        ids = np.array(sorted(cur), dtype=int)
        pos = np.array([cur[i] for i in ids]).reshape(-1, 2)
        vel = np.zeros_like(pos)
        for j, i in enumerate(ids):
            nxt = pos_of[k + 1].get(i) if k + 1 < len(pos_of) else None
            prv = pos_of[k - 1].get(i) if k > 0 else None
            if nxt is not None and prv is not None:
                vel[j] = (nxt - prv) / (2 * dt)
            elif nxt is not None:
                vel[j] = (nxt - pos[j]) / dt
            elif prv is not None:
                vel[j] = (pos[j] - prv) / dt
        frames.append((ids, pos, vel))
    scene = Scene(name, source, dt, frames)
    _set_robot_path(scene)
    return scene


def load_trajectory_file(path, dt: float = 0.4, name: str | None = None) -> Scene:
    """ETH/UCY .txt (whitespace) or CSV with header frame,ped_id,x,y."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = np.genfromtxt(path, delimiter=",", names=True)
        rows = np.stack([rows["frame"], rows["ped_id"], rows["x"], rows["y"]], 1)
    else:
        rows = np.loadtxt(path)
    return _from_rows(rows.astype(float), name or path.stem, f"dataset:{path.name}", dt)


def load_eth_ucy(sequence: str, dt: float = 0.4) -> Scene:
    return load_trajectory_file(DATA_DIR / f"{sequence}.txt", dt, sequence)


def equation_scene(duration_s: float = 300.0, dt: float = 0.4, seed: int = 0,
                   width: float = 16.0, height: float = 12.0) -> Scene:
    """Synthetic crowd with four phases (open -> counter-flow surge -> crossing -> dense mix).

    Pedestrians spawn as a Poisson process with phase-dependent rate and walk
    with constant velocity plus small noise. Not real data: used for stress
    testing and as RL training augmentation.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    # (spawn rate ped/s, fraction counter-flow (along x), fraction crossing (along y))
    phases = [(0.15, 0.5, 0.0), (1.4, 0.8, 0.1), (0.9, 0.2, 0.8), (1.8, 0.5, 0.5)]
    active, rows, next_id = [], [], 0
    for k in range(n):
        rate, counter, cross = phases[min(int(4 * k / n), 3)]
        for _ in range(rng.poisson(rate * dt)):
            speed = np.clip(rng.normal(1.25, 0.25), 0.5, 2.0)
            if rng.random() < cross:
                y0, sgn = (0.0, 1) if rng.random() < 0.5 else (height, -1)
                p, v = np.array([rng.uniform(2, width - 2), y0]), np.array([0.0, sgn * speed])
            else:
                x0, sgn = (0.0, 1) if rng.random() < 0.5 + 0.5 * (counter - 0.5) else (width, -1)
                p, v = np.array([x0, rng.uniform(3, height - 3)]), np.array([sgn * speed, 0.0])
            active.append([next_id, p, v]); next_id += 1
        still = []
        for pid, p, v in active:
            v = v + rng.normal(0, 0.05, 2)
            p = p + v * dt
            if -0.5 <= p[0] <= width + 0.5 and -0.5 <= p[1] <= height + 0.5:
                rows.append([k, pid, p[0], p[1]]); still.append([pid, p, v])
        active = still
    rows += [[0, -1, -100.0, -100.0], [n - 1, -1, -100.0, -100.0]]  # pin the timeline (far-away marker)
    scene = _from_rows(np.array(rows, float), f"equation_s{seed}", "synthetic_equation", dt)
    scene.path_start, scene.path_end = np.array([1.0, height / 2]), np.array([width - 1.0, height / 2])
    return scene


# ------------------------------------------------------------------ robot path
def _set_robot_path(scene: Scene):
    """Straight path along the dominant pedestrian-flow axis (maximises counter-flow
    interactions), spanning the 5th-95th percentile of where people actually walk."""
    pos = np.concatenate([p for _, p, _ in scene.frames if len(p)])
    vel = np.concatenate([v for _, _, v in scene.frames if len(v)])
    moving = vel[np.linalg.norm(vel, axis=1) > 0.3]
    _, _, vt = np.linalg.svd(moving - moving.mean(0), full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    centre = np.median(pos, 0)
    proj = (pos - centre) @ axis
    lo, hi = np.percentile(proj, [5, 95])
    scene.path_start, scene.path_end = centre + lo * axis, centre + hi * axis


def all_dataset_scenes(dt: float = 0.4):
    return {s: load_eth_ucy(s, dt) for s in ALL_SEQUENCES}
