"""Run one closed-loop episode: controller -> shield -> plant, and summarise it."""
from __future__ import annotations

import numpy as np


def run_episode(env, controller, shield=None):
    obs = env.reset()
    controller.reset()
    if shield is not None:
        shield.reset()
    log, done = [], False
    while not done:
        proposal = controller.act(obs)
        action, intervened = shield.filter(obs, proposal) if shield is not None else (proposal, False)
        obs, cost, done, info = env.step(action)
        info["proposal"], info["intervened"] = int(proposal), bool(intervened)
        log.append(info)
    return log


def summarise(log, dt=0.4):
    """Per-episode metrics (sums are converted to rates where that is fairer across scenes)."""
    g = lambda k: np.array([r[k] for r in log], dtype=float)
    n = len(log)
    hazard = g("hazard").astype(bool)
    return {
        "steps": n,
        "duration_s": n * dt,
        "mean_cost": g("cost").mean(),
        "distance_m": g("distance").sum(),
        "mean_speed_mps": g("v").mean(),
        "energy_j": g("energy_j").sum(),
        "energy_per_m_j": g("energy_j").sum() / max(g("distance").sum(), 1e-6),
        "mean_power_w": g("power_w").mean(),
        "mean_latency_ms": g("latency_ms").mean(),
        "p95_latency_ms": float(np.percentile(g("latency_ms"), 95)),
        "deadline_miss_rate": g("deadline_miss").mean(),
        "mean_time_to_detect_ms": g("detect_ms").mean(),
        "hazard_time_to_detect_ms": g("detect_ms")[hazard].mean() if hazard.any() else np.nan,
        "mean_map50": g("map50").mean(),
        "hazard_map50": g("map50")[hazard].mean() if hazard.any() else np.nan,
        "hazard_miss_prob": g("p_miss")[hazard].mean() if hazard.any() else 0.0,
        "hazard_steps": int(hazard.sum()),
        "mean_throughput_fps": g("throughput_fps").mean(),
        "mean_queue": g("queue").mean(),
        "frame_drop_rate": g("drop_frac").mean(),
        "mean_temp_c": g("temp").mean(),
        "max_temp_c": g("temp").max(),
        "time_over_soft_s": float((g("temp") > 80.0).sum() * dt),
        "throttle_s": float(g("throttled").sum() * dt),
        "safety_violations": int(g("violation").sum()),
        "violations_per_km": g("violation").sum() / max(g("distance").sum() / 1000, 1e-6),
        "mean_speed_excess_mps": g("speed_excess").mean(),
        "frontal_contacts": int(g("collision").sum()),
        "braking_limited_conflicts": int(g("braking_limited").sum()),
        "switches_per_min": g("switched").sum() / (n * dt / 60),
        "shield_interventions": int(g("intervened").sum()),
        "shield_intervention_rate": g("intervened").mean(),
    }
