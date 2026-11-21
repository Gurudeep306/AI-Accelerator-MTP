"""Robot-side perception features and the latency-aware speed constraint.

Key coupling (the reason the accelerator controller matters for safety):
the robot may only drive as fast as it can still stop before the nearest
in-path pedestrian, and the stopping distance includes the *time-to-detect*
of the perception pipeline plus actuation delay:

    t_detect = latency + queue wait + (confirm_hits / recall - 1) / fps
    d_stop(v) = v * (t_detect + t_act) + v^2 / (2 a_brake)
    v_stop = -a*tau + sqrt((a*tau)^2 + 2*a*d_free),   tau = t_detect + t_act

So a faster-but-weaker early exit (low recall) can *reduce* the safe speed,
and so can a configuration whose throughput falls below the camera rate.

    v_permit = min(v_nominal, v_zone, v_crowd(density), v_stop)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.params import RobotParams


@dataclass
class Percept:
    clearance_m: float          # longitudinal free distance to nearest in-path pedestrian
    density: float              # ped / m^2 within density radius
    crowd_norm: float           # density / reference, clipped to [0,1]
    n_view: int                 # pedestrians in the forward field of view
    hazard: bool                # an in-path pedestrian is within hazard distance
    contact: bool               # in-path pedestrian at body contact (frontal)
    closing_speed: float        # m/s, positive = nearest in-path pedestrian approaching
    min_clearance_forecast: float  # constant-velocity forecast over next 2 s


def perceive(pos, vel, robot_p, heading, v_robot, p: RobotParams, forecast_s=2.0, dt=0.4) -> Percept:
    if len(pos) == 0:
        return Percept(p.sensing_range_m, 0.0, 0.0, 0, False, False, 0.0, p.sensing_range_m)
    normal = np.array([-heading[1], heading[0]])
    rel = pos - robot_p
    dist = np.linalg.norm(rel, axis=1)
    along, lat = rel @ heading, rel @ normal
    in_view = (along > 0) & (dist < p.sensing_range_m) & (np.abs(lat) < along * np.tan(np.radians(60)) + 0.5)
    in_path = (along > 0) & (np.abs(lat) < p.corridor_half_width_m) & (along < p.sensing_range_m)
    clearance = p.sensing_range_m
    closing = 0.0
    if in_path.any():
        j = np.flatnonzero(in_path)[np.argmin(along[in_path])]
        clearance = max(along[j] - p.body_clearance_m, 0.0)
        closing = v_robot - float(vel[j] @ heading)
    density = float((dist < p.density_radius_m).sum()) / (np.pi * p.density_radius_m ** 2)
    # constant-velocity forecast (what an onboard tracker would provide)
    fc = clearance
    for t in np.arange(dt, forecast_s + 1e-9, dt):
        rp = robot_p + heading * v_robot * t
        r2 = pos + vel * t - rp
        a2, l2 = r2 @ heading, r2 @ normal
        m = (a2 > 0) & (np.abs(l2) < p.corridor_half_width_m)
        if m.any():
            fc = min(fc, max(a2[m].min() - p.body_clearance_m, 0.0))
    return Percept(clearance, density, min(density / p.crowd_density_ref, 1.0), int(in_view.sum()),
                   bool(in_path.any() and clearance < p.hazard_distance_m),
                   bool(in_path.any() and clearance <= 0.0), closing, fc)


def stopping_speed(clearance_m, reaction_s, p: RobotParams):
    """Largest v with v*tau + v^2/(2a) <= free distance (vectorised)."""
    d = np.maximum(np.asarray(clearance_m) - p.safety_buffer_m, 0.0)
    a, tau = p.max_brake, np.asarray(reaction_s)
    return -a * tau + np.sqrt((a * tau) ** 2 + 2 * a * d)


def crowd_speed(density, p: RobotParams):
    ratio = np.minimum(np.asarray(density) / p.crowd_density_ref, 1.0)
    return p.nominal_speed - ratio * (p.nominal_speed - p.crowd_speed_floor)


def permitted_speed(percept: Percept, detect_ms, p: RobotParams):
    """Speed cap given a time-to-detect (ms); vectorised over detect_ms."""
    tau = np.asarray(detect_ms) / 1000.0 + p.actuation_delay_s
    v_stop = stopping_speed(percept.clearance_m, tau, p)
    return np.minimum.reduce([np.full_like(v_stop, p.nominal_speed),
                              np.full_like(v_stop, p.zone_speed_limit),
                              np.full_like(v_stop, crowd_speed(percept.density, p)), v_stop])


def stopping_distance(v, detect_ms, p: RobotParams):
    tau = np.asarray(detect_ms) / 1000.0 + p.actuation_delay_s
    return v * tau + v ** 2 / (2 * p.max_brake)
