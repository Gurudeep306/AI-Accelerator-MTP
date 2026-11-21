"""Closed-loop environment: real crowd replay + robot + simulated edge accelerator.

One step = one ETH/UCY annotation period (0.4 s). In each step:
  1. the robot perceives the crowd (frame k) from its current position,
  2. the controller (through the safety shield) picks an operating point,
  3. the robot planner caps speed using the controller's latency contract
     (p90-predicted time-to-detect of the chosen operating point),
  4. the accelerator runs with a *sampled* latency (lognormal fitted to the
     measured p90, plus rare contention spikes) and a hardware thermal throttle,
  5. safety is judged with the *realised* time-to-detect. A step is a
     perception-induced violation when the commanded speed exceeds the speed
     that the realised time-to-detect permits (by > 0.05 m/s). Situations where
     a pedestrian cuts in closer than the robot can brake for are logged
     separately as braking-limited conflicts (crowd-induced, not controller-induced).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from accel.model import AcceleratorModel
from config.params import Params
from robot.safety import Percept, perceive, permitted_speed
from robot.scenes import Scene


@dataclass
class Observation:
    k: int
    percept: Percept
    temp: float
    queue: float
    throttled: bool
    v_robot: float
    prev_action: int
    ambient: float
    robot_p: np.ndarray
    heading: np.ndarray
    ped_pos: np.ndarray
    ped_vel: np.ndarray
    steps_left: int


def stage_cost(w, v_ratio, power_w, hazard_pmiss, violation, collision, deadline_miss,
               temp_over_soft, throttled, switched, drop_frac):
    """Shared objective (vectorised). Lower is better."""
    return (-w.progress * v_ratio + w.energy_per_w * power_w + w.hazard_miss * hazard_pmiss
            + w.safety_violation * violation + w.collision * collision + w.deadline_miss * deadline_miss
            + w.thermal_over_soft * temp_over_soft + w.throttle * throttled + w.switch * switched
            + w.frame_drop * drop_frac)


class CrowdAcceleratorEnv:
    def __init__(self, scene: Scene, params: Params, seed: int = 0, ambient_c: float | None = None,
                 initial_temp_c: float | None = None, start_k: int = 0, max_steps: int | None = None):
        self.scene, self.P = scene, params
        self.model = AcceleratorModel(params.accel, params.task)
        self.space = self.model.space
        self.rng = np.random.default_rng(seed)
        self.ambient = params.sim.ambient_c if ambient_c is None else ambient_c
        self.t0 = params.sim.initial_temp_c if initial_temp_c is None else initial_temp_c
        self.start_k = start_k
        self.end_k = len(scene) - 1 if max_steps is None else min(len(scene) - 1, start_k + max_steps)
        self.default_action = self.space.index(3, "FP16", 0.2, "final")
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> Observation:
        sc = self.scene
        self.k = self.start_k
        self.path_len = float(np.linalg.norm(sc.path_end - sc.path_start))
        self.axis = (sc.path_end - sc.path_start) / self.path_len
        self.s, self.direction = 0.0, 1.0
        self.v = 0.0
        self.temp, self.queue, self.throttled = self.t0, 0.0, False
        self.prev_action = self.default_action
        return self.observe()

    @property
    def robot_p(self):
        return self.scene.path_start + self.axis * self.s

    @property
    def heading(self):
        return self.axis * self.direction

    def observe(self) -> Observation:
        ids, pos, vel = self.scene.frames[self.k]
        self.percept = perceive(pos, vel, self.robot_p, self.heading, self.v, self.P.robot, dt=self.scene.dt)
        return Observation(self.k, self.percept, self.temp, self.queue, self.throttled, self.v,
                           self.prev_action, self.ambient, self.robot_p.copy(), self.heading.copy(),
                           pos, vel, self.end_k - self.k)

    def future_frames(self, horizon: int):
        """Ground-truth future crowd (only the oracle-preview MPC ablation uses this)."""
        return [self.scene.frames[min(self.k + h, len(self.scene) - 1)] for h in range(1, horizon + 1)]

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        P, m, rp, dt = self.P, self.model, self.P.robot, self.scene.dt
        pc = self.percept
        # (3) planner speed cap from the p90 latency contract of the chosen action
        sw_ms = float(m.reconfig_ms(self.prev_action, action))
        pred = m.step(action, self.temp, self.queue, dt, self.ambient, pc.n_view, pc.crowd_norm,
                      self.throttled, latency_mult=m.p90_mult(action), switch_ms=sw_ms)
        v_cmd = float(permitted_speed(pc, pred["detect_ms"], rp))
        # (4) realised accelerator behaviour
        mult = float(np.exp(self.rng.normal(0.0, m.a_sigma[action])))
        if self.rng.random() < P.accel.spike_probability:
            mult *= P.accel.spike_factor
        out = m.step(action, self.temp, self.queue, dt, self.ambient, pc.n_view, pc.crowd_norm,
                     self.throttled, latency_mult=mult, switch_ms=sw_ms)
        out = {k: float(v) for k, v in out.items()}
        v_floor = max(self.v - rp.max_brake * dt, 0.0)
        v_new = float(np.clip(v_cmd, v_floor, self.v + rp.max_accel * dt))
        braking_limited = v_cmd < v_floor - 1e-9
        # (5) safety with the realised time-to-detect
        v_safe_real = float(permitted_speed(pc, out["detect_ms"], rp))
        excess = max(min(v_new, max(v_cmd, v_floor)) - v_safe_real, 0.0) if not braking_limited else 0.0
        violation = bool(v_new > 0.05 and excess > 0.05)
        collision = bool(pc.contact and v_new > 0.05)
        deadline_miss = out["first_frame_ms"] > P.task.deadline_ms
        hazard_pmiss = out["p_miss"] if pc.hazard else 0.0
        switched = action != self.prev_action
        drop_frac = out["drops"] / (P.task.camera_fps * dt)
        was_throttled = self.throttled
        cost = float(stage_cost(P.cost, v_new / rp.nominal_speed, out["power_w"], hazard_pmiss, violation,
                                collision, deadline_miss, max(out["temp"] - P.accel.temp_soft_limit, 0.0),
                                was_throttled, switched, drop_frac))
        info = dict(k=self.k, t=self.k * dt, action=int(action), v=v_new, v_cmd=v_cmd,
                    clearance=pc.clearance_m, density=pc.density, n_view=pc.n_view, hazard=pc.hazard,
                    violation=violation, speed_excess=excess, braking_limited=braking_limited,
                    v_safe_real=v_safe_real,
                    collision=collision, deadline_miss=deadline_miss, hazard_pmiss=hazard_pmiss,
                    throttled=was_throttled, switched=switched, drop_frac=drop_frac, cost=cost,
                    distance=v_new * dt, reconfig_ms=sw_ms, **out)
        # advance robot (bounce at path ends) and accelerator state
        self.s += self.direction * v_new * dt
        if self.s >= self.path_len:
            self.s, self.direction = 2 * self.path_len - self.s, -1.0
        elif self.s <= 0:
            self.s, self.direction = -self.s, 1.0
        self.v, self.temp, self.queue = v_new, out["temp"], out["queue"]
        if self.temp >= P.accel.temp_throttle:
            self.throttled = True
        elif self.throttled and self.temp < P.accel.temp_release:
            self.throttled = False
        self.prev_action = int(action)
        self.k += 1
        done = self.k >= self.end_k
        return self.observe(), cost, done, info
