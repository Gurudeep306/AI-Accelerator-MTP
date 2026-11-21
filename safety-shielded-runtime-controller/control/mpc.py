"""Receding-horizon model predictive controller (exhaustive, vectorised).

Decision: a first move a1 (applied now) and a held move a2 for steps 2..H
(two-block move blocking), scored jointly for every (a1, a2) pair:

    J(a1, a2) = sum_{h=1..H} E[stage_cost_h]  +  gamma_tail * tail_cost(a2)

* crowd forecast: constant-velocity extrapolation of the tracked pedestrians
  (the information an onboard tracker has); `oracle=True` uses the true
  future crowd as an upper-bound ablation,
* accelerator: the same model as the plant with median latency; expected
  deadline misses use the lognormal jitter + spike distribution in closed form,
* tail: holds a2 for another `tail_s` seconds on a coarse grid so heat that
  would build up beyond the horizon (thermal time constant ~80 s) is priced,
* a1 is restricted to shield-admissible points (hard constraints), a2 to the
  Pareto-non-dominated operating points (65 of 135) to keep the search small.
"""
from __future__ import annotations

import numpy as np

from accel.model import AcceleratorModel
from config.params import Params
from control.baselines import Controller
from control.shield import Shield
from robot.safety import perceive, permitted_speed
from sim.env import stage_cost


def _q(z):
    """Gaussian upper-tail probability (Abramowitz-Stegun 7.1.26 via erfc)."""
    x = z / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erfc = y * np.exp(-x * x)
    return 0.5 * np.where(x >= 0, erfc, 2.0 - erfc)


def pareto_actions(m: AcceleratorModel):
    lat, net = m.latency_ms(np.arange(m.space.n))
    c = np.stack([m.a_busy_w * net, m.a_busy_w, lat, -m.a_recall, -m.a_map50], 1)
    return np.array([i for i in range(m.space.n)
                     if not any((c[j] <= c[i]).all() and (c[j] < c[i]).any() for j in range(m.space.n))])


class MPCController(Controller):
    def __init__(self, params: Params, horizon: int = 8, tail_s: float = 60.0, gamma_tail: float = 0.25,
                 oracle: bool = False, env=None):
        self.P, self.H, self.oracle, self.env = params, horizon, oracle, env
        self.name = "MPC-oracle" if oracle else "MPC"
        self.m = AcceleratorModel(params.accel, params.task)
        self.shield = Shield(params)
        self.a2_set = pareto_actions(self.m)
        self.tail_s, self.gamma_tail = tail_s, gamma_tail
        self.dt = 0.4

    def _forecast(self, obs):
        rp, dt = self.P.robot, self.dt
        frames = self.env.future_frames(self.H) if self.oracle else None
        out, v = [], obs.v_robot
        for h in range(1, self.H + 1):
            v = min(rp.nominal_speed, v + rp.max_accel * dt * 0.5)
            p_robot = obs.robot_p + obs.heading * v * dt * h
            if frames is not None:
                _, pos, vel = frames[h - 1]
            else:
                pos, vel = obs.ped_pos + obs.ped_vel * dt * h, obs.ped_vel
            out.append(perceive(pos, vel, p_robot, obs.heading, v, rp, forecast_s=0.0))
        return out

    def _expected_deadline_miss(self, lat_med, sigma):
        a = self.P.accel
        z = np.log(self.P.task.deadline_ms / lat_med) / sigma
        zs = z - np.log(a.spike_factor) / sigma
        return (1 - a.spike_probability) * _q(z) + a.spike_probability * _q(zs)

    def act(self, obs):
        P, m, w, dt, rp = self.P, self.m, self.P.cost, self.dt, self.P.robot
        mask, _ = self.shield.admissible(obs)
        a1_set = np.flatnonzero(mask)
        if len(a1_set) == 0:                       # let the shield's backup handle it
            a1_set = np.arange(m.space.n)
        A1 = np.repeat(a1_set, len(self.a2_set))
        A2 = np.tile(self.a2_set, len(a1_set))
        percepts = [obs.percept] + self._forecast(obs)[:-1]   # step h uses the percept at its start
        T = np.full(A1.shape, obs.temp)
        Q = np.full(A1.shape, obs.queue)
        thr = np.full(A1.shape, obs.throttled)
        v = np.full(A1.shape, obs.v_robot)
        J = np.zeros(A1.shape)
        sw1, sw2 = m.reconfig_ms(obs.prev_action, A1), m.reconfig_ms(A1, A2)
        for h, pc in enumerate(percepts):
            a = A1 if h == 0 else A2
            sw = sw1 if h == 0 else (sw2 if h == 1 else 0.0)
            out = m.step(a, T, Q, dt, obs.ambient, pc.n_view, pc.crowd_norm, thr, switch_ms=sw)
            p90 = m.step(a, T, Q, dt, obs.ambient, pc.n_view, pc.crowd_norm, thr, latency_mult=m.p90_mult(a),
                         switch_ms=sw)
            v_cmd = permitted_speed(pc, p90["detect_ms"], rp)
            v = np.clip(v_cmd, np.maximum(v - rp.max_brake * dt, 0), v + rp.max_accel * dt)
            switched = (A1 != obs.prev_action) if h == 0 else ((A2 != A1) if h == 1 else 0.0)
            J += stage_cost(w, v / rp.nominal_speed, out["power_w"], out["p_miss"] * pc.hazard, 0.0, 0.0,
                            self._expected_deadline_miss(out["first_frame_ms"], m.a_sigma[a]),
                            np.maximum(out["temp"] - P.accel.temp_soft_limit, 0.0), thr, switched,
                            out["drops"] / (P.task.camera_fps * dt))
            T, Q = out["temp"], out["queue"]
            thr = np.where(T >= P.accel.temp_throttle, True, np.where(T < P.accel.temp_release, False, thr))
        # thermal tail: hold a2 on a coarse 4 s grid
        coarse, n_tail = 4.0, int(self.tail_s / 4.0)
        pc = percepts[-1]
        tail = np.zeros(A1.shape)
        for _ in range(n_tail):
            out = m.step(A2, T, Q, coarse, obs.ambient, pc.n_view, pc.crowd_norm, thr)
            T = out["temp"]
            thr = np.where(T >= P.accel.temp_throttle, True, np.where(T < P.accel.temp_release, False, thr))
            tail += (coarse / dt) * (w.thermal_over_soft * np.maximum(T - P.accel.temp_soft_limit, 0)
                                     + w.throttle * thr + w.energy_per_w * out["power_w"])
        J += self.gamma_tail * tail / max(n_tail, 1)
        best = int(np.argmin(J))
        self.last_cost = float(J[best])
        return int(A1[best])
