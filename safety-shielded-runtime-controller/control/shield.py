"""Safety & Feasibility layer (run-time shield, Simplex-style).

Every controller's proposal passes through `Shield.filter`. An operating point
is *admissible* in the current state when, using the p90 latency contract:

  C1  deadline      p90 latency (+ reconfiguration overhead) <= task deadline
  C2  thermal       next-step temperature <= soft limit, and holding the action
                    for `hold_steps` keeps temperature below the throttle point
  C3  perception    if a pedestrian is in the hazard zone, the probability of
                    missing it during the step <= max_miss_probability
  C4  throughput    if a pedestrian is in the hazard zone, no frame drops
  C5  spike margin  if a pedestrian is in the hazard zone, a latency spike
                    (p90 x spike factor) must not lower the permitted speed by
                    more than `spike_speed_tol` m/s relative to the contract,
                    i.e. the point keeps head-room before its queue builds up

If the proposal is admissible it passes unchanged. Otherwise the shield applies
a *minimal intervention*: the admissible point closest in knob space (L1 over
frequency / precision / sparsity / exit index), ties broken by lower power. If
nothing is admissible it falls back lexicographically: thermal first, then
perception, then deadline (backup controller of the Simplex architecture).
"""
from __future__ import annotations

import numpy as np

from accel.model import AcceleratorModel
from config.params import Params
from robot.safety import permitted_speed


class Shield:
    def __init__(self, params: Params, hold_steps: int = 10, dt: float = 0.4, spike_speed_tol: float = 0.05):
        self.spike_speed_tol = spike_speed_tol
        self.P = params
        self.m = AcceleratorModel(params.accel, params.task)
        self.space = self.m.space
        self.hold_steps, self.dt = hold_steps, dt
        self.all = np.arange(self.space.n)
        self.knobs = np.stack([self.space.f, self.space.prec, self.space.sp, self.space.exit], 1)
        self.interventions = 0
        self.decisions = 0

    def reset(self):
        self.interventions = self.decisions = 0

    def evaluate(self, obs):
        """Constraint margins for all actions (vectorised)."""
        P, m, pc = self.P, self.m, obs.percept
        p90 = m.p90_mult(self.all)
        sw = m.reconfig_ms(obs.prev_action, self.all)
        out = m.step(self.all, obs.temp, obs.queue, self.dt, obs.ambient, pc.n_view, pc.crowd_norm,
                     obs.throttled, latency_mult=p90, switch_ms=sw)
        c1 = out["first_frame_ms"] <= P.task.deadline_ms
        t = out["temp"].copy()
        q = out["queue"]
        for _ in range(self.hold_steps - 1):
            nxt = m.step(self.all, t, q, self.dt, obs.ambient, pc.n_view, pc.crowd_norm, False)
            t, q = nxt["temp"], nxt["queue"]
        c2 = (out["temp"] <= P.accel.temp_soft_limit) & (t < P.accel.temp_throttle - 1.0)
        if pc.hazard:
            c3 = out["p_miss"] <= P.task.max_miss_probability
            c4 = out["drops"] <= 1e-9
            spike = m.step(self.all, obs.temp, obs.queue, self.dt, obs.ambient, pc.n_view, pc.crowd_norm,
                           obs.throttled, latency_mult=p90 * P.accel.spike_factor, switch_ms=sw)
            v_contract = permitted_speed(pc, out["detect_ms"], P.robot)
            v_spike = permitted_speed(pc, spike["detect_ms"], P.robot)
            c5 = (v_contract - v_spike) <= self.spike_speed_tol
        else:
            c3 = c4 = c5 = np.ones(self.space.n, bool)
        return dict(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, pred=out, t_hold=t)

    def admissible(self, obs):
        e = self.evaluate(obs)
        return e["c1"] & e["c2"] & e["c3"] & e["c4"] & e["c5"], e

    def filter(self, obs, proposal: int):
        self.decisions += 1
        mask, e = self.admissible(obs)
        if mask[proposal]:
            return int(proposal), False
        self.interventions += 1
        power = e["pred"]["power_w"]
        if mask.any():
            dist = np.abs(self.knobs - self.knobs[proposal]).sum(1).astype(float)
            score = np.where(mask, dist + 1e-3 * power, np.inf)
            return int(np.argmin(score)), True
        # backup: satisfy constraints in priority order, as far as possible
        keep = np.ones(self.space.n, bool)
        for c in ("c2", "c3", "c4", "c5", "c1"):
            if (keep & e[c]).any():
                keep &= e[c]
        if keep.all() or not e["c2"].any():          # nothing thermally safe -> coolest point
            return int(np.argmin(e["t_hold"])), True
        score = np.where(keep, e["pred"]["p_miss"] + 1e-3 * power, np.inf)
        return int(np.argmin(score)), True
