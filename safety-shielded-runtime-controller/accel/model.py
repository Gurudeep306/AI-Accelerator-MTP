"""Vectorised accelerator model: latency, accuracy/recall, power, thermal, queue.

Every function accepts numpy arrays of action indices, so the shield and the
MPC can score all 135 operating points (5 freq x 3 precision x 3 sparsity x
3 Anytime-YOLO exits) in one call.

Measured anchors: Anytime-YOLO (YOLOv8n) exit latency/mAP/recall on a GTX 1650
(BDD100K, 150 images) and YOLO11n FP32/FP16/INT8 on a T4. Everything else is a
documented model assumption (see config/params.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.params import AcceleratorParams, TaskParams

PROFILE_FILE = Path(__file__).with_name("anytime_yolo_profiles.json")


@dataclass
class ActionSpace:
    """Enumerates operating points. index = ((f*P + p)*S + s)*E + e."""
    p: AcceleratorParams

    def __post_init__(self):
        nf, np_, ns, ne = (len(self.p.freq_ghz), len(self.p.precisions),
                           len(self.p.sparsities), len(self.p.exits))
        grid = np.stack(np.meshgrid(np.arange(nf), np.arange(np_), np.arange(ns), np.arange(ne),
                                    indexing="ij"), -1).reshape(-1, 4)
        self.f, self.prec, self.sp, self.exit = (grid[:, i] for i in range(4))
        self.n = len(grid)

    def index(self, f: int, prec: str, sp: float, exit_: str) -> int:
        p = self.p
        return int((((f - 1) * len(p.precisions) + p.precisions.index(prec)) * len(p.sparsities)
                    + p.sparsities.index(sp)) * len(p.exits) + p.exits.index(exit_))

    def describe(self, a: int) -> str:
        p = self.p
        return (f"f{self.f[a] + 1}/{p.precisions[self.prec[a]]}/sp{p.sparsities[self.sp[a]]:.1f}"
                f"/{p.exits[self.exit[a]]}")

    def features(self, a) -> np.ndarray:
        """Normalised [freq, precision, sparsity, exit] in [0,1] (for RL / plots)."""
        a = np.asarray(a)
        return np.stack([self.f[a] / 4.0, self.prec[a] / 2.0, self.sp[a] / 2.0, self.exit[a] / 2.0], -1)


class AcceleratorModel:
    def __init__(self, p: AcceleratorParams, task: TaskParams):
        self.p, self.task = p, task
        self.space = ActionSpace(p)
        prof = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        ex = [prof[e] for e in p.exits]
        net = np.array([e["measured_network_latency_median_ms"] for e in ex])
        tot = np.array([e["measured_latency_median_ms"] for e in ex])
        p90 = np.array([e["measured_latency_p90_ms"] for e in ex])
        self.exit_net_ms = net * p.edge_slowdown
        self.exit_host_ms = tot - net                         # pre/post-processing on host
        self.exit_sigma = np.log(p90 / tot) / 1.2816          # lognormal jitter fitted to p90
        self.exit_map50 = np.array([e["measured_accuracy_map50"] for e in ex])
        self.exit_recall = np.array([e["measured_recall"] for e in ex])
        self.exit_gmacs = np.array([e["macs"] for e in ex]) / 1e9

        s, f = self.space, np.array(p.freq_ghz)
        self.a_net_ms = (self.exit_net_ms[s.exit] * (f[-1] / f[s.f])
                         * np.array(p.precision_latency)[s.prec] * np.array(p.sparsity_latency)[s.sp])
        self.a_host_ms = self.exit_host_ms[s.exit]
        self.a_sigma = self.exit_sigma[s.exit]
        v = np.array(p.volt)
        self.a_busy_w = (p.dyn_power_max_w * (f[s.f] / f[-1]) * (v[s.f] / v[-1]) ** 2
                         * np.array(p.precision_power)[s.prec] * np.array(p.sparsity_power)[s.sp])
        q = np.array(p.precision_accuracy)[s.prec] * np.array(p.sparsity_accuracy)[s.sp]
        self.a_map50 = self.exit_map50[s.exit] * q
        self.a_recall = self.exit_recall[s.exit] * q
        self.tau = p.thermal_resistance * p.thermal_capacitance
        self.throttle_action = s.index(1, "INT8", 0.4, "head1")
        self.min_power_action = int(np.argmin(self.a_busy_w * self.a_net_ms))

    # ------------------------------------------------------------------ pieces
    def latency_ms(self, a, n_objects=0.0, throttled=False):
        a = np.asarray(a)
        net = self.a_net_ms[a]
        if np.any(throttled):  # hardware forces the lowest frequency
            f = np.array(self.p.freq_ghz)
            net = np.where(throttled, net * f[self.space.f[a]] / f[0], net)
        return net + self.a_host_ms[a] + self.p.nms_ms_per_object * n_objects, net

    def reconfig_ms(self, a_from, a_to):
        s, p = self.space, self.p
        a_from, a_to = np.asarray(a_from), np.asarray(a_to)
        dvfs = (s.f[a_from] != s.f[a_to]) * p.dvfs_switch_ms
        swap = ((s.prec[a_from] != s.prec[a_to]) | (s.sp[a_from] != s.sp[a_to])) * p.engine_swap_ms
        return dvfs + swap

    def p90_mult(self, a):
        return np.exp(1.2816 * self.a_sigma[np.asarray(a)])

    def recall(self, a, crowd_norm=0.0):
        return self.a_recall[np.asarray(a)] * (1.0 - self.p.crowd_occlusion_penalty * crowd_norm)

    def map50(self, a, crowd_norm=0.0):
        return self.a_map50[np.asarray(a)] * (1.0 - self.p.crowd_occlusion_penalty * crowd_norm)

    def leak_w(self, temp):
        return self.p.leak_power_25c_w * np.exp(self.p.leak_temp_coeff * (np.asarray(temp) - 25.0))

    # --------------------------------------------------------------- one step
    def step(self, a, temp, queue, dt, ambient, n_objects=0.0, crowd_norm=0.0,
             throttled=False, latency_mult=1.0, switch_ms=0.0):
        """Advance the accelerator by one control period. Vectorised over a/temp/queue.

        latency_mult = 1 gives the median (prediction); the simulator passes a
        sampled lognormal multiplier to obtain realised behaviour.
        """
        a = np.asarray(a)
        lat, net = self.latency_ms(a, n_objects, throttled)
        lat, net = lat * latency_mult, net * latency_mult
        arrivals = self.task.camera_fps * dt
        capacity = np.maximum(dt * 1000.0 - switch_ms, 0.0) / lat
        processed = np.minimum(queue + arrivals, capacity)
        q_new = queue + arrivals - processed
        drops = np.maximum(q_new - self.task.queue_capacity, 0.0)
        q_new = np.minimum(q_new, self.task.queue_capacity)
        age_ms = lat * (1.0 + q_new) + switch_ms           # result age incl. waiting
        busy = np.minimum(processed * net / (dt * 1000.0), 1.0)
        busy_w = self.a_busy_w[a]
        if np.any(throttled):
            f, v = np.array(self.p.freq_ghz), np.array(self.p.volt)
            k = self.space.f[a]
            busy_w = np.where(throttled, busy_w * (f[0] / f[k]) * (v[0] / v[k]) ** 2, busy_w)
        power = self.p.idle_power_w + busy * busy_w + self.leak_w(temp)
        decay = np.exp(-dt / self.tau)
        t_new = ambient + (temp - ambient) * decay + self.p.thermal_resistance * power * (1 - decay)
        r = self.recall(a, crowd_norm)
        p_miss = (1.0 - r) ** np.maximum(processed, 0.0)
        fps = np.maximum(processed / dt, 1e-3)
        detect_ms = age_ms + 1000.0 * (self.task.confirm_hits / np.maximum(r, 1e-3) - 1.0) / fps
        return dict(latency_ms=lat, first_frame_ms=lat + switch_ms, age_ms=age_ms, detect_ms=detect_ms, processed=processed, queue=q_new,
                    drops=drops, power_w=power, temp=t_new, recall=r, p_miss=p_miss,
                    map50=self.map50(a, crowd_norm), throughput_fps=processed / dt,
                    energy_j=power * dt)
