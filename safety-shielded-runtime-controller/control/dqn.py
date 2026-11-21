"""Shielded deep RL controller: Double DQN with a branching-dueling head and n-step returns, in NumPy.

Two training modes:
  * DQN from scratch;
  * DQfD (Hester et al., AAAI 2018): pre-train on MPC demonstrations collected on
    the *training* scenes (TD loss + large-margin imitation loss), then keep
    learning online with the demonstrations kept in the replay mix. The result
    is a policy that decides in ~0.1 ms instead of the MPC's ~13 ms.

* Safe exploration by action masking: the agent only ever chooses among the
  shield-admissible operating points (a "preemptive shield", Alshiekh et al.,
  AAAI 2018), both when acting and when bootstrapping the Double-DQN target.
* Reward = -stage_cost (the same objective the MPC minimises).
* Trained on the ETH/UCY training sequences of a leave-one-scene-out fold plus
  synthetic equation-based scenes, with domain randomisation of ambient and
  initial temperature; evaluated on the held-out scene.
No deep-learning framework is needed (CPU-only, ~1 ms per training step).
"""
from __future__ import annotations

import pickle
from collections import deque

import numpy as np

from accel.model import ActionSpace
from config.params import Params
from control.baselines import Controller
from control.shield import Shield

OBS_DIM = 17


def featurise(obs, space: ActionSpace, P: Params) -> np.ndarray:
    pc = obs.percept
    return np.concatenate([
        [(obs.temp - obs.ambient) / 50.0, (obs.temp - P.accel.temp_soft_limit) / 10.0, float(obs.throttled),
         obs.queue / P.task.queue_capacity, pc.clearance_m / 8.0, pc.min_clearance_forecast / 8.0,
         min(pc.density, 2.0), pc.crowd_norm, min(pc.n_view, 30) / 20.0, float(pc.hazard),
         np.clip(pc.closing_speed, -2, 2) / 2.0, obs.v_robot / P.robot.nominal_speed, (obs.ambient - 35.0) / 10.0],
        space.features(obs.prev_action)]).astype(np.float32)


class BranchingDuelingMLP:
    """Q(s,a) = V(s) + sum_d A_d(s, a_d) + A_joint(s, a)   (all advantages centred).

    The factored branches (frequency / precision / sparsity / exit, 14 outputs,
    action-branching idea of Tavakoli et al., AAAI 2018) share what is learnt
    about one knob across all 135 joint points; the joint head captures
    interactions. Q is still formed for every joint action, so masking is exact.
    """

    def __init__(self, n_in, space: ActionSpace, hidden=128, seed=0):
        r = np.random.default_rng(seed)
        he = lambda i, o: (r.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(np.float32)
        sizes = [len(space.p.freq_ghz), len(space.p.precisions), len(space.p.sparsities), len(space.p.exits)]
        knobs = [space.f, space.prec, space.sp, space.exit]
        self.Mc = np.zeros((space.n, sum(sizes)), np.float32)
        off = 0
        for size, k in zip(sizes, knobs):
            self.Mc[:, off:off + size] = -1.0 / size
            self.Mc[np.arange(space.n), off + k] += 1.0
            off += size
        n_out, n_fac = space.n, sum(sizes)
        self.p = {"W1": he(n_in, hidden), "b1": np.zeros(hidden, np.float32),
                  "W2": he(hidden, hidden), "b2": np.zeros(hidden, np.float32),
                  "Wv": he(hidden, 1) * 0.1, "bv": np.zeros(1, np.float32),
                  "Wf": he(hidden, n_fac) * 0.1, "bf": np.zeros(n_fac, np.float32),
                  "Wa": he(hidden, n_out) * 0.01, "ba": np.zeros(n_out, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]; h1 = np.maximum(z1, 0)
        z2 = h1 @ p["W2"] + p["b2"]; h2 = np.maximum(z2, 0)
        val = h2 @ p["Wv"] + p["bv"]
        fac = h2 @ p["Wf"] + p["bf"]
        adv = h2 @ p["Wa"] + p["ba"]
        q = val + fac @ self.Mc.T + adv - adv.mean(1, keepdims=True)
        return q, (x, z1, h1, z2, h2)

    def backward(self, cache, dq):
        x, z1, h1, z2, h2 = cache
        p = self.p
        dval = dq.sum(1, keepdims=True)
        dfac = dq @ self.Mc
        dadv = dq - dq.mean(1, keepdims=True)
        g = {"Wv": h2.T @ dval, "bv": dval.sum(0), "Wf": h2.T @ dfac, "bf": dfac.sum(0),
             "Wa": h2.T @ dadv, "ba": dadv.sum(0)}
        dh2 = dval @ p["Wv"].T + dfac @ p["Wf"].T + dadv @ p["Wa"].T
        dz2 = dh2 * (z2 > 0)
        g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
        g["W1"], g["b1"] = x.T @ dz1, dz1.sum(0)
        return g

    def adam(self, g, lr, clip=10.0, b1=0.9, b2=0.999, eps=1e-8):
        norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in self.p:
            gk = g[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other, tau=1.0):
        for k in self.p:
            self.p[k] = (1 - tau) * self.p[k] + tau * other.p[k]


class DQNAgent:
    def __init__(self, params: Params, seed=0, gamma=0.92, n_step=3, lr=2.5e-4, batch=128,
                 buffer=100_000, tau=0.002, reward_scale=0.5):
        self.P = params
        self.space = ActionSpace(params.accel)
        self.nA = self.space.n
        self.q = BranchingDuelingMLP(OBS_DIM, self.space, seed=seed)
        self.qt = BranchingDuelingMLP(OBS_DIM, self.space, seed=seed)
        self.qt.copy_from(self.q)
        self.gamma, self.n_step, self.lr, self.batch, self.tau = gamma, n_step, lr, batch, tau
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.cap, self.ptr, self.size = buffer, 0, 0
        self.S = np.zeros((buffer, OBS_DIM), np.float32)
        self.A = np.zeros(buffer, np.int64)
        self.R = np.zeros(buffer, np.float32)
        self.S2 = np.zeros((buffer, OBS_DIM), np.float32)
        self.M2 = np.zeros((buffer, self.nA), bool)
        self.D = np.zeros(buffer, np.float32)
        self.demo = None                 # dict of arrays (S, A, R, S2, M2, D, M)
        self.demo_frac, self.margin, self.lambda_e = 0.25, 0.8, 1.0

    def set_demos(self, demo, demo_frac=0.25, margin=0.8, lambda_e=1.0):
        self.demo, self.demo_frac, self.margin, self.lambda_e = demo, demo_frac, margin, lambda_e

    def greedy(self, x, mask=None):
        q, _ = self.q.forward(x[None])
        q = q[0]
        if mask is not None and mask.any():
            q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def store(self, s, a, r, s2, m2, disc):
        """disc = gamma^k bootstrap factor (the task is continuing: window ends are truncations)."""
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.M2[i], self.D[i] = s, a, r, s2, m2, disc
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self, demo_only=False):
        nd = 0
        if self.demo is not None:
            nd = self.batch if (demo_only or self.size == 0) else int(self.batch * self.demo_frac)
        idx = self.rng.integers(0, self.size, self.batch - nd) if nd < self.batch else np.empty(0, int)
        parts = [(self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.M2[idx], self.D[idx])]
        if nd:
            j = self.rng.integers(0, len(self.demo["A"]), nd)
            dm = self.demo
            parts.append((dm["S"][j], dm["A"][j], dm["R"][j], dm["S2"][j], dm["M2"][j], dm["D"][j]))
        s, a, r, s2, m2, d = (np.concatenate(x) for x in zip(*parts))
        B, rows = len(a), np.arange(len(a))
        q2_online, _ = self.q.forward(s2)
        a_star = np.argmax(np.where(m2, q2_online, -np.inf), 1)
        q2_target, _ = self.qt.forward(s2)
        target = r + d * q2_target[rows, a_star]
        q, cache = self.q.forward(s)
        td = q[rows, a] - target
        dq = np.zeros_like(q)
        dq[rows, a] = np.clip(td, -1.0, 1.0) / B                     # Huber TD gradient
        if nd:                                                          # large-margin imitation loss
            k = rows[B - nd:]
            ms = self.demo["M"][j]
            qm = np.where(ms, q[k] + self.margin, -np.inf)
            qm[np.arange(nd), a[k]] -= self.margin                     # no margin on the expert action
            a_max = np.argmax(qm, 1)
            g = self.lambda_e / B
            np.add.at(dq, (k, a_max), g)
            np.add.at(dq, (k, a[k]), -g)
        self.q.adam(self.q.backward(cache, dq), self.lr)
        self.qt.copy_from(self.q, self.tau)
        return float(np.mean(np.abs(td)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q.p, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q.p = pickle.load(f)
        return self


def collect_demos(env_factory, expert_factory, params: Params, steps: int, seed: int = 0, gamma=0.92, n_step=3):
    """Roll out the expert (MPC) through the shield and store n-step transitions."""
    rng = np.random.default_rng(seed)
    space, shield = ActionSpace(params.accel), Shield(params)
    out = {k: [] for k in ("S", "A", "R", "S2", "M2", "D", "M")}
    t = 0
    while t < steps:
        env = env_factory(rng)
        expert = expert_factory(env)
        obs, done, nbuf = env.reset(), False, deque()
        mask, _ = shield.admissible(obs)
        x = featurise(obs, space, params)
        while not done and t < steps:
            action, _ = shield.filter(obs, expert.act(obs))
            obs2, cost, done, _ = env.step(action)
            mask2, _ = shield.admissible(obs2)
            x2 = featurise(obs2, space, params)
            nbuf.append((x, action, -cost * 0.5, mask))
            if len(nbuf) == n_step or done:
                while nbuf:
                    G = sum(gamma ** i * tr[2] for i, tr in enumerate(nbuf))
                    for k, v in zip(("S", "A", "R", "S2", "M2", "D", "M"),
                                    (nbuf[0][0], nbuf[0][1], G, x2, mask2, gamma ** len(nbuf), nbuf[0][3])):
                        out[k].append(v)
                    nbuf.popleft()
                    if not done:
                        break
            x, obs, mask = x2, obs2, mask2
            t += 1
    return {k: np.array(v) for k, v in out.items()}


def evaluate_agent(agent, val_envs, params):
    """Mean stage cost of the greedy masked policy on fixed validation windows."""
    from sim.runner import run_episode
    ctrl = DQNController(params, agent)
    return float(np.mean([np.mean([r["cost"] for r in run_episode(make(), ctrl)]) for make in val_envs]))


def train_dqn(env_factory, params: Params, total_steps=60_000, seed=0, warmup=2_000,
              eps_start=1.0, eps_end=0.05, eps_frac=0.5, log_every=10_000, verbose=True,
              val_envs=None, val_from=0.3, demos=None, pretrain_updates=20_000):
    """env_factory(rng) -> a fresh CrowdAcceleratorEnv (random scene/window/conditions).

    If `val_envs` (list of zero-arg env constructors built from *training*
    scenes) is given, the greedy policy is validated every `log_every` steps
    after `val_from * total_steps` and the best checkpoint is returned.
    """
    best = (np.inf, None)
    rng = np.random.default_rng(seed)
    agent = DQNAgent(params, seed=seed)
    history = []
    if demos is not None:
        agent.set_demos(demos)
        for i in range(pretrain_updates):
            agent.update(demo_only=True)
        if val_envs:
            val = evaluate_agent(agent, val_envs, params)
            best = (val, {k: v.copy() for k, v in agent.q.p.items()})
            history.append((0, float("nan"), val))
            if verbose:
                print(f"    [dqfd seed={seed}] pre-trained on {len(demos['A'])} demo transitions: val cost {val:7.4f}",
                      flush=True)
    shield = Shield(params)
    space = agent.space
    steps, ep_returns = 0, []
    while steps < total_steps:
        env = env_factory(rng)
        obs = env.reset()
        mask, _ = shield.admissible(obs)
        x = featurise(obs, space, params)
        nbuf, done, ret = deque(), False, 0.0
        while not done and steps < total_steps:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * steps / (eps_frac * total_steps))
            if rng.random() < eps:
                cand = np.flatnonzero(mask) if mask.any() else np.arange(space.n)
                proposal = int(rng.choice(cand))
            else:
                proposal = agent.greedy(x, mask)
            action, intervened = shield.filter(obs, proposal)
            obs2, cost, done, info = env.step(action)
            r = -(cost + params.cost.shield_intervention * intervened) * agent.reward_scale
            ret += -cost
            mask2, _ = shield.admissible(obs2)
            x2 = featurise(obs2, space, params)
            nbuf.append((x, action, r))
            if len(nbuf) == agent.n_step or done:
                while nbuf:
                    G = sum(agent.gamma ** i * t[2] for i, t in enumerate(nbuf))
                    agent.store(nbuf[0][0], nbuf[0][1], G, x2, mask2, agent.gamma ** len(nbuf))
                    nbuf.popleft()
                    if not done:
                        break
            x, obs, mask = x2, obs2, mask2
            steps += 1
            if agent.size >= warmup or (demos is not None and agent.size >= agent.batch):
                agent.update()
            if steps % log_every == 0:
                recent = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
                val = np.nan
                if val_envs and steps >= val_from * total_steps:
                    val = evaluate_agent(agent, val_envs, params)
                    if val < best[0]:
                        best = (val, {k: v.copy() for k, v in agent.q.p.items()})
                history.append((steps, recent, val))
                if verbose:
                    print(f"    [{'dqfd' if demos is not None else 'dqn'} seed={seed}] step {steps:>7}  eps={eps:.2f}  "
                          f"ep return {recent:8.2f}  val cost {val:7.4f}", flush=True)
        ep_returns.append(ret)
    if best[1] is not None:
        agent.q.p = best[1]
    return agent, history


class DQNController(Controller):
    def __init__(self, params: Params, agent: DQNAgent, masked: bool = True):
        self.P, self.agent, self.masked = params, agent, masked
        self.shield = Shield(params) if masked else None
        self.name = "RL" if masked else "RL (unmasked)"

    def act(self, obs):
        x = featurise(obs, self.agent.space, self.P)
        mask = self.shield.admissible(obs)[0] if self.masked else None
        return self.agent.greedy(x, mask)
