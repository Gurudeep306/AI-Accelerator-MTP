"""Shared experiment setup: folds, scene cache, controller factory."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from config.params import DEFAULT, Params
from control.baselines import FixedController, RuleBasedController
from control.dqn import DQNAgent, DQNController
from control.mpc import MPCController
from robot.scenes import SCENE_GROUPS, TRAIN_ONLY, equation_scene, load_eth_ucy
from sim.env import CrowdAcceleratorEnv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CONDITIONS = {"nominal (35 C)": 35.0, "hot (45 C)": 45.0}
CONTROLLERS = ["Fixed", "Rule-based", "MPC", "RL (DQfD)"]


@lru_cache(maxsize=None)
def scene(name: str):
    if name.startswith("equation"):
        return equation_scene(seed=int(name.split("_s")[-1]))
    return load_eth_ucy(name)


def training_sequences(test_group: str):
    """Leave-one-scene-out: every ETH/UCY sequence outside the test group."""
    seqs = [s for g, ss in SCENE_GROUPS.items() if g != test_group for s in ss] + TRAIN_ONLY
    return seqs + ["equation_s100", "equation_s101"]     # synthetic augmentation


def group_of(sequence: str) -> str:
    for g, ss in SCENE_GROUPS.items():
        if sequence in ss:
            return g
    return "synthetic"


def rl_model_path(fold: str, seed: int = 0, kind: str = "dqfd") -> Path:
    return MODELS / f"{kind}_fold-{fold}_seed{seed}.pkl"


def make_controller(name: str, params: Params, env, fold: str | None = None, rl_seed: int = 0):
    if name == "Fixed":
        return FixedController(params)
    if name == "Rule-based":
        return RuleBasedController(params)
    if name == "MPC":
        return MPCController(params, env=env)
    if name == "MPC-oracle":
        return MPCController(params, env=env, oracle=True)
    if name.startswith("RL"):
        kind = "dqn" if "scratch" in name else "dqfd"
        agent = DQNAgent(params).load(rl_model_path(fold, rl_seed, kind))
        ctrl = DQNController(params, agent, masked="unmasked" not in name)
        ctrl.name = name
        return ctrl
    raise ValueError(name)


def make_env(seq: str, params: Params = DEFAULT, seed: int = 0, ambient: float = 35.0, **kw):
    return CrowdAcceleratorEnv(scene(seq), params, seed=seed, ambient_c=ambient, **kw)


def random_window_factory(seqs, params: Params, window: int = 300):
    scenes = [scene(s) for s in seqs]

    def factory(rng: np.random.Generator):
        sc = scenes[rng.integers(len(scenes))]
        start = int(rng.integers(0, max(1, len(sc) - window)))
        return CrowdAcceleratorEnv(sc, params, seed=int(rng.integers(1 << 30)),
                                   ambient_c=float(rng.uniform(30.0, 48.0)),
                                   initial_temp_c=float(rng.uniform(40.0, 72.0)),
                                   start_k=start, max_steps=window)
    return factory
