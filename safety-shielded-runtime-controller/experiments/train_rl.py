"""Train the shielded RL agents, one per leave-one-scene-out fold (ETH/UCY protocol).

    python -m experiments.train_rl --kind dqfd --steps 100000 --workers 2   # main agent
    python -m experiments.train_rl --kind dqn  --steps 100000 --workers 2   # from-scratch ablation

DQfD first collects MPC demonstrations on every sequence (cached in
models/demos/); each fold then uses only the demonstrations of its *training*
sequences, so the held-out scene is never seen.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from config.params import DEFAULT
from control.dqn import collect_demos, train_dqn
from control.mpc import MPCController
from experiments.common import MODELS, make_env, random_window_factory, rl_model_path, training_sequences
from robot.scenes import ALL_SEQUENCES, SCENE_GROUPS

DEMO_DIR = MODELS / "demos"
VAL_SEQS = ("crowds_zara03", "uni_examples")      # training-only sequences -> no test leakage


def _collect(job):
    seq, steps = job
    path = DEMO_DIR / f"{seq}.npz"
    if not path.exists():
        d = collect_demos(random_window_factory([seq], DEFAULT), lambda env: MPCController(DEFAULT, env=env),
                          DEFAULT, steps, seed=hash(seq) % 10_000)
        np.savez_compressed(path, **d)
    return seq


def _load_demos(seqs):
    parts = [dict(np.load(DEMO_DIR / f"{s}.npz")) for s in seqs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _train(job):
    fold, seed, steps, kind = job
    seqs = training_sequences(fold)
    val = [(lambda s=s, a=a: make_env(s, DEFAULT, seed=7, ambient=a, max_steps=250))
           for s in VAL_SEQS for a in (35.0, 45.0)]
    kw = {}
    if kind == "dqfd":
        kw = dict(demos=_load_demos(seqs), eps_start=0.3, eps_end=0.02, warmup=0)
    t = time.time()
    agent, hist = train_dqn(random_window_factory(seqs, DEFAULT), DEFAULT, total_steps=steps, seed=seed,
                            val_envs=val, log_every=max(steps // 10, 1000), **kw)
    agent.save(rl_model_path(fold, seed, kind))
    return fold, seed, kind, time.time() - t, hist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["dqfd", "dqn"], default="dqfd")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--demo-steps", type=int, default=3_000, help="MPC demonstration steps per sequence")
    ap.add_argument("--folds", nargs="*", default=list(SCENE_GROUPS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    MODELS.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        if a.kind == "dqfd":
            DEMO_DIR.mkdir(parents=True, exist_ok=True)
            seqs = ALL_SEQUENCES + ["equation_s100", "equation_s101"]
            t = time.time()
            for s in pool.imap_unordered(_collect, [(s, a.demo_steps) for s in seqs]):
                print(f"  demos ready: {s}  ({time.time() - t:.0f} s)", flush=True)
        out = pool.map(_train, [(f, s, a.steps, a.kind) for f in a.folds for s in a.seeds], chunksize=1)
    log_path = MODELS / "training_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for f, s, k, t, h in out:
        log[f"{k}_{f}_seed{s}"] = {"seconds": t, "history": h}
        print(f"{k} fold {f:6s} seed {s}: {t / 60:.1f} min -> {rl_model_path(f, s, k).name}")
    log_path.write_text(json.dumps(log, indent=2, default=float))


if __name__ == "__main__":
    main()
