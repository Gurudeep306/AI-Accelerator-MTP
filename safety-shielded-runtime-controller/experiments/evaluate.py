"""Full evaluation: every controller x every held-out ETH/UCY sequence (+ synthetic
stress scene) x ambient condition x latency-jitter seed, plus ablations.

    python -m experiments.evaluate --seeds 0 1 2 --workers 2

Writes results/per_episode.csv, results/static_sweep.csv, results/timeseries/*.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np

from config.params import DEFAULT
from control.baselines import FixedController
from control.shield import Shield
from experiments.common import CONDITIONS, CONTROLLERS, RESULTS, group_of, make_controller, make_env
from robot.scenes import SCENE_GROUPS
from sim.runner import run_episode, summarise

TEST_SEQUENCES = [s for ss in SCENE_GROUPS.values() for s in ss]
SYNTHETIC = "equation_s0"
TIMESERIES = [("biwi_hotel", "hot (45 C)", 0), ("students003", "nominal (35 C)", 0)]


def _episode(job):
    ctrl_name, seq, cond, seed, shielded, keep_log = job
    env = make_env(seq, DEFAULT, seed=seed, ambient=CONDITIONS[cond])
    fold = group_of(seq) if seq != SYNTHETIC else "eth"
    ctrl = make_controller(ctrl_name, DEFAULT, env, fold=fold)
    t = time.time()
    log = run_episode(env, ctrl, Shield(DEFAULT) if shielded else None)
    row = {"controller": ctrl_name + ("" if shielded else " [no shield]"), "sequence": seq,
           "group": group_of(seq), "condition": cond, "seed": seed, "shielded": shielded,
           "wall_s": time.time() - t, **summarise(log)}
    return row, (log if keep_log else None)


def _static(job):
    a, seq, cond = job
    env = make_env(seq, DEFAULT, seed=0, ambient=CONDITIONS[cond])
    ctrl = FixedController(DEFAULT)
    ctrl.a = a
    s = summarise(run_episode(env, ctrl, None))
    return {"action": a, "label": env.space.describe(a), "sequence": seq, "condition": cond, **s}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args(argv)
    seqs = TEST_SEQUENCES + [SYNTHETIC]
    jobs = []
    for cond in CONDITIONS:
        for seq in seqs:
            for seed in a.seeds:
                for c in CONTROLLERS:
                    keep = (seq, cond, seed) in TIMESERIES
                    jobs.append((c, seq, cond, seed, True, keep))
    if not a.skip_ablations:
        for cond in CONDITIONS:
            for seq in TEST_SEQUENCES:
                jobs.append(("MPC-oracle", seq, cond, 0, True, False))
                jobs.append(("RL (DQN, scratch)", seq, cond, 0, True, False))
                for c in ("Fixed", "Rule-based", "RL (DQfD, unmasked)"):
                    jobs.append((c, seq, cond, 0, False, False))
    # longest jobs first for better load balance
    jobs.sort(key=lambda j: ("MPC" not in j[0], j[1] != "biwi_hotel"))
    print(f"running {len(jobs)} episodes on {a.workers} workers ...", flush=True)
    t = time.time()
    rows, logs = [], {}
    with Pool(a.workers) as pool:
        for i, (row, log) in enumerate(pool.imap_unordered(_episode, jobs), 1):
            rows.append(row)
            if log is not None:
                logs[(row["controller"], row["sequence"], row["condition"], row["seed"])] = log
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes  ({time.time() - t:.0f} s)", flush=True)
        rows.sort(key=lambda r: (r["condition"], r["sequence"], r["controller"], r["seed"]))
        write_csv(RESULTS / "per_episode.csv", rows)
        for (c, seq, cond, seed), log in logs.items():
            tag = f"{seq}_{cond.split()[0]}_{c.replace(' ', '').replace('(', '').replace(')', '')}"
            write_csv(RESULTS / "timeseries" / f"{tag}.csv", log)
        if not a.skip_sweep:
            sjobs = [(act, seq, cond) for cond in CONDITIONS for seq in TEST_SEQUENCES
                     for act in range(make_env(seq).space.n)]
            print(f"static operating-point sweep: {len(sjobs)} episodes ...", flush=True)
            srows = pool.map(_static, sjobs, chunksize=20)
            write_csv(RESULTS / "static_sweep.csv", srows)
    print(f"done in {(time.time() - t) / 60:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
