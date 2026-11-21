"""AI Accelerator Controller for a crowd-navigating robot — command-line entry point.

    python main.py demo                       # one scene, all controllers, quick look (~1 min)
    python main.py train   [--steps 150000]   # shielded DQN, one agent per ETH/UCY fold
    python main.py evaluate [--seeds 0 1 2]   # full closed-loop evaluation + ablations + static sweep
    python main.py report                     # tables (results/RESULTS.md) and figures (results/plots)
    python main.py all                        # train -> evaluate -> report
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def demo(sequence: str, ambient: float):
    from config.params import DEFAULT
    from control.shield import Shield
    from experiments.common import CONTROLLERS, group_of, make_controller, make_env, rl_model_path
    from sim.runner import run_episode, summarise

    print(f"Demo: {sequence}, ambient {ambient} C")
    keys = [("mean_cost", "cost"), ("energy_per_m_j", "J/m"), ("hazard_time_to_detect_ms", "t-detect ms"),
            ("deadline_miss_rate", "DL miss"), ("max_temp_c", "Tmax"), ("throttle_s", "throttle s"),
            ("safety_violations", "violations"), ("switches_per_min", "sw/min")]
    print(f"{'controller':12s}" + "".join(f"{h:>13s}" for _, h in keys))
    for name in CONTROLLERS:
        if name.startswith("RL") and not rl_model_path(group_of(sequence), 0).exists():
            print(f"{name:12s}  (no trained agent yet: run `python main.py train`)")
            continue
        env = make_env(sequence, DEFAULT, seed=0, ambient=ambient)
        s = summarise(run_episode(env, make_controller(name, DEFAULT, env, fold=group_of(sequence)), Shield(DEFAULT)))
        print(f"{name:12s}" + "".join(f"{float(s[k]):13.3f}" for k, _ in keys))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["demo", "train", "evaluate", "report", "all"])
    ap.add_argument("--steps", type=int, default=150_000, help="RL training steps per fold")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="evaluation jitter seeds")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sequence", default="crowds_zara01", help="demo scene")
    ap.add_argument("--ambient", type=float, default=45.0, help="demo ambient temperature (C)")
    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    if a.command == "demo":
        demo(a.sequence, a.ambient)
    if a.command in ("train", "all"):
        from experiments import train_rl
        train_rl.main(["--steps", str(a.steps), "--workers", str(a.workers)])
    if a.command in ("evaluate", "all"):
        from experiments import evaluate
        evaluate.main(["--workers", str(a.workers), "--seeds", *map(str, a.seeds)])
    if a.command in ("report", "all"):
        from experiments import report
        report.main()


if __name__ == "__main__":
    sys.exit(main())
