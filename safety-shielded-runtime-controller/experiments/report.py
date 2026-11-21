"""Aggregate results into tables (CSV + Markdown) and publication-style figures.

    python -m experiments.report
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import CONDITIONS, CONTROLLERS, MODELS, RESULTS  # noqa: E402
from robot.scenes import load_eth_ucy  # noqa: E402

PLOTS = RESULTS / "plots"
# validated categorical order (blue, orange, aqua, yellow, magenta, green); fixed per controller
COLORS = {"Fixed": "#2a78d6", "Rule-based": "#eb6834", "MPC": "#1baf7a", "RL (DQfD)": "#eda100",
          "MPC-oracle": "#e87ba4", "RL (DQN, scratch)": "#008300", "RL (DQfD, unmasked)": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})


def read_csv(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = float(v == "True")
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


# ------------------------------------------------------------------ aggregation
def aggregate(rows):
    """Pool episodes: rates weighted by steps (or by hazard steps / distance)."""
    S = lambda k: sum(r[k] for r in rows)
    W = lambda k, w="steps": sum(r[k] * r[w] for r in rows if not np.isnan(r[k])) / max(
        sum(r[w] for r in rows if not np.isnan(r[k])), 1e-9)
    dur, dist = S("duration_s"), S("distance_m")
    return {
        "episodes": len(rows),
        "mean_cost": W("mean_cost"),
        "energy_per_m_j": S("energy_j") / dist,
        "mean_power_w": W("mean_power_w"),
        "mean_speed_mps": dist / dur,
        "mean_latency_ms": W("mean_latency_ms"),
        "p95_latency_ms": W("p95_latency_ms"),
        "deadline_miss_pct": 100 * W("deadline_miss_rate"),
        "mean_map50": W("mean_map50"),
        "hazard_map50": W("hazard_map50", "hazard_steps"),
        "hazard_time_to_detect_ms": W("hazard_time_to_detect_ms", "hazard_steps"),
        "hazard_miss_prob_pct": 100 * W("hazard_miss_prob", "hazard_steps"),
        "mean_throughput_fps": W("mean_throughput_fps"),
        "mean_queue": W("mean_queue"),
        "frame_drop_pct": 100 * W("frame_drop_rate"),
        "mean_temp_c": W("mean_temp_c"),
        "max_temp_c": max(r["max_temp_c"] for r in rows),
        "over_soft_pct": 100 * S("time_over_soft_s") / dur,
        "throttled_pct": 100 * S("throttle_s") / dur,
        "violations_per_km": S("safety_violations") / (dist / 1000),
        "frontal_contacts_per_km": S("frontal_contacts") / (dist / 1000),
        "switches_per_min": sum(r["switches_per_min"] * r["duration_s"] / 60 for r in rows) / (dur / 60),
        "shield_intervention_pct": 100 * W("shield_intervention_rate"),
    }


def bootstrap_ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_vs_fixed(rows, ctrl, cond, metric, relative=False):
    """Paired episode differences controller - Fixed (same sequence, condition, seed)."""
    base = {(r["sequence"], r["seed"]): r for r in rows if r["controller"] == "Fixed" and r["condition"] == cond}
    d = []
    for r in rows:
        if r["controller"] == ctrl and r["condition"] == cond and (r["sequence"], r["seed"]) in base:
            b = base[(r["sequence"], r["seed"])][metric]
            d.append(100 * (r[metric] - b) / b if relative else r[metric] - b)
    return bootstrap_ci(d) if d else (np.nan,) * 3


# ---------------------------------------------------------------------- tables
TABLE_COLS = [("mean_cost", "Cost ↓", "{:+.3f}"), ("energy_per_m_j", "Energy J/m ↓", "{:.2f}"),
              ("mean_power_w", "Power W ↓", "{:.2f}"), ("mean_speed_mps", "Speed m/s ↑", "{:.3f}"),
              ("hazard_time_to_detect_ms", "Hazard t-detect ms ↓", "{:.0f}"),
              ("hazard_miss_prob_pct", "Hazard miss % ↓", "{:.2f}"),
              ("deadline_miss_pct", "Deadline miss % ↓", "{:.2f}"), ("max_temp_c", "Max T °C ↓", "{:.1f}"),
              ("throttled_pct", "Throttled % ↓", "{:.2f}"), ("violations_per_km", "Violations /km ↓", "{:.2f}"),
              ("switches_per_min", "Switches /min", "{:.1f}"),
              ("shield_intervention_pct", "Shield interv. %", "{:.1f}")]


def md_table(summary, controllers, cond):
    head = "| Controller | " + " | ".join(c[1] for c in TABLE_COLS) + " |"
    sep = "|---|" + "---|" * len(TABLE_COLS)
    lines = [head, sep]
    for c in controllers:
        if (c, cond) in summary:
            s = summary[(c, cond)]
            lines.append(f"| {c} | " + " | ".join(f.format(s[k]) for k, _, f in TABLE_COLS) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- figures
def _bars(ax, summary, metric, conds, ctrls, title, ylabel):
    width = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        vals = [summary[(c, cd)][metric] if (c, cd) in summary else np.nan for cd in conds]
        x = np.arange(len(conds)) + (i - (len(ctrls) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, color=COLORS[c], label=c)
    ax.set_xticks(range(len(conds)), [c.split(" (")[0] for c in conds])
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def figure_panel(summary, metrics, fname, suptitle, ctrls=CONTROLLERS):
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, (m, title, yl) in zip(np.atleast_1d(axes), metrics):
        _bars(ax, summary, m, conds, ctrls, title, yl)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ctrls), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(suptitle, y=1.15, fontsize=12, color=INK)
    fig.savefig(PLOTS / fname)
    plt.close(fig)


def figure_timeseries(path_glob, fname, title):
    files = sorted(RESULTS.glob(f"timeseries/{path_glob}*.csv"))
    if not files:
        return
    data = {}
    for f in files:
        rows = read_csv(f)
        name = next(c for c in CONTROLLERS if c.replace(" ", "").replace("(", "").replace(")", "") in f.stem)
        data[name] = rows
    panels = [("temp", "Temperature (°C)"), ("power_w", "Power (W)"), ("detect_ms", "Time-to-detect (ms)"),
              ("v", "Robot speed (m/s)")]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 11), sharex=True)
    ref = next(iter(data.values()))
    t = np.array([r["t"] for r in ref])
    axes[0].fill_between(t, 0, 8 * np.array([r["hazard"] for r in ref]), step="mid", color="#e4e3df",
                         label="hazard: in-path pedestrian < 3 m")
    axes[0].plot(t, np.minimum([r["clearance"] for r in ref], 8), color=MUTED, lw=1, label="in-path clearance (m)")
    axes[0].set_ylabel("Clearance (m)")
    axes[0].set_ylim(0, 8.5)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    for ax, (k, yl) in zip(axes[1:], panels):
        for c in [c for c in CONTROLLERS if c in data]:
            rows = data[c]
            tt = [r["t"] for r in rows]
            y = np.array([r[k] for r in rows])
            if k in ("power_w", "detect_ms"):          # 5-step moving average for legibility
                y = np.convolve(y, np.ones(5) / 5, mode="same")
            ax.plot(tt, y, color=COLORS[c], lw=1.4, label=c)
        ax.set_ylabel(yl)
    axes[1].axhline(80, color=MUTED, lw=1, ls="--")
    axes[1].axhline(85, color="#e34948", lw=1, ls=":")
    axes[1].text(t[-1], 80.5, "soft limit", ha="right", va="bottom", fontsize=8, color=MUTED)
    axes[1].text(t[-1], 85.5, "throttle", ha="right", va="bottom", fontsize=8, color=MUTED)
    axes[1].legend(loc="lower right", ncol=4, frameon=False, fontsize=8)
    axes[-1].set_xlabel("Scene time (s)")
    fig.suptitle(title, y=0.995, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(PLOTS / fname)
    plt.close(fig)


def figure_pareto(summary, sweep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, cond in zip(axes, CONDITIONS):
        pts = defaultdict(list)
        for r in sweep:
            if r["condition"] == cond:
                pts[r["label"]].append(r)
        xs, ys = [], []
        for lab, rr in pts.items():
            agg = aggregate(rr)
            xs.append(agg["energy_per_m_j"]); ys.append(agg["mean_cost"])
        ax.scatter(xs, ys, s=14, color="#b9b8b2", label="135 static operating points", zorder=1)
        for c, size in zip(CONTROLLERS, (90, 90, 200, 70)):    # MPC drawn larger: RL sits on top of it
            if (c, cond) in summary:
                s = summary[(c, cond)]
                ax.scatter(s["energy_per_m_j"], s["mean_cost"], s=size, color=COLORS[c], edgecolor="white",
                           linewidth=2, zorder=3, label=c)
        s = summary[("MPC", cond)]
        ax.annotate("MPC and RL (DQfD)\nnearly coincide", (s["energy_per_m_j"], s["mean_cost"]), xytext=(14, -22),
                    textcoords="offset points", fontsize=8, color=MUTED, ha="left")
        best = int(np.argmin(ys))
        ax.annotate("best static point\n(chosen in hindsight)", (xs[best], ys[best]), xytext=(25, 25),
                    textcoords="offset points", fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED))
        ax.set_title(cond, loc="left", fontsize=10)
        ax.set_xlabel("Energy per metre travelled (J/m)")
        ax.set_ylabel("Mean stage cost (lower is better)")
        ax.set_ylim(min(ys) - 0.15, np.percentile(ys, 70))
        ax.set_xlim(min(xs) * 0.9, np.percentile(xs, 90))
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Adaptive controllers vs every static operating point (held-out ETH/UCY scenes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "pareto_static_vs_adaptive.png")
    plt.close(fig)
    return pts


def figure_action_usage():
    files = sorted(RESULTS.glob("timeseries/biwi_hotel_hot*.csv"))
    if not files:
        return
    from accel.model import ActionSpace
    from config.params import DEFAULT
    sp = ActionSpace(DEFAULT.accel)
    knobs = [("Frequency level", sp.f, ["f1", "f2", "f3", "f4", "f5"]),
             ("Precision", sp.prec, list(DEFAULT.accel.precisions)),
             ("Sparsity", sp.sp, [str(s) for s in DEFAULT.accel.sparsities]),
             ("Anytime exit", sp.exit, list(DEFAULT.accel.exits))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    data = {}
    for c in CONTROLLERS:
        for f in files:
            if c.replace(" ", "").replace("(", "").replace(")", "") in f.stem:
                data[c] = np.array([int(r["action"]) for r in read_csv(f)])
    for ax, (title, arr, labels) in zip(axes, knobs):
        width = 0.8 / len(data)
        for i, (c, acts) in enumerate(data.items()):
            frac = np.bincount(arr[acts], minlength=len(labels)) / len(acts)
            ax.bar(np.arange(len(labels)) + (i - (len(data) - 1) / 2) * width, 100 * frac, width * 0.92,
                   color=COLORS[c], label=c)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("% of steps")
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.suptitle("Which knobs each controller uses (BIWI Hotel, hot ambient)", y=1.18, fontsize=12)
    fig.savefig(PLOTS / "knob_usage_hotel_hot.png")
    plt.close(fig)


def figure_training():
    p = MODELS / "training_log.json"
    if not p.exists():
        return
    log = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    pal = {"eth": "#2a78d6", "hotel": "#eb6834", "univ": "#1baf7a", "zara1": "#eda100", "zara2": "#e87ba4"}
    for k, v in sorted(log.items()):
        kind, fold = k.split("_")[0], k.split("_")[1]
        ls = "-" if kind == "dqfd" else "--"
        h = np.array([[x if x is not None else np.nan for x in row] for row in v["history"]], float)
        axes[0].plot(h[:, 0] / 1000, h[:, 1], color=pal[fold], lw=1.5, ls=ls, label=fold if kind == "dqfd" else None)
        axes[1].plot(h[:, 0] / 1000, h[:, 2], color=pal[fold], lw=1.5, ls=ls, marker="o", ms=3)
    axes[0].set_title("Training episode return (mean of last 20)", loc="left", fontsize=10)
    axes[1].set_title("Validation cost of greedy masked policy (lower is better; best kept)", loc="left", fontsize=10)
    axes[1].text(100, -0.265, "DQN from scratch (dashed)", ha="right", fontsize=8, color=MUTED)
    axes[1].text(100, -0.335, "DQfD: MPC demos + RL (solid)", ha="right", fontsize=8, color=MUTED)
    for ax in axes:
        ax.set_xlabel("Environment steps (thousands)")
    axes[0].legend(frameon=False, fontsize=8, title="held-out fold", ncol=1)
    fig.tight_layout()
    fig.savefig(PLOTS / "rl_training_curves.png")
    plt.close(fig)


def figure_scenes():
    names = ["biwi_eth", "biwi_hotel", "students003", "crowds_zara01"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, n in zip(axes, names):
        sc = load_eth_ucy(n)
        pos = np.concatenate([p for _, p, _ in sc.frames if len(p)])
        ax.scatter(pos[:, 0], pos[:, 1], s=1, color="#9fbfe8", alpha=0.35, rasterized=True)
        ax.plot(*np.stack([sc.path_start, sc.path_end]).T, color="#eb6834", lw=2.5, label="robot path")
        ax.set_title(f"{n}  ({sc.n_pedestrians} pedestrians, {len(sc) * sc.dt / 60:.1f} min)", fontsize=9, loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("ETH/UCY crowds replayed around the robot (blue = all recorded pedestrian positions)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "scenes_eth_ucy.png")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RESULTS / "per_episode.csv")
    test = [r for r in rows if r["group"] != "synthetic"]
    synth = [r for r in rows if r["group"] == "synthetic"]
    summary, synth_summary = {}, {}
    for c in sorted({r["controller"] for r in rows}):
        for cond in CONDITIONS:
            sel = [r for r in test if r["controller"] == c and r["condition"] == cond]
            if sel:
                summary[(c, cond)] = aggregate(sel)
            sel = [r for r in synth if r["controller"] == c and r["condition"] == cond]
            if sel:
                synth_summary[(c, cond)] = aggregate(sel)
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(next(iter(summary.values())))
        w = csv.writer(f)
        w.writerow(["controller", "condition", "scenes"] + keys)
        for (c, cond), s in sorted(summary.items()):
            w.writerow([c, cond, "ETH/UCY held-out"] + [s[k] for k in keys])
        for (c, cond), s in sorted(synth_summary.items()):
            w.writerow([c, cond, "synthetic stress"] + [s[k] for k in keys])

    # paired improvements vs Fixed
    paired = {}
    for c in CONTROLLERS[1:]:
        for cond in CONDITIONS:
            paired[(c, cond)] = {"cost": paired_vs_fixed(test, c, cond, "mean_cost"),
                                 "energy": paired_vs_fixed(test, c, cond, "energy_j", relative=True),
                                 "ttd": paired_vs_fixed(test, c, cond, "hazard_time_to_detect_ms", relative=True)}
    # per-scene cost table
    seqs = sorted({r["sequence"] for r in test})
    md = ["# Results", "",
          "All numbers are from the closed-loop simulator replaying the held-out ETH/UCY crowds "
          "(6 sequences, leave-one-scene-out for the RL agent). Accelerator power, temperature and latency are "
          "model estimates anchored to the measured Anytime-YOLO / YOLO11n profiles; they are not hardware measurements.", ""]
    for cond in CONDITIONS:
        n_ep = summary[("Fixed", cond)]["episodes"]
        md += [f"## Main comparison — {cond} ({n_ep} episodes per controller)", "",
               md_table(summary, CONTROLLERS, cond), ""]
    md += ["## Paired improvement vs Fixed (mean over episodes, bootstrap 95% CI)", "",
           "| Controller | Condition | Δ cost (neg = better) | Δ energy % | Δ hazard time-to-detect % |", "|---|---|---|---|---|"]
    fmt = lambda t, p="{:+.3f}": f"{p.format(t[0])} [{p.format(t[1])}, {p.format(t[2])}]"
    for (c, cond), d in paired.items():
        md.append(f"| {c} | {cond} | {fmt(d['cost'])} | {fmt(d['energy'], '{:+.1f}')} | {fmt(d['ttd'], '{:+.1f}')} |")
    md += ["", "## Mean stage cost per held-out sequence (nominal / hot)", "",
           "| Sequence | " + " | ".join(CONTROLLERS) + " |", "|---|" + "---|" * len(CONTROLLERS)]
    for s in seqs:
        cells = []
        for c in CONTROLLERS:
            v = [np.mean([r["mean_cost"] for r in test if r["sequence"] == s and r["controller"] == c
                          and r["condition"] == cond]) for cond in CONDITIONS]
            cells.append(f"{v[0]:+.3f} / {v[1]:+.3f}")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    if synth_summary:
        md += ["", "## Synthetic equation-based stress scene (never used for training)", ""]
        for cond in CONDITIONS:
            md += [f"### {cond}", "", md_table(synth_summary, CONTROLLERS, cond), ""]
    abl = [c for c in sorted({r["controller"] for r in rows}) if c not in CONTROLLERS]
    if abl:
        md += ["", "## Ablations (seed 0, held-out ETH/UCY)", "",
               "`[no shield]` = proposals applied without the safety layer; `MPC-oracle` = MPC with the true "
               "future crowd instead of a constant-velocity forecast.", ""]
        for cond in CONDITIONS:
            sub = {}
            for c in CONTROLLERS + abl:
                sel = [r for r in test if r["controller"] == c and r["condition"] == cond and r["seed"] == 0]
                if sel:
                    sub[(c, cond)] = aggregate(sel)
            md += [f"### {cond}", "", md_table(sub, CONTROLLERS + abl, cond), ""]
    sweep_path = RESULTS / "static_sweep.csv"
    if sweep_path.exists():
        sweep = read_csv(sweep_path)
        pts = figure_pareto(summary, sweep)
        md += ["", "## Best static operating point (chosen in hindsight on the test scenes)", ""]
        for cond in CONDITIONS:
            by = {lab: aggregate([r for r in sweep if r["label"] == lab and r["condition"] == cond])
                  for lab in {r["label"] for r in sweep}}
            lab = min(by, key=lambda k: by[k]["mean_cost"])
            md.append(f"- {cond}: `{lab}` → cost {by[lab]['mean_cost']:+.3f}, energy {by[lab]['energy_per_m_j']:.2f} J/m, "
                      f"hazard t-detect {by[lab]['hazard_time_to_detect_ms']:.0f} ms, throttled {by[lab]['throttled_pct']:.1f}%")
    (RESULTS / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    ctrls = [c for c in CONTROLLERS if (c, list(CONDITIONS)[0]) in summary]
    figure_panel(summary, [("mean_latency_ms", "Mean inference latency", "ms"),
                           ("energy_per_m_j", "Energy per metre", "J/m"),
                           ("hazard_map50", "mAP@50 during hazards", "mAP@50")],
                 "latency_energy_accuracy.png", "Latency / Energy / Accuracy", ctrls)
    figure_panel(summary, [("mean_temp_c", "Mean die temperature", "°C"),
                           ("mean_throughput_fps", "Mean throughput", "FPS"),
                           ("mean_queue", "Mean frame queue", "frames")],
                 "temperature_throughput_queue.png", "Temperature / Throughput / Queue", ctrls)
    figure_panel(summary, [("hazard_miss_prob_pct", "P(miss pedestrian) in hazard steps", "%"),
                           ("deadline_miss_pct", "Deadline misses", "% of steps"),
                           ("shield_intervention_pct", "Shield overrides", "% of steps"),
                           ("violations_per_km", "Perception-induced violations", "per km")],
                 "safety_constraint_violations.png", "Safety / Constraint violations", ctrls)
    figure_panel(summary, [("mean_cost", "Mean stage cost (lower is better)", "cost")],
                 "overall_cost.png", "Overall objective", ctrls)
    figure_timeseries("biwi_hotel_hot", "timeseries_hotel_hot.png",
                      "BIWI Hotel, hot ambient (45 °C): closed-loop behaviour of each controller")
    figure_timeseries("students003_nominal", "timeseries_univ_nominal.png",
                      "UCY Univ (students003), nominal ambient: closed-loop behaviour of each controller")
    figure_action_usage()
    figure_training()
    figure_scenes()
    print((RESULTS / "RESULTS.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
