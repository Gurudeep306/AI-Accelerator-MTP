"""Per-layer workload of the anytime detector, for accelerator design.

Run from the workspace root:
    .venv/Scripts/python docs/make_accelerator_workload.py
Writes docs/accelerator_workload.csv (one row per convolution, in execution
order) and docs/accelerator_workload_summary.md.

Weights are reported with BatchNorm folded into the convolution, which is how
an inference accelerator would hold them.
"""
import collections
import csv
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from anytime_yolo_interrupt.models.gelan_anytime import build_model  # noqa: E402

CHECKPOINT = os.path.join(ROOT, "runs", "anytime_interrupt", "train6_bdd70k", "last.pt")
IMG = 640
STAGES = ["backbone", "head1", "fpn", "head2", "pan", "head3", "refine", "head_final"]  # execution order


def stage_of(name: str) -> str:
    top = name.split(".")[0]
    if top in ("head1", "head2", "head3", "head_final"):
        return top
    if top == "stem" or top.endswith("_bb") or top in ("down_p3", "down_p4", "down_p5", "sppf_p5"):
        return "backbone"
    if top.endswith("_td") or top.startswith("reduce"):
        return "fpn"
    if top.endswith("_bu"):
        return "pan"
    if top.endswith("_final"):
        return "refine"
    raise ValueError(f"unknown stage for {name}")


model = build_model(10).eval()
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"])

# conv -> its BatchNorm (the Conv wrapper in models/common.py holds .conv and .bn)
bn_of = {m.conv: m.bn for m in model.modules() if type(m).__name__ == "Conv"}
act_of = {m.conv: type(m.act).__name__ for m in model.modules() if type(m).__name__ == "Conv"}

rows, other_ops = [], collections.Counter()


def conv_hook(name):
    def hook(mod, inp, out):
        k, s = mod.kernel_size[0], mod.stride[0]
        cin, cout = mod.in_channels, mod.out_channels
        hi, wi = inp[0].shape[2:]
        ho, wo = out.shape[2:]
        w = mod.weight.detach()
        if mod in bn_of:  # fold BatchNorm: w' = w * g / sqrt(var + eps), b' = beta - mean * g / sqrt(var + eps)
            bn = bn_of[mod]
            scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            w = w * scale.view(-1, 1, 1, 1)
            b = bn.bias.detach() - bn.running_mean * scale
        else:
            b = mod.bias.detach()
        rows.append(dict(
            order=len(rows), layer=name, stage=stage_of(name), kernel=k, stride=s, cin=cin, cout=cout,
            in_hw=f"{hi}x{wi}", out_hw=f"{ho}x{wo}",
            weights=w.numel(), biases=b.numel(), macs=ho * wo * cout * cin * k * k,
            in_bytes_int8=cin * hi * wi, out_bytes_int8=cout * ho * wo,
            activation="SiLU" if act_of.get(mod) == "SiLU" else "none",
            folded_weight_absmax=round(float(w.abs().max()), 4), folded_bias_absmax=round(float(b.abs().max()), 4),
        ))
    return hook


for name, mod in model.named_modules():
    if isinstance(mod, nn.Conv2d):
        mod.register_forward_hook(conv_hook(name))
    elif isinstance(mod, (nn.MaxPool2d, nn.Upsample)):
        mod.register_forward_hook(lambda m, i, o: other_ops.update([type(m).__name__]))

with torch.no_grad():
    model(torch.zeros(1, 3, IMG, IMG))

with open(os.path.join(HERE, "accelerator_workload.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# ---------------------------------------------------------------- summary
G, M = 1e9, 1e6
tot_macs = sum(r["macs"] for r in rows)
tot_w = sum(r["weights"] for r in rows)
tot_b = sum(r["biases"] for r in rows)
per = {s: [r for r in rows if r["stage"] == s] for s in STAGES}
macs = {s: sum(r["macs"] for r in per[s]) for s in STAGES}

lines = ["# Accelerator workload summary", "",
         f"Model: `train6_bdd70k/last.pt`, input 3 × {IMG} × {IMG}, batch 1. BatchNorm is folded into each "
         "convolution. One multiply-add (MAC) is one weight × one input value.", "",
         "Per-layer detail is in `accelerator_workload.csv` (one row per convolution, in execution order).", "",
         "## Totals", "",
         f"- Convolution layers: **{len(rows)}** ({sum(r['kernel'] == 3 for r in rows)} are 3×3, "
         f"{sum(r['kernel'] == 1 for r in rows)} are 1×1; {sum(r['stride'] == 2 for r in rows)} have stride 2)",
         f"- Multiply-adds per image: **{tot_macs / G:.2f} G**",
         f"- Weights: **{tot_w / M:.2f} M**, biases after folding: {tot_b:,} → about "
         f"**{(tot_w + tot_b) / M:.2f} MB** at 8 bits per value",
         f"- Layers followed by SiLU: {sum(r['activation'] == 'SiLU' for r in rows)}; "
         f"plain output layers (no activation): {sum(r['activation'] == 'none' for r in rows)}",
         f"- Other operations per image: {other_ops['MaxPool2d']} max-pool calls (5×5, stride 1), "
         f"{other_ops['Upsample']} nearest ×2 upsamples, 29 concatenations",
         f"- Channel widths used: {sorted({r['cin'] for r in rows} | {r['cout'] for r in rows})}", "",
         "## By stage, in execution order", "",
         "| Stage | Conv layers | 3×3 | 1×1 | Weights | Multiply-adds | Share |",
         "|---|---|---|---|---|---|---|"]
for s in STAGES:
    r = per[s]
    lines.append(f"| {s} | {len(r)} | {sum(x['kernel'] == 3 for x in r)} | {sum(x['kernel'] == 1 for x in r)} | "
                 f"{sum(x['weights'] for x in r) / M:.2f} M | {macs[s] / G:.3f} G | {100 * macs[s] / tot_macs:.1f}% |")
lines.append(f"| **total** | **{len(rows)}** | {sum(r['kernel'] == 3 for r in rows)} | "
             f"{sum(r['kernel'] == 1 for r in rows)} | **{tot_w / M:.2f} M** | **{tot_macs / G:.3f} G** | 100% |")

heads = macs["head1"] + macs["head2"] + macs["head3"] + macs["head_final"]
lines += ["", f"The four heads together are {heads / G:.2f} G multiply-adds, **{100 * heads / tot_macs:.0f}%** of the total.",
          "", "## Work needed before each answer", "",
          "| Answer | Sequential (as run today) | Only the body it needs + its own head |", "|---|---|---|"]
body_needed = {"head1": ["backbone"], "head2": ["backbone", "fpn"], "head3": ["backbone", "fpn", "pan"],
               "head_final": ["backbone", "fpn", "pan", "refine"]}
running = 0
for s in STAGES:
    running += macs[s]
    if s.startswith("head"):
        alone = sum(macs[b] for b in body_needed[s]) + macs[s]
        lines.append(f"| {s} | {running / G:.2f} G | {alone / G:.2f} G |")
lines += ["", "The right-hand column is what each answer would cost if earlier heads did not have to finish first, "
          "for example when a head runs in parallel with the next body stage on separate hardware.", "",
          "## By kernel size", "", "| Kernel | Layers | Multiply-adds | Share |", "|---|---|---|---|"]
for k in (3, 1):
    r = [x for x in rows if x["kernel"] == k]
    km = sum(x["macs"] for x in r)
    lines.append(f"| {k}×{k} | {len(r)} | {km / G:.2f} G | {100 * km / tot_macs:.1f}% |")

big_map = max(rows, key=lambda r: r["out_bytes_int8"])
big_mac = max(rows, key=lambda r: r["macs"])
big_w = max(rows, key=lambda r: r["weights"])
pyramid = 64 * 80 * 80 + 96 * 40 * 40 + 128 * 20 * 20
lines += ["", "## Memory", "",
          f"- Largest feature map: `{big_map['layer']}` output, {big_map['cout']} × {big_map['out_hw']} = "
          f"{big_map['out_bytes_int8'] / M:.2f} MB at 8 bits",
          f"- One set of P3 + P4 + P5 maps: {pyramid:,} values = {pyramid / M:.2f} MB at 8 bits. Each stage's set must "
          "stay available until both its head and the next stage have read it.",
          f"- Largest weight tensor: `{big_w['layer']}`, {big_w['cin']} → {big_w['cout']} channels, "
          f"{big_w['kernel']}×{big_w['kernel']}: {big_w['weights'] / 1e3:.0f} K weights",
          f"- Heaviest single layer: `{big_mac['layer']}`, {big_mac['macs'] / G:.3f} G multiply-adds", "",
          "## For quantisation", "",
          f"- Largest folded weight magnitude in any layer: {max(r['folded_weight_absmax'] for r in rows):.2f}; "
          f"smallest per-layer maximum: {min(r['folded_weight_absmax'] for r in rows):.3f}. The spread means "
          "per-layer (or per-channel) scaling is needed, not one scale for the whole model.",
          f"- Largest folded bias magnitude: {max(r['folded_bias_absmax'] for r in rows):.2f}",
          "- The CSV columns `folded_weight_absmax` and `folded_bias_absmax` give the range for every layer.",
          "- Accuracy after quantisation has **not** been measured; these are ranges only.", ""]

with open(os.path.join(HERE, "accelerator_workload_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
