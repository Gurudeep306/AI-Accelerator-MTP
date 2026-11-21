# The detector's source, exactly as it was given

**None of this was written here and none of it has been modified.** These are
the files from the anytime detector project. They are the input to the whole
study: step 1 imports them, runs the network, and records what every convolution
actually does.

Checksums for all of them are printed in the run header every time
`run_all.py` runs, so a result can be matched to the exact source it came from.

| File | What it is |
|---|---|
| `gelan_anytime.py` | the network. Stem, backbone, FPN, PAN, refinement, and the four detection heads with the incremental forward methods the study calls |
| `common.py` | the blocks it is built from: `Conv`, `ELANBlock`, `Down`, `Up`, `SPPF` |
| `detect_head.py` | one detection head. Three `ScaleHead`s, one per pyramid level, and the box decode |
| `postprocess.py` | decode, score threshold, non-maximum suppression, un-letterbox |
| `accelerator_workload.csv` | the author's own layer table, 151 rows |
| `accelerator_workload_summary.md` | the author's summary of it |
| `make_accelerator_workload.py` | the script that produced that table |
| `anytime_interrupt.yaml` | the run configuration |
| `train6_config.json` | the training run: 65,000 images, 30 epochs, head loss weights 0.3 / 0.5 / 0.7 / 1.0, the ten class names |
| `head_timing.json` | measured per-stage timings on a GTX 1650, 100 images |
| `latency_train6.json` | measured end-to-end latency per head, 150 images |

## How the study uses them

`src/mdl/` holds byte-identical copies of the four Python files, with an empty
`__init__.py` added so Python can import them as a package. The originals use
relative imports (`from .common import Conv`), which need one. That is the only
difference, and you can prove it:

```bash
for f in gelan_anytime.py common.py detect_head.py postprocess.py; do
    diff -q "model-source/$f" "src/mdl/$f" && echo "$f identical"
done
```

`accelerator_workload.csv` is used for exactly one thing: step 2 compares the
trace against it, field by field, as a gate. **No number from that file reaches
any result table.** If the check fails the run stops. Including it is the point:
a workload recovered by running the network and a workload recorded by its
author agree on all 1,208 compared fields.

`head_timing.json` and `latency_train6.json` are measurements on a GPU host, not
on an accelerator. They are not used in any result either. They are here because
they are the thing the study is contrasted against: that host reaches the first
answer in about 13.7 ms, which is the figure the accelerator has to beat.

## What is missing, and why it matters

The trained weights, `runs/anytime_interrupt/train6_bdd70k/last.pt`, are **not**
in this folder. That has two consequences the study states openly:

- **Sparsity is not simulated.** Skipping zero values needs real weights and
  real activations. Every latency here is therefore an upper bound in that
  respect.
- **Accuracy could not be computed here.** The mAP figures come from
  `../inputs/measured_accuracy.json`, copied from the project's own evaluation
  on 1,500 unseen images, with the source section recorded beside each number.

Neither affects the cycle counts. A cycle count depends on the shape of a
tensor, never on the values inside it, which is why the study runs the network
with random weights and a zero image and is still exact about timing.

The dataset images are not here either. `../tools/contact_sheet.py` builds the
figure for the paper from them if you can get a few frames and their label
files.
