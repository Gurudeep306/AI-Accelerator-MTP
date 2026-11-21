# Model mapping and execution

### Feedback-Controlled Safety-Critical AI Accelerator for Autonomous Systems

This is the *model mapping and execution* row of that project: taking a real
detection network, mapping it onto a systolic accelerator, measuring how long
every layer takes, and producing the operating-point table the runtime
controller needs. Without that table the controller has nothing to optimise over
and the accelerator has no specification to meet.

### On the tooling

**There is no simulator in this folder.** Every cycle count comes from
SCALE-Sim, the published cycle-level simulator for systolic arrays, installed
from its own source repository at a pinned commit. The word "anytime" in the
title describes the *detector* being studied, which returns four progressively
better answers from one pass. It does not describe a simulator, and nothing here
models an array.

What this folder adds around SCALE-Sim is three things and no more: it works out
what the detector's layers actually are by running it, it hands those layers to
SCALE-Sim in a way that does not get killed, and it adds SCALE-Sim's per-layer
cycles up. `python3 src/audit.py` prints, for every table, which file its
numbers live in and how they got there.

If you want SCALE-Sim with nothing of mine in the way at all, use
`scalesim-only/`. That is the simulator, plain topology files, and one shell
script.

Everything needed to reproduce the study on your own machine, from a clean
checkout, with one command.

---

## 0. What is in here

| Folder | What it holds |
|---|---|
| **`model-source/`** | **the detector's own code, unmodified.** The network, its blocks, the detection head, the post-processing, the author's layer table and training config. This is the input to everything. |
| `scalesim-only/` | SCALE-Sim with nothing of mine in the way: topology files and one shell script |
| `inputs/` | the measured accuracy and the target board's specification |
| `src/` | the six steps: trace, gate, memory, resolutions, SCALE-Sim sweep, tables |
| `tools/` | `contact_sheet.py`, which draws dataset frames for the paper |
| `out/` | everything a run produces, including `shape_cache/` with one file per SCALE-Sim run |

**Step by step instructions for running it are in Section 2 and 3 below.** The
companion PDF *Running the accelerator in SCALE-Sim* walks through the same
thing at the level of individual SCALE-Sim commands, with the real output of
each one.

---

## 1. What this does

It answers one question. Given a systolic accelerator of a chosen array size and
clock, **when is each of the four answers of the anytime detector ready, and
which of them land inside the 16 ms control deadline.**

It does that in six steps:

| Step | What it does | How long |
|---|---|---|
| 1 | Executes the detector and records every convolution it performs | ~60 s |
| 2 | Checks that trace, field by field, against the shipped layer table | instant |
| 3 | Intercepts every tensor operation and finds the peak live activation set | ~30 s |
| 4 | Repeats step 1 at five input sizes | ~90 s |
| 5 | Runs SCALE-Sim over **every layer**, for each array size and dataflow | ~1 h, cached |
| 6 | Totals the simulated cycles and prints the result tables | instant |

---

## 2. Install

You need Python 3.9 or newer. Check with `python3 --version`.

```
cd scalesim-study
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs SCALE-Sim from its **source repository**, not from
PyPI. The released package is behind the repository, and the run header records
the exact commit so a result can always be matched to the simulator that
produced it.

On Windows the third line is `.venv\Scripts\activate` instead.

The install pulls PyTorch, which is large, around 2 GB. Do this the day before,
not five minutes before you have to present.

Confirm it worked without running anything:

```
python3 run_all.py --check
```

That prints your machine, your Python, your package versions, and a SHA-256
checksum of every input file. If something is missing it says so and stops.

---

## 3. Run

```
python3 run_all.py
```

The whole study, about 22 minutes, most of it step 5.

```
python3 run_all.py --quick
```

Steps 1, 2, 3, 4 and 6, about 3 minutes. Simulates nothing new, so the tables
cover only the configurations already in `out/shape_cache/` and print a dash for
the rest.

```
python3 run_all.py --keep-sim
```

Re-runs the trace and the tables against the SCALE-Sim results already cached,
without simulating anything new. Use this after changing an input in `inputs/`.

**Nothing is ever simulated twice.** Every distinct layer shape is cached in
`out/shape_cache/` under its shape and configuration, so stopping the sweep and
restarting it costs nothing, and later sweeps over other array sizes or input
resolutions reuse whatever overlaps. Delete the folder to force a clean re-run.

Two things make a cycle-accurate sweep of this size possible at all, and both
are worth knowing about because a naive attempt fails:

- **SCALE-Sim does not release memory between the layers of one call.** Over 151
  layers the resident set climbs past 2 GB and the process is killed by the
  operating system with no traceback. Every shape therefore runs in its own
  process, which caps the memory and lets one oversized shape fail alone.
- **The detector has 151 layers but only 37 distinct shapes**, because its four
  heads are structurally identical. Simulating shapes rather than layers is
  about 4.3 times less work for an identical answer. That is checked, not
  assumed: see section J of the output.

Both write the full transcript to `out/simulation_output.txt` while printing it
to the screen, so what is shown live and what ends up in the file are the same
thing.

---

## 4. What comes out

```
out/simulation_output.txt   the whole transcript, sections A to J
out/trace.json              every convolution, as executed
out/crosscheck.json         the result of the gate in step 2
out/memory.json             peak live activation set
out/resolutions.json        the detector at five input sizes
out/validation.json         the SCALE-Sim comparison (full runs only)
out/results.json            every number the paper's tables are built from
out/scalesim/               SCALE-Sim's own reports, one folder per run
out/dataset_sheet.png       real frames with labels, if you ran the helper below
```

`out/results.json` is the file to hand back if the paper has to be rebuilt from
your run rather than from anyone else's.

---

## 5. Showing it

Suggested order, which takes about five minutes at the desk.

**First, show where the inputs came from.** Open `PROVENANCE.md`. It lists every
input file, who produced it, and which numbers in the study it feeds. Run
`python3 run_all.py --check` so the checksums appear on screen. The point being
made is that the detector source is not yours and is unmodified, and the accuracy
figures are measured by someone else and are not simulated here.

**Then run it.** `python3 run_all.py --quick`. Three minutes, and the tables
appear live.

**Stop at step 2 and let it be read.** This is the strongest moment in the demo.
1,208 fields compared, zero mismatches. It means the workload was recovered by
running the network, independently of the table that ships with it, and the two
agree exactly. Everything after that rests on it.

**Point at section A.** The four heads are 77 per cent of the frame. That is the
fact the whole design argument turns on and it is not obvious in advance.

**Point at section D.** The best case the target board can reach is 175.5 ms for
the cheapest answer, against a 16 ms deadline. Eleven times over.

**Point at section F.** Memory costs 1.13 ms against 234 ms of arithmetic. So the
bottleneck is the array, not the bus, which tells you where not to spend effort.

**Point at section G.** Eight settings of thirty-two are admissible, and the
fourth answer is dominated by the third on both latency and accuracy.

**If asked how far the numbers can be trusted, show section J.** It names the
exact simulator build, the layer and shape counts, the reuse factor, and the
number of shapes that returned two different cycle counts, which should be zero.

---

## 5b. Showing SCALE-Sim itself

If someone wants to see the simulator rather than a wrapper around it, there is
a folder for exactly that:

```
cd examples
python3 -m scalesim.scale -c accelerator.cfg -t head1_p5.csv \
  -l head1_p5_layout.csv -p ./results -i conv -s N
cat results/head1_p5/COMPUTE_REPORT.csv
```

Four minutes, seven layers, SCALE-Sim's own command line and its own reports.
`examples/README.md` says what to point at, and `examples/expected_console.log`
holds the output from this machine so you can diff against it.

The companion document *Running the accelerator in SCALE-Sim* walks through
every one of those files line by line.

---

## 5a. Putting your run in the paper

The paper is built from `out/results.json` and `out/simulation_output.txt`, not
from numbers typed into it. Every table and every figure is generated from those
two files, so rebuilding the paper against your run makes every number in it
yours.

Do a full run, then keep the whole `out/` folder. That folder is the handoff. It
contains the transcript your teacher saw on screen and the machine-readable file
the tables are generated from, and the run header inside it records your machine,
your Python, your package versions and a checksum of every input, so the two can
always be matched to each other.

---

## 6. Two questions you should expect

**"Are these real weights?"**

No, and it does not matter for latency. The detector is built with random
weights and fed a zeros image. A cycle count depends only on tensor shapes,
never on the values inside them, and step 2 proves the shapes are right. It
does matter for two things the study therefore does not claim: sparsity is not
simulated, and accuracy is not simulated. The accuracy column is measured by the
person who trained the network, on 1,500 images it had never seen, and the file
it is copied from is `inputs/measured_accuracy.json`.

**"Is any of this modelled rather than simulated?"**

No. Every cycle count is SCALE-Sim's own, for that exact layer shape and that
exact array and dataflow. The only arithmetic applied to a simulated number is
addition, to total the per-layer cycles over a stage or over the stages an
answer needs. A configuration that was not simulated prints a dash rather than
an estimate.

The one adjustment is the derating in section D, which scales simulated cycles
to the utilisation measured in the accelerator's own Verilog. Both halves of
that ratio are printed so it can be checked or removed.

---

## 7. If something breaks

**`ModuleNotFoundError: torch`** — the virtual environment is not active.
Re-run `source .venv/bin/activate`.

**Step 5 throws `TypeError: only 0-dimensional arrays ...`** — SCALE-Sim 3.0.0
has a bug where it calls `int(max(numpy_array))`. Step 5 patches it
automatically on first run and saves the original beside it as
`double_buffered_scratchpad_mem.py.orig`. If the patch cannot find the line it
warns you rather than failing silently.

**Step 5 is killed, or the machine swaps** — this is expected on a small
machine for the largest shapes, and it is handled: each shape runs in its own
process, so a failure is reported and the sweep continues. Shapes that did
succeed stay cached. Re-run and only the missing ones are attempted.

**Step 5 is taking too long** — stop it. Everything already cached is kept, and
`--keep-sim` builds the tables from it. The array sizes simulated are listed in
section J, and sizes that were not simulated print a dash rather than a guess.

**Step 2 reports mismatches** — stop. Something in `inputs/` does not match the
code in `src/mdl/`. Do not use any number after that point until it is resolved.

---

## 7a. Showing real frames from the dataset

The simulation needs no images, because a cycle count depends on the shape of a
tensor and not on what is inside it, so none ship with this package. A
demonstration is better with two or three real frames on screen though. Ask your
teammate for a few images from the validation split together with their label
files, put them in a folder, and run:

```
pip install pillow
python3 tools/contact_sheet.py <that folder>
```

It writes `out/dataset_sheet.png`: the frames with their labelled boxes drawn
on, and under each one the object count and exactly how many pixels of grey the
letterbox adds. It reads both layouts, images and labels together in one folder
or the `images/` and `labels/` pair the training code uses.

---

## 7b. Proving where the numbers came from

```
python3 src/audit.py
```

It computes nothing. It opens the files the results were built from and reports,
for each table in the paper, which file holds its numbers and how they got
there. It picks three layers at random and names the SCALE-Sim file each one
reads, counts how many of the detector's distinct shapes have a simulator result
on disk, confirms that no shape returned two different cycle counts, and prints
the one adjustment in the whole study with both halves of its ratio.

It also prints what is *not* claimed, which is the part worth reading twice.

---

## 8. Changing the study

Nothing is hardcoded in the scripts. To change the board, the clocks, the array
sizes or the input sizes, edit `inputs/platform.json`. To change the accuracy
column, edit `inputs/measured_accuracy.json`. Then re-run. The tables follow.

