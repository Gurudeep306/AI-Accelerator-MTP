# Start here

**Model mapping and execution** &mdash; Feedback-Controlled
Safety-Critical AI Accelerator for Autonomous Systems.

Four commands get you a result. Everything else in this folder is detail.

```bash
# 1. make a virtual environment and install SCALE-Sim from source
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt    # ~2 GB, do this the night before

# 2. check the machine before trusting it. Runs nothing.
python3 run_all.py --check

# 3. the short run: everything except the SCALE-Sim sweep, about 3 minutes
python3 run_all.py --quick

# 4. the full run: adds the sweep, about an hour, cached so it never repeats
python3 run_all.py
```

The transcript lands in `out/simulation_output.txt` while it prints, so what you
show on screen and what you keep are the same file.

---

## If someone wants to see SCALE-Sim itself, with none of my code

```bash
cd scalesim-only
chmod +x run.sh
./run.sh accel_12x12.cfg demo_head1_p5.csv 4
```

Four minutes. SCALE-Sim's own command line, plain topology files, and one shell
script that cuts them into slices and adds up the reports. Read `scalesim-only/README.md`.

---

## If someone asks whether the numbers are really the simulator's

```bash
python3 src/audit.py
```

It computes nothing. It opens the files and reports, for every table in the
paper, which file holds its numbers. It names the SCALE-Sim file behind three
layers picked at random, counts how many of the detector's distinct shapes have
a simulator result on disk, and prints what the study does **not** claim.

---

## The specified 4x4 mesh

```bash
# SCALE-Sim on the specification v1.1 mesh (4x4, output-stationary)
python3 src/sim_spec44.py 2        # INT8, two lanes per PE   (about 2 hours)
python3 src/sim_spec44.py 4        # INT4, four lanes per PE  (about 1 hour)

# the closed-form cycle equations against every SCALE-Sim result on disk
python3 tools/check_formula.py
```

The results of both 4x4 runs are already in `out/spec44_cache/`.

---

## Where everything is

| You want | Look in |
|---|---|
| the detector's own code | `model-source/` — unmodified, with a README explaining each file |
| SCALE-Sim on its own | `scalesim-only/` |
| the measured accuracy, and where it came from | `inputs/measured_accuracy.json` |
| the target board's specification | `inputs/platform.json` |
| the six steps of the study | `src/step1_trace.py` … `src/step6_results.py` |
| one SCALE-Sim run per layer shape | `out/shape_cache/` |
| every number the paper's tables use | `out/results.json` |
| the full transcript | `out/simulation_output.txt` |
| real dataset frames for the paper | `tools/contact_sheet.py`, once you have a few images |

---

## The paper

The write-up of this work is [`Model-Mapping-and-Execution.pdf`](../Model-Mapping-and-Execution.pdf), one level up from this folder.

---|---|
| *Model-Mapping-Running-In-SCALE-Sim* | every SCALE-Sim command, step by step, with the real output of each |
| *Model-Mapping-How-To-Present-It* | the five-minute demonstration, the dataset explained, the questions you will be asked |
| *Model-Being-Mapped-Explained* | the detector itself, block by block |
| *Model-Mapping-and-Execution* | the paper |

---

## Two things that will bite you, both in SCALE-Sim and both real

**`-s N` does nothing.** It is supposed to suppress the per-cycle traces. In
`scalesim/scale.py` the flag is parsed into a variable and then
`save_disk_space=False` is passed regardless, so the variable is never used.
Seven small layers wrote 704 MB. Clean up with
`find results -name '*_TRACE.csv' -delete`. This package calls the Python API
instead, where the flag is honoured.

**A long run is killed with no error.** SCALE-Sim does not release memory
between the layers of one call, so over 151 layers it passes 2.3 GB and the
operating system kills it: no traceback, no message, no reports. It looks as
though the simulator cannot handle the network. It can, one slice at a time.
