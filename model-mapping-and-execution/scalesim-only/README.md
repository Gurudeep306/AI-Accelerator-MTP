# Running the detector on SCALE-Sim

*Model mapping and execution &mdash; Feedback-Controlled
Safety-Critical AI Accelerator for Autonomous Systems.*

SCALE-Sim and bash. Nothing else. No code of mine runs except a shell script
that writes topology files, calls the simulator, and adds up its reports.

---

## 1. Install SCALE-Sim

Install from the **source repository**, not from PyPI. The released package
(3.0.0, August 2025) is behind the repository, and the version string says
3.0.0 either way, so the only way to know what you have is to pin a commit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/scalesim-project/SCALE-Sim.git@9f98c4371055a54c75209c2e02b640b897550532"
```

Check what you actually got:

```bash
python3 -c "import importlib.metadata as m; \
  print(m.version('scalesim')); \
  print(m.distribution('scalesim').read_text('direct_url.json'))"
```

```
3.0.0
{"url": "https://github.com/scalesim-project/SCALE-Sim.git", "vcs_info":
 {"commit_id": "9f98c4371055a54c75209c2e02b640b897550532", ...}}
```

### Patch one bug first

SCALE-Sim calls `int(max(numpy_array))` in its scratchpad memory model, which
throws. One line, and the original is kept beside it:

```bash
python3 - <<'EOF'
import scalesim, pathlib, shutil
p = pathlib.Path(scalesim.__file__).parent / 'memory/double_buffered_scratchpad_mem.py'
s = p.read_text()
old = 'int(max(ofmap_serviced_cycles))'
new = 'int(np.max(np.asarray(ofmap_serviced_cycles)))'
if old in s:
    shutil.copy(p, str(p) + '.orig')
    p.write_text(s.replace(old, new))
    print('patched, original kept as .orig')
else:
    print('already patched')
EOF
```

---

## 2. What is in this folder

| File | What it is |
|---|---|
| `accel_8x8.cfg` | the array that has been built |
| `accel_12x12.cfg` | the largest array the target part can hold |
| `accel_32x32.cfg`, `accel_64x64.cfg` | arrays big enough to meet the deadline |
| `demo_head1_p5.csv` | 7 layers: the deepest level of the first head. **Four minutes.** |
| `answer_head1.csv` | 48 layers: everything the first answer needs |
| `full_network.csv` | all 151 convolutions |
| `*_layout.csv` | the neutral layout file v3 requires, one per topology |
| `run.sh` | writes the slices, calls SCALE-Sim, totals the reports |

Every topology is already pre-padded. SCALE-Sim convolves without implicit
padding while the detector pads to keep its resolution, so a 3x3 layer on a 20
by 20 map is written as 22. The rule is `+ (k - 1)`. Miss it and every 3x3 layer
comes out about ten per cent too cheap.

---

## 3. The four-minute run

```bash
chmod +x run.sh
./run.sh accel_12x12.cfg demo_head1_p5.csv 4
```

Real output:

```
network   : demo_head1_p5.csv  (7 layers)
machine   : accel_12x12.cfg
slices    : 4 layers each
results   : out_accel_12x12_demo_head1_p5/

  layers   1..4         1,833,212 cycles
  layers   5..7            14,319 cycles

layers simulated   : 7
total cycles       : 1,847,531
stall cycles       : 0
mean layer util    : 67.0 %

time for one frame, before any derating
  at  100 MHz :     18.48 ms   (  54.13 fps)
  at  150 MHz :     12.32 ms   (  81.19 fps)
  at  200 MHz :      9.24 ms   ( 108.25 fps)
  at  400 MHz :      4.62 ms   ( 216.51 fps)
```

**That is one pyramid level of one head out of four, and at 150 MHz it already
costs 12.32 ms of a 16 ms budget.** The whole network is 151 layers.

---

## 4. Read SCALE-Sim's own report

```bash
cat out_accel_12x12_demo_head1_p5/COMPUTE_ALL.csv
```

```
LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %,
0, 476863, 458303, 0, 89.37318760732528, 96.96969696969663, 87.16377255702903,
1, 476863, 458303, 0, 89.37318760732528, 96.96969696969663, 87.16377255702903,
2, 476863, 458303, 0, 89.37318760732528, 96.96969696969663, 87.16377255702903,
3, 476863, 458303, 0, 89.37318760732528, 96.96969696969663, 87.16377255702903,
4,  13681,   4773, 0, 74.49309774891171, 80.80808080808080, 72.63647713085916,
5,  10381,   4773, 0,  7.44930977489117,  8.08080808080808,  7.26364771308591,
6,  11481,   4773, 0, 29.79723909956468, 32.32323232323232, 29.05459085234366,
```

Two things to point at.

**Rows 0 to 3 are identical.** Four convolutions of the same shape, the same
cycle count every time. The detector has 151 layers but only 37 distinct shapes,
because its four heads are structurally identical.

**Row 5 is 7.45 per cent utilised.** That is the objectness convolution, 128
channels into one. In a weight-stationary mapping the output channels go across
the columns of the array, so eleven of twelve columns have nothing to do. It
costs almost nothing in multiply-adds and a great deal in time, and you cannot
see it in any arithmetic count. Only a simulator shows it.

Use the third column, `Total Cycles`. The second includes filling the buffers
before compute starts, which is right for a layer running alone and wrong for
layers running back to back.

The other three reports sit beside it:

```bash
ls out_accel_12x12_demo_head1_p5/slices/s001/run/
# BANDWIDTH_REPORT.csv  COMPUTE_REPORT.csv  DETAILED_ACCESS_REPORT.csv  TIME_REPORT.csv

cat out_accel_12x12_demo_head1_p5/slices/s001/run/BANDWIDTH_REPORT.csv | cut -c1-110
```

Average words per cycle for each operand, on the buffers and on the external
bus. The input side pulls about 9.8 words per cycle from DRAM while the weight
side pulls 0.36, which is the signature of weight-stationary: load the weights
once, then stream.

---

## 5. The whole network

```bash
./run.sh accel_12x12.cfg full_network.csv 8
```

151 layers, about an hour. For the first answer only, which is the cheapest
thing the accelerator can deliver:

```bash
./run.sh accel_12x12.cfg answer_head1.csv 8
```

Compare array sizes by changing one argument:

```bash
for cfg in accel_8x8.cfg accel_12x12.cfg accel_32x32.cfg accel_64x64.cfg; do
    ./run.sh "$cfg" answer_head1.csv 8
done
```

---

## 6. Two things that will bite you

### The `-s` flag does nothing, and it will fill your disk

`-s N` is supposed to suppress the per-cycle memory traces. In
`scalesim/scale.py` the flag is parsed into a variable and then
`save_disk_space=False` is passed regardless, so the variable is never used.
Seven small layers produced **704 MB** of traces. `run.sh` deletes them after
each slice. If you call the simulator yourself:

```bash
find results -name '*_TRACE.csv' -delete
```

### A long run is killed with no error at all

SCALE-Sim does not release memory between the layers of one call. Over 151
layers the resident set climbs past 2.3 GB and the operating system kills the
process. There is no traceback, no message, and no reports. It looks as though
the simulator cannot handle the network. It can; it cannot be handed all of it
at once.

Watch it happen, if you want to show it. The third argument of `0` means one
call for everything:

```bash
./run.sh accel_12x12.cfg full_network.csv 0 &
while kill -0 $! 2>/dev/null; do
    ps -o rss= -C python3 | awk '{s+=$1} END {printf "%.2f GB\n", s/1048576}'
    sleep 10
done
```

Then run it again with `8` and it finishes.

---

## 7. Turning cycles into the answer

Three steps, and only the first involves the simulator.

1. **Total the layers an answer needs.** `run.sh` prints this. An answer does
   not need the whole network: the first head needs the backbone and itself,
   which is `answer_head1.csv`.

2. **Apply the derating.** A simulator models an idealised memory system. On an
   8x8 array SCALE-Sim reports 97.9 per cent utilisation on this workload, while
   the accelerator's own Verilog measures 60.1 per cent, because a real array
   also spends cycles loading weights between folds. Multiply by `97.9 / 60.1 =
   1.63`.

3. **Divide by the clock and compare against 16 ms.**

On 12x12 at 200 MHz the cheapest of the four answers lands at 175.5 ms against a
16 ms budget. Eleven times over, on the largest array the board can hold, with
the cheapest answer it can give.
