#!/usr/bin/env python3
"""Run ONE layer shape through SCALE-Sim, live, and show every input and output.

    python3 demo/one_layer.py --toy                 # tiny layer, 95 cycles, shows the traces
    python3 demo/one_layer.py 20 3 1 64 64          # a real detector layer (H, k, stride, Cin, Cout), INT8
    python3 demo/one_layer.py 20 3 1 64 64 --int4   # the same layer in INT4

Run from the project folder (the one that contains src/ and out/).
"""
import argparse, contextlib, csv, io, json, math, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from sim_config import CFG, LAYOUT_HEADER, LAYOUT_ROW, patch_scalesim

B, D, G, Y, C, R = '\033[1m', '\033[2m', '\033[32m', '\033[33m', '\033[36m', '\033[0m'
def head(t): print(f'\n{B}{C}── {t} {"─" * max(0, 66 - len(t))}{R}')
def show(path, n=None):
    lines = path.read_text().splitlines()
    for ln in lines[:n] if n else lines: print('   ' + ln)
    if n and len(lines) > n: print(f'   {D}... ({len(lines) - n:,} more lines){R}')

ap = argparse.ArgumentParser()
ap.add_argument('shape', nargs='*', type=int, help='H k stride Cin Cout')
ap.add_argument('--int4', action='store_true')
ap.add_argument('--toy', action='store_true')
a = ap.parse_args()
if a.toy:
    ih, k, s, cin, cout, lanes = 4, 3, 1, 4, 4, 2
else:
    if len(a.shape) != 5: ap.error('give 5 numbers: H k stride Cin Cout, or --toy')
    ih, k, s, cin, cout = a.shape
    lanes = 4 if a.int4 else 2

head('1. The layer, as the trace recorded it')
print(f'   input {ih}x{ih}x{cin}, kernel {k}x{k}, stride {s}, padding {(k - 1) // 2}, {cout} filters'
      f'   ->  format {"INT4" if lanes == 4 else "INT8"} ({lanes} lanes per PE)')

head('2. Fit it to our chip')
ph, cinp = ih + (k - 1), math.ceil(cin / lanes)
print(f'   padding : SCALE-Sim does not pad, so IFMAP size = {ih} + ({k} - 1) = {ph}')
print(f'   lanes   : a PE does {lanes} channels per cycle, so channels given = ceil({cin}/{lanes}) = {cinp}')

name = f'{"toy" if a.toy else "layer"}_i{ih}_k{k}_s{s}_c{cin}_f{cout}_L{lanes}'
work = ROOT / 'demo' / 'runs' / name
work.mkdir(parents=True, exist_ok=True)
(work / 'topology.csv').write_text('Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,Channels,Num Filter,Strides,\n'
                                   f'L,{ph},{ph},{k},{k},{cinp},{cout},{s},\n')
(work / 'layout.csv').write_text(LAYOUT_HEADER + f'L,{LAYOUT_ROW}\n')
(work / 'run.cfg').write_text(CFG.format(name='run', rows=4, cols=4, df='os'))

head('3. The three files SCALE-Sim reads')
print(f'{Y}   run.cfg{R}  (the 4x4 output-stationary array)'); show(work / 'run.cfg', 22)
print(f'{Y}   topology.csv{R}  (this layer)'); show(work / 'topology.csv')

head('4. Run SCALE-Sim')
with contextlib.redirect_stdout(io.StringIO()):
    patch_scalesim()
from scalesim.scale_sim import scalesim
print(f'   {D}scalesim(config=run.cfg, topology=topology.csv, layout=layout.csv).run_scale(){R}')
t0 = time.time()
sim = scalesim(save_disk_space=not a.toy, verbose=False, config=str(work / 'run.cfg'),
               topology=str(work / 'topology.csv'), layout=str(work / 'layout.csv'))
with contextlib.redirect_stdout(io.StringIO()):
    sim.run_scale(top_path=str(work))
print(f'   {G}done in {time.time() - t0:.1f} s{R}   reports in {work.relative_to(ROOT)}/run/')

head('5. What SCALE-Sim reported (COMPUTE_REPORT.csv)')
row = next(csv.DictReader((work / 'run' / 'COMPUTE_REPORT.csv').open()))
row = {x.strip(): y.strip() for x, y in row.items() if x}
for key in ['Total Cycles', 'Stall Cycles', 'Overall Util %', 'Mapping Efficiency %', 'Compute Util %']:
    print(f'   {key:24} {float(row[key]):>14,.2f}' if '%' in key else f'   {key:24} {int(float(row[key])):>14,}')
cyc = int(float(row['Total Cycles']))
if a.toy:
    print(f'\n{Y}   IFMAP_SRAM_TRACE.csv{R}  (cycle, address read by array row 0..3; -1 = nothing)')
    show(work / 'run' / 'layer0' / 'IFMAP_SRAM_TRACE.csv', 8)

head('6. Check it with the formula')
oh = math.ceil((ph - k) / s) + 1
M, N, Kp = oh * oh, cout, k * k * cinp
f = math.ceil(M / 4) * math.ceil(N / 4) * (Kp + 6) - 1
print(f"   M = {oh}x{oh} = {M:,}   N = {N}   K' = {k}x{k}x{cinp} = {Kp}")
print(f"   cycles = ceil(M/4) x ceil(N/4) x (K' + 6) - 1 = {math.ceil(M/4):,} x {math.ceil(N/4)} x {Kp + 6} - 1 = {f:,}")
print(f'   SCALE-Sim = {cyc:,}   ' + (f'{G}{B}EQUAL{R}' if f == cyc else f'{Y}DIFFERENT{R}'))
print(f'   time at 200 MHz = {cyc:,} / 200,000,000 = {cyc / 2e5:.2f} ms')

cache = ROOT / 'out' / 'spec44_cache' / f'L{lanes}_i{ih}_k{k}_s{s}_c{cin}_f{cout}.json'
if cache.exists():
    old = json.loads(cache.read_text())['cycles']
    print(f'   stored result from the full run: {int(old):,}   ' + (f'{G}same{R}' if int(old) == cyc else f'{Y}differs{R}'))
