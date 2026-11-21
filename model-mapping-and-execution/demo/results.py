#!/usr/bin/env python3
"""Show how 151 layers become 37 SCALE-Sim results, and the latency of each exit.

    python3 demo/results.py           # both parts
    python3 demo/results.py shapes    # only: 151 layers -> 37 shapes, with the list
    python3 demo/results.py exits     # only: latency of each exit

Run from the project folder (the one that contains src/ and out/).
"""
import collections, json, pathlib, signal, sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet when piped into head

ROOT = pathlib.Path(__file__).resolve().parent.parent
B, D, C, R = '\033[1m', '\033[2m', '\033[36m', '\033[0m'
log = json.loads((ROOT / 'out' / 'trace.json').read_text())['log']
NEEDS = {1: ['backbone', 'head1'], 2: ['backbone', 'fpn', 'head2'], 3: ['backbone', 'fpn', 'pan', 'head3'],
         4: ['backbone', 'fpn', 'pan', 'refine', 'head_final']}


def cycles(L, lanes):
    """SCALE-Sim's stored result for this layer's shape (lanes 2 = INT8, 4 = INT4)."""
    f = ROOT / 'out' / 'spec44_cache' / f"L{lanes}_i{L['ih']}_k{L['k']}_s{L['s']}_c{L['cin']}_f{L['cout']}.json"
    return json.loads(f.read_text())['cycles']


part = sys.argv[1] if len(sys.argv) > 1 else 'all'
shapes = collections.Counter((L['ih'], L['k'], L['s'], L['cin'], L['cout']) for L in log)

if part in ('all', 'shapes'):
    print(f'\n{B}{C}── 151 layers -> {len(shapes)} unique shapes{R}')
    print(f'   layers traced: {len(log)}    unique (H, k, s, Cin, Cout): {len(shapes)}    reuse: {len(log) / len(shapes):.2f}x')
    print(f'   {D}{"shape (H,k,s,Cin,Cout)":28}{"copies":>7}{"INT8 cycles":>14}{"INT4 cycles":>14}{R}')
    for sh, n in sorted(shapes.items(), key=lambda x: -x[1]):
        L = dict(zip(['ih', 'k', 's', 'cin', 'cout'], sh))
        print(f'   {str(sh):28}{n:>7}{int(cycles(L, 2)):>14,}{int(cycles(L, 4)):>14,}')

if part in ('all', 'exits'):
    print(f'\n{B}{C}── Latency of each exit (sum of its layers, SCALE-Sim cycles){R}')
    print(f'   {D}{"exit":6}{"layers":>7}{"INT8 cycles":>15}{"ms @200MHz":>12}{"INT4 cycles":>15}{"ms @200MHz":>12}{"fps INT8/INT4":>16}{R}')
    for e, stages in NEEDS.items():
        ls = [L for L in log if L['stage'] in stages]
        c8 = sum(cycles(L, 2) for L in ls)
        c4 = sum(cycles(L, 4) for L in ls)
        print(f'   {e:<6}{len(ls):>7}{int(c8):>15,}{c8 / 2e5:>12.1f}{int(c4):>15,}{c4 / 2e5:>12.1f}{2e8 / c8:>9.2f} / {2e8 / c4:.2f}')
    print(f'   {D}ms = cycles / (200 MHz x 1000).  At 100 or 150 MHz, divide by that clock instead.{R}\n')
