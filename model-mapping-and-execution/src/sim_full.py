# -*- coding: utf-8 -*-
"""Drive SCALE-Sim across the whole detector, one configuration at a time.

Every cycle count this produces is SCALE-Sim's own, layer by layer. Nothing is
modelled, extrapolated or scaled. The only arithmetic performed here is adding
SCALE-Sim's per-layer numbers together to get per-stage and per-answer totals.

Layers are handed over in slices, each in its own process, because SCALE-Sim
holds on to memory across the layers of a single call and is killed by the
operating system long before 151 layers are done. A slice is sized by how much
work it contains rather than by how many layers, since one 640 by 640 layer
costs more than forty 20 by 20 ones.

    python3 src/sim_full.py <rows> <cols> <dataflow> [size] [--budget N]
"""
import json
import subprocess
import sys
import time

from common_paths import OUT, SRC, read, save
from sim_chunk import layers_for

# Work in a slice, in units of output-pixel times operand-row. Tuned so that a
# slice stays near a gigabyte, which is what keeps the process alive.
BUDGET = 60_000_000
MAX_LAYERS = 12


def cost(r):
    oh = (r['ih'] + r['k'] - 1 - r['k']) // r['s'] + 1
    return oh * oh * (r['k'] * r['k'] * r['cin'])


def slices(layers, budget=BUDGET):
    out, cur, acc = [], [], 0
    for i, r in enumerate(layers):
        c = cost(r)
        if cur and (acc + c > budget or len(cur) >= MAX_LAYERS):
            out.append((cur[0], cur[-1] + 1))
            cur, acc = [], 0
        cur.append(i)
        acc += c
    if cur:
        out.append((cur[0], cur[-1] + 1))
    return out


def main():
    rows, cols = int(sys.argv[1]), int(sys.argv[2])
    df = sys.argv[3]
    size = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith('-') else '640'
    budget = BUDGET
    if '--budget' in sys.argv:
        budget = int(sys.argv[sys.argv.index('--budget') + 1])

    layers = layers_for(size)
    sl = slices(layers, budget)
    name = f'{rows}x{cols}_{df}_{size}'

    print(f'\nSCALE-Sim over the whole detector: array {rows}x{cols}, '
          f'dataflow {df}, input {size}')
    print(f'{len(layers)} layers in {len(sl)} slices, each in its own process\n',
          flush=True)

    t0 = time.time()
    per, failed = [], []
    for lo, hi in sl:
        cf = OUT / f'chunk_{rows}x{cols}_{df}_{size}_{lo:03d}.json'
        if cf.exists():
            d = json.loads(cf.read_text())
            per += d['per_layer']
            print(f'  layers {lo:3d}-{hi-1:3d}  (already done)', flush=True)
            continue
        p = subprocess.run([sys.executable, str(SRC / 'sim_chunk.py'),
                            str(rows), str(cols), df, size, str(lo), str(hi)],
                           capture_output=True, text=True)
        if p.returncode != 0 or not cf.exists():
            why = (p.stderr.strip().splitlines() or ['killed, most likely out of memory'])[-1]
            print(f'  layers {lo:3d}-{hi-1:3d}  FAILED: {why}', flush=True)
            failed.append((lo, hi))
            continue
        sys.stdout.write(p.stdout)
        sys.stdout.flush()
        per += json.loads(cf.read_text())['per_layer']

    el = time.time() - t0
    if failed:
        print(f'\n{len(failed)} slices did not finish. Re-run with a smaller '
              f'--budget to split them further;')
        print('slices that already succeeded are cached and will not be repeated.')
        return 1

    total = sum(p['cycles'] for p in per)
    stage = {}
    for p in per:
        a = stage.setdefault(p['stage'], dict(cycles=0.0, macs=0, convs=0,
                                              wbytes=0, obytes=0))
        a['cycles'] += p['cycles']
        a['macs'] += p['macs']
        a['convs'] += 1
        a['wbytes'] += p['wbytes']
        a['obytes'] += p['obytes']

    save(f'sim_{name}.json', dict(name=name, rows=rows, cols=cols, df=df,
                                  size=size, seconds=el, total_cycles=total,
                                  per_layer=per, stage=stage))
    mac = sum(p['macs'] for p in per)
    print(f'\n{len(per)} layers, {total:,.0f} cycles, '
          f'array utilisation {100*mac/(total*rows*cols):.1f}%, {el:.0f}s')
    print(f'wrote out/sim_{name}.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
