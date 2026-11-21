# -*- coding: utf-8 -*-
"""Run SCALE-Sim for every distinct layer shape, and cache the result.

Two facts make this the right way round.

The first is that the detector repeats itself. Its four detection heads are
identical in structure and differ only in their trained weights, and the pyramid
blocks recur, so the 151 convolutions at 640 by 640 cover only 37 distinct
shapes. Across all five input sizes there are 183. Simulating shapes rather than
layers is therefore about four times less work for exactly the same answer.

The second is that SCALE-Sim gives a layer the same cycle count whenever the
shape and the configuration are the same. It is not an assumption here: the
package simulates every one of the 151 layers in place for one configuration and
checks that repeated shapes came back identical. That check is printed, and if
it ever failed this whole approach would be abandoned rather than patched.

Results are cached per shape in out/shape_cache/, keyed by shape and
configuration, so nothing is ever simulated twice no matter how many array
sizes, dataflows or input resolutions are swept afterwards. Deleting the folder
forces a clean re-run.

    python3 src/sim_shapes.py <rows> <cols> <dataflow> [sizes...]
"""
import contextlib
import hashlib
import io
import json
import resource
import subprocess
import sys
import time

from common_paths import OUT, SRC, read

CACHE = OUT / 'shape_cache'


def key(sh, rows, cols, df):
    ih, k, s, cin, cout = sh
    return f'{rows}x{cols}_{df}_i{ih}_k{k}_s{s}_c{cin}_f{cout}'


def shape_of(r):
    return (r['ih'], r['k'], r['s'], r['cin'], r['cout'])


def layers_for(size):
    if str(size) == '640':
        return read('trace.json')['log']
    return read('resolutions.json')[str(size)]['layers']


def cost(sh):
    ih, k, s, cin, cout = sh
    oh = (ih + k - 1 - k) // s + 1
    return oh * oh * k * k * cin * cout


def load(sh, rows, cols, df):
    p = CACHE / (key(sh, rows, cols, df) + '.json')
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def main():
    rows, cols, df = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sizes = sys.argv[4:] or ['640']
    CACHE.mkdir(exist_ok=True)

    shapes = []
    for sz in sizes:
        for r in layers_for(sz):
            sh = shape_of(r)
            if sh not in shapes:
                shapes.append(sh)
    shapes.sort(key=cost)

    todo = [sh for sh in shapes if load(sh, rows, cols, df) is None]
    print(f'array {rows}x{cols}, dataflow {df}, input sizes {", ".join(sizes)}')
    print(f'{len(shapes)} distinct shapes, {len(shapes)-len(todo)} already '
          f'cached, {len(todo)} to simulate', flush=True)
    if not todo:
        return 0

    t0 = time.time()
    done = 0
    for sh in todo:
        p = subprocess.run(
            [sys.executable, str(SRC / 'sim_one.py'), str(rows), str(cols), df,
             *[str(x) for x in sh]], capture_output=True, text=True)
        done += 1
        if p.returncode != 0 or load(sh, rows, cols, df) is None:
            why = (p.stderr.strip().splitlines()
                   or ['killed, most likely out of memory'])[-1][:90]
            print(f'  [{done:3d}/{len(todo)}] {str(sh):28} FAILED  {why}',
                  flush=True)
            continue
        d = load(sh, rows, cols, df)
        print(f'  [{done:3d}/{len(todo)}] {str(sh):28} '
              f'{d["cycles"]:12,.0f} cycles  {d["util"]:5.1f}% util  '
              f'{d["seconds"]:6.1f}s', flush=True)

    print(f'\n{len(todo)} shapes in {time.time()-t0:.0f}s, cached in '
          f'out/shape_cache/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
