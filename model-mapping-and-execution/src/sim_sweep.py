# -*- coding: utf-8 -*-
"""Sweep SCALE-Sim over the configurations the study needs.

Largest arrays first, because they are the quickest, so the cheap results are
on disk early and an interrupted sweep still leaves something usable. Everything
is cached per shape, so stopping and restarting costs nothing.

    python3 src/sim_sweep.py             the configurations the tables need
    python3 src/sim_sweep.py --tier 1    only the weight-stationary set at 640
"""
import subprocess
import sys
import time

from common_paths import SRC, load

ARRAYS = [128, 64, 32, 24, 16, 12, 8]


def main():
    plat = load('platform.json')['sweep']
    sizes = [str(z) for z in plat['input_sizes']]
    big = sizes[0]

    tier = None
    if '--tier' in sys.argv:
        tier = int(sys.argv[sys.argv.index('--tier') + 1])

    jobs = []
    # 1  the dataflow every result uses, at the trained resolution
    for n in ARRAYS:
        jobs.append((1, n, 'ws', [big]))
    # 2  the other two dataflows, for the design-space table
    for n in ARRAYS:
        for df in ('os', 'is'):
            jobs.append((2, n, df, [big]))
    # 3  the smaller input sizes, for the resolution tables
    for n in ARRAYS:
        jobs.append((3, n, 'ws', sizes[1:]))

    if tier:
        jobs = [j for j in jobs if j[0] == tier]

    print(f'{len(jobs)} configurations to cover\n', flush=True)
    t0 = time.time()
    for i, (t, n, df, szs) in enumerate(jobs, 1):
        print(f'--- [{i}/{len(jobs)}] tier {t}: {n}x{n} {df} '
              f'{",".join(szs)}', flush=True)
        p = subprocess.run([sys.executable, str(SRC / 'sim_shapes.py'),
                            str(n), str(n), df, *szs])
        if p.returncode != 0:
            print(f'    configuration failed, continuing', flush=True)
    print(f'\nsweep finished in {(time.time()-t0)/60:.1f} minutes')


if __name__ == '__main__':
    raise SystemExit(main())
