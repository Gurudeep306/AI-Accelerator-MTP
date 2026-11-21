# -*- coding: utf-8 -*-
"""Worker: one layer shape through SCALE-Sim, in its own process.

A separate process per shape is not fastidiousness. SCALE-Sim holds on to memory
across the layers of a single call, so a long run climbs until the operating
system kills it with no traceback at all. One process per shape puts a hard
ceiling on that, and a shape that is too large for the machine fails alone
instead of taking the run down with it.

    python3 src/sim_one.py <rows> <cols> <dataflow> <ih> <k> <s> <cin> <cout>
"""
import contextlib
import csv
import io
import json
import resource
import shutil
import sys
import time

from common_paths import OUT
from sim_config import CFG, LAYOUT_HEADER, LAYOUT_ROW, patch_scalesim
from sim_shapes import CACHE, key


def main():
    rows, cols, df = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ih, k, s, cin, cout = (int(x) for x in sys.argv[4:9])
    sh = (ih, k, s, cin, cout)
    name = key(sh, rows, cols, df)

    with contextlib.redirect_stdout(io.StringIO()):
        patch_scalesim()
    from scalesim.scale_sim import scalesim

    work = OUT / 'scalesim_runs' / name
    work.mkdir(parents=True, exist_ok=True)

    # SCALE-Sim convolves without implicit padding; the detector pads to keep
    # its resolution, so the map is enlarged by k-1 before it is handed over.
    ph = ih + (k - 1)

    topo = work / 'topology.csv'
    topo.write_text('Layer name,IFMAP Height,IFMAP Width,Filter Height,'
                    'Filter Width,Channels,Num Filter,Strides,\n'
                    f'L,{ph},{ph},{k},{k},{cin},{cout},{s},\n')
    lay = work / 'layout.csv'
    lay.write_text(LAYOUT_HEADER + f'L,{LAYOUT_ROW}\n')
    cfg = work / 'run.cfg'
    cfg.write_text(CFG.format(name='run', rows=rows, cols=cols, df=df))

    t0 = time.time()
    sim = scalesim(save_disk_space=True, verbose=False, config=str(cfg),
                   topology=str(topo), layout=str(lay))
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run_scale(top_path=str(work))
    el = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    rep = work / 'run' / 'COMPUTE_REPORT.csv'
    with rep.open() as f:
        row = next(csv.DictReader(f))
    kk = {a.strip(): b.strip() for a, b in row.items() if a}

    bw = {}
    bwf = work / 'run' / 'BANDWIDTH_REPORT.csv'
    if bwf.exists():
        with bwf.open() as f:
            b = next(csv.DictReader(f))
        bw = {a.strip(): b[a].strip() for a in b if a and a.strip()}

    oh = (ph - k) // s + 1
    macs = oh * oh * k * k * cin * cout
    cyc = float(kk['Total Cycles'])

    out = dict(shape=list(sh), rows=rows, cols=cols, df=df,
               padded_ifmap=ph, out_hw=oh, macs=macs,
               cycles=cyc,
               stall_cycles=float(kk.get('Stall Cycles', 0) or 0),
               util=float(kk.get('Overall Util %', 0) or 0),
               mapping_eff=float(kk.get('Mapping Efficiency %', 0) or 0),
               compute_util=float(kk.get('Compute Util %', 0) or 0),
               bandwidth=bw, seconds=el, peak_rss_gb=peak,
               simulator='SCALE-Sim, scalesim-project/SCALE-Sim')

    CACHE.mkdir(exist_ok=True)
    (CACHE / (name + '.json')).write_text(json.dumps(out))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
