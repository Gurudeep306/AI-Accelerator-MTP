# -*- coding: utf-8 -*-
"""Worker: run SCALE-Sim over one slice of the detector's layers.

SCALE-Sim does not release memory between the layers of a single call. Over 151
layers at 640 by 640 the resident set climbs past what an ordinary machine has
and the process is killed, silently, with no traceback. That is the whole reason
a naive full-network run fails.

The fix is not to avoid SCALE-Sim. It is to hand it a few layers at a time in a
fresh process, so the memory goes back to the operating system between slices.
This file is one slice. sim_full.py drives it and stitches the pieces together.

    python3 src/sim_chunk.py <rows> <cols> <dataflow> <size> <first> <last>
"""
import contextlib
import csv
import io
import json
import resource
import sys
import time

from common_paths import OUT, read, save
from sim_config import CFG, LAYOUT_HEADER, LAYOUT_ROW, patch_scalesim


def layers_for(size):
    if str(size) == '640':
        return read('trace.json')['log']
    return read('resolutions.json')[str(size)]['layers']


def main():
    rows, cols = int(sys.argv[1]), int(sys.argv[2])
    df, size = sys.argv[3], sys.argv[4]
    lo, hi = int(sys.argv[5]), int(sys.argv[6])

    patch_scalesim()
    from scalesim.scale_sim import scalesim

    allL = layers_for(size)
    part = allL[lo:hi]
    tag = f'{rows}x{cols}_{df}_{size}_{lo:03d}'

    work = OUT / 'scalesim_full'
    work.mkdir(exist_ok=True)

    topo = work / f'topo_{tag}.csv'
    with topo.open('w') as f:
        f.write('Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,'
                'Channels,Num Filter,Strides,\n')
        for i, r in enumerate(part):
            ph = r['ih'] + (r['k'] - 1)        # SCALE-Sim convolves 'valid'
            f.write(f'L{lo+i:03d},{ph},{ph},{r["k"]},{r["k"]},'
                    f'{r["cin"]},{r["cout"]},{r["s"]},\n')
    lay = work / f'layout_{tag}.csv'
    lay.write_text(LAYOUT_HEADER + ''.join(
        f'L{lo+i:03d},{LAYOUT_ROW}\n' for i in range(len(part))))

    cfg = work / f'{tag}.cfg'
    cfg.write_text(CFG.format(name=tag, rows=rows, cols=cols, df=df))

    t0 = time.time()
    s = scalesim(save_disk_space=True, verbose=False, config=str(cfg),
                 topology=str(topo), layout=str(lay))
    with contextlib.redirect_stdout(io.StringIO()):
        s.run_scale(top_path=str(work))
    el = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    got = []
    with (work / tag / 'COMPUTE_REPORT.csv').open() as f:
        for row in csv.DictReader(f):
            k = {kk.strip(): vv.strip() for kk, vv in row.items() if kk}
            got.append(dict(cycles=float(k['Total Cycles']),
                            util=float(k.get('Overall Util %', 0) or 0),
                            mapping=float(k.get('Mapping Efficiency %', 0) or 0)))

    bw = {}
    bwf = work / tag / 'BANDWIDTH_REPORT.csv'
    if bwf.exists():
        with bwf.open() as f:
            rd = list(csv.DictReader(f))
        for i, row in enumerate(rd):
            k = {kk.strip(): vv.strip() for kk, vv in row.items() if kk}
            bw[i] = {kk: vv for kk, vv in k.items() if 'DRAM' in kk or 'SRAM' in kk}

    out = dict(tag=tag, lo=lo, hi=hi, seconds=el, peak_rss_gb=peak,
               per_layer=[dict(index=lo + i, stage=r['stage'], ih=r['ih'],
                               k=r['k'], s=r['s'], cin=r['cin'], cout=r['cout'],
                               macs=r['macs'], wbytes=r['wbytes'],
                               obytes=r['obytes'], **g, bw=bw.get(i, {}))
                          for i, (r, g) in enumerate(zip(part, got))])
    save(f'chunk_{tag}.json', out)
    print(f'  layers {lo:3d}-{hi-1:3d}  {sum(g["cycles"] for g in got):14,.0f} '
          f'cycles  {el:7.1f}s  peak {peak:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
