# -*- coding: utf-8 -*-
"""SCALE-Sim on the specification v1.1 mesh: 4x4, output-stationary.

    python3 src/sim_spec44.py 2   # INT8, two lanes per PE
    python3 src/sim_spec44.py 4   # INT4, four lanes per PE
"""
# Run SCALE-Sim on the specification's 4x4 output-stationary mesh.
# Lane packing: an INT8 PE multiplies 2 K-elements per cycle, INT4 4, so the
# reduction length K = k*k*cin is presented to SCALE-Sim as k*k*ceil(cin/lanes).
import json, math, os, subprocess, sys, time, pathlib
PK = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PK / 'src'))
from sim_shapes import CACHE, key, shape_of, layers_for, cost
lanes = int(sys.argv[1])
OUTD = PK / 'out' / 'spec44_cache'; OUTD.mkdir(exist_ok=True)
shapes = []
for r in layers_for('640'):
    sh = shape_of(r)
    if sh not in shapes: shapes.append(sh)
shapes.sort(key=cost)
for sh in shapes:
    ih, k, s, cin, cout = sh
    dst = OUTD / f'L{lanes}_i{ih}_k{k}_s{s}_c{cin}_f{cout}.json'
    if dst.exists(): continue
    cinp = math.ceil(cin / lanes)
    # SCALE-Sim's memory grows about 1 KB per simulated cycle. A layer whose
    # filters would need more than ~3.5 M cycles is split into filter groups of
    # a multiple of 4 (the array width); in output-stationary mode the groups
    # are separate folds, so the layer total is the sum of the parts plus one
    # cycle per extra part (checked against an unsplit run in check_split.py).
    oh = math.ceil((ih + k - 1 - k) / s) + 1
    est = math.ceil(oh*oh/4) * math.ceil(cout/4) * (k*k*cinp + 6)
    parts = max(1, math.ceil(est / float(os.environ.get("SPLIT", "3.5e6"))))
    per = math.ceil(math.ceil(cout / parts) / 4) * 4
    groups = []
    left = cout
    while left > 0:
        groups.append(min(per, left)); left -= per
    subs = []
    ok = True
    for g in groups:
        psh = (ih, k, s, cinp, g)
        src = CACHE / (key(psh, 4, 4, 'os') + '.json')
        if not src.exists():
            p = subprocess.run([sys.executable, str(PK/'src'/'sim_one.py'), '4', '4', 'os', *map(str, psh)],
                               capture_output=True, text=True, cwd=str(PK))
        if not src.exists():
            print(f'{sh} lanes={lanes} part {g} FAILED', flush=True); ok = False; break
        subs.append(json.loads(src.read_text()))
    if not ok:
        continue
    d = dict(subs[0])
    d['cycles'] = sum(x['cycles'] for x in subs) + (len(subs) - 1)
    d['seconds'] = sum(x['seconds'] for x in subs)
    d['peak_rss_gb'] = max(x['peak_rss_gb'] for x in subs)
    d['macs'] = sum(x['macs'] for x in subs)
    d['util'] = sum(x['util'] * x['cycles'] for x in subs) / sum(x['cycles'] for x in subs)
    d['parts'] = [ {'cout': g, 'cycles': x['cycles']} for g, x in zip(groups, subs)]
    d['shape'] = [ih, k, s, cinp, cout]
    d.update(orig_shape=list(sh), lanes=lanes, packed_cin=cinp,
             macs_true=d['macs'] // cinp * cin)
    dst.write_text(json.dumps(d))
    print(f'{sh} lanes={lanes} cycles={d["cycles"]:,.0f} util={d["util"]:.1f} t={d["seconds"]:.0f}s rss={d["peak_rss_gb"]:.2f}GB', flush=True)
print('DONE', lanes)
