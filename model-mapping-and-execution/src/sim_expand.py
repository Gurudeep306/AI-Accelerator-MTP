# -*- coding: utf-8 -*-
"""Turn the cache of per-shape SCALE-Sim results into per-layer, per-stage and
per-answer totals.

The only arithmetic here is addition. Every cycle count is SCALE-Sim's own, for
that exact layer shape and that exact configuration. Nothing is modelled,
interpolated or scaled, and a configuration that was never simulated is reported
as missing rather than estimated.
"""
from common_paths import read
from sim_shapes import layers_for, load, shape_of

STAGES = ['backbone', 'head1', 'fpn', 'head2', 'pan', 'head3', 'refine',
          'head_final']
HEADS = ['head1', 'head2', 'head3', 'final']
NEEDS = {'head1': ['backbone', 'head1'],
         'head2': ['backbone', 'fpn', 'head2'],
         'head3': ['backbone', 'fpn', 'pan', 'head3'],
         'final': ['backbone', 'fpn', 'pan', 'refine', 'head_final']}
UPTO = {'head1': STAGES[:2], 'head2': STAGES[:4],
        'head3': STAGES[:6], 'final': STAGES[:8]}


def have(rows, cols, df, size):
    """True when every layer of this configuration has been simulated."""
    return all(load(shape_of(r), rows, cols, df) is not None
               for r in layers_for(size))


def per_layer(rows, cols, df, size):
    out = []
    for i, r in enumerate(layers_for(size)):
        d = load(shape_of(r), rows, cols, df)
        if d is None:
            return None
        out.append(dict(index=i, stage=r['stage'], shape=list(shape_of(r)),
                        macs=r['macs'], wbytes=r['wbytes'], obytes=r['obytes'],
                        cycles=d['cycles'], util=d['util'],
                        stall=d.get('stall_cycles', 0),
                        mapping_eff=d.get('mapping_eff', 0)))
    return out


def by_stage(rows, cols, df, size):
    pl = per_layer(rows, cols, df, size)
    if pl is None:
        return None
    st = {}
    for p in pl:
        a = st.setdefault(p['stage'], dict(cycles=0.0, macs=0, convs=0,
                                           wbytes=0, obytes=0, stall=0.0))
        a['cycles'] += p['cycles']
        a['macs'] += p['macs']
        a['convs'] += 1
        a['wbytes'] += p['wbytes']
        a['obytes'] += p['obytes']
        a['stall'] += p['stall']
    return st


def answers(rows, cols, df, size, derate=1.0):
    """Cycles at which each of the four answers is ready, both schedules."""
    st = by_stage(rows, cols, df, size)
    if st is None:
        return None
    out = {}
    for h in HEADS:
        out[h] = dict(
            minimal=sum(st[s]['cycles'] for s in NEEDS[h]) * derate,
            sequential=sum(st[s]['cycles'] for s in UPTO[h]) * derate,
            macs_minimal=sum(st[s]['macs'] for s in NEEDS[h]),
            macs_sequential=sum(st[s]['macs'] for s in UPTO[h]),
            weights=sum(st[s]['wbytes'] for s in NEEDS[h]))
    return out


def total(rows, cols, df, size):
    st = by_stage(rows, cols, df, size)
    return None if st is None else sum(v['cycles'] for v in st.values())


def duplicate_check(rows, cols, df, size):
    """Every shape that occurs more than once must have one cycle count."""
    seen = {}
    for r in layers_for(size):
        sh = shape_of(r)
        d = load(sh, rows, cols, df)
        if d is None:
            continue
        seen.setdefault(sh, set()).add(d['cycles'])
    repeated = {k: v for k, v in seen.items() if v}
    bad = {k: v for k, v in repeated.items() if len(v) > 1}
    n_layers = len(layers_for(size))
    return dict(layers=n_layers, shapes=len(seen),
                reuse_factor=(n_layers / len(seen)) if seen else 0,
                inconsistent=len(bad))
