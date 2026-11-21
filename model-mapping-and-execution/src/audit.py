# -*- coding: utf-8 -*-
"""Where every number in the results came from.

This computes nothing. It opens the files the results were built from and
reports, for each table in the paper, which file holds its numbers and how they
got there. Run it when someone asks whether the latencies are really the
simulator's.

    python3 src/audit.py
"""
import json
import random

from common_paths import OUT, load, read, rule
from sim_shapes import CACHE, key, layers_for, shape_of
import sim_expand as E


def main():
    R = read('results.json')
    TR = read('trace.json')
    CR = read('crosscheck.json')

    rule('AUDIT   WHERE EVERY NUMBER CAME FROM')

    cached = sorted(CACHE.glob('*.json'))
    print(f'per-shape SCALE-Sim results on disk : {len(cached)}')
    if cached:
        one = json.loads(cached[0].read_text())
        print(f'simulator recorded in them          : {one["simulator"]}')
    print(f'results.json says the simulator was : {R.get("simulator")}')

    print('\nEvery table in the paper, and the file its numbers live in')
    print(f'{"table":8} {"what it holds":34} {"produced by":28} {"file":22}')
    rows = [
        ('I', 'layer counts, MACs, weights', 'executing the detector', 'out/trace.json'),
        ('II', 'simulator build, shape counts', 'the sweep, plus the checks', 'out/results.json'),
        ('III', 'MACs per answer', 'executing the detector', 'out/trace.json'),
        ('IV', 'cycles per frame', 'SCALE-Sim, per layer', 'out/shape_cache/'),
        ('V', 'latency on the platform', 'SCALE-Sim, then x derate', 'out/shape_cache/'),
        ('VI', 'bytes moved, compute time', 'traced bytes, SCALE-Sim time', 'both'),
        ('VII', 'latency and accuracy', 'SCALE-Sim and measured mAP', 'both'),
        ('VIII', 'smallest array that fits', 'SCALE-Sim, per array size', 'out/shape_cache/'),
    ]
    for a, b, c, d in rows:
        print(f'{a:8} {b:34} {c:28} {d:22}')

    print('\nThe accuracy column is not simulated and could not be. It is copied')
    print('from inputs/measured_accuracy.json, which records the section of the')
    print('source document each figure came from.')

    rule('CHECK 1   EVERY LATENCY TRACES TO A SIMULATOR FILE')
    arr = R['built']['array'][0]
    big = str(load('platform.json')['sweep']['input_sizes'][0])
    layers = layers_for(big)
    missing, seen = [], {}
    for r in layers:
        sh = shape_of(r)
        f = CACHE / (key(sh, arr, arr, 'ws') + '.json')
        if not f.exists():
            missing.append(sh)
        else:
            seen[sh] = f
    print(f'layers in the detector              : {len(layers)}')
    print(f'distinct shapes behind them         : {len(seen) + len(set(missing))}')
    print(f'shapes with a SCALE-Sim file        : {len(seen)}')
    print(f'shapes with no file                 : {len(set(missing))}')

    print('\nThree layers picked at random, and the file each one reads:')
    random.seed(0)
    for r in random.sample(layers, min(3, len(layers))):
        sh = shape_of(r)
        f = seen.get(sh)
        if not f:
            continue
        d = json.loads(f.read_text())
        print(f'  layer in {r["stage"]:11} shape {str(sh):26}')
        print(f'    file      out/shape_cache/{f.name}')
        print(f'    cycles    {d["cycles"]:,.0f}   util {d["util"]:.1f}%   '
              f'simulated in {d["seconds"]:.1f}s')

    rule('CHECK 2   NOTHING IS INTERPOLATED BETWEEN SHAPES')
    dc = E.duplicate_check(arr, arr, 'ws', big)
    print(f'shapes that occur more than once    : reuse factor {dc["reuse_factor"]:.2f}x')
    print(f'shapes returning two cycle counts   : {dc["inconsistent"]}')
    print('\nA shape is simulated once and the result is reused wherever that')
    print('shape appears. That is a lookup, not an interpolation: no cycle count')
    print('is ever derived from a different shape, a different array or a')
    print('different dataflow.')

    rule('CHECK 3   THE WORKLOAD MATCHES THE SHIPPED LAYER TABLE')
    print(f'fields compared                     : {CR["fields"]:,}')
    print(f'fields that disagree                : {CR["mismatches"]}')
    print(f'verdict                             : {CR["verdict"]}')

    rule('THE ONE NUMBER THAT IS NEITHER TRACED NOR SIMULATED')
    print(f'SCALE-Sim utilisation at {R["built"]["array"][0]}x{R["built"]["array"][1]}'
          f'          : {R["sim_util"]:.1f}%')
    print(f'utilisation measured in the RTL     : {100*R["measured_util"]:.1f}%')
    print(f'derating applied to platform rows   : x{R["derate"]:.2f}')
    print('\nThat ratio is the only adjustment made to a simulated cycle count')
    print('anywhere in the study. Both halves of it are printed so it can be')
    print('checked, or divided back out. Tables IV and VIII do not carry it.')

    rule('WHAT IS NOT CLAIMED')
    for line in (
        'Sparsity is not simulated. It needs the trained weights, which are not',
        '  here, so the latencies are upper bounds in that respect.',
        'Accuracy does not move with the hardware setting. A reduced numeric',
        '  precision would change it, and quantifying that needs a calibration run.',
        'Post-processing is excluded. It runs on a fixed candidate grid, so it is',
        '  bounded and content-independent, but it is not zero.',
        'The derating was measured at one array size and is assumed to hold at',
        '  others.',
    ):
        print(line)


if __name__ == '__main__':
    main()
