# -*- coding: utf-8 -*-
"""STEP 2  Check the trace against the layer table shipped with the detector.

This is the gate. If it fails, nothing downstream can be trusted, because every
later number is built on the shapes recovered in step 1. Eight fields are
compared for every one of the 151 convolutions.
"""
import csv

from common_paths import INPUTS, rule, read, save

FIELDS = ['kernel', 'stride', 'cin', 'cout', 'in_hw', 'macs', 'weights', 'stage']


def main():
    rule('STEP 2   CROSS-CHECK AGAINST THE SHIPPED LAYER TABLE')

    mine = read('trace.json')['log']
    theirs = list(csv.DictReader((INPUTS / 'accelerator_workload.csv').open()))

    print(f'rows in accelerator_workload.csv : {len(theirs)}')
    print(f'convolutions traced in step 1    : {len(mine)}')
    if len(mine) != len(theirs):
        print('\nMISMATCH in row count. Stopping.')
        raise SystemExit(1)

    bad = []
    for i, (a, b) in enumerate(zip(mine, theirs)):
        ih = int(str(b['in_hw']).split('x')[0])
        pairs = [
            ('kernel', a['k'], int(str(b['kernel']).split('x')[0])),
            ('stride', a['s'], int(b['stride'])),
            ('cin', a['cin'], int(b['cin'])),
            ('cout', a['cout'], int(b['cout'])),
            ('in_hw', a['ih'], ih),
            ('macs', a['macs'], int(b['macs'])),
            ('weights', a['wbytes'], int(b['weights'])),
            ('stage', a['stage'], b['stage']),
        ]
        for f, m, t in pairs:
            if m != t:
                bad.append((i, b.get('layer', '?'), f, m, t))

    n = len(mine) * len(FIELDS)
    print(f'\nlayers compared                  : {len(mine)}')
    print(f'fields compared                  : {n:,}')
    print(f'mismatches                       : {len(bad)}')
    for r in bad[:20]:
        print('    layer', r[0], r[1], 'field', r[2], 'traced', r[3], 'table', r[4])

    mt = sum(a['macs'] for a in mine)
    mh = sum(int(b['macs']) for b in theirs)
    wt = sum(a['wbytes'] for a in mine)
    wh = sum(int(b['weights']) for b in theirs)
    bias = sum(int(b['biases']) for b in theirs)
    print(f'\nmultiply-add total   traced {mt:,}   table {mh:,}   difference {mt-mh}')
    print(f'weight total         traced {wt:,}   table {wh:,}   difference {wt-wh}')
    print(f'folded biases, from the table    : {bias:,}')
    print(f'weights plus biases at int8      : {(wh+bias)/1024:,.1f} KB')

    verdict = 'PASS' if not bad and mt == mh else 'FAIL'
    print(f'\nverdict                          : {verdict}')
    save('crosscheck.json', dict(fields=n, mismatches=len(bad), verdict=verdict,
                                 bias_count=bias))
    if verdict == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
