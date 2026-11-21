# -*- coding: utf-8 -*-
"""Check the closed-form cycle equations against every SCALE-Sim result on disk.

    WS: ceil(K/R) ceil(N/C) (2R + C + M - 2) - 1
    OS: ceil(M/R) ceil(N/C) (R + C + K - 2) - 1
    IS: ceil(K/R) ceil(M/C) (2R + C + N - 2) - 1

M uses SCALE-Sim's own output size, ceil((H + k - 1 - k) / s) + 1.

    python3 tools/check_formula.py
"""
import collections, glob, json, math, pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / 'out'


def model(df, R, C, M, N, K):
    if df == 'ws':
        return math.ceil(K / R) * math.ceil(N / C) * (2 * R + C + M - 2) - 1
    if df == 'os':
        return math.ceil(M / R) * math.ceil(N / C) * (R + C + K - 2) - 1
    return math.ceil(K / R) * math.ceil(M / C) * (2 * R + C + N - 2) - 1


cnt, ok = collections.Counter(), collections.Counter()
for f in glob.glob(str(OUT / 'shape_cache' / '*.json')):
    d = json.loads(open(f).read())
    ih, k, s, cin, cout = d['shape']
    o = math.ceil((d['padded_ifmap'] - k) / s) + 1
    key = f"{d['rows']}x{d['cols']} {d['df'].upper()}"
    cnt[key] += 1
    ok[key] += model(d['df'], d['rows'], d['cols'], o * o, cout, k * k * cin) == d['cycles']
for f in glob.glob(str(OUT / 'spec44_cache' / '*.json')):
    d = json.loads(open(f).read())
    ih, k, s, cin, cout = d['orig_shape']
    o = math.ceil((ih + k - 1 - k) / s) + 1
    key = f"4x4 OS spec, {d['lanes']} lanes"
    cnt[key] += 1
    ok[key] += model('os', 4, 4, o * o, cout, k * k * d['packed_cin']) == d['cycles']
print(f'{"configuration":24} {"shapes":>7} {"equal to the cycle":>20}')
for k in sorted(cnt):
    print(f'{k:24} {cnt[k]:7} {ok[k]:20}')
print(f'\ntotal {sum(cnt.values())} shapes, {sum(ok.values())} equal to the cycle')
