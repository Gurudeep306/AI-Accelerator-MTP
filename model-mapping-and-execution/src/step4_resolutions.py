# -*- coding: utf-8 -*-
"""STEP 4  Run the same detector at five input sizes.

Resolution is treated as a measured axis rather than a scaling assumption. The
detector is executed again at each size, with the same hooks as step 1, so the
multiply-add counts for 320 by 320 are recovered the same way as those for 640
by 640 and are not divided down from them.
"""
from common_paths import load, rule, save, read
from mdl.gelan_anytime import build_model
from step1_trace import run_once, STAGES, NUM_CLASSES


def main():
    rule('STEP 4   THE SAME DETECTOR AT FIVE INPUT SIZES')

    sizes = load('platform.json')['sweep']['input_sizes']
    model = build_model(NUM_CLASSES).eval()
    out = {}

    for S in sizes:
        LOG, cells, _ = run_once(model, S)
        per = {s: sum(r['macs'] for r in LOG if r['stage'] == s) for s in STAGES}
        out[str(S)] = dict(total=sum(r['macs'] for r in LOG), per_stage=per, cells=cells,
                           layers=[dict(stage=r['stage'], macs=r['macs'], ih=r['ih'],
                                        k=r['k'], s=r['s'], cin=r['cin'], cout=r['cout'],
                                        wbytes=r['wbytes'], obytes=r['obytes'])
                                   for r in LOG])

    base = out[str(sizes[0])]['total']
    print(f'{"input":>9} {"convs":>6} {"MACs (G)":>10} {"vs 640":>8} '
          f'{"cells/head":>11} {"head 1 only (G)":>16}')
    for S in sizes:
        o = out[str(S)]
        h1 = o['per_stage']['backbone'] + o['per_stage']['head1']
        print(f'{S:5d} sq {len(o["layers"]):6d} {o["total"]/1e9:10.3f} '
              f'{o["total"]/base:7.2f}x {o["cells"]:11,} {h1/1e9:16.3f}')

    print('\nThe layer count never changes. Only the feature maps shrink, so the')
    print('cost falls roughly with the pixel count but not exactly with it.')

    save('resolutions.json', out)
    print('\nwrote out/resolutions.json')


if __name__ == '__main__':
    main()
