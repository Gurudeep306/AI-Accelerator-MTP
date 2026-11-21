# -*- coding: utf-8 -*-
"""STEP 1  Recover the workload by executing the detector.

Nothing here is read from a table. A forward hook is attached to every
convolution in the detector, the four incremental forward methods are called
in turn so that the stage boundaries are real, and each hook records the
shapes that actually passed through its layer.

The weights are random and the input is zeros. That is deliberate and it is
sound, because a cycle count depends only on tensor shapes, never on the
values inside them. It is also the reason this package cannot measure
sparsity or accuracy, which need the trained checkpoint.
"""
import torch
import torch.nn as nn

from common_paths import rule, save
from mdl.gelan_anytime import build_model

NUM_CLASSES = 10
IMG = 640
STAGES = ['backbone', 'head1', 'fpn', 'head2', 'pan', 'head3', 'refine', 'head_final']


def run_once(model, img, want_log=True):
    """Push one image size through the detector and return every convolution."""
    LOG, STAGE, handles = [], ['?'], []

    def mk(mod):
        def f(m, inp, out):
            x = inp[0]
            kh, kw = m.kernel_size
            g = m.groups
            macs = out.shape[2] * out.shape[3] * out.shape[1] * (x.shape[1] // g) * kh * kw
            LOG.append(dict(stage=STAGE[0], ih=x.shape[2], iw=x.shape[3], cin=x.shape[1],
                            k=kh, s=m.stride[0], p=m.padding[0],
                            oh=out.shape[2], ow=out.shape[3], cout=out.shape[1],
                            macs=macs, wbytes=m.weight.numel(),
                            ibytes=x.numel(), obytes=out.numel()))
        return f

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(mk(m)))

    x = torch.zeros(1, 3, img, img)
    with torch.no_grad():
        STAGE[0] = 'backbone'
        a, b, c = model.forward_backbone(x)
        STAGE[0] = 'head1'
        o1 = model.head1([a, b, c])
        STAGE[0] = 'fpn'
        d, e, f = model.forward_topdown(a, b, c)
        STAGE[0] = 'head2'
        model.head2([d, e, f])
        STAGE[0] = 'pan'
        g, h, i = model.forward_bottomup(d, e, f)
        STAGE[0] = 'head3'
        model.head3([g, h, i])
        STAGE[0] = 'refine'
        j, k, l = model.forward_final_refine(g, h, i)
        STAGE[0] = 'head_final'
        model.head_final([j, k, l])

    for z in handles:
        z.remove()
    cells = sum(cl.shape[2] * cl.shape[3] for cl, _, _ in o1)
    return LOG, cells, o1


def main():
    rule('STEP 1   THE WORKLOAD, RECOVERED BY EXECUTING THE DETECTOR')

    model = build_model(NUM_CLASSES).eval()
    n_par = sum(p.numel() for p in model.parameters())
    n_bn = sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm2d)
               for p in m.parameters())
    n_cw = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))

    print(f'parameters                  : {n_par:,}')
    print(f'  of which BatchNorm        : {n_bn:,}')
    print(f'  convolution weights       : {n_cw:,}')

    LOG, cells, o1 = run_once(model, IMG)
    tot = sum(r['macs'] for r in LOG)

    print(f'\nconvolutions executed       : {len(LOG)}')
    print(f'  3x3                       : {sum(1 for r in LOG if r["k"] == 3)}')
    print(f'  1x1                       : {sum(1 for r in LOG if r["k"] == 1)}')
    print(f'  stride 2                  : {sum(1 for r in LOG if r["s"] == 2)}')
    print(f'multiply-adds per frame     : {tot:,}  ({tot/1e9:.4f} G)')

    print('\nper stage, in execution order')
    print(f'{"stage":12} {"convs":>6} {"MACs (G)":>10} {"share":>7} '
          f'{"weights (KB)":>13} {"output (KB)":>12}')
    stats = {}
    for s in STAGES:
        rs = [r for r in LOG if r['stage'] == s]
        m = sum(r['macs'] for r in rs)
        w = sum(r['wbytes'] for r in rs)
        o = sum(r['obytes'] for r in rs)
        stats[s] = dict(convs=len(rs), macs=m, weights=w, out=o,
                        k3=sum(1 for r in rs if r['k'] == 3),
                        k1=sum(1 for r in rs if r['k'] == 1),
                        s2=sum(1 for r in rs if r['s'] == 2))
        print(f'{s:12} {len(rs):6d} {m/1e9:10.4f} {100*m/tot:6.1f}% '
              f'{w/1024:13,.1f} {o/1024:12,.1f}')
    wt = sum(v['weights'] for v in stats.values())
    print(f'{"total":12} {len(LOG):6d} {tot/1e9:10.4f} {100.0:6.1f}% '
          f'{wt/1024:13,.1f} {sum(v["out"] for v in stats.values())/1024:12,.1f}')

    heads = ['head1', 'head2', 'head3', 'head_final']
    hm = sum(stats[h]['macs'] for h in heads)
    hw = sum(stats[h]['weights'] for h in heads)
    print(f'\nthe four heads together     : {hm/1e9:.3f} G, {100*hm/tot:.0f}% of the frame')
    print(f'head weights of all weights : {100*hw/wt:.0f}%')

    print('\nhead output shape, one head, three pyramid levels')
    for i, (cls, obj, reg) in enumerate(o1):
        print(f'  stride {model.strides[i]:2d}  cls {tuple(cls.shape)}  '
              f'obj {tuple(obj.shape)}  reg {tuple(reg.shape)}')
    print(f'  grid cells per head       : {cells:,}')
    print(f'  values per cell           : {NUM_CLASSES} class + 1 object + 4 box '
          f'= {NUM_CLASSES + 5}')

    save('trace.json', dict(log=LOG, stats=stats, total_macs=tot, params=n_par,
                            bn_params=n_bn, conv_weights=n_cw, cells=cells))
    print('\nwrote out/trace.json')
    return tot


if __name__ == '__main__':
    main()
