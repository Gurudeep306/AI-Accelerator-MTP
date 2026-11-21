# -*- coding: utf-8 -*-
"""Real liveness analysis of the activation working set.

Every tensor operation the friend's model performs is intercepted, so the
graph is the one his code actually builds -- including the concatenations
inside each ELAN block and the two upsamples, which a layer table cannot
show. Then a standard last-use pass gives the peak number of activation
bytes that must be live at once. Reported at 1 byte per value (int8).
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from common_paths import rule, save
from mdl.gelan_anytime import build_model
EV = []          # (op, [input tensor ids], [(output id, numel)])
SIZE = {}


class Trace(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        def ids(o):
            r = []
            if isinstance(o, torch.Tensor):
                r.append(id(o))
            elif isinstance(o, (list, tuple)):
                for x in o:
                    r += ids(x)
            return r

        ins = []
        for a in list(args) + list(kwargs.values()):
            ins += ids(a)
        outs = []
        for i, o in zip(ids(out), [out] if isinstance(out, torch.Tensor) else
                        (out if isinstance(out, (list, tuple)) else [])):
            pass
        ot = out if isinstance(out, (list, tuple)) else [out]
        for o in ot:
            if isinstance(o, torch.Tensor):
                SIZE[id(o)] = o.numel()
                outs.append(id(o))
        EV.append((str(func), ins, outs))
        return out


model = build_model(10).eval()
PARAM = {id(p) for p in model.parameters()}
BUF = {id(b) for b in model.buffers()}

x = torch.zeros(1, 3, 640, 640)
SIZE[id(x)] = x.numel()

with torch.no_grad(), Trace():
    p3b, p4b, p5b = model.forward_backbone(x)
    o1 = model.head1([p3b, p4b, p5b])
    p3t, p4t, p5t = model.forward_topdown(p3b, p4b, p5b)
    o2 = model.head2([p3t, p4t, p5t])
    p3u, p4u, p5u = model.forward_bottomup(p3t, p4t, p5t)
    o3 = model.head3([p3u, p4u, p5u])
    p3f, p4f, p5f = model.forward_final_refine(p3u, p4u, p5u)
    o4 = model.head_final([p3f, p4f, p5f])

# last use of every activation tensor
last = {}
first = {}
for t, (op, ins, outs) in enumerate(EV):
    for i in ins + outs:
        if i in PARAM or i in BUF:
            continue
        last[i] = t
        first.setdefault(i, t)
first[id(x)] = 0

live, peak, peak_at = set(), 0, 0
curve = []
for t, (op, ins, outs) in enumerate(EV):
    for o in outs:
        if o not in PARAM and o not in BUF:
            live.add(o)
    b = sum(SIZE.get(i, 0) for i in live)
    curve.append(b)
    if b > peak:
        peak, peak_at = b, t
    for i in list(live):
        if last.get(i, -1) <= t:
            live.discard(i)

print(f'tensor operations traced     : {len(EV):,}')
print(f'distinct activation tensors  : {len(last):,}')
print(f'peak live activation (int8)  : {peak:,} bytes = {peak/1024:.1f} KB')
print(f'peak reached at op           : {peak_at} ({EV[peak_at][0]})')
print(f'weights + folded bias (int8) : 5,438,500 bytes = 5.19 MB')

# the tensors that must survive a stage boundary
CARRY = {
    'after backbone  -> head1 / fpn': [p3b, p4b, p5b],
    'after fpn       -> head2 / pan': [p3t, p4t, p5t],
    'after pan       -> head3 / refine': [p3u, p4u, p5u],
    'after refine    -> final head': [p3f, p4f, p5f],
}
print('\nwhat has to survive each stage boundary, at int8')
for k, ts in CARRY.items():
    b = sum(t.numel() for t in ts)
    shp = ', '.join(f'{tuple(t.shape)[1:]}' for t in ts)
    print(f'  {k:36} {b:8,} B = {b/1024:6.1f} KB   {shp}')

big = sorted(((SIZE[i], i) for i in last), reverse=True)[:6]
print('\nthe six largest single activation tensors')
for b, i in big:
    print(f'  {b:9,} values  = {b/1024:7.1f} KB at int8')

json.dump(dict(peak_live=peak, n_ops=len(EV), n_tensors=len(last),
               curve=curve,
               carry={k: sum(t.numel() for t in v) for k, v in CARRY.items()}),
          open(SIM / 'mem.json', 'w'))
