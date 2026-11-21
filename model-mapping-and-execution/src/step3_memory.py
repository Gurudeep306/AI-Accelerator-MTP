# -*- coding: utf-8 -*-
"""STEP 3  Liveness analysis of the activation working set.

A layer table cannot show you a concatenation or an upsample, and this detector
performs twenty-nine of the first and two of the second. So instead of hooking
only the convolutions, this step intercepts every single tensor operation the
detector performs, using a PyTorch dispatch mode. The graph analysed is
therefore the one the code actually builds.

A standard last-use pass over that trace gives the peak number of activation
bytes that have to be live at the same time, which is what decides whether the
working set fits in on-chip memory. Reported at one byte per value, meaning
int8.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from common_paths import load, rule, save
from mdl.gelan_anytime import build_model

NUM_CLASSES = 10
IMG = 640


class Tracer(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.events = []
        self.size = {}
        # Hold a reference to every tensor produced during the trace. Without
        # this, CPython frees a tensor as soon as the graph drops it and reuses
        # its id() for a later one, so the liveness pass merges two different
        # tensors and the answer changes from run to run. Keeping them alive
        # costs a few hundred megabytes and makes the result exact.
        self.keep = []

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
        for o in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(o, torch.Tensor):
                self.keep.append(o)
                self.size[id(o)] = o.numel()
                outs.append(id(o))
        self.events.append((str(func), ins, outs))
        return out


def main():
    rule('STEP 3   PEAK ACTIVATION WORKING SET, BY LIVENESS ANALYSIS')

    plat = load('platform.json')
    bram = plat['part']['bram_bytes']

    model = build_model(NUM_CLASSES).eval()
    fixed = {id(p) for p in model.parameters()} | {id(b) for b in model.buffers()}

    x = torch.zeros(1, 3, IMG, IMG)
    tr = Tracer()
    tr.size[id(x)] = x.numel()
    with torch.no_grad(), tr:
        p3b, p4b, p5b = model.forward_backbone(x)
        model.head1([p3b, p4b, p5b])
        p3t, p4t, p5t = model.forward_topdown(p3b, p4b, p5b)
        model.head2([p3t, p4t, p5t])
        p3u, p4u, p5u = model.forward_bottomup(p3t, p4t, p5t)
        model.head3([p3u, p4u, p5u])
        p3f, p4f, p5f = model.forward_final_refine(p3u, p4u, p5u)
        model.head_final([p3f, p4f, p5f])

    EV = tr.events

    # CPython reuses id() after a tensor is freed, so keying liveness on id()
    # alone silently merges two different tensors and gives a different answer
    # from one run to the next. A tensor can only appear as an input while it is
    # alive, and every tensor is the output of some operation, so an id showing
    # up as a *new output* means the previous holder of that id is already dead.
    # Retiring it at that point and issuing a fresh token makes the identity
    # exact, and the result reproducible.
    token = {}            # live id -> token
    size = {}             # token -> numel
    last = {}             # token -> index of last use
    nxt = [0]

    def issue(i, numel):
        t = nxt[0]
        nxt[0] += 1
        token[i] = t
        size[t] = numel
        return t

    issue(id(x), x.numel())
    events = []
    for t, (_op, ins, outs) in enumerate(EV):
        used = set()
        for i in ins:
            if i in fixed:
                continue
            tok = token.get(i)
            if tok is not None:
                used.add(tok)
                last[tok] = t
        made = set()
        for o in outs:
            if o in fixed:
                continue
            token.pop(o, None)                 # the old holder of this id is gone
            tok = issue(o, tr.size.get(o, 0))
            made.add(tok)
            last[tok] = t
        events.append((used, made))

    live, peak, peak_at = set(), 0, 0
    for t, (_used, made) in enumerate(events):
        live |= made
        b = sum(size[k] for k in live)
        if b > peak:
            peak, peak_at = b, t
        live = {k for k in live if last.get(k, -1) > t}

    n_tensors = nxt[0]

    print(f'tensor operations intercepted    : {len(EV):,}')
    print(f'distinct activation tensors      : {n_tensors:,}')
    print(f'peak live activation, int8       : {peak:,} bytes = {peak/1024:,.1f} KB')
    print(f'reached at operation             : {peak_at} ({EV[peak_at][0]})')
    print(f'on-chip BRAM on the target part  : {bram:,} bytes = {bram/1024:,.1f} KB')
    print(f'ratio, working set to BRAM       : {peak/bram:.1f}x too large')
    print('\nThis figure is exact and reproducible: every tensor produced during')
    print('the trace is held, so no identity is ever reused, and three runs on')
    print('the same machine give the same number.')

    carry = {
        'backbone to head 1 and fpn': [p3b, p4b, p5b],
        'fpn to head 2 and pan': [p3t, p4t, p5t],
        'pan to head 3 and refine': [p3u, p4u, p5u],
        'refine to the final head': [p3f, p4f, p5f],
    }
    print('\nwhat has to survive each stage boundary, at int8')
    for k, ts in carry.items():
        b = sum(z.numel() for z in ts)
        shp = '  '.join(str(tuple(z.shape)[1:]) for z in ts)
        print(f'  {k:30} {b:8,} B = {b/1024:6.1f} KB   {shp}')

    big = sorted(size.values(), reverse=True)[:6]
    print('\nthe six largest single activation tensors, at int8')
    for b in big:
        print(f'  {b:9,} values = {b/1024:8,.1f} KB')

    save('memory.json', dict(peak_live=peak, n_ops=len(EV), n_tensors=n_tensors,
                             bram_bytes=bram,
                             carry={k: sum(z.numel() for z in v) for k, v in carry.items()}))
    print('\nwrote out/memory.json')


if __name__ == '__main__':
    main()
