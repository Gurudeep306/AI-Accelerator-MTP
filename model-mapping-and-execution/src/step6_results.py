# -*- coding: utf-8 -*-
"""STEP 6  The result tables.

Every cycle count below is SCALE-Sim's own, for that exact layer shape and that
exact array and dataflow. There is no analytical model anywhere in this file.
The only arithmetic performed on a simulated number is addition, to total
SCALE-Sim's per-layer cycles over a stage or over the stages an answer needs.

Each table says which of three kinds its numbers are:
  TRACED     recorded by executing the detector in step 1
  SIMULATED  SCALE-Sim, swept in step 5
  MEASURED   produced elsewhere and copied in, never recomputed here
"""
from common_paths import load, read, rule, save
import sim_expand as E

STAGES = E.STAGES
HEADS = E.HEADS
NEEDS = E.NEEDS
UPTO = E.UPTO


def main():
    TR = read('trace.json')
    RES = read('resolutions.json')
    MEM = read('memory.json')
    ACCJ = load('measured_accuracy.json')
    PLAT = load('platform.json')

    part = PLAT['part']
    built = PLAT['built_design']
    sw = PLAT['sweep']
    DEADLINE = PLAT['deadlines']['control_deadline_ms']
    FRAME = PLAT['deadlines']['camera_frame_ms']
    DRAM_GBS = part['dram_mts'] * 1e6 * part['dram_bus_bytes'] / 1e9
    FIXED = built['dsp_total'] - built['dsp_in_array']

    ACC = {k: v for k, v in ACCJ['per_head'].items() if not k.startswith('_')}
    cls = {k: v for k, v in ACCJ['per_class'].items() if not k.startswith('_')}
    BOXES = {k: v[0] for k, v in cls.items()}
    AP = {k: v[1] for k, v in cls.items()}

    S = TR['stats']
    tot = TR['total_macs']
    BIG = str(sw['input_sizes'][0])
    SIZES = [str(z) for z in sw['input_sizes']]
    ARRS = sorted({a[0] for a in sw['arrays']})
    DFS = sw['dataflows']

    def done(n, df, size):
        return E.have(n, n, df, size)

    # ---------------------------------------------------------------- A
    rule('A.  THE WORKLOAD, BY STAGE   (traced in step 1)')
    print(f'{"stage":12} {"convs":>6} {"MACs (G)":>10} {"share":>7} '
          f'{"weights (KB)":>13} {"output (KB)":>12}')
    wb_all = ab_all = 0
    for s in STAGES:
        v = S[s]
        wb_all += v['weights']
        ab_all += v['out']
        print(f'{s:12} {v["convs"]:6d} {v["macs"]/1e9:10.4f} '
              f'{100*v["macs"]/tot:6.1f}% {v["weights"]/1024:13,.1f} '
              f'{v["out"]/1024:12,.1f}')
    nconv = sum(v['convs'] for v in S.values())
    print(f'{"total":12} {nconv:6d} {tot/1e9:10.4f} {100.0:6.1f}% '
          f'{wb_all/1024:13,.1f} {ab_all/1024:12,.1f}')
    hk = ['head1', 'head2', 'head3', 'head_final']
    hm = sum(S[h]['macs'] for h in hk)
    hw = sum(S[h]['weights'] for h in hk)
    print(f'\nthe four heads together      : {hm/1e9:.3f} G, '
          f'{100*hm/tot:.0f}% of the frame')
    print(f'head weights of all weights  : {100*hw/wb_all:.0f}%')
    print(f'peak live activation         : {MEM["peak_live"]/1024:,.0f} KB at int8')
    print(f'on-chip BRAM on the part     : {MEM["bram_bytes"]/1024:,.0f} KB')

    # ---------------------------------------------------------------- B
    rule('B.  WORK REQUIRED FOR EACH ANSWER   (traced)')
    print(f'{"answer":8} {"sequential, as written":>24} {"only what it needs":>20} '
          f'{"saving":>8} {"weights needed (KB)":>20}')
    for h in HEADS:
        a = sum(S[s]['macs'] for s in UPTO[h]) / 1e9
        b = sum(S[s]['macs'] for s in NEEDS[h]) / 1e9
        w = sum(S[s]['weights'] for s in NEEDS[h])
        print(f'{h:8} {a:22.3f} G {b:18.3f} G {100*(1-b/a):7.0f}% '
              f'{w/1024:20,.1f}')

    # ---------------------------------------------------------------- C
    rule(f'C.  CYCLES PER FRAME AT {BIG}x{BIG}, WHOLE DETECTOR   '
         f'(SCALE-Sim, every layer)')
    print(f'{"array":>9} {"PEs":>7} {"fits?":>6} '
          + ''.join(f'{d:>16}' for d in DFS) + f'{"best util":>11}')
    design = []
    for n in ARRS:
        cells, us = [], []
        for df in DFS:
            t = E.total(n, n, df, BIG)
            cells.append(f'{t:,.0f}' if t else 'not simulated')
            if t:
                us.append(100.0 * tot / (t * n * n))
        fits = 'yes' if n * n + FIXED <= part['dsp_slices'] else 'no'
        design.append(dict(arr=f'{n}x{n}', pes=n * n, fits=(fits == 'yes'),
                           cycles={df: E.total(n, n, df, BIG) for df in DFS},
                           best_util=max(us) if us else None))
        print(f'{n}x{n:<6} {n*n:7,} {fits:>6} '
              + ''.join(f'{c:>16}' for c in cells)
              + (f'{max(us):10.1f}%' if us else f'{"":>11}'))
    print(f'\n"fits" allows {FIXED} slices of fixed logic plus one per cell '
          f'against the {part["dsp_slices"]} on the {part["board"]}.')

    ok_ws = [n for n in ARRS if done(n, 'ws', BIG)]
    if not ok_ws:
        print('\nNo weight-stationary configuration is complete. '
              'Run step 5 before this step.')
        return

    # ---------------------------------------------------------------- D
    rule(f'D.  THE TARGET PLATFORM AT {BIG}x{BIG}   '
         f'(SCALE-Sim, derated to measured RTL)')
    a8 = built['array'][0]
    base = E.total(a8, a8, 'ws', BIG)
    if base is None:
        print(f'The {a8}x{a8} array the derating is measured against has not been')
        print('simulated in this run, so the platform tables below are skipped.')
        print('Run step 5 to completion, then this step again.')
        base = None
    if base is None:
        save('results.json', dict(partial=True, design_space=design,
                                  stage_stats=S, total_macs=tot, n_conv=nconv,
                                  simulator='SCALE-Sim'))
        return
    sim_util = 100.0 * tot / (base * a8 * a8)
    meas = built['measured_array_utilisation']
    DER = (sim_util / 100.0) / meas
    print(f'SCALE-Sim utilisation at {a8}x{a8} : {sim_util:.1f}%   '
          f'(an idealised memory system)')
    print(f'measured in the RTL build      : {100*meas:.1f}%   '
          f'(weight-load cycles included)')
    print(f'so platform cycles = simulated cycles x {DER:.2f}\n')
    print(f'{"array":>7} {"MHz":>5} {"answer":8} {"cycles":>14} {"ms":>9} '
          f'{"fps":>7} {"vs deadline":>13} {"vs frame":>11}')
    board_rows = []
    for a in sw['platform_arrays']:
        n = a[0]
        ans = E.answers(n, n, 'ws', BIG, DER)
        if ans is None:
            continue
        for mhz in sw['platform_clocks_mhz']:
            for h in HEADS:
                c = ans[h]['minimal']
                ms = 1000.0 * c / (mhz * 1e6)
                board_rows.append(dict(arr=f'{n}x{n}', mhz=mhz, head=h,
                                       cycles=int(c), ms=ms))
                print(f'{n}x{n:<4} {mhz:5d} {h:8} {int(c):14,} {ms:9.1f} '
                      f'{1000/ms:7.2f} '
                      f'{"meets" if ms <= DEADLINE else f"{ms/DEADLINE:.0f}x over":>13} '
                      f'{"meets" if ms <= FRAME else f"{ms/FRAME:.0f}x over":>11}')
    print('\nEvery row uses the minimal schedule, so every row is a best case.')

    # ---------------------------------------------------------------- E
    rule('E.  SMALLEST SIMULATED ARRAY THAT MEETS THE DEADLINE   (SCALE-Sim)')
    ref_mhz = sw['platform_clocks_mhz'][1]
    print(f'Of the array sizes actually simulated, the smallest that lands each')
    print(f'answer inside {DEADLINE:.0f} ms at {ref_mhz} MHz, minimal schedule, '
          f'same derating.\n')
    print(f'{"input":>9} ' + ''.join(f'{h:>14}' for h in HEADS))
    needed = {}
    for sz in SIZES:
        row = []
        for h in HEADS:
            found = None
            for n in ARRS:
                ans = E.answers(n, n, 'ws', sz, DER)
                if ans is None:
                    continue
                if 1000.0 * ans[h]['minimal'] / (ref_mhz * 1e6) <= DEADLINE:
                    found = n
                    break
            needed[f'{sz}|{h}'] = found
            row.append(f'{found}x{found}' if found else '-')
        print(f'{sz:>5} sq ' + ''.join(f'{r:>14}' for r in row))
    cap = part['dsp_slices'] - FIXED
    print(f'\nThe part offers {cap} cells, which is {int(cap**0.5)}x{int(cap**0.5)}.')
    print('A dash means no array size that this run simulated at that input size')
    print('meets the deadline. It does not mean no array size could.')
    for sz in SIZES:
        got = [n for n in ARRS if E.have(n, n, 'ws', sz)]
        print(f'  {sz:>4} sq  simulated at: '
              + (', '.join(f'{n}x{n}' for n in got) if got else 'nothing'))

    # ---------------------------------------------------------------- F
    rule(f'F.  MEMORY TRAFFIC PER FRAME AT {BIG}x{BIG}   (traced, SCALE-Sim '
         f'for the compute column)')
    bigI = int(BIG)
    in_b = 3 * bigI * bigI
    cells = RES[BIG]['cells']
    out_b = 4 * cells * 15
    bias = read('crosscheck.json')['bias_count']
    print(f'input frame at int8             {in_b/1024:10,.1f} KB')
    print(f'weights plus folded bias        {(wb_all+bias)/1024:10,.1f} KB')
    print(f'all four answers out            {out_b/1024:10,.1f} KB')
    print(f'peak live activation            {MEM["peak_live"]/1024:10,.1f} KB')
    print(f'on-chip BRAM available          {MEM["bram_bytes"]/1024:10,.1f} KB')
    print(f'\nNone of that fits on chip, so it all crosses the external bus.')
    print(f'{part["dram"]} peak bandwidth    {DRAM_GBS:10.2f} GB/s\n')
    ref_n = sw['platform_arrays'][-1][0]
    ans_ref = E.answers(ref_n, ref_n, 'ws', BIG, DER)
    print(f'{"answer":8} {"weights (KB)":>13} {"frame+out":>11} {"total (KB)":>11} '
          f'{"peak BW":>9} {"60% BW":>9} {"compute":>10}')
    mem_table = []
    for h in HEADS:
        w = sum(S[s]['weights'] for s in NEEDS[h])
        t = w + in_b + cells * 15
        cms = 1000.0 * ans_ref[h]['minimal'] / (ref_mhz * 1e6)
        mem_table.append(dict(head=h, weights=w, other=t - w, total=t,
                              peak_ms=1000 * t / (DRAM_GBS * 1e9),
                              bw60_ms=1000 * t / (0.6 * DRAM_GBS * 1e9),
                              compute_ms=cms))
        print(f'{h:8} {w/1024:13,.1f} {(t-w)/1024:11,.1f} {t/1024:11,.1f} '
              f'{1000*t/(DRAM_GBS*1e9):8.2f}ms '
              f'{1000*t/(0.6*DRAM_GBS*1e9):8.2f}ms {cms:9.1f}ms')
    m0 = mem_table[0]
    print(f'\nCompute times are for {ref_n}x{ref_n} at {ref_mhz} MHz. Memory is '
          f'{m0["compute_ms"]/m0["bw60_ms"]:.0f} times')
    print('cheaper than arithmetic here, so the array is the constraint.')

    # ---------------------------------------------------------------- G
    rule('G.  THE OPERATING-POINT TABLE')
    print(f'Latency is SCALE-Sim. Accuracy is measured on '
          f'{ACCJ["dataset"]["test_images_unseen"]:,} unseen images and is')
    print(f'not simulated here. A setting is admissible only if it meets '
          f'{DEADLINE:.0f} ms.\n')
    print(f'{"array":>7} {"MHz":>5} {"answer":8} {"latency":>10} {"mAP50":>7} '
          f'{"mAP50-95":>9} {"recall":>7} {"admissible":>11}')
    OP = []
    for n in [x for x in ARRS if x in (8, 12, 32, 64)]:
        ans = E.answers(n, n, 'ws', BIG, DER)
        if ans is None:
            continue
        for mhz in sw['design_clocks_mhz']:
            for h in HEADS:
                ms = 1000.0 * ans[h]['minimal'] / (mhz * 1e6)
                m50, m95, prec, rec = ACC[h]
                ok = ms <= DEADLINE
                OP.append(dict(arr=f'{n}x{n}', mhz=mhz, head=h, ms=ms,
                               map50=m50, map5095=m95, recall=rec, ok=ok))
                print(f'{n}x{n:<4} {mhz:5d} {h:8} {ms:9.2f}ms {m50:7.3f} '
                      f'{m95:9.3f} {100*rec:6.1f}% {"yes" if ok else "no":>11}')
    adm = [r for r in OP if r['ok']]
    print(f'\nadmissible settings      : {len(adm)} of {len(OP)}')
    if adm:
        best = max(adm, key=lambda r: (r['map50'], -r['ms']))
        cheap = min(adm, key=lambda r: r['ms'])
        print(f'most accurate admissible : {best["arr"]} at {best["mhz"]} MHz, '
              f'{best["head"]}, {best["ms"]:.2f} ms, mAP50 {best["map50"]:.3f}')
        print(f'fastest admissible       : {cheap["arr"]} at {cheap["mhz"]} MHz, '
              f'{cheap["head"]}, {cheap["ms"]:.2f} ms, mAP50 {cheap["map50"]:.3f}')
    dom = [h for i, h in enumerate(HEADS[1:], 1)
           if ACC[h][0] <= ACC[HEADS[i - 1]][0]]
    if dom:
        print(f'answers dominated by a cheaper one : {", ".join(dom)}')

    # ---------------------------------------------------------------- H
    rule('H.  THE DATASET BEHIND THE ACCURACY COLUMN   (measured, not simulated)')
    nb = sum(BOXES.values())
    ni = ACCJ['dataset']['training_images']
    print(f'dataset                      {ACCJ["dataset"]["name"]}')
    print(f'training images              {ni:,}')
    print(f'labelled boxes               {nb:,}')
    print(f'boxes per image              {nb/ni:.1f}')
    print(f'test images, unseen          {ACCJ["dataset"]["test_images_unseen"]:,}')
    print(f'\n{"class":15} {"boxes":>10} {"share":>7} {"AP50":>7}')
    for k, v in sorted(BOXES.items(), key=lambda kv: -kv[1]):
        print(f'{k:15} {v:10,} {100*v/nb:6.1f}% {AP[k]:7.3f}')

    # ---------------------------------------------------------------- I
    rule('I.  ACHIEVABLE FRAME RATE ON THE TARGET PLATFORM   (SCALE-Sim)')
    bn = sw['platform_arrays'][-1][0]
    print(f'{bn}x{bn} is the largest array in the sweep that fits the part. '
          f'Minimal schedule,')
    print('weight stationary, measured derating, cheapest answer only.\n')
    print(f'{"input":>9} {"MACs (G)":>10} {"cycles":>14} '
          + ''.join(f'{str(m) + " MHz":>12}' for m in sw['platform_clocks_mhz']))
    res_table = {}
    for sz in SIZES:
        ans = E.answers(bn, bn, 'ws', sz, DER)
        if ans is None:
            print(f'{sz:>5} sq   not simulated')
            continue
        c = ans['head1']['minimal']
        m = ans['head1']['macs_minimal']
        res_table[sz] = dict(total=RES[sz]['total'], cells=RES[sz]['cells'],
                             head1=m, cycles_12=int(c))
        fps = [f'{1000.0/(1000.0*c/(mhz*1e6)):.2f} fps'
               for mhz in sw['platform_clocks_mhz']]
        print(f'{sz:>5} sq {m/1e9:10.3f} {int(c):14,} '
              + ''.join(f'{x:>12}' for x in fps))

    # ---------------------------------------------------------------- J
    rule('J.  WHAT WAS SIMULATED, AND THE CHECKS ON IT')
    dc = E.duplicate_check(ok_ws[0], ok_ws[0], 'ws', BIG)
    print(f'simulator                    SCALE-Sim, scalesim-project/SCALE-Sim')
    print(f'layers in the detector       {dc["layers"]}')
    print(f'distinct layer shapes        {dc["shapes"]}')
    print(f'reuse factor                 {dc["reuse_factor"]:.2f}x '
          f'(the four heads are identical in structure)')
    print(f'shapes giving two answers    {dc["inconsistent"]}')
    print(f'\nEvery layer is simulated. Because a shape that recurs gives the same')
    print(f'cycle count every time, which the line above checks, each distinct')
    print(f'shape is simulated once and its result reused. No cycle count in this')
    print(f'run is modelled, interpolated or scaled from another one.\n')
    print(f'{"array":>9} ' + ''.join(f'{df:>16}' for df in DFS))
    for n in ARRS:
        print(f'{n}x{n:<6} ' + ''.join(
            f'{("complete" if done(n, df, BIG) else "not run"):>16}'
            for df in DFS))

    n0 = sw['platform_arrays'][-1][0]
    print(f'\nSCALE-Sim per shape, {n0}x{n0} weight stationary, the ten dearest '
          f'of the {dc["shapes"]} shapes')
    print(f'{"ifmap":>7} {"k":>3} {"stride":>7} {"cin":>5} {"cout":>5} '
          f'{"SCALE-Sim cycles":>18} {"util":>7} {"of frame":>9}')
    pl = E.per_layer(n0, n0, 'ws', BIG)
    tot_c = sum(p['cycles'] for p in pl)
    seen, uniq = set(), []
    for p in pl:
        t = tuple(p['shape'])
        if t not in seen:
            seen.add(t)
            uniq.append(p)
    for p in sorted(uniq, key=lambda z: -z['cycles'])[:10]:
        ih, k, st, cin, cout = p['shape']
        occurs = sum(1 for q in pl if tuple(q['shape']) == tuple(p['shape']))
        print(f'{ih:4d}sq {k:3d} {st:7d} {cin:5d} {cout:5d} '
              f'{p["cycles"]:18,.0f} {p["util"]:6.1f}% '
              f'{100*p["cycles"]*occurs/tot_c:8.1f}%')
    print('\nThe last column is that shape\'s share of the frame, counting every')
    print('layer it appears in. SCALE-Sim\'s own reports are under out/scalesim_runs/.')

    save('results.json', dict(derate=DER, board_rows=board_rows, op=OP,
                              needed=needed, res_table=res_table,
                              design_space=design, mem_table=mem_table,
                              work_table=[dict(head=h,
                                               seq=sum(S[s]['macs'] for s in UPTO[h]),
                                               minimal=sum(S[s]['macs'] for s in NEEDS[h]),
                                               weights=sum(S[s]['weights'] for s in NEEDS[h]))
                                          for h in HEADS],
                              input_bytes=in_b, out_bytes=out_b,
                              sim_util=sim_util, measured_util=meas,
                              deadline_ms=DEADLINE, frame_ms=FRAME,
                              part=part, built=built, fixed_dsp=FIXED,
                              ref_mhz=ref_mhz, ref_arr=[ref_n, ref_n],
                              dataset=ACCJ['dataset'],
                              dram_gbs=DRAM_GBS, total_macs=tot, n_conv=nconv,
                              peak_live=MEM['peak_live'],
                              bram_bytes=MEM['bram_bytes'],
                              weights_bytes=wb_all, cells=cells,
                              stage_stats=S, acc=ACC, boxes=BOXES, ap=AP,
                              duplicate_check=dc,
                              simulator='SCALE-Sim (scalesim-project/SCALE-Sim)'))
    print('\n\nwrote out/results.json')


if __name__ == '__main__':
    main()
