# -*- coding: utf-8 -*-
"""The SCALE-Sim configuration every run in this package uses.

One architecture preset, varied only in array geometry and dataflow, so that
two configurations differ in exactly the thing being compared and nothing else.
The layout file is neutral: SCALE-Sim v3 requires one even when custom layout is
switched off, and all its factors are 1 so it imposes no ordering of its own.
"""
import pathlib


CFG = """[general]
run_name = {name}

[architecture_presets]
ArrayHeight:    {rows}
ArrayWidth:     {cols}
IfmapSramSzkB:    256
FilterSramSzkB:   256
OfmapSramSzkB:    128
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Dataflow : {df}
Bandwidth : 10
ReadRequestBuffer: 128
WriteRequestBuffer: 128

[layout]
IfmapCustomLayout: False
FilterCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 4
IfmapSRAMBankPort: 2
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 4
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : csr
OptimizedMapping : false
BlockSize : 4
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
"""

LAYOUT_HEADER = (
    'Layer name,IFMAP Height Intraline Factor,IFMAP Width Intraline Factor,'
    'Filter Height Intraline Factor,Filter Width Intraline Factor,'
    'Channel Intraline Factor,Num Filter Intraline Factor,'
    'IFMAP Height Intraline Order,IFMAP Width Intraline Order,'
    'Channel Intraline Order,IFMAP Height Interline Order,'
    'IFMAP Width Interline Order,Channel Interline Order,'
    'Num Filter Intraline Order,Channel Intraline Order2,'
    'Filter Height Intraline Order,Filter Width Intraline Order,'
    'Num Filter Interline Order,Channel Interline Order2,'
    'Filter Height Interline Order,Filter Width Interline Order,\n')
LAYOUT_ROW = '1,1,1,1,1,1,1,2,3,1,2,3,1,2,3,4,1,2,3,4,'


def patch_scalesim():
    """SCALE-Sim 3.0.0 calls int(max(numpy_array)) and throws. Fix it in place."""
    import pathlib
    import scalesim
    f = (pathlib.Path(scalesim.__file__).parent / 'memory'
         / 'double_buffered_scratchpad_mem.py')
    s = f.read_text()
    old = 'self.total_cycles = int(max(ofmap_serviced_cycles))'
    new = 'self.total_cycles = int(np.max(np.asarray(ofmap_serviced_cycles)))'
    if old in s:
        f.with_suffix('.py.orig').write_text(s)
        f.write_text(s.replace(old, new))
        print(f'patched {f.name} (original saved alongside as .py.orig)')
    elif new in s:
        print(f'{f.name} already patched')
    else:
        print(f'WARNING: could not find the line to patch in {f.name}. '
              f'If this step crashes, that is why.')


