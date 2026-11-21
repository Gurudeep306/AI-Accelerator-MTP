# Reconfigurable AI Accelerator: RTL and verification

Verilog RTL for a descriptor-driven, single-tile 4×4 systolic accelerator with packed signed INT8 / INT4 arithmetic, its self-checking testbenches, and the simulation output.

The report for this work is [`Reconfigurable_AI_Accelerator_Project_Report.pdf`](../Reconfigurable_AI_Accelerator_Project_Report.pdf), one level up from this folder.

## What is in here

| Folder | What it holds |
|---|---|
| `rtl/pe/` | signed multiplier, lane multiplier, processing element, tile feeder, post-processors |
| `rtl/array/` | the systolic array (`systolic_array_v2.v` is the registered 4×4 mesh) |
| `rtl/memory/` | address generators and ping-pong SRAM (`*_v2.v` are the current versions) |
| `rtl/control/` | descriptor engine, sequencer FSM, confidence evaluator, control unit |
| `rtl/bus/` | AXI4-Lite slave |
| `rtl/phase2_accelerator_core_v2.v` | the integrated single-tile core |
| `tb/` | self-checking testbenches, one per module plus the integrated core |
| `scripts/` | regression scripts (Icarus Verilog) and `extract_vcd_signals.py` |
| `sim-output/` | waveform of the integrated core (`.vcd`), GTKWave views (`.gtkw`), the signal list, and recorded verification results |

Files without `_v2` are the earlier versions, kept for reference.

## Running it

Needs [Icarus Verilog](https://steveicarus.github.io/iverilog/) (`iverilog`, `vvp`) and Bash. From this folder:

```bash
./scripts/run_regression.sh               # multiplier and processing element
./scripts/run_day6_descriptor.sh          # descriptor format and fetch
./scripts/run_day7_sequencer.sh           # sequencer FSM
./scripts/run_day8_memory_array.sh        # AGUs, ping-pong SRAM, systolic array
./scripts/run_day9_phase2_integration.sh  # integrated core and mesh post-processor
```

To look at the waveform:

```bash
cd sim-output
gtkwave phase2_accelerator_core.vcd phase2_accelerator_demo.gtkw
python3 ../scripts/extract_vcd_signals.py   # lists every signal in the VCD
```

## Scope

One 4×4 output tile, verified in RTL simulation. AXI4/DMA integration, multi-tile scheduling and FPGA synthesis are future work.
