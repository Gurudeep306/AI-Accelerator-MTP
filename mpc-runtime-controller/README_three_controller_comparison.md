# Three-controller experiment package

This package compares three control strategies using a common MPC core:

- **A — History-based MPC** (`history_mpc_controller.py`): uses completed-frame workload history to forecast upcoming workload.
- **B — Frame-preview MPC** (`frame_preview_mpc.py`): uses inexpensive features from the incoming frame before full inference.
- **C — Within-frame adaptive MPC** (`within_frame_mpc_controller.py`): observes Layer 1 and chooses a candidate configuration for Layer 2 onward.

## Important distinction

These are three strategies, not three independently validated hardware controllers. They share `practical_mpc_controller.py`, which requires:
1. a calibrated per-action performance predictor,
2. an action set with quality validated against labelled data,
3. calibrated thermal parameters before thermal MPC is enabled,
4. a hardware adapter that applies decisions safely.

No synthetic latency, energy, or quality measurements are bundled as if they were real.

## Required common experiment setup

Use the same:
- frame trace and arrival timestamps,
- model and accelerator configuration set,
- measured latency/energy/quality table,
- power, temperature, deadline and quality limits,
- initial state and safety fallback.

Measure:
- total energy and energy/frame,
- end-to-end latency and deadline misses,
- quality against labels,
- predictor error,
- controller decision time,
- preview overhead for B,
- layer-boundary reconfiguration overhead for C.

## Run / integration

The modules expose classes and intentionally leave hardware I/O to the integrator. Run:
```bash
python history_mpc_controller.py
python within_frame_mpc_controller.py
python frame_preview_mpc.py --help
```

The first two commands print integration notes; they do not fabricate a run result. To run experiments, instantiate the shared `MPCController` with a real calibrated predictor and quality-validated actions, then pass it to the corresponding adapter.

## Safety and interpretation

- Controller A has a cold-start problem until enough completed-frame observations exist.
- Controller B's preview may cost time and energy; include that overhead in evaluation.
- Controller C is only possible if the accelerator supports the relevant configuration changes at a safe layer boundary. If the hardware cannot change a knob mid-frame, C is not deployable as designed.
- The MAC-ratio option in Controller C is a workload proxy only. It is not an accurate latency or energy model by itself.
- A controller passing software tests does not establish real-time safety or hardware readiness.
