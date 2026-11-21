# Closed-Loop Reconfigurable AI Accelerator Framework

A multi-timescale closed-loop predictive control framework for dynamic deep neural network (DNN) acceleration on reconfigurable hardware.

---

## 📌 Executive Overview

Modern edge and automotive perception models exhibit dynamic, input-dependent computation (varying bounding box counts, conditional routing, dynamic resolution, and sparsity). Conventional fixed-schedule or reactive DVFS (Dynamic Voltage and Frequency Scaling) controllers face an unavoidable trade-off:
1. **Conservative Over-Provisioning (Static High):** Meets deadlines but wastes up to 40% energy.
2. **Reactive Feedback (Controller A / PID):** Reacts only *after* latency lags, causing high deadline miss rates under bursty workloads.

This repository implements a **multi-timescale predictive closed-loop controller**:
- **Controller B (Inter-frame Preview MPC):** Predicts upcoming frame workload over a receding horizon ($H=5$) and selects optimal baseline hardware operating regimes.
- **Controller C (Intra-frame Fine-grained MPC):** Dynamically adjusts sub-frame compute modes, frequency steps, and array utilization with sub-millisecond physical feedback.

The system is rigorously validated using both synthetic traces and real-world perception traces derived from **YOLOv8n inference on nuScenes autonomous driving camera footage**.

---

## 🏛️ System Architecture & Controllers

```
   Raw Video / Sensor Ingestion (nuScenes CAM_FRONT)
                        │
                        ▼
       Workload Feature Extraction & Profiling
   (Object counts, dynamic layer MACs, patch density)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
Inter-Frame Preview MPC          Intra-Frame Fine MPC
    (Controller B)                  (Controller C)
• Receding Horizon (H=5)        • Sub-frame layer tracking
• Beam-Search Trajectory Opt    • Physical telemetry feedback
• Mode / Frequency envelope     • Dynamic throttle / boost
       └────────────────┬────────────────┘
                        │
                        ▼
         Reconfigurable Hardware Accelerator
       (Replay Harness / Physical HW Model)
```

### Controller Inventory
- **`practical_mpc_controller.py`**: Production-grade receding horizon MPC with beam search pruning ($B=4$), cold-start recovery, and strict mathematical feasibility guarantees.
- **`within_frame_mpc_controller.py`**: Intra-frame Controller C with layer-by-layer dynamic frequency adjustment and real-time execution slip compensation.
- **`frame_preview_mpc.py`**: Frame-level Preview MPC utilizing workload forecasting.
- **`history_mpc_controller.py`**: Autoregressive historical baseline MPC.
- **`unified_replay_harness.py`**: Discrete-event accelerator hardware simulator supporting dual-mode DVFS, reconfiguration delays, and execution timeline accounting.

---

## 📂 Repository Structure

```
.
├── practical_mpc_controller.py          # Core receding-horizon MPC engine
├── within_frame_mpc_controller.py       # Layer-level intra-frame controller
├── frame_preview_mpc.py                 # Inter-frame preview MPC
├── history_mpc_controller.py            # Historical autoregressive MPC baseline
├── unified_replay_harness.py            # Hardware replay simulator
├── workload_prediction_interface.py     # Workload feature & predictor API
├── workload_ground_truth.py             # Ground-truth workload extractor
│
├── build_real_yolo_profiles_and_traces.py # nuScenes + YOLOv8 trace pipeline
├── train_real_yolo_workload_models.py     # Workload model training
├── train_controller_b_tier_classifier.py  # Tier classifier for Controller B
├── run_real_yolo_replay.py              # End-to-end YOLOv8 replay benchmark
├── run_phase4_replay.py                 # Multi-scene benchmark harness
│
├── data/                                # Dataset directory (nuScenes mini)
├── traces/                              # Calibrated synthetic & real traces
├── results/                             # Benchmark CSVs, JSONs, and metrics
│   ├── EXP-01/                          # Workload prediction experiments
│   ├── colab_experiment/                # Physical Colab GPU latency profiles
│   ├── milestone_b/                     # Initial profiling milestone results
│   └── phase4/                          # Closed-loop replay comparisons
│
├── tests/                               # Pytest test suite
│   ├── test_timeline_algebra.py         # Hardware timeline accounting tests
│   ├── test_mpc_feasibility_guarantee.py # Mathematical feasibility proofs
│   ├── test_deadline_miss_evaluation.py # Deadline miss boundary checks
│   └── test_consistency_and_e2e_feasibility.py # Integration tests
```

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
Requires Python 3.10+:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt # (or install numpy, scipy, pandas, ultralytics, pytest)
```

### 2. Run Test Suite
Validate timeline algebra, feasibility guarantees, and controller consistency:
```bash
pytest
```

### 3. Run Real-World YOLO Replay Benchmark
Simulate closed-loop control across real nuScenes driving traces under 33.3ms (30 FPS) and 50ms deadlines:
```bash
python run_real_yolo_replay.py
```

### 4. Train & Evaluate Workload Models
```bash
python train_real_yolo_workload_models.py
python train_controller_b_tier_classifier.py
```

---

## 📊 Key Experimental Findings

1. **Deadlock Elimination:** Dynamic feasibility clamping guarantees $0\%$ deadline misses even under $3\sigma$ bursty workload shocks.
2. **Energy Efficiency:** Compared to Static High (over-provisioning), the multi-timescale closed-loop controller achieves **28% to 37% energy reduction** while maintaining zero deadline misses.
3. **Control Overhead:** MPC decision time is kept under **0.4 ms** via beam-search pruning, well within the sub-frame budget.
