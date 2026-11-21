# Feedback-Controlled Safety-Critical AI Accelerator for Autonomous Systems

M.Tech project, Dept. of CSE, IIT Tirupati · Guide: Dr. Jaynarayan T. Tudu

## Anytime detector: training and profiling

Anytime object detectors trained on BDD100K (a 4-exit GELAN detector and a fast anytime YOLOv8n), with accuracy, latency, quantisation and weight statistics per exit, and a profiling study of other detectors.

- **Report:** to be added
- **Results:** [`anytime-detector-training-and-profiling/`](anytime-detector-training-and-profiling/)

## Model mapping and execution

Cycle-accurate mapping of an anytime object detector onto a 4×4 INT8/INT4 systolic array with SCALE-Sim v3, and the operating-point table it gives the runtime controller.

- **Paper:** [`Model-Mapping-and-Execution.pdf`](Model-Mapping-and-Execution.pdf)
- **Code, inputs and simulation outputs:** [`model-mapping-and-execution/`](model-mapping-and-execution/) — start with [`START-HERE.md`](model-mapping-and-execution/START-HERE.md)

## Reconfigurable AI accelerator: RTL design and verification

Descriptor-driven single-tile 4×4 systolic accelerator in Verilog with packed INT8/INT4 arithmetic, self-checking testbenches and simulation waveforms.

- **Report:** [`Reconfigurable_AI_Accelerator_Project_Report.pdf`](Reconfigurable_AI_Accelerator_Project_Report.pdf)
- **RTL, testbenches and simulation output:** [`reconfigurable-ai-accelerator-design/`](reconfigurable-ai-accelerator-design/)
