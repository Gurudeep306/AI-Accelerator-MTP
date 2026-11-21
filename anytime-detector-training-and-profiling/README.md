# Anytime detector: training and profiling results

Measured results for the anytime object detectors trained on BDD100K road scenes, plus a profiling study of other detectors. Every number here was measured or counted; the README in each folder names the file it comes from.

| Folder | What it holds |
|---|---|
| `gelan-4-exit-detector/` | the GELAN detector with 4 exits, trained from scratch: parameter study (exit × FP32/FP16/INT8 × GPU/CPU → accuracy, latency, size, compute) and per-layer weight statistics with histograms |
| `yolov8n-fast-anytime-detector/` | the fast anytime YOLOv8n (frozen backbone + 2 early heads): accuracy per head, latency, quantisation, per-class accuracy, training curves, parameter study, weight statistics |
| `model-comparison/` | all five models measured in one run on the same GPU and images: accuracy per head, latency, FPS, size and compute, INT8 loss, weight ranges |
| `detector-profiling/` | profiling table of other detectors on BDD100K and KITTI across FP32/FP16/INT8 and devices (`master.csv`, `master.xlsx`), weights per model, and trade-off plots |

Charts are given as PNG and SVG; the numbers behind them are in the `.csv` and `.xlsx` files.
