# Accelerator workload summary

Model: `train6_bdd70k/last.pt`, input 3 × 640 × 640, batch 1. BatchNorm is folded into each convolution. One multiply-add (MAC) is one weight × one input value.

Per-layer detail is in `accelerator_workload.csv` (one row per convolution, in execution order).

## Totals

- Convolution layers: **151** (75 are 3×3, 76 are 1×1; 7 have stride 2)
- Multiply-adds per image: **8.89 G**
- Weights: **5.43 M**, biases after folding: 8,692 → about **5.44 MB** at 8 bits per value
- Layers followed by SiLU: 115; plain output layers (no activation): 36
- Other operations per image: 3 max-pool calls (5×5, stride 1), 2 nearest ×2 upsamples, 29 concatenations
- Channel widths used: [1, 3, 4, 10, 16, 32, 48, 64, 96, 128, 144, 192, 256]

## By stage, in execution order

| Stage | Conv layers | 3×3 | 1×1 | Weights | Multiply-adds | Share |
|---|---|---|---|---|---|---|
| backbone | 27 | 13 | 14 | 0.46 M | 0.982 G | 11.1% |
| head1 | 21 | 12 | 9 | 1.07 M | 1.720 G | 19.3% |
| fpn | 12 | 4 | 8 | 0.13 M | 0.363 G | 4.1% |
| head2 | 21 | 12 | 9 | 1.07 M | 1.720 G | 19.3% |
| pan | 16 | 7 | 9 | 0.40 M | 0.438 G | 4.9% |
| head3 | 21 | 12 | 9 | 1.07 M | 1.720 G | 19.3% |
| refine | 12 | 3 | 9 | 0.14 M | 0.226 G | 2.5% |
| head_final | 21 | 12 | 9 | 1.07 M | 1.720 G | 19.3% |
| **total** | **151** | 75 | 76 | **5.43 M** | **8.888 G** | 100% |

The four heads together are 6.88 G multiply-adds, **77%** of the total.

## Work needed before each answer

| Answer | Sequential (as run today) | Only the body it needs + its own head |
|---|---|---|
| head1 | 2.70 G | 2.70 G |
| head2 | 4.78 G | 3.06 G |
| head3 | 6.94 G | 3.50 G |
| head_final | 8.89 G | 3.73 G |

The right-hand column is what each answer would cost if earlier heads did not have to finish first, for example when a head runs in parallel with the next body stage on separate hardware.

## By kernel size

| Kernel | Layers | Multiply-adds | Share |
|---|---|---|---|
| 3×3 | 75 | 8.17 G | 91.9% |
| 1×1 | 76 | 0.72 G | 8.1% |

## Memory

- Largest feature map: `stem.0.conv` output, 16 × 320x320 = 1.64 MB at 8 bits
- One set of P3 + P4 + P5 maps: 614,400 values = 0.61 MB at 8 bits. Each stage's set must stay available until both its head and the next stage have read it.
- Largest weight tensor: `head1.scale_heads.2.cls_stem.0.conv`, 128 → 128 channels, 3×3: 147 K weights
- Heaviest single layer: `head1.scale_heads.0.cls_stem.0.conv`, 0.236 G multiply-adds

## For quantisation

- Largest folded weight magnitude in any layer: 12.39; smallest per-layer maximum: 0.162. The spread means per-layer (or per-channel) scaling is needed, not one scale for the whole model.
- Largest folded bias magnitude: 2.35
- The CSV columns `folded_weight_absmax` and `folded_bias_absmax` give the range for every layer.
- Accuracy after quantisation has **not** been measured; these are ranges only.
