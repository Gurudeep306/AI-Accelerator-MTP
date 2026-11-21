# Weight value distribution of every layer (fast anytime YOLOv8n)

Two CSV files describe the weight values of all **102 convolution
layers** of the fast model (`runs/anytime_v8/bdd1500_fast/weights/last.pt`),
**3,217,088 weights** in total. Every number is computed directly from the
checkpoint by `docs/make_weight_distribution.py --model anytime_v8`.

The columns are the same as for train6; see
[`gelan-4-exit-detector/weight_distribution/README.md`](../../gelan-4-exit-detector/weight_distribution/README.md)
for the meaning of each column of `layer_weight_stats.csv` and
`layer_weight_histogram.csv`.

## The layers

Layers are listed in **execution order**, with every head computed:
backbone, Head 1, FPN, Head 2, PAN, Final head.

| Stage (`stage` column) | Layers | Weights | What it is |
|---|---|---|---|
| `backbone` | 27 | 1,267,632 | YOLOv8n layers 0 to 9 |
| `head1` | 19 | 43,856 | Early head 1 (`early.0`), depthwise |
| `fpn` | 8 | 184,320 | YOLOv8n layers 10 to 15 |
| `head2` | 19 | 43,856 | Early head 2 (`early.1`), depthwise |
| `pan` | 10 | 798,720 | YOLOv8n layers 16 to 21 |
| `head_final` | 19 | 878,704 | YOLOv8n's Detect head (layer 22) |

- **81 layers** have a BatchNorm after them. The other 21 are the 18
  output convolutions of the three heads and the 3 DFL layers.
- **Each head has a DFL layer** (`*.dfl.conv`, 16 weights). It is a 1×1
  convolution whose fixed weights are 0, 1, …, 15. It turns the 16 box-edge
  scores into one distance, and it is never trained. That is why the
  largest weight of every head stage shows as 15.0.
- **Depthwise layers:** in each early-head branch, the layer named
  `….0.0.conv` is the depthwise 3×3 convolution. Its `cin` is the full
  channel count, but each filter sees only one channel, so `weights` =
  `cout × 9`, not `cout × cin × 9`.

## Two versions of each layer

| Version | What it is |
|---|---|
| `trained` | The weights as stored in the checkpoint |
| `folded` | BatchNorm folded in: `w' = w × γ / √(var + ε)`. This is what an accelerator or the INT8 model multiplies with |

Ultralytics stores checkpoints in FP16, so the `trained` values are FP16
numbers (for example 0.400391). The statistics are computed from them in
double precision.

## What the data shows

- **The YOLOv8n part is unchanged.** Layers `model.0` to `model.22` are
  bit-for-bit identical to the fine-tuned YOLOv8n
  (`runs/yolov8n_compare/bdd1500_ft/weights/best.pt`): the largest
  difference over all 355 shared tensors, BatchNorm statistics included,
  is 0. Training only changed the two early heads.
- **Folding changes the ranges a lot.** The first layer, `model.0.conv`,
  has trained weights between −0.514 and 0.483, but folded weights
  between −21.03 and 16.59.
- **The depthwise layers have large folded weights.** The 12 depthwise
  3×3 layers of the early heads reach a folded magnitude of 4.08 to
  12.27, against 0.59 to 1.12 for the 1×1 layers right after them. Each
  depthwise filter has only 9 weights, so BatchNorm scales them up.
- **The ranges differ greatly between layers.** Leaving out the 3 fixed
  DFL layers, the largest folded weight per layer goes from 0.205
  (`model.22.cv3.0.2`) to 21.03 (`model.0.conv`), about 103×. A
  fixed-point accelerator therefore needs at least one scale per layer.

## Histogram pictures (`histograms/`)

Drawn from `layer_weight_histogram.csv` by `docs/make_weight_histograms.py`.

| File | Contents |
|---|---|
| `histograms/overview_trained_page_1.png` … `_5.png` | 24 layers per page, weights as trained |
| `histograms/overview_folded_page_1.png` … `_5.png` | 24 layers per page, BatchNorm folded in |
| `histograms/all_layers.pdf` | All 10 overview pages in one file |
| `histograms/layers/<order>_<layer>.png` | One picture per layer (102 files): trained and folded side by side |

Each panel uses that layer's own 64 bins, so the x-axes differ between
panels. The dark vertical line marks 0.

## How the files were checked

- `layer_weight_stats.csv` has 102 rows. `layer_weight_histogram.csv` has
  13,056 rows (102 layers × 2 versions × 64 bins).
- Every layer's histogram counts add up to its `weights`.
- The model has 3,225,226 parameters in total. The CSVs count only
  convolution weights, so BatchNorm parameters and biases are not
  included.

## Regenerate

```bash
.venv\Scripts\python docs/make_weight_distribution.py --model anytime_v8 --checkpoint runs/anytime_v8/bdd1500_fast/weights/last.pt --out docs/anytime_v8_fast/weight_distribution
```
```bash
.venv\Scripts\python docs/make_weight_histograms.py --src docs/anytime_v8_fast/weight_distribution --model-name "anytime YOLOv8n (fast)"
```
