# Weight value distribution of every layer

Two CSV files describe the weight values of all **151 convolution
layers** of the trained model (`runs/anytime_interrupt/train6_bdd70k/last.pt`),
**5,429,808 weights** in total. Every number is computed directly from the
checkpoint by `docs/make_weight_distribution.py`.

Layers are listed in **execution order**, the same order and names as
`docs/accelerator_workload.csv`, so the two files can be joined on `order`
or `layer`.

## Two versions of each layer's weights

| Version | What it is |
|---|---|
| `trained` | The convolution weights exactly as stored in the checkpoint |
| `folded` | The weights with BatchNorm folded in: `w' = w × γ / √(var + ε)`. This is what an accelerator or the INT8 model actually multiplies with |

115 layers have a BatchNorm after them; for the other 36 (the
`cls_pred`, `reg_pred` and `obj_pred` output layers of the heads) the two
versions are identical.

## `layer_weight_stats.csv`: one row per layer

| Column | Meaning |
|---|---|
| `order` | Position in execution order, 0 to 150 |
| `layer` | Layer name in the model |
| `stage` | backbone, head1, fpn, head2, pan, head3, refine or head_final |
| `kernel`, `stride` | Kernel size (1 or 3) and stride (1 or 2) |
| `cin`, `cout` | Input and output channels |
| `weights` | Number of weights = cout × cin × kernel × kernel |
| `batchnorm_folded` | True if a BatchNorm was folded into this layer |
| `trained_min` … `trained_max` | Smallest and largest weight, as trained |
| `trained_mean`, `trained_std` | Average and standard deviation |
| `trained_abs_mean`, `trained_abs_max` | Average and largest magnitude |
| `trained_p1` … `trained_p99` | Percentiles 1, 5, 25, 50 (median), 75, 95 and 99: for example 99% of the weights are below `p99` |
| `folded_*` | The same statistics for the folded weights |
| `folded_bias_min`, `folded_bias_max` | Range of the folded biases (bias = β − mean × γ / √(var + ε), or the layer's own bias for the output layers) |
| `folded_channel_absmax_min`, `folded_channel_absmax_max` | Each output channel has its own largest magnitude; these are the smallest and largest of those. A big gap means one INT8 scale for the whole layer would waste precision on most channels |
| `folded_int8_step_per_tensor` | INT8 rounding step if the whole layer shares one scale: `folded_abs_max / 127` |
| `folded_int8_step_per_channel_mean` | Average INT8 step with one scale per output channel (what the INT8 model in this project uses) |
| `folded_share_below_1pct_of_absmax` | Fraction of weights smaller than 1% of the layer's largest magnitude; with a per-layer INT8 scale, these round to 0 or ±1 |

## `layer_weight_histogram.csv`: the shape of each distribution

One row per bin: 151 layers × 2 versions × 64 bins = 19,328 rows.

| Column | Meaning |
|---|---|
| `order`, `layer` | As above |
| `version` | `trained` or `folded` |
| `bin` | Bin number, 0 to 63 |
| `bin_low`, `bin_high` | Bin edges. The 64 bins are equal width and span that layer's own minimum to maximum, so bins differ between layers |
| `count` | Number of weights in the bin; the counts of a layer add up to its `weights` |

To plot one layer, filter on `layer` and `version` and draw `count`
against the bin centre, `(bin_low + bin_high) / 2`.

## What the data shows

- **Folding changes the ranges a lot.** The first layer, `stem.0.conv`,
  has trained weights between −1.11 and 0.97, but folded weights between
  −11.28 and 12.39.
- **The ranges differ greatly between layers.** The largest folded weight
  per layer goes from 0.161 (`head1.scale_heads.2.cls_pred`) to 12.39
  (`stem.0.conv`), about 77×. A fixed-point accelerator therefore needs a
  separate scale for each layer, at least.

## Histogram pictures (`histograms/`)

Drawn from `layer_weight_histogram.csv` by `docs/make_weight_histograms.py`,
so they show exactly the values in the CSV.

| File | Contents |
|---|---|
| `histograms/overview_trained_page_1.png` … `_7.png` | 24 layers per page, weights as trained |
| `histograms/overview_folded_page_1.png` … `_7.png` | 24 layers per page, BatchNorm folded in |
| `histograms/all_layers.pdf` | All 14 overview pages in one file |
| `histograms/layers/<order>_<layer>.png` | One picture per layer (151 files): trained and folded side by side, with min, max, mean, std, 1st and 99th percentile |

Each panel uses that layer's own 64 bins, so the x-axes differ between
panels; compare shapes, and read the range from the axis or the CSV. The
dark vertical line marks 0.

## How the files were checked

- Same 151 layers in the same order as `docs/accelerator_workload.csv`.
- `folded_abs_max` matches that file's `folded_weight_absmax` (it is
  rounded to 4 decimals there).
- The total of `weights` is 5,429,808, the same as the workload file.
  The model's full parameter count, 5,447,012, is higher because it also
  includes BatchNorm parameters and the output-layer biases.
- Every layer's histogram counts add up to its `weights`.

Numbers in the CSVs have 6 significant digits.

## Regenerate

```bash
.venv\Scripts\python docs/make_weight_distribution.py
```
```bash
.venv\Scripts\python docs/make_weight_histograms.py
```
