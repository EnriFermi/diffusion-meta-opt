# ResNet18-Slim zoo timing and storage preflight

Date: 2026-08-18.

Purpose: estimate the reconstructed WeightCLIP zoo cost on the current single
NVIDIA H100 NVL before implementation/production. This is a synthetic-input
throughput preflight, not evidence about model quality.

## Source facts

- Official repository: `https://github.com/HSG-AIML/weightCLIP.git`, inspected
  at commit `be080677a6eceacdbe3b2823caffe3c0cc73fa7e`.
- Architecture: released `ResNet18Slim(width_mult=0.5)`, about 2.8M parameters.
- Exact 10-class state-dict serialization measured locally: 11,209,800 bytes.
- Primary population follows the released ResNet config: 10 datasets x 50
  independent lineages x 45 training epochs; primary snapshots are the two
  terminal zero-based indices 43/44, i.e. one-based epochs 44/45.
- Published train-set sizes sum to 124,817 images. Total final-zoo
  training-example exposures are
  `124,817 * 50 * 45 = 280,838,250`.

## Local concurrency preflight

The released process-parallel trainer was run unmodified in dry-run mode on
cached synthetic tensors with the official model, SGD, batch 256, 8,192 train
images, 1,024 validation images and 1,024 test images per lineage. The progress
timer excludes the parent Python import/startup and includes child startup,
training and final validation/test forwards.

| Concurrent model processes | Lineages | Progress wall time | Observation |
|---:|---:|---:|---|
| 1 | 8 | 13.7 s | GPU underfilled / startup-heavy |
| 2 | 16 | 16.6 s | best tier |
| 4 | 16 | 16.8 s | best tier |
| 8 | 16 | 19.4 s | slower from contention |

Production starts at four processes and can fall back to two after one
real-dataset epoch. Scaling the synthetic estimate from 20 to 50 final
lineages/dataset gives about ten hours of pure final-zoo training. The released
shell's mandatory two-seed/seven-LR sweep adds 14 sweep lineages for every 50
final lineages, about 28% extra compute. Including real startup, every-epoch
FP32 checkpoint I/O, final evaluation and integrity review, the pre-probe
production planning range is **14--17 hours**. Replace this extrapolation with
the measured ETA from the two real 45-epoch probes.

## Matrix/tile census (classifier and all BN keys excluded)

The official body contains 20 convolution matrices and 2,790,240 coded scalar
weights after exact im2col-style reshape `[in_channels*k_h*k_w, out_channels]`.

| Tile `(d_in x d_out)` | Tiles/model | Valid fill | Matrices fitting one tile |
|---|---:|---:|---:|
| `64x64` | 694 | 98.16% | 2/20 |
| `96x96` | 379 | 79.88% | 2/20 |
| `128x64` | 354 | 96.22% | 2/20 |
| `128x128` | 193 | 88.24% | 3/20 |
| `256x128` | 106 | 80.33% | 3/20 |
| `256x256` | 73 | 58.32% | 4/20 |

Square `128x128` is the current primary: it reduces the code count to 193 with
11.76% padding. `96x96` is dominated by it, requiring 379 codes with 20.12%
padding. Rectangular `128x64` is more storage-efficient but remains only an
explicit technical fallback if `128x128` fails the production-shaped
memory/throughput gate. A short loss trial cannot select between them because
the known AE loss has delayed phase transitions.

## Storage budget

Archive model-only FP32 states at initialization and every one of 45 epochs: 46
states per lineage. For 500 lineages:

`11,209,800 * 46 * 500 = 257,825,400,000 bytes = 240.12 GiB`.

A rolling optimizer/scheduler checkpoint is kept only for active resume and is
not duplicated in the archive. Cached image tensors are expected to add about
2 GB. The user explicitly superseded the earlier 50 GiB target in favor of
retaining every epoch in FP32 and compressing later. Half-epoch archival would
be about 206 GB decimal and is not planned.

An exact lossless delta archive was also tested on one locally trained
45-checkpoint trajectory. Compressing every full state with zlib level 1 gave
only `1.076x`; XORing each FP32 byte stream against the previous epoch before
compression gave `1.281x` (504,441,000 raw bytes to 393,690,070 bytes). The
all-epoch population would need over `2.1x` compression to fit alongside data
under the old 50 GiB target, so the plan does not rely on speculative lossless
compression. Raw every-epoch FP32 is the operating point; compression is a
later storage operation.

## Split provenance

The official config uses `RandomSplitter` with 70/15/15 ratios. Its own class
docstring warns that checkpoints from the same model may leak across splits.
The earlier `16/2/2` proposal was ours, not official. The revised proposal uses
lineage-safe `35/7/8` per dataset, yielding 700/140/160 primary checkpoints. No
fixed pair of datasets is used for flow meta-validation: validation aggregates
seven lineages from each of all ten source datasets, and the separate internal
test aggregates eight from each.
