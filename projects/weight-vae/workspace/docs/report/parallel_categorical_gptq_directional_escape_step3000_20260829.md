# Parallel categorical GPTQ: directional diagnostics at step 3000

Date: 2026-08-29 UTC.

## Failure and question

The backward objective appears nearly flat after the initial convergence to the
categorical prior, while the detached behavioral-direction, structural-direction,
and same-layout direction-InfoNCE diagnostics improve.  The question is whether
that is a logging/batch artifact, scale-only learning, or real sample-conditional
learning hidden by the scalar ordinal average.

## Validity and formulas

- The production trainer remained untouched.  The checkpoint probe was
  read-only and gated the rolling checkpoint as exact step 3000, committed
  logical cursor 96000.
- The backward loss is the sum of code and scale weighted cumulative ordinal
  log losses.  Code loss averages scalar predictions over every valid GPTQ code
  coordinate.
- Detached structural direction and direction InfoNCE are computed from hard
  `argmax` code IDs.  Each p16 signed-code vector is unit-normalized.  The
  predicted per-output scale therefore cancels exactly from these two
  diagnostics.
- InfoNCE negatives have identical tile row, tile column, input mask, and output
  mask.  A position-only decoder prior cannot create a persistent own-target
  versus negative similarity gap within these groups.
- Behavioral direction is computed after `X @ W_hat`; unlike patch direction,
  it can also benefit from relative scale accuracy across output columns.
- Step-1000 and step-3000 checkpoint probes use different next-cursor batches
  (32000 and 96000).  Cross-checkpoint probe deltas are therefore **not a paired
  model comparison**.  The rolling-window trends below cover hundreds of
  successive batches and are the evidence for the temporal trend; each exact
  probe establishes only its checkpoint's functional state.

## Competing mechanisms and predictions

1. **Batch/group-mix artifact.** Raw InfoNCE can change when eligible group sizes
   change.  It does not predict a simultaneous rise in diagonal-minus-negative
   similarity, top-1 accuracy, latent interaction, and both direction cosines
   while eligibility stays stable.
2. **Scale-only learning.** It can improve behavioral direction, but cannot
   improve scale-invariant p16 structural direction or direction InfoNCE.
3. **Position-prior refinement.** It can lower ordinal loss but cannot identify
   the own target among exact-layout negatives and predicts little sensitivity
   to exact-layout `z` swaps.
4. **Late partial escape from the prior basin.** It predicts increasing
   sample-by-slot latent structure, less-uniform latent routing, more
   sample-centered decoder writes, a nonzero exact-layout `z`-swap code-loss
   gap, and improved own-target direction discrimination.  Because the ordinal
   loss averages all scalar coordinates and the direction diagnostics
   unit-normalize sparse hard vectors, the ordinal improvement can be much
   smaller than the apparent directional improvement.

## Rolling results

Means from `train_metrics.jsonl`:

| Window | code ordinal | latent interaction RMS ratio | structural dir | behavioral dir | InfoNCE | diag minus hardest negative | InfoNCE top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| steps 500--990 | 0.082642 | 0.0447 | 0.9644 | 0.9755 | 0.8846 | 0.00130 | 0.441 |
| steps 1000--1490 | 0.082532 | 0.0552 | 0.9598 | 0.9664 | 0.8610 | 0.00504 | 0.476 |
| steps 1500--1990 | 0.082513 | 0.0750 | 0.9582 | 0.9661 | 0.8555 | 0.00636 | 0.496 |
| steps 2000--2490 | 0.082691 | 0.1332 | 0.9534 | 0.9621 | 0.8230 | 0.01171 | 0.536 |
| steps 2500--2990 | 0.082351 | 0.1860 | 0.9529 | 0.9569 | 0.8020 | 0.01622 | 0.571 |
| steps 3000--3380 | 0.082345 | 0.2371 | 0.9458 | 0.9486 | 0.7758 | 0.02394 | 0.625 |

InfoNCE eligibility remains approximately 0.65--0.68 across these windows.
The code loss is not literally constant: it improves about 0.00030, or 0.36%,
from the 500--990 window to the current window, but its per-batch standard
deviation is about 0.001.  The small trend is visually buried by target-mix
noise.  Code accuracy and MAE remain nearly flat, showing that the new signal is
mostly a reorganization of which hard errors occur, not broad scalar-token
accuracy yet.

## Exact step-3000 functional state

On logical samples 96000--96031:

- Original code ordinal loss is 0.082462.  An exact-layout cyclic `z` swap makes
  it 0.083275 (+0.000814, +0.99%).  Thus the soft code distribution now contains
  sample-conditional information even though only 1.43% of hard code argmaxes
  cross a decision boundary under that swap.
- Original hard NRMSE is 0.999850; swapped `z` gives 1.000538.  This is a real
  but extremely small reconstruction gain.
- Scale is strongly conditional: its ordinal loss changes from 0.003127 to
  0.072647 under the same swap.
- Latent centered variance decomposes into 81.65% sample main effect, 11.02%
  slot main effect, and 7.34% sample-by-slot interaction.  This is not the fully
  repeated-slot state diagnosed at step 1000.
- Decoder attention is no longer uniform.  For decoder block 2, normalized
  entropy is 0.9630 and mean maximum probability is 0.0806, versus a uniform
  probability of 1/32 = 0.03125.  The categorical tail has entropy 0.9642 and
  maximum probability 0.0691.
- The categorical-tail attention write has exact-layout sample-centered RMS
  0.1742, 42.0% of total write RMS.  The decoder is therefore transmitting
  sample-dependent latent content, not only a coordinate template.

## Narrow conclusion

The falling direction diagnostics are not explained by scale alone, a pure
position prior, or the varying InfoNCE group count.  They are evidence of a
late **partial escape**: the latent geometry and attention routing have started
to carry sample-conditional code information after the early prior solution.

This does not yet mean useful reconstruction.  Unit normalization makes the
hard direction diagnostics highly sensitive to a sparse set of newly nonzero
or rearranged code decisions, whereas the optimized ordinal loss averages all
scalar coordinates and hard NRMSE remains essentially at the zero-output
baseline.  The supported statement is therefore “real but weak conditional
learning,” not “the collapse is solved.”

## Artifacts

- Exact checkpoint probe:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/mechanistic_probe_step3000.json`
- Raw live metrics:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/train_metrics.jsonl`
- Raw gradient telemetry:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/gradient_telemetry.jsonl`
- Step-3000 rolling prior summary:
  `docs/report/parallel_categorical_gptq_prior_review_step3000_20260829.json`
- Implemented categorical loss and hard decode:
  `training/weightclip_benchmark/run_parallel_categorical_gptq_700m_production.py`
- Implemented old direction and InfoNCE diagnostics:
  `training/weightclip_benchmark/run_direct_normalized_scaled_700m_production.py`
