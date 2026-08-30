# Parallel categorical GPTQ scale audit at step 5000

Date: 2026-08-29 UTC.

## Failure definition

The live scale branch reports only about 6--7% exact 256-way accuracy and about
5--6 bins of absolute error.  The audit asks whether scale is not learning,
whether those metrics are misleading because the bins are very fine, and which
part of the 128-output scale field the model actually carries through `z`.

The production trainer was inspected read-only and was not signalled, stopped,
or modified.

## Validity and implementation facts

- The physical checkpoint probe opened the atomic rolling checkpoint at exact
  step 5000, cursor 160000, and evaluated logical samples 160000--160031.  It
  contained 3168 valid output-scale targets and 19 samples in seven exact-layout
  groups.
- The GPTQ target scale is exactly `max(abs(W_output_row)) / 7`; it is not an
  activation-dependent or noisy target.  The encoder receives the same value as
  standardized `log2(scale)`, repeated on the four p32 input chunks belonging to
  that output row.  See
  `training/weightclip_benchmark/run_parallel_categorical_gptq_700m_production.py:409-414`
  and
  `training/weightclip_benchmark/run_direct_normalized_scaled_700m_production.py:532-539`.
- The scale decoder is a bias-free `Linear(1536, 256)` after pooling the eight
  p16 child states of each output row.  It therefore predicts 128 independent
  scale distributions, not one explicitly shared scale.  See
  `run_parallel_categorical_gptq_700m_production.py:333-345`.
- The 256 bin centers span log2 scale `[-12, 0]`, so adjacent centers differ by
  0.047059 log2, a multiplicative ratio of 1.03316.  Intrinsic target-binning
  error is only 0.01160 log2 at p50 and 0.02327 at p99 (about 0.8% and 1.6%).
  Thus bin width does not explain the model's current physical error.
- Old behavioral/structural scale metrics are computed from the complete hard
  decoded matrix `signed_code * predicted_scale`.  They are confounded by the
  mostly-zero/incorrect code argmax and are not clean scale-head metrics.  See
  `run_parallel_categorical_gptq_700m_production.py:713-734`.

## Trajectory

All window entries below are means over the online production batches except
the final fixed-checkpoint probe.

| State | Scale ordinal | Exact acc. | Off-by-one | MAE bins | MSE bins | Norm. entropy |
|---|---:|---:|---:|---:|---:|---:|
| Step 1 | .111326 | .00934 | .02255 | 65.16 | 5940.59 | .9430 |
| Steps 510--1000 | .002797 | .06287 | .19027 | 5.91 | 69.08 | .1961 |
| Steps 2010--3000 | .002647 | .06571 | .19261 | 5.82 | 66.85 | .1884 |
| Steps 3010--4000 | .002560 | .06638 | .19457 | 5.77 | 66.36 | .2087 |
| Steps 4010--4890 | .002468 | .06592 | .19566 | 5.71 | 64.24 | .2147 |
| Step-5000 probe | .001995 | .06155 | .19918 | 5.42 | 55.51 | .2100 |

Scale learned rapidly at the beginning and continues to improve slowly.  It is
not dead, but exact high-resolution recovery has plateaued.

## Physical accuracy and the shortcut discriminator

| Predictor on the step-5000 batch | MAE bins | Rel. error p50 | Rel. error mean | Rel. error p90 | log2 correlation |
|---|---:|---:|---:|---:|---:|
| Sealed fit global mode, class 122 | 11.95 | 26.8% | 48.3% | 148.9% | 0 |
| Oracle per-sample median, broadcast to all outputs | **4.89** | **11.4%** | **16.4%** | **34.2%** | .901 |
| Model | 5.42 | 12.9% | 17.0% | 35.6% | .890 |

The target and predicted fields were then decomposed into the sample mean and
the output-row residual around that mean:

- sample-mean predicted/target correlation: **.988**;
- sample-mean absolute error: **.0919 log2**;
- within-sample/output-row predicted/target correlation: **.064**;
- within-sample residual RMSE: **.323 log2**;
- between-sample mean variation accounts for **70.2%** of target log-scale
  variance on this batch.

This is the discriminating result.  The model has learned the easy
sample-global radius extremely well, but it has learned almost none of the
128-output relative scale pattern.  A one-number-per-sample oracle is slightly
better than the model on the very scale metric it is meant to predict.

The exact-layout swap does not contradict this.  On the eligible subset,
original versus donor-`z` scale loss is `.001975 -> .006214` and MAE is
`5.37 -> 8.30` bins.  Donor samples have different global radii even when their
layout is identical, so this intervention establishes sample conditioning but
does not establish recovery of the within-sample 128-vector.  The decomposition
shows what that conditioning mostly contains.

## Loss geometry and gradients

The implemented cumulative ordinal loss weights thresholds by

`abs(2 * (threshold - target) + 1) / (K - 1)^2`.

For a sharp prediction displaced by `d` bins, the crossed-threshold point cost
is `d^2 / (K - 1)^2`.  Scale therefore uses denominator `255^2 = 65025`, while
the 15-way code loss uses `14^2 = 196`.  At the current scale MAE of about 5.4
bins, the point-distance part is only about `4.5e-4` of the maximum-range cost.
The scale component consequently contributes only about 2.7--3.3% of total
loss after step 500 even though 13--35% physical scale errors remain.  See
`run_parallel_categorical_gptq_700m_production.py:501-551` and `:1100-1111`.

This normalization is a plausible enabling mechanism for the sample-mean
shortcut, but it is not evidence of a dead scale graph:

| Window | Scale / code RMS at exact `z` | Scale-head / code-head grad RMS | Code-scale cosine at `z` |
|---|---:|---:|---:|
| 500--1000 | 2.14x | .74x | .027 |
| 1100--2000 | 2.05x | .71x | .038 |
| 2100--3000 | 1.62x | .60x | .001 |
| 3100--4000 | 1.92x | .74x | .005 |
| 4100--5000 | 1.70x | .47x | -.012 |

The scale gradient arriving at `z` is still larger than the code gradient, the
private scale head is live, and the two gradients are nearly orthogonal rather
than antagonistic.  Therefore "scale loses because its scalar loss is small"
is incomplete.  The supported mechanism is that the available scale credit
has converged to the high-variance, easily transmitted sample-global statistic,
while the objective/routing combination supplies too little useful pressure to
recover output-relative detail.

## Competing mechanisms

1. **No scale learning / global prior only:** excluded.  The model strongly
   beats the sealed global mode and exact-layout `z` swaps worsen it.
2. **Scale-bin discretization floor:** excluded.  Intrinsic binning error is
   about an order of magnitude smaller than model error.
3. **Sample-global shortcut with missing output-relative carrier:** strongly
   supported by `.988` sample-mean correlation versus `.064` within-sample
   correlation and by the broadcast oracle matching the model.
4. **Dead or code-opposed scale gradients:** excluded for the current state.
   Scale gradients are finite, larger than code at `z`, and nearly orthogonal.
5. **Argmax is a poor decision rule for the learned soft ordinal
   distribution:** remains viable.  The loss trains cumulative probabilities,
   while logging/dequantization uses categorical argmax.  Expected-bin or
   CDF-median decoding was not stored in this probe, so its possible gain is not
   established.  Prediction entropy is low but nonzero (normalized `.21`, about
   1.16 nats), so this is worth checking from future logits without another
   expensive bank scan.
6. **The shared categorical tail / eight-child pooling itself destroys the
   relative scale field:** not isolated.  The pooling has the correct
   per-output grouping, but no intervention separates it from the bottleneck
   routing and loss geometry.

## Narrow conclusion

The user's concern is directionally correct, but "scale does not learn" is too
broad.  The branch learns the matrix-level radius almost perfectly and fails on
the output-row-relative scale vector.  Low exact accuracy partly reflects fine
3.3%-spaced bins, but physical errors of median 12.9% and p90 35.6% are genuinely
material.  The leading mechanism is a sample-global shortcut encouraged by a
max-range-normalized ordinal objective and the current latent routing; neither
quantization noise nor dead/conflicting gradients explains it.

The simplest model-side correction suggested by these facts is to keep GPTQ
codes categorical but make scale a one-scalar-per-output `log2(scale)` regression
readout, ideally from the pre-p16 shared state, with an error scale tied to the
observed training standard deviation rather than the full 12-octave bin range.
That proposal has not yet been experimentally validated.  Before changing the
live run, the cheap missing check is expected-bin/CDF-median decoding from saved
scale logits; it can determine how much of the apparent error is only the hard
argmax decision rule.

## Artifacts

- Physical checkpoint audit:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/scale_audit_checkpoint_latest_20260829.json`
- Raw online metrics:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/train_metrics.jsonl`
- Raw gradient telemetry:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/gradient_telemetry.jsonl`
- Sealed fit/held-out prior counts:
  `docs/report/parallel_categorical_gptq_prior_counts_20260829.json`
- Audit script:
  `scripts/audit_parallel_categorical_scale_checkpoint.py`
