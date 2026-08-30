# Parallel categorical GPTQ: mechanistic current-state probe at step 98,000

Date: 2026-08-30 UTC

## Scope and validity

This was a bounded read-only probe.  The live trainer was not signalled, paused,
or mutated.  The exact rolling resume checkpoint opened by the probe had:

- schema `weightclip_parallel_categorical_gptq_700m_production_v1`;
- step `98,000`;
- committed cursor `3,136,000`;
- valid cursor contract: `98,000 * 32 = 3,136,000`;
- checkpoint stat at open: size `8,361,775,579`, mtime_ns
  `1788084726998484015`.

The paired probe batch contained 64 samples beginning at logical index 3,136,000.
Fifty-five samples belonged to 16 exact-layout groups and were used for all
same-layout interventions.  The raw machine-readable evidence is:

- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/current_state_mechanistic_probe_latest_20260830.json`
- implementation:
  `training/weightclip_benchmark/analyze_parallel_categorical_current_state.py`

## Failure definition

The model is no longer at the zero-output baseline, but its reconstructed weight
payload remains far from the GPTQ teacher:

| Decode | Raw NRMSE |
|---|---:|
| zero output | 1.0000 |
| predicted hard codes + predicted hard scales | 0.8856 |
| GPTQ codes + continuous GPTQ scales | 0.1677 |

The current question is which internal path causes the remaining gap: latent
collapse, decoder ignoring `z`, scale collapse, code decoding, attention routing,
or calibration-activation overfit.

## Competing mechanisms and unique predictions

1. **Latent collapse / decoder ignores `z`.**  Predicts low latent rank and little
   change after exact-layout sample swaps.
2. **Scale remains sample-global only.**  Predicts high sample-mean correlation
   but near-zero row-centered and sample-by-row correlations; oracle codes plus
   predicted scale should remain very poor.
3. **The code distribution contains useful information hidden mainly by argmax.**
   Predicts a large gain from probability-mean code decode while the code loss
   and `z` swap show conditional information.
4. **The code head has learned only a prior.**  Predicts negligible same-layout
   `z`-swap effect and accuracy/MAE close to an input-independent central code.
5. **Attention has collapsed into a useless route.**  Predicts saturated maps
   together with little sample-specific write signal and weak `z` dependence.
   A different possibility is successful deterministic ownership: saturated,
   mostly sample-invariant routes whose values and writes remain sample-specific.
6. **Activation calibration overfit.**  Predicts materially worse held-out than
   calibration operator reconstruction.

## Results

### 1. The latent is not collapsed, and the decoder strongly uses it

Centered latent variance decomposes as:

| Component | Variance fraction | RMS |
|---|---:|---:|
| slot main effect | 72.89% | 3.1565 |
| sample x slot interaction | 22.07% | 1.7368 |
| sample main effect | 5.04% | 0.8299 |

The centered latent matrix has effective rank 19.67, with 21/28/43 singular
directions needed for 90/95/99% energy.  The interaction component itself has
effective rank 31.03.  This is incompatible with all slots or all samples having
the same latent.

The direct same-layout intervention is stronger evidence.  Swapping complete
`z` tensors between samples:

| Metric | Original `z` | Swapped `z` | Change |
|---|---:|---:|---:|
| code ordinal loss | 0.06956 | 0.10839 | 1.558x |
| scale ordinal loss | 0.001278 | 0.05006 | 39.17x |
| code hard argmax changed | - | - | 31.69% |
| scale hard argmax changed | - | - | 91.10% |

Therefore neither a decoder-independent code prior nor latent collapse describes
the current checkpoint.  Unlike the early run, both heads now carry strong
sample-conditional information through `z`.

### 2. Scale has recovered row-level structure; it is imperfect but not the current dominant bottleneck

For probability-mean log2-scale decode:

| Scale statistic | Value |
|---|---:|
| all-row Pearson | 0.9522 |
| all-row R2 | 0.9066 |
| log2 RMSE | 0.2614 |
| log2 MAE | 0.1861 |
| median multiplicative error | 1.100x |
| exact-layout sample-mean correlation | 0.9994 |
| exact-layout row-centered correlation | 0.7255 |
| exact-layout sample x row correlation | 0.7414 |

The remaining scale residual is shrinkage rather than collapse: predicted
row-centered RMS is 0.2586 versus target 0.3772, and predicted interaction RMS is
0.2202 versus target 0.3134.  Both are about 69-70% of target amplitude.

Crossed decodes localize the reconstruction gap:

| Codes | Scale | Raw NRMSE |
|---|---|---:|
| oracle | oracle continuous | 0.1677 |
| oracle | predicted expected | 0.2287 |
| predicted hard | oracle continuous | 0.8845 |
| predicted hard | predicted hard | 0.8856 |

Thus scale error is material when codes are correct (`0.1677 -> 0.2287`), but it
barely changes the current poor-code reconstruction (`0.8845 -> 0.8856`).  The
current end-to-end bottleneck is code content, not scale.

Hard, probability-mean, and probability-median scale decoding are almost
equivalent: RMSE 0.2638, 0.2614, and 0.2624 log2 respectively.  A hard/expected
decode mismatch is not the scale failure.

### 3. The code head is conditional but strongly center-seeking

On exact-layout samples, target code 0 (class 7) occurs in 17.02% of components,
whereas the model's hard prediction is code 0 in 79.40%.  Target nonzero fraction
is 82.98%; predicted nonzero fraction is only 20.60%.

The current hard code statistics are:

- accuracy 18.45%;
- off-by-one accuracy 50.95%;
- MAE 1.776 bins;
- target-class probability 0.1490;
- target-class NLL 2.3102.

A constant code-0 point estimate on this same target batch would have accuracy
17.02%, off-by-one accuracy 47.48%, and MAE 1.936 bins.  The model is better than
that constant estimate, but only modestly in hard space.  This does not mean its
soft distribution is input-independent: the exact-layout `z` swap increases code
loss by 55.8% and changes 31.7% of code argmaxes.

Probability-mean code decoding improves NRMSE with oracle scale only from 0.8845
to 0.8778.  Therefore argmax/mode mismatch is real but explains only 0.0066 NRMSE;
the larger problem is that the conditional code distribution remains too broad
and centered to recover the magnitude of most nonzero GPTQ codes.

### 4. Attention is highly saturated, but current evidence does not establish it as harmful

Attention changed qualitatively from the near-uniform early checkpoint:

| Block | Normalized entropy | Mean max probability | Logit std | sample-specific attention-map variance | sample-specific write variance |
|---|---:|---:|---:|---:|---:|
| decoder 1 | 0.333 | 0.643 | 3.73 | 8.8% | 23.4% |
| decoder 2 | 0.639 | 0.357 | 2.53 | 28.7% | 16.2% |
| decoder 3 | 0.447 | 0.525 | 3.02 | 32.4% | 34.0% |
| decoder 4 | 0.293 | 0.664 | 6.19 | 37.3% | 40.6% |
| decoder 5 | 0.172 | 0.833 | 5.84 | 17.9% | 35.3% |
| categorical tail | 0.053 | 0.946 | 11.19 | 7.6% | 29.6% |

The tail is nearly one-hot and mostly uses the same route for samples with the
same layout, but its writes remain sample-specific.  Together with the strong
`z`-swap effect, this is consistent with a learned deterministic owner/routing
map over sample-specific latent values.  Saturation is therefore a risk and a
proximal geometry fact, not yet a supported cause of poor reconstruction.

To distinguish useful ownership from route collapse, the missing discriminator
is slot-load/owner utilization (how many latent slots receive traffic) plus a
small read-only route-temperature perturbation.  Entropy alone cannot make the
causal claim.

### 5. The model is not fitting only calibration activations

Operator reconstruction for the current hard decode is 0.01911 on calibration
activations and 0.01902 on held-out activations (full 0.01907).  The zero baseline
is 0.03149/0.03137.  There is no held-out penalty here, so calibration-only
activation overfit is excluded on this batch.

Recent live gradient telemetry also has no dead path or strong code-scale
antagonism: at steps 98,300-98,700, `dL_code/dz` RMS is roughly
`5.7e-8..6.5e-8`, `dL_scale/dz` RMS `0.9e-8..3.7e-8`, and cosine
`-0.002..+0.132`.  Scale has lower current pressure because it is much closer to
solved, not because its gradient is cancelled by code.

## Supported conclusion

The early latent-collapse/global-scale diagnosis is stale at step 98,000.  The
encoder now produces a ranked, sample-by-slot representation; the decoder strongly
depends on it; and scale contains real row and interaction structure.  The
remaining reconstruction failure is localized primarily to the categorical code
payload: the learned conditional distributions improve ordinal loss and carry
sample identity, but their point estimates overwhelmingly choose zero and retain
too little nonzero magnitude.  Changing only the scale head cannot materially fix
the current end-to-end NRMSE.

The deep decoder has also converged to near-discrete routing.  That may be the
mechanism that enabled the escape from the early uniform-attention state, or it
may later become a capacity bottleneck; this probe does not distinguish those
possibilities.

## What is not established

- This is one exact checkpoint and one paired batch.  It does not establish that
  the apparent late online NRMSE worsening is a true temporal regression.  A
  same-batch step-90k versus step-98k comparison is required.
- Attention saturation is not established as causal without owner-load and
  route perturbation measurements.
- The probe distinguishes hard-decode mismatch from insufficient soft precision,
  but does not yet distinguish whether the remaining broad code distribution is
  caused by the ordinal objective, information capacity, or optimization speed.

## Next discriminating measurements

1. On the same fixed probe batch, compare step 90k and step 98k model states.
2. Record per-block latent-slot traffic and owner utilization, then evaluate a
   bounded inference-only attention-temperature sweep.
3. Compare the existing ordinal distribution with its probability-mean point
   estimate and with a loss-aligned direct ordinal point estimator.  The current
   expected-code result already bounds the maximum easy argmax-only gain as small.

