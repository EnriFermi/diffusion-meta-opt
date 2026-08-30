# Parallel categorical GPTQ production: hostile review at step 1000

Date: 2026-08-29 UTC.

## Question and failure definition

The review asks whether the early decrease of the new categorical objective is
evidence that the autoencoder is reconstructing sample-specific GPTQ codes, or
whether the decoder has taken an input-independent class-prior shortcut.

For this snapshot, useful conditional learning would require at least one of:

- code accuracy/error materially better than a held-out input-independent prior;
- hard decoded raw NRMSE below the zero-prediction value of 1;
- an improvement in the detached direction diagnostics;
- sample-specific information remaining active in the latent bottleneck.

## Validity checks

- Live schema/config: `weightclip_parallel_categorical_gptq_700m_production_v1`,
  seed 42, batch 32, fixed LR 5e-5. Resolved config is at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/resolved_config.json`.
- The trainer was alive and finite at step 1000. Every required parameter group
  had a nonzero gradient in the stored step-1000 telemetry.
- Prior targets used the same emitted/permuted production stream and the exact
  `_mask_aware_gptq_targets` implementation. Ten batches were split into five
  fit and five held-out batches (160 samples each).
- The code distribution was stable across the split: zero-code mass was
  0.169625 on fit and 0.169671 on held-out.
- Limitation: the target-prior split covers logical samples 0..319, while the
  live ten-row window is steps 910..1000. It is not a same-example comparison.
  Agreement across loss, entropy, accuracy, off-by-one accuracy, MAE, and MSE is
  therefore the relevant discriminator.
- Opening the production bank for the prior probe read approximately 42 GiB.
  The exact counts are sealed for reuse; this scan should not be repeated for
  routine monitoring.

## Competing mechanisms and predictions

### H1: input-independent code-prior shortcut

The coordinate query residual and categorical head can emit a fixed ordered
class distribution without using sample-specific `z`. The implemented ordinal
loss has target-dependent threshold weights. Its Bayes-optimal unconditional
distribution is therefore not the empirical marginal, but it is analytically
computable from class counts.

Predictions: live code loss and entropy match that analytic optimum; hard code
metrics match constant zero; hard NRMSE is 1; two-way-centered latent variation
collapses; head gradients remain much larger than the direct gradient entering
`z`.

### H2: genuine conditional code learning hidden by an inaccurate scale head

Predictions: code accuracy/error beats the unconditional code baseline even if
raw NRMSE remains poor; detached direction losses should begin improving.

### H3: scale-only conditional learning

Predictions: scale-bin metrics beat their global prior, while code metrics do
not. Because dequantization is `code * scale`, zero hard codes still yield an
approximately zero weight prediction and NRMSE 1.

### H4: the GPTQ target itself has an NRMSE floor near 1

Prediction: teacher-code dequantization would also have NRMSE near 1.

### H5: numerical/runtime failure or completely dead graph

Predictions: nonfinite values, missing/dead gradient groups, or a dead trainer.

## Results

Exact machine-readable results are in
`docs/report/parallel_categorical_gptq_prior_review_step1000_20260829.json`.

| Quantity | Input-independent held-out baseline | Live mean, steps 910..1000 |
|---|---:|---:|
| Code ordinal loss | 0.083400 | 0.082586 |
| Normalized code entropy | 0.736152 | 0.734060 |
| Code accuracy | 0.169671 | 0.170523 |
| Code off-by-one accuracy | 0.470856 | 0.474102 |
| Code MAE, bins | 1.947682 | 1.938532 |
| Code MSE, bins squared | 6.303195 | 6.261739 |
| Hard raw NRMSE | 1 for exact zero prediction | 1.001639 |

The analytic unconditional distribution has argmax class 7, which is signed
GPTQ code zero. Its normalized entropy is 0.736152. The live code entropy is
0.734060 and all four hard code metrics match the constant-zero baseline within
ordinary batch variation. This simultaneously matching signature is much more
specific than merely observing 17% accuracy.

The transition occurred almost immediately:

- step 1: code loss 0.164047, code accuracy 0.063435, entropy 0.903398,
  hard NRMSE 30.5438;
- step 10: code loss 0.085405, code accuracy 0.164612, entropy 0.678302,
  hard NRMSE 1.0347;
- step 1000 window: values remain at the unconditional-prior solution.

The scale branch is different. Its held-out global-mode baseline has accuracy
0.03363, off-by-one accuracy 0.09750, MAE 14.20 bins, and MSE 395.82. Live scale
metrics are accuracy 0.06131, off-by-one accuracy 0.18927, MAE 5.85, and MSE
67.17. Thus the scale head is extracting useful conditional or layout signal.
It cannot rescue reconstruction while the code argmax is zero.

Detached diagnostics agree with a zero/uncorrelated weight prediction:

- hard raw NRMSE: 1.00164;
- behavioral direction: 0.97195;
- structural direction: 0.96945;
- actual operator RMS diagnostic: 0.02871, versus roughly 0.0012--0.0015 for
  teacher GPTQ dequantization in the live rows.

The bottleneck geometry moved in the predicted direction. The two-way-centered
sample-by-slot interaction RMS ratio fell from 0.8741 at initialization to
0.0724 at step 10 and 0.0478 in the step-1000 window. At step 1000, direct
gradient RMS at `z` was 2.50e-9 from code and 5.63e-9 from scale. Per-parameter
gradient RMS was 1.12e-5 in the code head, 9.71e-6 in the scale head, 2.02e-8 in
the Distribution Encoder, and 1.52e-7 in encoder block 13. These group RMS
values are not interchangeable measures of total update size, but they localize
the strong attenuation of sample-specific supervision before the encoder.

## Excluded and remaining mechanisms

- H2 is not supported at step 1000: code metrics do not beat the prior and the
  old direction diagnostics do not improve.
- H4 is excluded for the current batches: teacher GPTQ raw NRMSE is roughly
  0.15--0.17, far below 1.
- H5 is excluded as a current runtime explanation: the run is finite and every
  required gradient group is live.
- H1 is directly supported by the six-way match to the analytic prior.
- H3 is also supported: scale learning is real but currently reconstruction-
  irrelevant because hard codes are zero.
- It is not established that this state is irreversible. Gradients are nonzero
  and the latent interaction ratio recovered slightly from about 0.027 around
  steps 60--100 to about 0.048 at step 1000. A later escape remains possible.

## Narrow verdict

At step 1000, the apparent code-loss improvement is not evidence of
sample-specific weight reconstruction. The code branch has converged almost
exactly to the Bayes-optimal input-independent prior for the implemented loss,
while only the scale branch shows conditional learning. This is a scientific
failure mode, not a fatal runtime failure. Under the user's instruction, the
production trainer should remain untouched and be checked later for departure
from this precisely quantified prior baseline.

## Minor telemetry defect

`code_valid_count` and `scale_valid_count` are logged as 1.0 because the code
constructs the count with `valid.new_tensor(float(valid.sum()))` where `valid`
has Boolean dtype. This does not affect the loss or optimizer, but those two
logged count fields are invalid.

## Evidence paths

- Raw live metrics:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/train_metrics.jsonl`
- Raw gradient telemetry:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/gradient_telemetry.jsonl`
- Target counts and collection protocol:
  `docs/report/parallel_categorical_gptq_prior_counts_20260829.json`
- Derived step-1000 result:
  `docs/report/parallel_categorical_gptq_prior_review_step1000_20260829.json`
- Re-analysis script:
  `scripts/analyze_parallel_categorical_gptq_prior.py`
