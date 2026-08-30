# Parallel categorical GPTQ: paired step-90k/98k/100k mechanism audit

Date: 2026-08-30 UTC

## Scope and validity

This was a read-only audit. The live trainer was not signalled, paused, or
mutated. The exact persistent step-100,000 model was preserved by hardlink
immediately after its atomic replacement:

- step 90,000:
  `/var/tmp/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/model_step90000_hostile_snapshot.pt`;
- step 100,000:
  `/var/tmp/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/model_step100000_paired_snapshot.pt`.

Both payloads have the production schema and exact requested step. Persistent
model-only checkpoints do not contain a cursor field; the reported cursors are
therefore explicitly marked as derived from `step * production_batch_size`, not
misrepresented as payload fields.

The decisive paired probe used the exact same B64 production samples, logical
indices `3,136,000..3,136,063`, for both checkpoints. This is also exactly the
batch used by the earlier step-98,000 rolling-checkpoint audit. Thus the
90k/98k/100k comparison below has no batch, layout, target-tokenization, dtype,
or decode confound. Step 98,000 is comparable because its artifact records the
exact cursor, indices, config and BF16/FP32 evaluation contract. The only format
difference is rolling-resume versus persistent model-only checkpoint.

To test whether the 90k-to-100k result was one-batch noise, a second fixed panel
used eight consecutive B32 batches (256 samples), starting at the same logical
index. It used the same two checkpoint states and production loader.

Primary artifacts:

- paired B64 JSON:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/paired_90k_100k_mechanistic_probe_20260830.json`;
- eight-B32 panel JSON:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/paired_90k_100k_core_panel_8x32_20260830.json`;
- step-98 JSON:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/current_state_mechanistic_probe_latest_20260830.json`;
- implementations:
  `training/weightclip_benchmark/analyze_parallel_categorical_paired_90k_100k.py`
  and
  `training/weightclip_benchmark/analyze_parallel_categorical_paired_panel.py`.

## Failure definition

Success means stable progress toward the GPTQ teacher: ordinal loss, hard and
probability-mean reconstruction, operator reconstruction, and conditional use
of `z` should improve together. The current failure is not NaN, dead graph, or
global latent collapse. It is that the production objective makes slow average
progress while the code point estimate remains overwhelmingly zero and useful
row/sample-conditioned code and scale structure is not stable across
checkpoints.

## Competing mechanisms and discriminating predictions

1. **Global latent collapse / decoder ignores `z`.** Predicts low latent rank,
   negligible code and scale changes under same-layout `z` swaps, and one-slot
   attention traffic.
2. **Harmful attention saturation.** Predicts that inference-only diffusion of
   the tail routing improves ordinal loss or reconstruction, and/or that winner
   traffic has collapsed to one or very few unused slots.
3. **Useful sparse ownership.** Predicts saturated but layout-consistent routes,
   substantial slot utilization, and little benefit from changing temperature.
4. **Pure constant-LR churn.** Predicts no robust endpoint progress across a
   paired multi-batch panel, with roughly symmetric improvements/regressions.
5. **Slow progress plus high checkpoint variance/forgetting.** Predicts broad
   endpoint improvement together with a non-monotonic intermediate checkpoint
   on an exactly shared batch.
6. **Ordinal-distribution/point-reconstruction mismatch.** Predicts improving
   ordinal/NLL values while the mode becomes more center-seeking and hard or
   expected reconstruction improves much less. It also predicts that merely
   replacing argmax with the probability mean cannot recover the gap.
7. **Stable sample-global scale shortcut.** Predicts high sample-mean scale
   dependence but persistently absent row interaction. A transient structured
   carrier instead predicts large checkpoint-to-checkpoint changes in row-scale
   correlation under the same architecture.

## Results

### 1. There is broad but small 90k-to-100k progress; it is not pure churn

Across the eight paired B32 batches:

| Metric | step 90k | step 100k | paired result |
|---|---:|---:|---:|
| code ordinal | 0.085450 | 0.083850 | better on 7/8 batches |
| scale ordinal | 0.002574 | 0.002481 | better on 7/8 batches |
| hard NRMSE | 1.02525 | 1.01451 | better on 8/8 batches |
| expected-code/scale NRMSE | 1.02607 | 1.01447 | better on 8/8 batches |
| oracle-code + expected-scale NRMSE | 0.30698 | 0.30107 | improved |
| operator metric, batch mean | 0.029884 | 0.029464 | better on 6/8 batches |

Therefore the narrow endpoint conclusion is slow broad improvement, not a
stationary random walk. The improvement is nevertheless small: both hard and
expected reconstruction remain slightly worse than the zero-output NRMSE of
1.0 at step 100k.

### 2. The step-98 checkpoint was a real same-batch structured transient

On the exact shared B64 batch:

| Metric | step 90k | step 98k | step 100k |
|---|---:|---:|---:|
| code ordinal, exact-layout subset | 0.08617 | **0.06956** | 0.08439 |
| scale ordinal, exact-layout subset | 0.002794 | **0.001278** | 0.002678 |
| hard NRMSE, full B64 | 1.02267 | **0.88561** | 1.01333 |
| expected-code/scale NRMSE, full B64 | 1.02358 | **0.87899** | 1.01318 |
| oracle-code + expected-scale NRMSE | 0.30663 | **0.22874** | 0.29871 |
| code target probability, exact-layout | 0.13566 | **0.14902** | 0.13550 |
| scale row-centered correlation | 0.0169 | **0.7255** | 0.0396 |
| scale interaction correlation | 0.0115 | **0.7414** | 0.0375 |
| code loss ratio after same-layout `z` swap | 1.018x | **1.558x** | 1.018x |
| scale loss ratio after same-layout `z` swap | 15.00x | **39.17x** | 14.73x |

This is not a hard-argmax artifact: probability-mean reconstruction shows the
same 98k peak. It is also not only scale: both code conditional dependence and
row-scale structure peak together. By step 100k, the model has returned close
to its 90k weak-conditional state on this batch despite modest net endpoint
progress on the broader panel.

The preserved evidence proves non-monotonic checkpoint behaviour on this exact
batch. It does **not** prove that step 98k was globally better on all production
batches, because the overwritten rolling step-98 checkpoint cannot now be run
on the additional 192 panel samples. Thus “constant LR causes global churn” is
a leading hypothesis, not an established sole cause.

### 3. The code objective is improving while its mode becomes more degenerate

Across the 256-sample panel:

| Code statistic | step 90k | step 100k |
|---|---:|---:|
| target-class NLL | 2.4979 | **2.4584** |
| mean target probability | 0.13585 | 0.13536 |
| hard MAE, bins | 1.9591 | **1.9479** |
| prediction equal to zero code | 87.23% | **90.57%** |
| target equal to zero code | 17.05% | 17.05% |

The NLL and ordinal risk improve, but the modal decision becomes even more
center-seeking. Probability-mean decoding is not a hidden fix: at step 100k its
NRMSE is 1.01447 versus 1.01451 for argmax. At the structured step-98 state the
model predicted zero on 79.40% of exact-layout components and expected decoding
only improved NRMSE from 0.88561 to 0.87899. Therefore argmax mismatch is real
but small; the distribution itself still carries too little precise signed
magnitude.

The supported mechanism is an objective geometry with a cheap central/broad
solution: reducing average ordinal threshold risk and NLL does not require a
stable high-amplitude sample-conditioned point estimate. This explains why
scalar training metrics can improve while reconstruction remains near zero and
why a useful conditional carrier can be forgotten without an overwhelming
penalty. It does not imply the objective is completely uninformative: the 98k
state had both lower ordinal loss and better reconstruction.

### 4. Tail saturation is not the immediate bottleneck

Tail routing is nearly discrete but stable between endpoints:

| Tail statistic | step 90k | step 100k |
|---|---:|---:|
| normalized attention entropy | 0.0519 | 0.0492 |
| mean maximum probability | 0.9478 | 0.9484 |
| effective winner slots | 9.31 | 9.31 |
| winner slots with at least 0.1% load | 16 | 15 |
| maximum winner load | 19.60% | 19.01% |
| effective slots by soft mass | 10.00 | 9.98 |
| same-layout agreement to group mode | 91.0% | 92.2% |

All 32 slots receive nonzero soft mass. Decoder blocks 1--4 use all 32 slots
above the 0.1% winner-load threshold, with 18.7--25.7 effective winner slots at
step 100k; concentration appears mainly in block 5 and the tail. This is sparse
ownership, not a one-slot routing collapse.

The step-100k inference-only temperature intervention is causal evidence:

| Routing intervention | code ordinal | hard NRMSE |
|---|---:|---:|
| trained routing | **0.084390** | **1.013327** |
| tail logits x0.5 | 0.084742 | 1.013685 |
| tail logits x2 | 0.084398 | 1.013808 |
| tail logits x4 | 0.084451 | 1.014190 |
| all decoder/tail logits x2 | 0.084524 | 1.013845 |

Every perturbation is neutral-to-worse and all effects are below 0.001 NRMSE.
Consequently attention saturation is a geometry fact and possible long-horizon
risk, but it is excluded as the immediate cause that a simple temperature
change could fix. The stable tail load also cannot explain the large 98k-to-100k
quality loss by itself.

### 5. Global latent collapse is excluded, but the code carrier is weak at 100k

At step 100k, centered latent variance is 4.38% sample main, 74.18% slot main,
and 21.45% sample-by-slot interaction. The interaction has effective rank 31.13
and latent RMS is 3.83. At step 90k the corresponding interaction fraction is
22.54%, effective rank 29.97, and RMS 3.67. Therefore neither all slots nor all
samples have collapsed.

Functional use differs by head. At step 100k, same-layout `z` swaps change 94.3%
of scale argmaxes and multiply scale loss by 14.7, but change only 19.5% of code
argmaxes and increase code loss by 1.8%. The scale head still reads strong
sample-global information; the code head currently extracts weak target-specific
information. At step 98k the code loss increased by 55.8% under the same
intervention, proving that the architecture can temporarily form a much stronger
conditional code carrier.

## Excluded mechanisms

- **Wrong checkpoint/batch/config:** excluded by exact step/schema checks and
  identical recorded logical indices and target pipeline.
- **Global latent collapse:** excluded by ANOVA/rank and strong scale `z`-swap
  dependence.
- **Decoder ignores `z` completely:** excluded by scale dependence and code
  argmax changes, although current code-loss dependence is weak.
- **Single-slot attention collapse:** excluded by winner and soft-mass load.
- **Simple harmful-temperature saturation:** excluded by the bounded causal
  temperature sweep.
- **Argmax alone hides a good soft reconstruction:** excluded by expected decode
  remaining near the hard result.
- **Pure endpoint churn with no learning:** excluded by the eight-batch paired
  90k-to-100k improvement.

## Narrow supported conclusion

The model is making slow average progress, but its optimized categorical
objective permits a strongly center-seeking solution whose point reconstruction
remains near zero. The latent space and routing have not globally collapsed.
Instead, sample-conditioned code and row-scale carriers are unstable: they were
strong at step 98k on the exact fixed batch and weak at both 90k and 100k, while
tail ownership itself stayed essentially unchanged.

The best-supported mechanism is therefore **weakly rewarded conditional
amplitude plus checkpoint-scale forgetting around a center-seeking ordinal
solution**, not attention saturation. Constant LR is a plausible operational
driver of the forgetting, but the available evidence does not yet identify it
as the sole causal mechanism because the step-98 checkpoint cannot be evaluated
on the full panel and no paired LR-decay intervention has been run.

The next decisive experiment is not another attention architecture change. It
is a paired continuation from the same checkpoint with current constant LR
versus a substantially reduced/decayed LR, evaluated on a sealed multi-batch
panel at frequent checkpoints. The discriminator is whether reduced LR retains
conditional `z`-swap dependence, row-scale correlation and NRMSE gains without
sacrificing ordinal progress. A second independent discriminator is an
objective change that directly penalizes center-seeking point error; that tests
objective geometry separately from optimizer noise.
