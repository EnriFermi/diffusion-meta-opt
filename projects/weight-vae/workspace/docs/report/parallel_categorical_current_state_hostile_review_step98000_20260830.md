# Hostile review: parallel categorical GPTQ at step 98,000

Date: 2026-08-30 UTC

## Verdict

The current model is neither latent-collapsed nor an input-independent
code-prior model.  The old sample-global-only scale diagnosis is also stale.
The remaining end-to-end error is presently dominated by the categorical code
payload: the decoder uses sample-specific `z`, but its hard predictions are
still zero in 79.40% of valid coordinates while the target is zero in only
17.02%.

Attention is extremely sharp, especially in the categorical tail, but the
available evidence does **not** establish sharp attention as the cause.  It is
equally consistent with a useful deterministic ownership map.  A slot-load and
temperature intervention is still required before changing the architecture on
that basis.

## Validity and failure definition

- Exact read-only checkpoint: step 98,000, cursor 3,136,000; the cursor contract
  `step * batch_size` is valid.
- Fixed probe batch: 64 samples, including 55 samples in 16 exact-layout groups.
- Current hard reconstruction NRMSE is 0.8856, versus 1.0 for zero output and
  0.1677 for the GPTQ teacher.
- The live process is finite. Across the logged run, maximum pre-clip gradient
  norm is 2.472, below the clip threshold 5.0; no logged step was clipped.
- Online trajectory windows are not paired fixed-evaluation sets. They can show
  a broad plateau, but cannot prove the apparent 90k--98k regression.

Primary evidence:

- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/current_state_mechanistic_probe_latest_20260830.json`
- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/train_metrics.jsonl`
- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/gradient_telemetry.jsonl`
- `training/weightclip_benchmark/analyze_parallel_categorical_current_state.py`

The generated `train_loss_curve.png` was inspected, but its raw overplotting and
axis range make it unsuitable for subtle late-trend claims; the window numbers
below come from the JSONL.

## Trajectory: escape happened, then largely plateaued

| Online step window | code ordinal | scale ordinal | hard NRMSE | behavioral direction | structural direction |
|---|---:|---:|---:|---:|---:|
| 20k--40k | .077207 | .001707 | .9129 | .6735 | .7750 |
| 40k--60k | .076575 | .001657 | .9083 | .6664 | .7677 |
| 60k--80k | .076621 | .001659 | .9111 | .6658 | .7683 |
| 80k--98k | .076053 | .001681 | .9116 | .6616 | .7646 |

Thus “continued slow escape” is supported through roughly 30k--40k, but is not
a good description of the subsequent end-to-end trajectory. The code objective
still moves slightly; hard reconstruction has not improved materially after
about 40k.

## Falsification of competing narratives

### Full latent collapse or marginal/fingerprint-only decoding: excluded

The centered latent has effective rank 19.67; its sample-by-slot interaction has
effective rank 31.03 and carries 22.07% of centered variance. Exact-layout
sample swapping changes 31.69% of code argmaxes and raises code loss
`.06956 -> .10839` (1.56x). Removing the interaction raises code loss to
`.08528`. This is substantive conditional computation, not merely a latent ID
fingerprint attached to an unchanged output prior.

The growing raw latent RMS is not evidence of renewed collapse. Decoder K/V are
formed after affine-free RMS normalization of `from_latent(z)` in every
cross-attention block, so uniform scale growth is largely a gauge direction.
Moreover, absolute sample-by-slot interaction RMS has grown, not vanished.

### Row-scale routing failure: excluded as the current dominant bottleneck

At step 98k, expected log-scale has Pearson .952 and R2 .907. Exact-layout
sample-mean, row-centered, and sample-by-row correlations are .999, .726, and
.741. This is a qualitative recovery from the step-5k row-centered correlation
of .064.

Crossed reconstructions localize the remaining error:

| Codes | Scales | NRMSE |
|---|---|---:|
| oracle | oracle | .1677 |
| oracle | predicted | .2287 |
| predicted | oracle | .8845 |
| predicted | predicted | .8856 |

Scale is imperfect, but fixing it alone changes the current prediction by only
about .0011 NRMSE. Code content is the active bottleneck.

### Argmax/ordinal decision-rule mismatch: too small to explain the failure

Probability-mean code decoding with oracle scale improves NRMSE only
`.8845 -> .8778`. Hard, probability-mean, and median scale decodes are likewise
nearly identical. Therefore a better readout rule cannot recover the missing
payload.

The ordinal objective itself remains a viable *training* mechanism. Its
cost-sensitive cumulative score rewards small lattice errors and permits a
strong center-seeking solution under incomplete conditioning. The observed
79.40% predicted-zero fraction, broad normalized entropy .706, and only .149
mean probability on the target class match that prediction. However this probe
does not distinguish objective-induced shrinkage from a representational or
optimization limit, because no same-model counterfactual objective was tested.

### Numerical optimizer stall: excluded; constant-LR churn remains viable

At step 98k, Adam's bias-corrected effective update RMS is nontrivial. The
nominal per-step update is about 1.4e-4 of parameter RMS for the scale head,
2.7e-4 for the code head, and 3.7e-4--4.4e-4 for representative trunk tensors.
Between the preserved step-90k model and step 98k, relative parameter movement
is 5.7% (`to_latent`), 8.0% (`from_latent`), 6.9% (code head), 6.7% (scale
head), 14.8% (output queries), and 7.6% (latent slots). The optimizer is not
frozen.

Since those large parameter changes coexist with an end-to-end plateau,
high-noise constant-LR update churn is still viable. It is not established
without evaluating steps 90k and 98k on the same fixed batch. The step-90k
checkpoint was preserved for that discriminator at:

`/var/tmp/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/model_step90000_hostile_snapshot.pt`

## Attention: striking localization, not yet a causal mechanism

The categorical tail has normalized entropy .053, mean maximum probability
.946, and logit standard deviation 11.19. Decoder block 5 is also sharp
(.172/.833/5.84). Yet tail writes retain 29.6% exact-layout sample-specific
variance, and swapping `z` strongly damages both heads. This falsifies the
simple claim “attention saturated, therefore the decoder ignores latent
content.”

Two mechanisms remain:

1. **Useful deterministic ownership.** Stable, sharp and reasonably balanced
   routes assign output coordinates to latent slots; sample-specific values do
   the actual decoding.
2. **Frozen/imbalanced routing bottleneck.** Winner-take-all maps underuse slots
   or lock queries to suboptimal owners, starving Q/K of useful reassignment
   gradients and limiting conditional code precision.

The unique discriminator is per-slot winner load/utilization plus an
inference-only temperature perturbation (and ideally its Q/K gradient effect).
A bounded attempt was terminated during the repository's full bank SHA scan to
respect the time cap; it produced no scientific result. Therefore attention
saturation must remain a leading risk, not a found cause.

## Narrow supported conclusion

The early prior/collapse basin has been escaped. At step 98k, the model has a
real, ranked latent carrier, strong sample-conditioned decoding, and useful
row-wise scale. What remains is a code-payload precision/coverage failure:
conditional logits are still heavily shrunk toward code zero, so most nonzero
GPTQ magnitude is not emitted.

The present evidence narrows the cause to three live alternatives:

1. the ordinal objective's center-seeking geometry under a lossy conditional
   representation;
2. a sharp but potentially imbalanced/frozen attention ownership map;
3. a constant-LR noise floor/update churn after the initial escape.

None of those three is yet uniquely identified. The two cheapest decisive
checks are (a) same-batch step90k-vs-step98k evaluation and (b) slot-load plus
temperature/QK-gradient probing. Architecture or loss changes based solely on
the observed attention entropy would be premature.
