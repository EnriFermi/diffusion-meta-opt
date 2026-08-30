# Parallel categorical GPTQ: would the previous latent hinge fix the collapse?

Date: 2026-08-29 UTC. Read-only review; the live trainer was not signalled or
modified.

## Failure being addressed

At checkpoint 1,000, the categorical code branch matches the analytic
input-independent prior and hard reconstruction is at the zero-output baseline.
The latent is approximately

`z[b,s] = global + sample_main[b] + slot_main[s] + interaction[b,s]`,

with 98.69% / 0.72% / 0.59% of centered variance in the sample-main,
slot-main, and sample-by-slot interaction terms. Decoder cross-attention is
nearly uniform over 32 slots. Exact-layout latent swapping changes only 2.12%
of hard code predictions and increases code loss by 0.37%, while it strongly
damages scale. The failure is therefore specifically missing conditional code
use, not absence of every sample signal.

Primary evidence:

- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/mechanistic_probe_step1000.json`
- `/home/coder/project/docs/report/parallel_categorical_gptq_hostile_review_step1000_20260829.md`
- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/train_metrics.jsonl`

## What the previous hinge actually constrains

The exact old auxiliary constructs the two-way-centered residual

`r[b,s] = z[b,s] - mean_s(z[b,*]) - mean_b(z[*,s]) + mean_bs(z)`

and applies cross-sample and within-sample RMS floors to `r / stopgrad(RMS(z))`.
It removes the global, sample-main, and slot-main components. It therefore
pushes only sample-by-slot interaction variance; it does not require that the
decoder use that variance or that it contain GPTQ-code information.

With uniform decoder attention, the relevant local algebra is decisive. If the
value map is linear,

`read_q = mean_s V(z[b,s])`, while `mean_s r[b,s] = 0`.

Thus the component directly enlarged by the hinge is in the nullspace of an
exactly uniform linear read. The implemented value path applies tokenwise
RMSNorm before its linear projection, so cancellation is exact only for the
linearized path around identical slots; nonlinear higher-order effects can
remain. Nonzero interaction can also indirectly create Q/K routing gradients
by making values unequal, so the hinge can serve as a symmetry breaker, but it
gives no task-aligned gradient to Q/K and can be satisfied by a low-rank hash or
decoder-irrelevant variation.

## Direct evidence against sufficiency

1. In the previous production intervention using this exact hinge, its own
   geometry target was satisfied: checkpoint-1,000 interaction RMS ratio was
   0.542 and the loss was zero. Nevertheless, matched-to-swapped direction loss
   changed by only 0.021%, predicted direction remained a common template, and
   the unchanged reconstruction objective did not improve materially. Within-
   sample effective rank remained only 1.77 median. Evidence:
   `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_500k_v1/analysis_review.md`.

2. In the current categorical run, the detached old hinge would be 0.05019 at
   step 1,000, so it would oppose the early geometric collapse. But by live step
   2,080 the interaction RMS ratio recovered naturally to 0.10456, slightly
   above the 0.10 global margin, while hard raw NRMSE was still 1.00026 and code
   loss/accuracy remained at the prior solution. The exact per-axis hinge was
   still 0.00943 because weak samples/slots remained, but the key point is that
   nontrivial interaction variance and failed conditional reconstruction
   coexist in this architecture too.

3. The previous hinge can dominate precisely the wrong subspace. In the old
   run, its active step-600 gradient RMS at z was 1.235e-7. Current code credit
   at z is only about 2.50e-9 at step 1,000. These are different runs and not a
   calibrated same-state ratio, but they expose the risk: coefficient 1 can
   train arbitrary interaction geometry much more strongly than code content.
   The old run also had a transient clipped step when the hinge activated.

## Competing fix mechanisms and predictions

### H1: reuse only the old latent hinge

Prediction: interaction statistics rebound above 0.10, and Q/K gradients may
increase because values are no longer identical. However code likelihood can
remain at the unconditional prior, attention can remain near-uniform, and the
exact-layout code swap gap can remain negligible. This is plausible as a
symmetry-breaking guardrail, not established as a reconstruction fix.

### H2: task-coupled same-layout code retrieval

For every exact-layout group, score prediction `i` against every true GPTQ code
field `j` using negative mean ordinal loss, then apply row-wise cross-entropy
with the diagonal as the positive. Retain the ordinary absolute code and scale
losses.

When all predictions are the same prior, the retrieval objective sends each
sample its own target-minus-group-average code gradient. Exact-layout address
queries are identical within a group, and the current decoder has no direct
sample-specific Distribution-Encoder path, so the only route that can solve
the discrimination is `z`. This directly targets conditional code information,
not arbitrary latent spread. Pairwise scoring reuses the existing forward
logits; it does not require a second decoder pass.

Unique success predictions:

- the same-layout retrieval diagonal beats chance;
- `dL_code/dz` does not collapse by hundreds-fold while the head keeps learning;
- correct-z code loss becomes materially better than shuffled-z code loss;
- live code metrics beat the sealed unconditional prior and hard NRMSE drops
  below 1.

Failure modes to monitor:

- negative gaming: retrieval improves while matched absolute ordinal loss and
  NRMSE do not; keeping the absolute loss and gating on those metrics rejects it;
- finite-bank memorization: training retrieval improves but held-out matrices
  do not;
- temperature/sequence-scale saturation: retrieval gradients vanish or
  dominate before conditional metrics improve;
- exact-layout coverage is incomplete, so report eligible fraction and do not
  interpret all-batch averages as direct retrieval supervision.

### H3: carrier-free terminal readout

Make the final code head read a terminal cross-attention write rather than the
residual query state. This removes the direct query-to-logit prior carrier while
retaining queries for addressing. It is a sensible architectural fallback if
H2 produces healthy code-specific gradients but those gradients still fail at
the attention read. It is not sufficient by theorem: any classifier can still
represent a global class prior, and a uniform read of broadcast values remains
uninformative.

## Narrow recommendation

Do not restart with the previous coefficient-1 latent hinge as the sole fix.
It would likely prevent the exact early interaction statistic from falling as
far, but existing evidence shows that it can be satisfied in a decoder-null,
low-rank, or task-irrelevant subspace.

The smallest mechanism-matched next intervention is the same-layout,
target-aware code retrieval loss on the final categorical predictions, together
with the existing absolute code and scale objectives. The old hinge may be
retained only as a secondary symmetry-breaking guardrail if desired, but it
must not be the primary claim or gate. If retrieval establishes a strong
code-gradient at z yet the attention read remains uniform and conditional code
metrics do not improve, then the next required change is a carrier-free
terminal cross-attention readout rather than a stronger latent-variance margin.
