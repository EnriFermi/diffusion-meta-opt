# Gauge-fixed direction path: independent causal review

Snapshot: 2026-08-29 15:50 UTC. The live run was not signalled, paused, or
otherwise mutated. Exact extracted comparison rows are in
`docs/notes/gauge_fixed_direction_causal_review_20260829.csv`.

## Failure definition

In the fresh fixed-Frobenius direction-head run, the head norm constraint works
numerically, but the direction head rapidly loses stable rank, valid raw-logit
norms grow, and gradients from both absolute direction supervision and the
direction InfoNCE weaken sharply before reaching the latent/encoder. Direction
loss itself is not solved.

Primary live evidence:

- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_gauge_fixed_500k_v1/train_metrics.jsonl`
- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_gauge_fixed_500k_v1/gradient_telemetry.jsonl`

Exact same-start unconstrained comparison:

- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_500k_v1/train_metrics.jsonl`
- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_500k_v1/gradient_telemetry.jsonl`

## Narrow observations

The fixed radius stays at about 0.08062065. Nevertheless, between gradient
steps 1 and 100:

- spectral norm rises from 0.02196 to 0.04006;
- stable rank falls from 13.48 to 4.05, and reaches 2.45 at step 200;
- row norms do not individually collapse: at step 200 their min/median/max are
  0.01445 / 0.01925 / 0.02526. The failure is principally row correlation / a
  concentrated singular spectrum, not a single dead output row;
- raw-logit p01/median/p99 move from 0.0717 / 0.0862 / 0.0929 to
  1.2048 / 1.3280 / 1.4552 at step 100;
- the direction objective is 1.900 at step 1, 1.890 at step 100, and 1.982 at
  step 200. Vanishing gradients therefore do not mean the task was solved.

At the latent, the absolute-direction gradient RMS falls from 3.292e-6 to
1.653e-7 at step 100 and 2.727e-8 at step 200. InfoNCE falls from 6.057e-6 to
1.455e-7 at step 100, then is 3.108e-7 at step 200. Scale is still live:
4.082e-7 at step 100 and 1.553e-7 at step 200.

The first unmistakable direction-only failure is downstream of the split:

- direction-head absolute gradient: 0.6107 -> 0.0170 -> 0.00651;
- direction-tail Q absolute gradient: 1.485e-5 -> 3.963e-7 -> 1.759e-7;
- direction-tail V absolute gradient: 3.672e-4 -> 1.855e-6 -> 1.516e-6.

Thus scale-vs-direction conflict in the shared trunk cannot be the sole cause:
the gradient is already severely attenuated inside the direction-only
head/tail. In the shared encoder it becomes an amplifier: at step 200 the
encoder-1 Q scale gradient is about 20 times the absolute-direction gradient
and their cosine is -0.776.

## Leading proximal mechanism

Let `l = W h` be a 16-dimensional raw direction logit and
`u = l / ||l||` the prediction. Backward through this normalization is

`grad_l = (I - u u^T) grad_u / ||l||`.

Two facts matter together:

1. The hard Frobenius sphere fixes only the sum of squared singular values. It
   does not stop the matrix from moving norm budget into one or two singular
   modes.
2. The decoder can align normalized hidden states with the corresponding
   high-gain right-singular subspace. Then logits align with the matching
   left-singular subspace and
   become large. The normalization Jacobian removes precisely that radial/top
   logit component. Multiplication by `W^T` can return only through the
   remaining angular singular channels.

This is the leading explanation of the otherwise paradoxical combination: the
head matrix itself still receives a gradient, while the gradient into direction
hidden/tail weakens. It also explains why a global radius alone was
insufficient. Current telemetry is consistent with this mechanism but does not
directly capture the activation gradients needed to prove the boundary.

The hidden-alignment implication is already strongly suggested, though not
directly measured. Direction hidden is passed through non-affine RMSNorm, so
its norm is approximately `sqrt(1536)`. The ratio
`median raw norm / (spectral norm * sqrt(1536))` rises from about 0.099 at step
1 to 0.846 at step 100. This is close to the operator-norm upper bound and is
incompatible with raw-logit growth being only a global head scaling effect. It
shows concentration in a high-gain singular subspace; it does not, by itself,
prove alignment to one specific top singular vector.

Crucially, uniformly scaling `W` cannot by itself kill the gradient into `h`:
the `1 / ||l||` factor is cancelled by the same scale in `W^T`. Anisotropy plus
alignment, not global norm, is required for this upstream bottleneck.

## Competing mechanisms and discriminating predictions

### A. Fixed norm-budget plus Adam/retraction drives singular concentration

Prediction: large per-step angular motion of the small-radius head, a sizable
radial component in the Adam-preconditioned update before retraction, and
transfer of squared singular mass from non-top to top modes. Current telemetry
only proves post-retraction norm and first-moment tangency; it does not log the
pre-retraction update or retraction angle. This is a viable trigger, not yet a
proved root cause.

### B. Task/data already drive the unconstrained head toward the same low-rank
carrier

Prediction: at matched old checkpoints, the composite affine-RMSNorm/head
matrix also loses stable rank and its hidden states align with top modes. The
old run has no singular-spectrum/raw-logit telemetry and no retained step-100
checkpoint. Current same-step loss/gradient comparison therefore cannot
distinguish intrinsic task pressure from a failure introduced by the sphere.
The old run also shows direction-gradient decay by step 200, so the sphere did
not create the entire pathology; it changed and initially intensified it.

### C. Absolute direction and InfoNCE cancel

Prediction: each objective has a healthy gradient but their cosine approaches
-1. Contradicted as the sole cause: each objective is separately small at
step 100; their cosine is +0.375 at the head and +0.088 at the latent. Some
tail/shared tensors have negative cosines, so local cancellation can amplify
the failure but does not originate it.

### D. Scale loss kills direction in the shared path

Prediction: direction remains healthy in the private head/tail and disappears
only after merging into the shared decoder/encoder. Contradicted as the sole
cause by the private direction-tail decline. Strong negative scale/direction
cosines and scale dominance at step 200 show it is a downstream amplifier.

### E. Whole-latent collapse

Prediction: latent interaction statistics cross the anti-collapse margin and
the hinge remains active. Not supported: overall latent RMS grows and the
two-way interaction statistics generally remain above the 0.10 margin. A
direction-specific latent subspace could still collapse; the current aggregate
hinge does not measure it.

### F. Attention saturation / QK death is the primary cause

Prediction: attention entropy/logits saturate before raw logits grow, while a
direct gradient at direction hidden remains healthy. Current artifacts do not
log attention entropy, QK norms, direction-hidden gradients, or their temporal
order. Q/K gradients are small, but V gradients and the output-normalization
path also collapse, so QK saturation is not established as the initiating
mechanism.

### G. Logging or decomposition bug

The gradient telemetry uses separate `torch.autograd.grad` calls for exact
weighted direction, scale, and InfoNCE objectives. Component parity is zero,
all values are finite, and the two variants match at step 1. A logging bug is
not a viable explanation for the main trend.

## Hostile boundary check and missing decisive probe

What is supported now is a subsystem localization, not an exact activation
boundary: the earliest clearly affected direction-exclusive subsystem is
`direction tail -> direction head -> unit normalization`. At step 100 the
gauge-fixed and old runs have similar absolute head-weight gradient RMS
(0.0170 versus 0.0152), but the gauge-fixed tail-Q gradient is 5.2 times lower
and encoder-1-Q gradient is about 95 times lower. This is compatible with a
bad `W^T` angular transfer, but it could also include a changed tail/attention
Jacobian. Parameter-gradient RMS values across differently sized modules cannot
prove that the exact death occurs at the normalization boundary. It is not
supported to name either that boundary or a particular attention block as the
found root without activation-gradient captures.

At the step-1000 checkpoint, the decisive read-only probe should capture, on
one fixed batch and separately for absolute direction and InfoNCE:

- all head singular values;
- direction-hidden projection energy on every right singular vector;
- raw-logit, `grad_u`, and `grad_l` projection energy on every left singular
  vector;
- the two transfer ratios `||grad_l|| / ||grad_u||` (unit-normalization
  Jacobian) and `||W^T grad_l|| / ||grad_l||` (head angular return), followed by
  gradients at tail input, shared state, latent, and encoder output;
- pre-retraction Adam update radial fraction, retraction angle, and singular
  mass transfer;
- attention entropy/QK-logit percentiles only to test the remaining attention
  alternative.

This can establish the current proximal death boundary: if the drop occurs in
the first two ratios and singular-mode decomposition, the spectral-alignment
story is correct; if `grad_h` is healthy but tail/shared gradients fail, the
attention/Jacobian alternative wins. It can also compute a virtual next Adam
update from the saved moments without mutating the run to test retraction
geometry.

One step-1000 gauge checkpoint cannot establish whether the sphere caused the
singular concentration. The old unconstrained run did not retain a matched
step-1000 model with equivalent telemetry. That causal trigger question still
requires either a matched old checkpoint/replay or an independently retained
control state. Step 1000 can prove the proximal mechanism, not the
between-variant cause.

## Relevant source

- Direction head and unit normalization: `projects/weight-vae/workspace/training/weightclip_benchmark/run_direct_normalized_scaled_700m_production.py`, lines 800-839.
- Fixed composite radius and affine-free RMSNorm: same file, lines 888-924.
- Retraction and moment projection: same file, lines 1155-1238 and 2963-2972.
- Separate loss-gradient telemetry: same file, lines 1483-1565 and 1960-2019.
- Both structural and activation direction losses are cosine-like normalized
  objectives: `projects/weight-vae/workspace/big_vae/models/big_weight_vae_parts/loss_mixin.py`, lines 158-172 and 423-432.
