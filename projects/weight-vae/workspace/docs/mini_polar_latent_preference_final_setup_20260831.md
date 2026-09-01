# Mini Polar Latent-Preference Experiment: Final Setup

Date: 2026-08-31

Status: design frozen for implementation; no new training launched by this
document.

## 1. Question

Can we make the mini polar autoencoder learn informative latents early and make
the decoder use those latents, without degrading the original pre-categorical
behavioral plus structural regression objective?

The experiment separates two mechanisms:

1. encoder representation pressure: `z` must distinguish matched operator
   instances instead of collapsing to an uninformative code;
2. decoder conditional-use pressure: the original losses must be lower with the
   matching `z` than with a hard mismatched `z`.

The stopped step-72,611 checkpoint is an evaluation reference, not the start of
the main causal run. At that checkpoint a random latent roll already raises the
fixed-probe loss by `+2.4625`, and latent entropy effective rank is `98.05`.
Starting from it would not test prevention of the early weak-latent phase.

## 2. Invariants

Keep unchanged across every causal-panel arm:

- exact 9,938,689-parameter mini polar model before the auxiliary head;
- tile `32x32`, patch size `16`, hidden dim `192`, 16 latent slots of dim 48;
- Distribution Encoder and all existing decoder conditioning paths;
- original coordinate-routed objective:
  `behavioral(direction + 10*scale) + structural(direction + 10*scale)`;
- raw operator MSE remains diagnostics-only and has backward coefficient zero;
- AdamW, LR `5e-5`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0.01`;
- batch size 128, seed 42, BF16 autocast, TF32, clip norm 5;
- same sealed WeightCLIP manifest and the same marginal dataset/checkpoint/layer
  coverage.

The baseline model must be constructed with seed 42 before the auxiliary head
is initialized with its own seed. This preserves identical base-model starting
weights.

## 3. Hard-paired batch contract

The causal panel uses 64 deterministic pairs per B128 batch. Members of a pair
must match on all decoder-visible nuisance variables:

- dataset;
- checkpoint index / training epoch;
- exact `layer_key`;
- operator operation, depth and role;
- parent tile row and column;
- p32 subtile input and output indices;
- input/output mask signatures;
- the same graph gauge transformation.

They must differ in:

- lineage;
- checkpoint SHA;
- underlying weight tensor and activation context.

Primary training negatives are different-lineage/same-epoch pairs. The
same-lineage/adjacent-epoch pair is reserved for evaluation because it can be
nearly identical and therefore become a false negative.

Pairing is deterministic from seed 42 and balanced so every lineage appears
equally often as the first and second member. Both members receive the normal
base regression loss. One direction per pair receives the extra negative
decoder pass, selected by a deterministic balanced bit; this gives 64 negative
decodes per step rather than 128.

### Gauge handling

The first 3k-step causal panel is canonical-only for all arms. This makes exact
tile-coordinate matching unambiguous. It changes the view schedule, so a
paired-sampler control arm is mandatory; historical baseline metrics are
context, not the formal control.

If the combined objective passes the causal gate, the confirmation loader must
apply one shared valid graph gauge to both members before tiling. Reusing equal
`view_index` values from different checkpoints is forbidden because their
actual permutations differ. The shared-gauge transformation must pass the
existing exact inverse/equivalence check before the confirmation launch.

No fallback to random, different-layer or different-coordinate negatives is
allowed. Missing pair coverage is a sampler error, not a reason to weaken the
negative definition.

## 4. Representation head and loss

The training-only projection head is:

```text
z [B,16,48]
 -> RMSNorm over dim 48
 -> flatten [B,768]
 -> Linear(768,256)
 -> GELU
 -> Dropout(0.10)
 -> Linear(256,128)
 -> L2 normalize
```

It adds about 0.23M parameters and is discarded for decoding/evaluation.

Two independent projector-dropout passes of the same `z` form the positive
pair. The paired sample from Section 3 is the hard negative. Use symmetric
binary InfoNCE in both directions with temperature `0.10`. Do not use “same
layer” or “same dataset” as a positive label: that would encourage layer or
dataset prototypes while removing instance-specific operator information.

Call the resulting mean loss `L_repr`.

## 5. Decoder latent-preference loss

For anchor `i` and its matched hard negative `j`:

```text
L_pos_i = original per-example routed loss of target i under D(z_i, c_i)
L_neg_i = the same routed loss of target i under D(stopgrad(z_j), c_i)

gap_i = (L_neg_i - L_pos_i) / stopgrad(L_pos_i + 1e-6)

L_pref_i = tau * softplus((margin - gap_i) / tau)
```

Use:

- relative margin `0.50` (the mismatched loss should be at least 50% worse);
- smooth-hinge temperature `tau=0.10`;
- mean over the 64 anchors selected for the negative pass.

The positive branch is the ordinary full model graph. In the negative branch:

- the foreign latent is detached;
- target `c_i`, coordinates and masks are retained;
- the already-constructed direct-conditioning/query state is detached before
  the shared decoder blocks, so negative pressure cannot train the Distribution
  Encoder or coordinate/query conditioner into a cheap mismatch detector;
- gradients still reach the shared decoder blocks, direction/scale tails and
  output heads.

The finite margin is essential. It stops negative pressure after sufficient
separation. An unbounded `L_pos-L_neg` objective is forbidden because it can win
by making the negative branch arbitrarily pathological.

The preference comparison uses a within-example reduction of the same four
routed terms. A focused B=1 parity test must prove exact agreement with the
historical helper. The ordinary positive `L_base` continues to call the
historical batch helper unchanged; its batch-level behavioral weighting is not
silently replaced by a mean of within-example reductions.

## 6. Total objective and coefficient calibration

Never replace the original loss. Optimize:

```text
L_total = L_base + lambda_pref * L_pref + lambda_repr * L_repr
```

Determine static coefficients once on a sealed fixed paired probe:

- choose `lambda_pref` so its initial combined encoder+decoder gradient norm is
  0.25 times the base-loss gradient norm;
- choose `lambda_repr` so its encoder gradient norm is 0.10 times the base-loss
  encoder gradient norm;
- clamp resolved `lambda_pref` to `[0.02, 10.0]`;
- clamp resolved `lambda_repr` to `[0.001, 0.20]`;
- store raw and weighted component gradient ledgers and the resolved numbers in
  `loss_calibration.json`.

The online coefficients are static after calibration. Ramp both linearly from
zero to their resolved values during optimizer steps 1 through 1,000. This
keeps the base initialization exact and avoids an auxiliary-gradient shock.

If any weighted auxiliary/base gradient ratio exceeds 0.75 at a telemetry step,
the run is invalid and stops with a diagnostic checkpoint. Do not silently
renormalize coefficients online.

## 7. Causal panel

All arms start from the exact same seed-42 model initialization, head seed,
paired data plan and logical cursor:

1. `paired_control`: original loss only; the head is absent;
2. `repr_only`: `L_base + lambda_repr*L_repr`;
3. `pref_only`: `L_base + lambda_pref*L_pref`;
4. `combined`: all three terms.

Run one 20-step contract smoke, then 3,000 optimizer steps per arm. The small
panel is the experiment-selection gate, not a production baseline.

The panel distinguishes:

- head-only improvement: representation pressure is sufficient;
- preference-only improvement: the decoder-use constraint is sufficient;
- combined-only improvement: both sides of the encoder/decoder coordination
  failure must be addressed;
- no improvement or worse positive loss: the proposed mechanism is not
  supported under the matched-negative intervention.

Extend the winning arm to 30,000 steps, covering the historical two transition
regions. Extend to step 72,611 only after the 30k review shows a real benefit.

Before a long confirmation launch, an independent reviewer must inspect the
final implementation/config and give explicit GO.

### 7.1 User-authorized production override (2026-08-31)

The user explicitly approved launching the combined arm directly for the full
500,000-step horizon, without stopping at 3k or 30k and without another user
approval gate. The short panel remains the scientifically cleaner ablation but
is superseded for this operational launch. A production-faithful B128 smoke
showed that the originally proposed `lambda_pref <= 0.50` clamp would reduce
the requested 0.25 gradient-ratio target to about 0.017. The final static
calibration cap is therefore 10.0; the resolved smoke value was 7.2933 and was
not clamped. The 1,000-step linear ramp and 0.75 diagnostic stop remain.

## 8. Required metrics

Keep every historical loss, structural, behavioral, gradient, rate and VRAM
metric. Add:

- positive and negative total loss;
- positive and negative behavioral direction/scale;
- positive and negative structural direction/scale;
- absolute and relative preference gap: mean, p10, median and p90;
- fraction satisfying the 0.50 margin;
- hard-pair coverage (must be 100%), lineage/checkpoint mismatch checks and
  target-pair weight distance;
- positive projector cosine, paired-negative cosine, InfoNCE loss and pairwise
  retrieval accuracy;
- latent batch std, cross-sample cosine, entropy/stable rank and top energy
  fraction;
- fixed-probe interventions: paired latent swap, random latent roll, context
  roll and zero-latent ablation;
- gradient norms split by base, representation and preference objective and by
  encoder, latent bridge, decoder trunk, direction tail and scale tail;
- stage timings: data wait, host-to-device, encoder, positive decoder, negative
  decoder, loss, backward, optimizer and logging.

Log all metrics to JSONL and Comet. Write readable plots for base quality,
preference gap, representation geometry, component gradients and throughput.

## 9. Scientific gates

At 3,000 steps, a candidate passes only if:

- positive base loss is no more than 5% worse than `paired_control` at equal
  examples;
- matched relative-gap median is at least 0.50 and p10 is positive;
- the margin is not paid by only one loss family (no single direction/scale or
  behavioral/structural component contributes more than 80% of the median
  absolute gap);
- latent effective rank and variance are not below the paired control;
- paired-swap sensitivity improves without a comparable deterioration in the
  normal fixed-probe loss;
- no NaNs, missing gradients, repeated samples, metadata mismatches or invalid
  pairs occur.

The conclusion must remain narrow: this panel can establish earlier
instance-specific latent use under the paired WeightCLIP protocol. It cannot by
itself establish better downstream weight generation.

## 10. Math-preserving speed changes

Apply the same runtime changes to all panel arms:

- background CPU batch producer with a bounded queue of four batches;
- pin assembled batch tensors before transfer;
- overlap batch preparation and host-to-device transfer with GPU work;
- remove unconditional per-step `torch.cuda.synchronize`; synchronize only at
  telemetry boundaries and before checkpoint/stop;
- keep ordinary AdamW for the causal panel; fused AdamW and `torch.compile` are
  separate speed interventions and must not be mixed into this loss comparison;
- decode negatives for only 64 anchors per B128 step.

The optimized `paired_control` should target at least 1.5x the historical
median step throughput. A loss arm must retain at least 70% of optimized control
throughput. If either threshold fails, profile the recorded stage timings before
launching 30k.

## 11. Artifacts and checkpointing

Use unique roots per arm. Every resume checkpoint stores:

- base model and projection head when present;
- optimizer state including the new head group;
- RNG state;
- paired sampler cursor and balanced direction bit;
- resolved loss calibration and full config;
- phase step and total examples consumed.

The original stopped run and its checkpoints are immutable references:

- `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1`
- `/dev/shm/weightclip_mini_polar_regression_10m_p16_tile32_b128_500k_v1/resume_latest.pt`
