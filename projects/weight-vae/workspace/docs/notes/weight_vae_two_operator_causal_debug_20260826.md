# Weight VAE two-operator causal debugging — 2026-08-26

## Narrow conclusion

For this exact two-operator memorization task, the primary bottleneck is
localized to the old **encoder weight-content interface and stack**. The tested
protocols do not support an intrinsically incapable decoder, the structural
loss implementation, batch averaging, Adam epsilon, or the mandatory latent
bridge as the primary explanation.

In V5/V6, raw weight information is mixed with large sample-common and
X-conditioned components before and during the Perceiver encoder. The resulting
latent is almost identical across two nearly orthogonal weight matrices. The
decoder therefore learns a global or position-indexed template, and tile-level
directional gradients at the common state mostly cancel. A direct, high-rank,
bias-free raw-W readout (V8) supplies a clean W-specific latent to the otherwise
unchanged V6 bridge/PosFiLM/decoder and overfits both operators within 500 steps:

| Metric | V5/V6 failure | V8 step 500 |
|---|---:|---:|
| mean full-operator direction loss | `0.9839` V5 / `0.7671` V6 | `0.04073` |
| maximum per-operator direction loss | `0.9863` V5 / `0.7672` V6 | `0.04239` |
| maximum NRMSE | `1.426` V5 / `1.264` V6 | `0.3461` |
| minimum paired-z swap direction penalty | approximately `0` | `0.95659` |
| minimum fixed-X W-swap direction penalty | not available / approximately `0` behaviorally | `0.95669` |
| raw-z paired relative delta | about `0.0012` V6 | `1.41087` |

This establishes that a clean raw-W content path is sufficient to solve the
two-operator problem and that the existing downstream decoder can use it. It
does **not** yet establish generalization to a large operator bank or isolate
which individual old-encoder component is necessary and sufficient, because
V8 jointly bypasses the tokenizer and Perceiver content stack. In particular,
the evidence does not identify sample-common bias, X/value mixing, or a
specific Perceiver operation as the sole root cause.

## Failure definition

The smoke dataset contains exactly two distinct canonical matrices from the
same layer, each reconstructed from all nine tiles:

- shape: `1152 x 128`;
- layer: `layer3.0.conv2.weight`;
- matrix cosine: `0.000869672`;
- physical batch: all 18 tiles, one optimizer step per exact cycle;
- active objective: structural direction `1.0`, structural scale `0.1`.

Failure means that matched full-operator direction loss does not improve while
swapping the operators' latents has negligible effect. Success must improve
both operators and show a positive latent-swap penalty; a common or
position-only template is not success.

The run-bound resolved two-operator manifest is:

`/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/two_full_operator_manifest.json`

## Validity checks

- Both operators are present in every native B18 cycle; tile stitching is
  bitwise exact and logical order repeats deterministically.
- The two matrices are nearly orthogonal, so a single target template cannot
  fit both to low directional loss.
- The same structural loss/evaluator can optimize an unrestricted output
  tensor to near-zero direction loss, excluding a loss or masking defect.
- V8 starts from random initialization and preserves the V6
  bridge/decoder/head initialization under a local RNG fork.
- The exact B18 preflight runs before the first optimizer step. It verified 768
  unique routes, strict IEEE-FP32 linearity of the V8 readout at fixed routing,
  exact zero W -> zero z -> zero bridge behavior, and gradients far above Adam
  epsilon for Q, K, output, bridge, and direction head. The unchanged
  downstream decoder is not globally zero-preserving; its zero-W output is
  logged rather than assumed to be zero.
- The final run completed exactly 500 steps with no NaN, Inf, traceback, or
  clipping event. The training log ends with `Training completed successfully`.
- Two earlier V8 attempts were stopped before optimizer step 1 and preserved
  at
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step.invalid_step0_routing_scale_bf16_gate`
  and
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step.invalid_step0_tf32_gate`.
  They contain step-0 diagnostics but no training checkpoint and are not used
  as training evidence. They identified and fixed, respectively, real Q/K
  routing-scale takeover and a TF32-vs-IEEE-FP32 diagnostic mismatch.

Exact preflight:

`/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/v8_exact_b18_step0_preflight.json`

Key preflight values:

| Check | Value |
|---|---:|
| routes / unique argmax | `768 / 768` |
| median effective keys per route | `2.468` |
| strict TF32-disabled FP32 superposition error | `8.11e-8` |
| native BF16 superposition error | `0.003740` |
| Q / K gradient RMS | `1.84e-5 / 1.69e-5` |
| readout output gradient RMS | `1.83e-4` |
| bridge V / output gradient RMS | `1.29e-4 / 9.89e-5` |
| direction-head gradient RMS | `1.30e-3` |
| smallest gradient / Adam epsilon | `1691x` |

## Competing causal mechanisms and unique predictions

| Mechanism | Unique prediction | Result |
|---|---|---|
| Loss, mask, data, or accumulation bug | Free output cannot fit; operator/tile inventories or reductions disagree | Excluded by free-output control, exact stitching, exact cycle, and matched evaluator |
| Full-B18 cross-sample cancellation is the primary cause | Singleton or homogeneous updates escape while the mixed batch fails | Excluded as primary in the tested arm: cyclic-singleton still ends near direction loss `0.98` and swap gap `~0` |
| Adam `eps=1e-8` permanently floors the content path | Exact state fork with `eps=1e-12` restores operator identity | Excluded as the root explanation in the tested fork: both arms stay at mean direction `~0.984`, swap gap `~0` |
| Mandatory bridge destroys a good encoder code | Strong W identity exists before bridge but disappears after it | Excluded as the initial loss point for V6: raw z is already collapsed; equal-RMS z perturbations pass through the bridge |
| Missing positional carrier is sufficient | PosFiLM enables W-specific reconstruction | Excluded as a sufficient fix: V6 learns the shared per-query oracle (`~0.7657`) but z-swap remains zero |
| Encoder creates a sample-common/X-heavy code and dilutes W content | A/B variation collapses across encoder depth; zero-W resembles matched; direct W content fixes it | Supported by V6/V7 probes and decisively by V8 rescue |
| V8's temporary plateau is a permanent optimizer/readout death | W identity or gradients disappear and do not recover | Excluded: the same optimizer exits the plateau after step 200 and reaches `0.0407` |

## Discriminating experiment ladder

### V5: mandatory bridge with no positional value carrier

At step 2500 the model matches the one-global-direction oracle:

- operator direction losses: `0.98154`, `0.98645`;
- raw-z A/B cosine: `0.999996`;
- raw-z relative delta: `0.003062`;
- bridge attention entropy: `0.999941` normalized;
- bridge output query-centered energy: `1.105e-6`;
- latent swap changes output by only `0.001322` relative.

The latent projection `[Wk; Wv]` is full column rank with condition number
`4.01`, so projection nullspace is not the mechanism. Individual tile
directional gradients at the common bridge state cancel at `0.23578`, nearly
the random-vector reference `1/sqrt(18)=0.23570`.

Evidence:

`/mnt/shared/weightclip_benchmark/ae_v5_two_full_operator_overfit_5000step/debug/causal_lane_a_step2500/report.md`

### V6: fixed positional FiLM

PosFiLM restores query identity and quickly learns a shared position-indexed
template, but does not learn operator identity:

- final step 500 mean direction loss: `0.76705`;
- per-query shared-template oracle: `0.765695`;
- minimum z-swap direction delta: `-1.45e-4`;
- raw-z paired relative delta: `0.001231`.

The encoder probe localizes the content loss:

- tokenizer `y_proj.bias` RMS / W-dependent contribution RMS: `77.37x`;
- paired relative signal: tokenizer `0.03510` -> Perceiver layer 0 `0.003451`
  -> raw z `0.001203`;
- latent RMS grows `0.395 -> 3.311` while batch-centered energy reaches only
  `7.15e-7`;
- zero-W, W-swap, and X-swap all change raw z by only about `0.1%`.

The bridge preserves or slightly amplifies the already tiny z delta, so it is
not the point where W identity is first lost.

Evidence:

`/mnt/shared/weightclip_benchmark/ae_v6_two_full_operator_posfilm_overfit_2000step/debug/encoder_identity_step500_r3/report.md`

The step-500 model and evaluation are valid. The configured longer V6 run was
interrupted while saving resume state after that evaluation, so this evidence
does not imply completion of the nominal 2,000-step horizon.

### Cancellation and optimizer discriminators

The cyclic-singleton arm changes the intra-cycle stochastic/Adam trajectory
but still ends at mean direction `0.97952`, NRMSE `1.424`, and z-swap delta
`8.9e-7`. Therefore gradient cancellation is a real consequence of the common
state, but not the primary cause: removing full-cycle averaging does not create
a useful representation.

Evidence:

`/mnt/shared/weightclip_benchmark/ae_v6_two_op_causal_cyclic_singleton_504step/checkpoints/train/weightclip_ae_v6_two_op_causal_cyclic_singleton_504step/stage_1/two_operator_metrics.jsonl`

Exact step-2500 Adam-state forks also fail to escape:

- `eps=1e-8`: final mean direction `0.984131`, swap delta `-5.9e-6`;
- `eps=1e-12`: final mean direction `0.983946`, swap delta `-6.6e-7`.

Evidence roots:

`/mnt/shared/weightclip_benchmark/ae_v5_two_op_eps_fork_step2500/eps_1e-8`

`/mnt/shared/weightclip_benchmark/ae_v5_two_op_eps_fork_step2500/eps_1e-12`

### V7: mandatory cross-refresh

V7 removes latent residual persistence and returns only each layer's normalized
cross-read. Through the last stored evaluation at step 300, it fails because
layers 0-8 can influence the final latent only through subsequent query
routing:

- layer-0 cross Q/K/V/output gradients are approximately `1e-20..1e-18` at
  step 1 and exactly zero at steps 100/200/300;
- layer 9 remains live;
- zero-W raw RMS is essentially identical to matched raw RMS;
- W-swap sensitivity is smaller than X-swap sensitivity at layers 0 and 9;
- layer-9 pre-RMS raw state grows `2.14x` by step 300 while centered energy
  falls `7.36x`;
- step-300 mean direction loss is `0.99706`, swap delta approximately zero.

The arm was intentionally stopped after the valid step-300 evaluation and has
no checkpoint. It demonstrates a routing-only gradient-null topology plus
persistent upstream common-token content and shows no rescue through step 300;
a delayed rescue by step 500 was not tested.

Evidence:

`/mnt/shared/weightclip_benchmark/ae_v7_two_op_causal_mandatory_cross_refresh_500step/checkpoints/train/weightclip_ae_v7_two_op_causal_mandatory_cross_refresh_500step/stage_1/two_operator_metrics.jsonl`

`/mnt/shared/weightclip_benchmark/ae_v7_two_op_causal_mandatory_cross_refresh_500step/checkpoints/train/weightclip_ae_v7_two_op_causal_mandatory_cross_refresh_500step/stage_1/grad_layer_rms.csv`

## V8 intervention

V8 replaces the old encoder content stack with one clean readout:

1. Values are the raw signed 16-element W patches, with no value projection or
   bias.
2. There are 24 independent routes per latent and 32 latents, yielding 768
   distinct routing rows. This removes the 512-rank ceiling of the rejected
   single-route draft and gives a theoretical fixed-X content capacity of up
   to 12,288 coordinates; the exact full-map numerical rank was not measured
   in the live run.
3. Fixed narrow anchors guarantee complete spatial coverage. X enters only as
   a bounded Q/K routing perturbation; it never enters the value stream.
4. The output mix is bias-free and identity-initialized.
5. There is no residual, gate, RMS normalization, or dropout on the W-value
   readout path.
6. The existing V6 mandatory bridge, PosFiLM, decoder stack, and heads are kept
   unchanged and bitwise seed-matched.

Implementation:

- `big_vae/models/vae_shared.py`: `CleanContentReadoutV8`;
- `big_vae/models/big_weight_vae_parts/encoding_mixin.py`: direct raw-W readout;
- `training/big_vae/two_operator_overfit.py`: matched, z-swap, fixed-X W-swap,
  zero-W, routing, and arithmetic diagnostics.

## V8 results

| Step | mean dir | max dir | max NRMSE | paired-z swap min | fixed-X W-swap min | Q/K routing saturation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1.00025` | `1.00087` | `23.217` | `-0.00016` | `-0.00011` | `0.000` |
| 1 | `0.97153` | `0.97318` | `6.818` | `0.01743` | `0.01743` | `0.000` |
| 100 | `0.96430` | `0.96579` | `1.413` | `0.02937` | `0.02937` | `1.000` |
| 200 | `0.96428` | `0.96578` | `1.413` | `0.02941` | `0.02939` | `0.958` |
| 300 | `0.81383` | `0.81895` | `1.367` | `0.17593` | `0.17595` | `0.000` |
| 400 | `0.26473` | `0.27006` | `0.764` | `0.72914` | `0.72923` | `0.000` |
| 500 | `0.04073` | `0.04239` | `0.346` | `0.95659` | `0.95669` | `0.000` |

At step 500, replacing W while holding X fixed gives direction losses
`0.99908` and `0.99663`, while matched losses are `0.04239` and `0.03907`.
Zero-W losses are `0.99364` and `0.99111`. This is decisive W-specific use,
not a common or positional-template solution.

Evidence:

`/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/checkpoints/train/weightclip_ae_v8_two_op_clean_content_readout_500step/stage_1/two_operator_metrics.jsonl`

`/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/checkpoints/train/weightclip_ae_v8_two_op_clean_content_readout_500step/stage_1/grad_layer_rms.csv`

### The step-100/200 plateau

The plateau is not a loss of W identity:

- raw-z paired delta remains `~1.4108` throughout;
- all 768 fixed routes remain distinct;
- fixed-X W-swap already costs `~0.0294` direction loss.

The learned X-only router transiently saturates:

- step 100 raw/bounded score RMS `2.374 / 0.10008`, saturation `1.0`;
- step 200 `1.377 / 0.09991`, saturation `0.958`;
- Q/K gradient RMS falls below Adam epsilon at those two probes.

By step 300 the learned score unsaturates (`0.02463 / 0.02050`, saturation
zero), and gradients recover by about 4.3-4.6 orders for Q/K, 3.3 orders for
the readout/bridge, and 2.0 orders for the direction head. At the same time the
decoder begins aligning the already available W code with the targets. By step
500 the learned X perturbation is small (`bounded RMS=0.01658`) and the fixed
anchors dominate numerically.

Router saturation coincides with the plateau and remains a viable contributing
mechanism, but its necessity and the causal split from downstream escape out
of the common-template stationary basin are **not established** by this one
trajectory. They recover together. A fixed-routing ablation would distinguish
them cheaply.

## Mechanisms excluded as primary explanations in the tested protocols

- **Decoder intrinsically incapable of using latents:** excluded by V8 with the same
  bridge/PosFiLM/decoder.
- **Mandatory bridge rank/nullspace as the root failure:** excluded by the
  full-rank projection audit and V8 success.
- **Pure batch-gradient cancellation:** excluded as primary by the singleton
  arm; cancellation remains a downstream amplifier of the common
  representation.
- **Adam epsilon as the root cause:** excluded by exact epsilon forks and by V8
  escaping its transient plateau with the same optimizer; epsilon can still
  modulate already-small updates.
- **Missing positional identity alone:** excluded by V6 learning only the
  shared per-query template.
- **Scale loss dominating direction:** the successful run uses the same
  direction `1.0`, scale `0.1` objective as the failed smokes; scale is already
  small while direction later falls by more than 20x.
- **Stale config, wrong checkpoint, wrong data, or logging reduction:** excluded
  by the fresh deterministic run, startup ledgers, exact evaluator, and
  intervention controls.

## Remaining uncertainty and precommit status

The experiment proves two-operator memorization, not operator-bank
generalization. V8 also jointly removes several old-encoder components, so it
does not by itself distinguish tokenizer bias, X/value mixing, Perceiver
fixed-RMS updates, and recurrent content dilution as independent causes.

The automated `early_success` checker was intentionally disabled for this
causal arm. The resolved config retains a reference threshold block with
`consecutive_evals=3` and `min_step=500`; consequently no automated success
flag existed for this run, and that runtime configuration could not accumulate
a three-row streak within 500 steps.

For the separate scientific sustained-evidence check, the frozen thresholds
were applied manually to the evaluation rows at 300/400/500 without runtime
min-step gating. Steps 400 and 500 pass strongly, but step 300 has maximum
direction loss `0.81895` against a threshold of `0.80`. Thus only two
consecutive rows pass and the three-evaluation scientific criterion is **not
confirmed**, even though the terminal result is decisive. No threshold was
changed post hoc.

## Recommended next experiments

1. **Fixed-routing V8 ablation, <=500 steps.** Freeze Q/K or remove the learned
   X perturbation. Prediction: if it avoids the 100-200 saturation and fits at
   least as fast, learned X routing is unnecessary for this scale.
2. **Scale operator count while retaining the 2,000-step cap.** Use 16, then 64
   canonical matrices with an operator-disjoint train/validation split. This
   tests whether the clean content interface generalizes rather than merely
   serving as a high-capacity memorizer. Precommit quantitative matched and
   swap thresholds with uncertainty intervals.
3. **Factor the old encoder changes.** Use a same-seed small factorial over at
   least tokenizer bias/projection and content depth/normalization, rather than
   relying only on one-at-a-time add-backs that can miss interactions. Preserve
   the fixed-X W-swap gate in every arm.
4. **Production criterion.** Do not launch a long operator-bank run until
   matched quality and both paired-z and fixed-X W-swap sensitivity exceed
   precommitted quantitative thresholds on held-out operators. A train-loss
   decrease alone is insufficient.

## Final artifacts

- Immutable model checkpoint (model-only, no exact optimizer resume):
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/checkpoints/train/weightclip_ae_v8_two_op_clean_content_readout_500step/stage_1/step_0000500.pt`
- Resolved config:
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/resolved_run_config.json`
- Run log:
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/runs/weightclip_ae_v8_two_op_clean_content_readout_500step/logs/train_rank0.log`
- Exact B18 preflight:
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/v8_exact_b18_step0_preflight.json`
- Full evaluation trajectory:
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/checkpoints/train/weightclip_ae_v8_two_op_clean_content_readout_500step/stage_1/two_operator_metrics.jsonl`
- Gradient trajectory:
  `/mnt/shared/weightclip_benchmark/ae_v8_two_op_clean_content_readout_500step/checkpoints/train/weightclip_ae_v8_two_op_clean_content_readout_500step/stage_1/grad_layer_rms.csv`
