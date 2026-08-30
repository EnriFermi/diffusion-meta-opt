# V5 two-operator causal lane A — step 2500

## Narrow verdict

The fixed checkpoint is trapped at an almost single global direction template. The immediate structural cause is the mandatory bridge topology combined with collapsed latents: the bridge uses the 1024 query states only as attention routing, attention is almost uniform over 32 nearly identical latent slots, and the returned value state loses essentially all absolute query identity. RoPE-only decoder self-attention cannot reconstruct that missing carrier. Directional gradients at the common bridge state are nearly random across the 18 tiles and cancel at the expected `1/sqrt(18)` rate.

This is not caused by a nullspace or rank defect in `[Wk; Wv]`.

## Validity

- Exact model checkpoint: `step_0002500.pt`, step 2500, 654,993,537 parameters.
- Exact deterministic canonical two-operator cycle: logical tiles 0–17, alternating A/B.
- BF16 inference/VJP on GPU0; zero optimizer steps.
- Directional VJPs use the differentiable total loss with all non-direction lambdas set to zero. The initial incomplete attempt using detached telemetry was stopped before forward and preserved separately as `causal_lane_a_step2500.invalid_detached_loss`.

## Discriminating results

### Common-template basin

- Model directional loss: A `0.981543`, B `0.986445`.
- Optimal one-global-direction target oracle: `0.984004`.
- Optimal per-query-position template shared across all 18 samples: `0.765695`.
- Optimal per-sample global direction: `0.964439`.
- Identity oracle: `4.71e-7`.

The model matches the global-template optimum while leaving a large query-position-only improvement unused.

### Query carrier deletion

| Node | raw RMS | query-centered RMS | centered energy fraction |
|---|---:|---:|---:|
| decoder query input | 47.7119 | 2.68837 | 3.204e-3 |
| bridge output | 4.47841 | 0.004708 | 1.105e-6 |
| decoder layer 0 output | 8.57452 | 0.007801 | 8.275e-7 |
| decoder layer 7 output | 19.3144 | 0.017286 | 8.009e-7 |
| final q norm | 1.00118 | 0.000895 | 7.984e-7 |
| direction head | 3.87479 | 0.002995 | 5.975e-7 |

The bridge reduces centered RMS by 571x and centered energy fraction by about 2900x. Decoder residual RMS then grows 4.48→19.31 without restoring relative query variation.

Bridge attention is effectively uniform: normalized entropy `0.999941`; mean maximum probability `0.032565` versus uniform `1/32 = 0.03125`; paired A/B attention cosine `0.999992`.

### Operator identity

- Raw encoder z: paired A/B cosine `0.999996`, relative delta `0.003062`.
- Bridge value projection: relative A/B delta `0.003974`.
- Matched↔swapped bridge output: relative delta `0.003872`.
- Matched↔swapped final output: relative delta `0.001322`.

Identity is already weak in z and is attenuated further by the common decoder trajectory.

### Excluded projection-nullspace mechanism

`[Wk; Wv]` is `1024x384`, numerical rank 384 at relative thresholds from `1e-3` through `1e-6`, condition number `4.01`, and measured z-difference nullspace fraction `2.4e-8`. Mean gain on z differences is `1.76`. The latent projection is well-conditioned and full-column-rank.

### Gradient cancellation

- Individual bridge-output directional gradient L2: mean `2.345e-5`.
- Pairwise off-diagonal cosine: `4.0e-5`.
- Paired A/B cosine: `-6.18e-4`.
- Cancellation ratio `||sum g_i|| / sum ||g_i|| = 0.235778`, essentially `1/sqrt(18) = 0.235702`.
- VJP RMS: bridge output `2.28e-9`, raw z `1.19–1.47e-8`, decoder query input only `1.50e-13`.

The value path still carries a z gradient, but the per-tile directions are mutually unrelated at the shared common state and cancel in shared parameters. The query-routing Jacobian is nearly dead.

### Frozen positional-carrier intervention

The zero-step intervention `q_hidden := bridge_output + decoder_query_pos_emb` changes bridge-input-to-decoder centered RMS `0.004708→0.123964` and centered energy fraction `1.105e-6→7.663e-4`. The carrier persists through layer 7 and raises final-q-norm centered RMS `0.000895→0.006921`.

It does not improve directional loss or z-gradient immediately with already-collapsed frozen downstream weights. Therefore it establishes that the missing carrier can mechanically prevent the symmetry, but it is not evidence that checkpoint surgery repairs the trained model. It should be tested from fresh initialization.

## Code topology

- `decoding_mixin.py:200-214`: V5 deliberately drops the query residual and passes only bridge values to the decoder.
- `decoding_mixin.py:622-637`: `q_pos_emb` is built but not used in the V5 output path.
- `vae_shared.py:1070-1084`: bridge output is only attention-weighted latent values followed by `out_proj`.
- `vae_shared.py:1140-1174`: decoder positions affect Q/K rotation, not an additive value/state carrier.

## Artifacts

- `report.json`: configuration, oracle, rowspace, attention, cancellation, and intervention summary.
- `activation_deltas.csv`: 103-node paired activation trace.
- `directional_vjp.csv`: per-operator directional VJP at 103 nodes.
- `matched_swapped_deltas.csv`: matched↔swapped trace.
- `qpos_intervention_activations.csv`: frozen positional-carrier trace.
- `progress.log`: runtime stages.
