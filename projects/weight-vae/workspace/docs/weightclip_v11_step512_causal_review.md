# V11 exact64 step-512 causal review

Date: 2026-08-27 UTC

## Failure definition

The fresh `four_trunk_complement_v11` exact64 run was required at step 512 to:

- reduce held-out mean normalized complement MSE to at most 80% of its step-0 value;
- make the full prediction's mean directional loss lower than the immutable protected floor;
- retain nontrivial W sensitivity in every private trunk.

The run stopped fail-closed at step 512 because the first two conditions failed. This is a valid scientific NO-GO, not an OOM, NaN, numerical-geometry failure, or an invalid comparison.

## Bound run and evidence

Run root:

`/mnt/shared/weightclip_benchmark/ae_v11_exact64_four_trunk_complement_v1_1984step`

Primary evidence:

| Artifact | SHA-256 |
|---|---|
| `operator_set_metrics.jsonl` | `884d80874b2e319b23488ea30623cd3dc393322c0cdbc20163c296a48f11e5ab` |
| `v11_exact_b18_step1_gradients_v1.json` | `57a7998d34f8713de475934bd975fad6cb56faca0c7f201c266ad1e67d12e4f0` |
| `checkpoints/.../stage_1/grad_layer_rms.csv` | `1d7b3b76680c6a130ae73dd4d9a6bded8dc33168f3becd65a2021819cff3aede` |
| `runs/.../logs/train_rank0.log` | `3ce1320172654b718f0a26bad2cfddbc9b693d02f4d970ecccf4298d4c190b63` |
| `runs/.../crashes/fatal/train_rank_0_pid3826111_1787799756124.json` | `265d97762b3747763d4e58319ef790af05370ee22660942ce96e5f947de835ff` |

There is intentionally no model checkpoint: the policy allowed only a final step-1984 weights-only checkpoint, and the scientific gate stopped the run at step 512.

Postmortem root:

`/mnt/shared/weightclip_benchmark/ae_v11_postmortem_bundle_v3_20260827T043130Z`

| Postmortem artifact | SHA-256 |
|---|---|
| `report.json` | `823020a441c5b293fecb33bc70c9f7106216ebe75e46720c0c507e027deefc0c` |
| `COMPLETE.json` | `06e6842f6c1336bf71eb6ad20fb060e2691da4609b9ff672e989465d755a7142` |
| `analysis/gradient_panel.json` | `1198e322f77ed07380501300c6f75459fc663e3ee8a36feb514c0cd841b6f738` |
| `analysis/gradient_closure.json` | `e01e5bafcc1d8e4bbcda79fa6aeadaba4af11d4f3655f557fd9b8b7177287281` |
| `analysis/ridge_report.json` | `b5e54ae2b4b56d6c7b6ae332508554f43f38a15f5d40f47abea42e1c4cf3a145` |

The postmortem deterministically replayed the failed run to post-update step 256: all six archived signature values had absolute delta zero and the committed cursor was 4608. Model, 474-entry Adam state, scheduler, scaler, and Python/NumPy/CPU/CUDA RNG were restored. Analysis used `torch.autograd.grad` only; before/after digests over all 1,259 parameters and buffers were identical (`3f648d50...`), there were zero optimizer steps, and the 4.75-GB `/dev/shm` snapshot was deleted before `COMPLETE.json` was written. All 24 BF16 component-closure checks passed.

## Validity checks

- Frozen source received two independent launch GOs with no P0/P1 findings.
- The selection and schedule remained `6fb8ad70...` and `7826db6a...`.
- Step 0 passed exact basis, mask, BF16-feed, native/manual replay, row-space, zero-W, and zero-z gates.
- Step 1 passed before the first optimizer update: 293/293 strategic groups and all 474 trainable tensors (235,073,152 parameters) had finite, nonzero gradients; unused/nonfinite/whole-zero counts were all zero.
- The four serialized trunk-code gradients were distinct: maximum absolute cross-trunk cosine was 0.11591, excluding V10's exact identical-gradient topology.
- Every persisted V11 metric was finite. Routing near-cap fractions stayed zero and all 40 blocks remained represented in gradient telemetry.

## Results

| Step | Matched direction | Fixed floor direction | Complement normalized MSE | Residual/floor RMS | Mean cross-trunk linear CKA |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.22958355 | 0.22843819 | 1.00360910 | 0.03663482 | 0.48828 |
| 256 | 0.22881456 | 0.22843819 | 1.00116138 | 0.02219338 | 0.99355 |
| 512 | 0.22844907 | 0.22843819 | 1.00002610 | 0.00689350 | 0.99011 |

At step 512, complement MSE was 0.99643 of the step-0 value rather than at most 0.8. The full output was still worse than the fixed floor by 1.09e-5. The residual response was suppressed by about 81%, and the causal za-shuffle effect remained microscopic (minimum direction delta -3.98e-6).

The encoder path was not dead. At step 512, the four trunk code RMS values were about 0.0158-0.0179 and their fixed-X W-pair relative deltas were 0.213-0.233. The protected zc shuffle minimum direction gap remained 0.761.

Gradient telemetry supports active retreat: the residual head retained the largest gradient while the upstream bridge and trunk gradients became much smaller. The model found the short optimization path of shrinking the decoder-visible residual back toward the already-good floor.

## Competing mechanisms and unique predictions

### H1: local structural/complement gradient conflict or underweighting

The globally correct complement improves both objectives, but the two local gradients around the random residual may oppose each other, and the complement term has weight 0.1.

Unique predictions:

- same-batch `g_struct` and `g_comp` have negative cosine or a small resultant in residual-path groups;
- a gradient-scale-matched complement-only fork escapes the floor retreat while the baseline does not.

Result: excluded as a primary mechanism at the sampled step-256 state. Across eight disjoint B18 batches, structural and complement gradients were strongly aligned: the all-trainable cosine was 0.99804, and major path groups were 0.998-0.9995. With the configured 0.1 coefficient, complement gradient norm was about 0.32 of structural gradient norm and reinforced it. No group passed the conservative cancellation-evidence gate after accounting for measured BF16 separate-backward nonadditivity. A larger complement coefficient remains an untested operational intervention, but antagonistic objectives do not explain this failure.

### H2: decoder/readout Jacobian bottleneck

The trunks retain W-specific information, but the mandatory bridge/decoder does not turn za into the target complement and instead learns to attenuate its output.

Unique predictions:

- frozen za has useful held-out linear predictability for the target complement;
- optimizing free za through the frozen decoder succeeds only weakly, while jointly adapting za and decoder succeeds;
- normalized za-to-residual response is much smaller than latent perturbation size.

Result: the downstream-ignore half is supported, but the claim that the decoder ignored an already useful linear code is not. The causal za intervention was microscopic and backward sensitivity was concentrated in the residual head, yet the operator-held-out ridge probe found no linearly generalizable target-complement signal in za. A nonlinear code may still exist, so a nonlinear frozen-code probe is the missing discriminator.

### H3: informationally redundant trunks

Removing the exact V10 sum nullspace did not force specialization. Sample-structure CKA rose from 0.488 to about 0.99.

Unique predictions:

- an all-four held-out readout is no better than the best single-trunk readout;
- leave-one-trunk-out readouts lose little predictive performance;
- shuffled-code controls fail while single/all-trunk readouts remain similar.

Result: useful informational redundancy is not established. On an operator-grouped 48/16 split, test normalized MSE was 1.00025-1.00041 for individual trunks, 1.00032 for all four concatenated, and 1.00030-1.00036 for leave-one-trunk-out fits; all selected the maximum ridge penalty. Full concat was no better than the best single trunk, and no leave-out caused a material loss. However, every learned readout had negative R-squared and even a matched Gaussian control (1.00003) beat them. The supported statement is therefore that none of the trunks demonstrates held-out linear complement information or a unique linear contribution—not that they redundantly encode useful information. Nonlinear information remains untested.

### H4: fixed za capacity or complement target is intrinsically unreachable

The 2,048-dimensional adaptive bottleneck or production decoder may be unable to represent the required 6,144-dimensional per-tile complement well enough.

Unique prediction:

- even jointly optimized per-tile free za and decoder cannot substantially reduce held-out complement MSE.

Status: viable but currently unsupported.

### H5: inter-operator batch-gradient conflict

Different operator pairs may request incompatible residual updates, leaving a small full-batch resultant.

Unique predictions:

- complement-gradient cosines across disjoint B18 batches are low or negative and the aggregate resultant is small;
- single-batch or per-operator optimization improves while the shared full stream does not.

Result: excluded as a primary mechanism at step 256. Authoritative direct production gradients across eight disjoint B18 batches had all-trainable pairwise cosines 0.741-0.948 and resultant-over-sum-norm 0.936. Bridge and trunk resultants were 0.945-0.971; even serialized-code resultants were 0.697-0.759. There is batch heterogeneity, but not the negative or near-zero resultant required for a cancellation explanation.

## Excluded mechanisms

- **Invalid geometry or masking:** exact FP32/BF16, orthogonality, row-space, mask, and replay gates passed.
- **Routing saturation:** near-bound fractions remained zero and routing supports/argmax telemetry stayed live.
- **Dead-at-initialization branch:** every strategic group and every trainable tensor had a finite nonzero step-1 gradient and representable Adam update.
- **V10's exact common-code gradient symmetry:** private concatenated slices had distinct gradients at step 1.
- **Loss of all W identity:** every trunk retained substantial fixed-X W-pair sensitivity.

## Completed discriminator

The bounded V3 postmortem completed the shared deterministic replay, eight-B18 direct-gradient panel, and operator-grouped ridge panel. Direct `(structural + 0.1 * complement) / 3` gradients were authoritative; separately rounded BF16 component gradients were used only for attribution, with their closure defect measured explicitly.

The evidence excludes objective antagonism and severe inter-batch cancellation, and it shows that step-256 za does not linearly predict the held-out target complement. Gradient energy is also highly downstream-concentrated: residual-head norm was 2.886 versus decoder 0.403, bridge 0.0222, and individual trunks 0.0108-0.0156. These are raw objective-gradient norms, not Adam update magnitudes.

The originally proposed optimization forks were deliberately omitted to keep the final diagnostic bounded. Consequently, fixed bottleneck capacity, nonlinear code sufficiency, direction of causality between poor trunk codes and head retreat, and the effect of changing the 0.1 complement coefficient remain unresolved.

## Narrow conclusion

V11 removed the exact V10 output-sum degeneracy and made all four private branches gradient-live, but that additional formal capacity was not sufficient. The narrow supported proximal mechanism is a floor-dominated shallow-readout shortcut coupled to a W-responsive but target-poor adaptive representation: the residual head learns to attenuate the production residual, while za has no demonstrated held-out linear information about `P_perp(W)`. Classical negative structural/complement conflict, routing saturation, dead gradients, and shared-parameter batch cancellation are excluded as primary causes. High cross-trunk CKA is real, but the experiment does not establish redundancy among useful features because none of the learned codes passes the linear target-information test.

The direction of causality is not fully identified: weak target alignment in the trunks may drive head retreat, or the easy floor/head path may starve the trunks of useful credit. A frozen-step256 nonlinear probe paired with a direct production-path `za -> complement` auxiliary intervention is the smallest remaining discriminator. A free-code plus decoder oracle is still required before blaming the fixed 2,048-dimensional capacity.
