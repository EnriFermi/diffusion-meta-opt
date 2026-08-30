# Source-only latent-code factorial design

Date: 2026-08-16

Implementation: `experiments/source_latent_code_factorial.py`

Frozen post-run analyzer: `experiments/analyze_source_latent_code_factorial.py`

Parent evidence is reused verbatim from
`artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2`.
The launcher accepts a parent argument only to fail clearly on the wrong path:
the parent realpath and SHA256 of every required artifact are exact constants.
The exact deterministic-AE checkpoint, transitive evaluator/model source files,
held-out source-bank realpath, 72 ViT-B matrices, disjoint Flickr30k activation
panels A/B, source role-by-depth templates, and two tilings are all sealed before
the first model forward. There is no target-domain option, import, or data path.

The launcher remains audit-only unless `--execute` is supplied. Audit-only mode
does more than metadata inspection: on CPU it loads all 72 held-out records,
checks the A/B and model contracts, reconstructs both locked tilings, and writes
the complete intervention manifest. It does not load or execute the Weight-AE.

## Exact intervention point

For this canonical deterministic AE, the relevant path is

```text
latent_slots = encoder(W_tile, C_enc)
z_dec = latent_norm(flatten(latent_slots))       # [tiles, 512]
W_hat_tile = decoder(z_dec, C_dec_patch, position)
```

The checkpoint/evaluator contract fixes deterministic AE mode, latent decoder
KV, no query hint, no direct encoder-token decoder path, and the z shortcut off.
The runtime preflight checks attention and FFN shapes at batch sizes one and two.
It compares the latent-slot path with explicit `z_dec`, actual encoder tokens
with `None` and zero tokens, and default decoding with explicitly disabled
shortcut decoding. These comparisons must be bit-exact before the factorial.

Zeroing raw latent slots would leave the learned `latent_norm` bias. The zero
arm therefore replaces `z_dec` itself with exact zeros. Decoder biases may
remain, but their output is then independent of W.

## Locked 2 × 4 × 4 factorial

There are 32 formal cells per matrix and tiling:

| Axis | Levels | Meaning |
|---|---|---|
| `C_enc` | `cell`, `native` | source role-depth mean or held-out A activation context, used only by the encoder |
| W-code | `correct`, `permuted_within_row`, `deranged_block`, `zero` | target tile code; same-matrix misplaced tile code; depth+6 block code; no W-dependent code |
| `C_dec` | `cell`, `native`, `wrong_role_same_depth`, `same_role_wrong_depth` | correct cell C_patch; held-out A C_patch; role-only error; depth-only error |

There are 28 unique decoder calls: 24 nonzero-code calls and four zero-code
calls. The zero calls are copied across the two formal encoder labels, and the
copied prediction hashes must be identical. Across 72 matrices and two tilings,
the exact formal grid contains 4,608 unique rows.

Only `C_patch` enters this decoder. Claims about `C_dec` are therefore about
decoder-side patch context, not replacement of the full distribution encoding.

## Primary and secondary wrong-code interventions

The primary control is `permuted_within_row`. Within every target matrix and
each row-tile group, the encoded z vectors are cyclically shifted by half of the
column-tile count. It is a bijection with no fixed points. Thus it preserves:

- the exact target matrix, block, role, and depth;
- the target `C_enc`, tiling, row group, code multiset, and decoder context;
- code marginal norms and distributions exactly.

It changes only which column tile receives which target-matrix code. A positive
result against this control is the evidence needed for an exact/tile-specific
load-bearing-code claim. A positive result only against another block cannot
support that stronger wording.

For native C, every column tile in one row group receives the same activation
slice `X_context_A[:, row_indices]`. The full run persists a bit-exact runtime
check that `C_var`, `C_patch`, and `C_pooled` are identical across all columns
of every row group, with row-template hashes. This is what makes the native
within-row permutation W-only rather than a hidden C intervention.

`deranged_block` is secondary. It uses W from the same role at depth
`(d + 6) mod 12`, retiled with the target partition and re-encoded under the
target `C_enc`. This probes block-specific W information but changes the source
matrix. Its six paired depth orbits `(d, d+6)` are the bootstrap clusters; it is
not analyzed as 12 independent depths.

For both controls, every target/control code tensor must have unequal SHA256,
finite nonzero norms, finite cosine and relative-distance diagnostics, and a
strictly nonzero per-tile difference. The global permutation manifest also
records target and donor tile identities and proves the no-fixed-point rule.

## Separate decoder-context interventions

The old joint wrong-role-plus-wrong-depth arm is split into two balanced maps:

- `wrong_role_same_depth`: role cycles by +1, depth is fixed;
- `same_role_wrong_depth`: role is fixed, depth cycles by +6.

Every wrong C_patch tensor must have a different hash and positive L2 distance
from its target cell tensor. The launcher freezes a 144-row table with absolute
and relative L2 distance, cosine, and norm diagnostics. This separates role and
depth addressing and makes effect size versus actual template distance visible.

## Exact primary decision rule

The primary metric is raw full-matrix `E_X`; no held-out-fitted gain is used.
For each required row, success means both:

```text
point E_X(comparator) / E_X(correct) >= 1.05
one-sided paired ratio bootstrap L95 > 1.0
```

The primary fixed-C load-bearing screen passes only if that conjunction holds
for:

- both primary controls: within-row permutation and zero;
- both macro-over-role and micro aggregations;
- both locked tilings;
- the cell/cell path.

This is eight jointly required rows. The analogous eight native/native rows are
a robustness tier; passing all 16 is the stronger context-robust tier, not the
sole definition of primary success. Depth+6 is secondary and cannot rescue
failure against within-row permutation.

Directional/operator-structure wording has an additional frozen gate: under
cell/cell, the paired 5th percentile of
`operator_cosine(correct) - operator_cosine(permuted_within_row)` must exceed
zero for macro and micro on both tilings. Its point effect is always reported;
no post-hoc magnitude threshold is invented.

A non-pass is not automatically falsification. If a required row has ratio
`U95 < 1.05`, that endpoint excludes an effect of at least 5%; otherwise its
failure is inconclusive at the frozen precision. Role-local, micro-only,
sub-5%, or one-tiling effects remain exploratory.

The 2×2 correct-code `C_enc × C_dec` cells are also frozen for raw E_X, E_W,
operator cosine, and weight cosine. The interaction estimand is

```text
NN - NC - CN + CC
```

at role, macro, and micro levels. It diagnoses context matching or a possible
context-dependent chart/gauge, but does not replace the paired code rule.

## Mandatory validity and evidence

The full run writes, among other artifacts:

- `factorial_matrix_sufficient_stats.csv`: 4,608 unique formal rows with W/X
  quadratic sufficient statistics, raw errors, cosines, norm ratios, source and
  intervention identities, and prediction hashes;
- `factorial_aggregate_metrics.csv`: all role, macro, and micro aggregates;
- `paired_code_effects.csv`: 768 paired estimates with point ratios and 5th/95th
  bootstrap percentiles; depth+6 rows explicitly name six-orbit clustering;
- `paired_code_cosine_effects.csv`: 768 independently reviewable cosine
  contrasts with the same primary/secondary cluster rules;
- `paired_c_effects.csv`: 320 paired encoder-C, decoder-C, and interaction
  contrasts over raw errors and cosines;
- `latent_correct_donor_summary.csv`: 576 matrix/context/control assertions;
- `latent_correct_donor_tiles.csv`: 165,888 per-tile cosine, norm, and distance
  diagnostics;
- `native_row_condition_invariance.csv`: 144 bit-exact native C invariance rows;
- `code_derangement_manifest.csv`: 82,944 precomputed target/control mappings
  across the two locked tilings;
- exact template, checkpoint, parent, code-dependency, decoder-path, factorial,
  parity, and zero-reuse manifests.

The cell/correct/cell and native/correct/native arms must reproduce all 288
corresponding parent predictions with identical SHA256. Their sufficient
statistics must also satisfy absolute tolerance `1e-9` and relative tolerance
`1e-12`. This is a hard gate, not merely a logged comparison.

The frozen analyzer must match the SHA recorded before the run. It revalidates
the SHA and byte size of every runner CSV/JSON/PT before reading it. It then
independently recomputes all 512 aggregate rows and all primary error, cosine,
and C bootstrap point estimates/quantiles directly from the 4,608 matrix rows,
requiring maximum absolute disagreement at most `1e-12`. It recomputes the
fixed-C primary, native robustness, context-robust, and directional tiers and
writes:

- a correct-code 2×2 C table and interaction plot;
- paired effect ratio plots with both thresholds;
- role-by-depth heatmaps for primary within-row and secondary depth+6 effects;
- latent cosine, norm-ratio, and relative-distance summaries/plots;
- separate wrong-role and wrong-depth effects versus template distance.

Automatic analysis completion still requires a human inspection of the plots,
metrics, and suspicious/heterogeneous cells before any scientific conclusion.

## Discriminating outcomes

- Passing the eight-row primary screen establishes on this source ViT-B panel that the
  decoder uses correctly assigned, tile-specific W information through z; the
  fixed-C reconstruction is not explained by C and decoder priors alone.
- Correct beating zero and depth+6 but not within-row permutation supports only
  coarse matrix/block information, not exact tile addressing.
- Correct approximately matching zero leaves a decoder/template-prior account
  viable. It excludes the frozen >=5% claim only where ratio U95 is below 1.05;
  otherwise the corresponding endpoint is inconclusive.
- A win only for matched C encoder/decoder contexts is consistent with a
  context-dependent representation or chart; it is not evidence for a
  universal W-only code.
- A wrong-role effect without a wrong-depth effect, or vice versa, localizes
  semantic addressing. Effect-versus-distance plots check whether one arm was
  simply a larger numerical perturbation.

A positive source-only result establishes only a causal W/tile-specific z
contribution on this panel. It does not establish good absolute reconstruction
(the existing fixed-C raw macro error is already poor), cross-domain transfer,
or a global model-weight manifold. It validates a mechanism needed before a
target-domain transfer experiment; it is not itself transfer evidence.

## Runtime planning

At batch 64 on one H100, plan approximately 45–70 minutes for the full source
run plus several minutes for serialization and frozen analysis. The design has
two tilings, 72 matrices, target/depth+6 encodes under cell/native C, and 28
unique decoder arms per matrix. Use an 80-minute safe envelope and retain the
nightly window for rerun/forensics if a validity gate fails. This is an estimate,
not measured timing for the final implementation.

## Commands

Fresh CPU audit-only preflight:

```bash
/home/coder/.conda/envs/diff-meta-opt312/bin/python \
  experiments/source_latent_code_factorial.py \
  --output-dir artifacts/crossmodal_united_structure/source_latent_code_factorial_audit_frozen_20260816
```

Full source-only GPU run (requires explicit authorization):

```bash
/home/coder/.conda/envs/diff-meta-opt312/bin/python \
  experiments/source_latent_code_factorial.py \
  --execute \
  --device cuda:0 \
  --batch-size 64 \
  --output-dir artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816
```

Frozen post-run analysis:

```bash
/home/coder/.conda/envs/diff-meta-opt312/bin/python \
  experiments/analyze_source_latent_code_factorial.py \
  --run-dir artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816
```
