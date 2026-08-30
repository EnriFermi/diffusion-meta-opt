# Frozen global-context latent-geometry ablation

Date frozen: 2026-08-16 UTC

Status: **FROZEN DESIGN, AMENDED PRE-FORWARD; NO WEIGHT-AE FORWARD AUTHORIZED BY THIS NOTE**.

Pre-forward amendments (2026-08-16 UTC):

1. The original frozen version, SHA-256
   `cc97f6c4db35e840d713eec3ef7444c929edd0ea9a7bb3c7a25585ea3a94d302`,
   included role and depth in the common-tiling RNG identity. That shared
   role/depth-dependent nonlinear preprocessing could itself inject aligned
   label structure across panels. Before runner completion, contract creation,
   or any learned/untrained Weight-AE forward, the identity was changed to
   depend on matrix shape only.
2. The resulting version, SHA-256
   `94fb796baebbc0b1b47ca9eb81d3752712d563982c2882082ed8c0bd2bd823ae`,
   left a literal ambiguity between the primary-path metadata prohibition and
   the predeclared role/depth-conditioned `learned_cell` positive reference.
   The dataflow lock below now states the sole, segregated exception explicitly;
   `learned_cell` remains ineligible for the primary no-injection claim and for
   both decision levels.
3. The resulting version, SHA-256
   `44e06c5e7408715314677a4b94a8411e0a5974a69382e8fd0bea64e47534f819`,
   listed `zero_weight_sanity.json` without defining its computation or
   authority. The W-only, no-model-forward algebraic check and its exact pass
   rule are now frozen below, closing any path to an extra ad-hoc forward or
   outcome gate.

Seeds, populations, representations, metrics, thresholds, artifact grids, and
counts are unchanged. These amendments remove validity/specification confounds;
none was informed by an experimental outcome.

This note freezes the scientific protocol before implementation and before any
new learned or untrained Weight-AE forward. Execution remains blocked until a
runner, analyzer, external preexecution contract, exact implementation hashes,
and an independent pre-forward review exist. Outcome inspection may not change
the population, features, probes, seeds, nulls, thresholds, or plots below.

## Claim boundary

The Level A claim is deliberately narrow:

> Conditional on three fixed ViT-style vision checkpoints and one fixed,
> source-activation-derived chart condition, the observed role/depth latent
> organization does not require sample-specific activation or label injection.

Level A is not a population-generalization claim. Leave-one-checkpoint-out
diagnostics, depth-block resampling, and QAP permutations characterize these
three fixed checkpoints only. They do not create an IID sample of tasks,
models, datasets, depths, or tilings.

Level B asks the VAE-specific question: does the learned encoder organize the
fixed checkpoints better than predeclared raw-weight and same-architecture
untrained controls? It is also conditional on the fixed checkpoints.

Neither level establishes cross-domain structure. All three cores are
ViT-style vision encoders. Dataset, task, model, and checkpoint identity remain
perfectly confounded within each panel.

## Competing hypotheses

1. **H1: sample-specific activation/label injection explains the existing
   structure.** Replacing all role/depth conditions by one global condition
   destroys cross-checkpoint role/depth correspondence.
2. **H2: fixed-chart weight organization.** With one identical global
   condition, learned codes remain repeatable across tilings and show matching
   role/depth organization across the three fixed checkpoints.
3. **H3: cheap raw statistics or tiling/shape artifacts.** Global learned codes
   look organized, but a simple raw descriptor or fixed CountSketch matches
   them; equal-shape attention-role prediction is weak.
4. **H4: architecture prior rather than learned representation.** A freshly
   initialized, same-architecture encoder matches the learned encoder.
5. **H5: the generic source condition selects a useful chart.** Learned
   global-C succeeds while learned zero-C weakens. Zero-C is an OOD secondary
   diagnostic; its failure does not falsify H2.
6. **H6: checkpoint-specific deformation.** Panel separation is visually
   strong, but matched role/depth relations and fixed cross-checkpoint probes
   fail.

The protocol tests H1-H6 in one run by sharing the exact weights, tilings,
features, and evaluation code.

## Immutable inputs

The implementation must bind and rehash these files before the first model
forward:

| Input | Path | SHA-256 |
|---|---|---|
| Deterministic learned AE | `artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt` | `d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00` |
| Source condition templates | `artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/source_condition_templates.pt` | `8fc6c61bb6baae4e7b1d618133ec651a91386d90c66f540182faa7dfb1655f99` |
| Source held-out manifest | `artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/heldout_panel_manifest.json` | `10020aeabfcce6fe2df3003a7a626f21735cd0763c664e178f27ebcbf42ff985` |
| Source cell-code seed 1 | `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/factorial_code_cache_seed_26081601.pt` | `a585c2ecfe4d1e560c79dca19518543dbdfb5f897bfa413b3ef5148f01995c8f` |
| Source cell-code seed 2 | `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/factorial_code_cache_seed_26081602.pt` | `c56f1ed3ccd0038f3f78ec767facd32cc0bba490f2e31229ec5d7b18a211bfd6` |
| Beans panel | `artifacts/crossmodal_united_structure/prospective_geometry_matched_panels_20260816/panels/beans/panel.pt` | `a47ddd182f09a56fc42d0fe887d0f2717b1972d46afbd1c92c80be13bbc9e6b6` |
| TrOCR/SROIE panel | `artifacts/crossmodal_united_structure/prospective_geometry_matched_panels_20260816/panels/trocr_sroie/panel.pt` | `ded6e7a84c1b34f1fe2d7fd1bdf6ccf041df23c1e20a710197815e0bf99f91a9` |
| Candidate tiling indices | `artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_20260816/tiling_indices.pt` | `a404812555e756e1b4fd380b99bc523530f2eade4a3ab0ddb974dfefba7ec25c` |
| Candidate cell-C codes | `artifacts/crossmodal_united_structure/prospective_geometry_matched_replication_20260816/correct_latent_codes.pt` | `b2b677f8c445c0f129dcd3b9819fdbfd65eb4659af7fecb35f89f4705a1e4daa` |

The exact global condition is `templates["global"]`:

- `c_var` tensor SHA-256:
  `3681bfd5033ade56c65f6d090ce41dea33866295580b1c85d679c6b3ae06303c`;
- `c_patch` tensor SHA-256:
  `b8755c89a668b4bd844f4bb53511951e5cd85b56d91bd537f28c5655df0829e4`.

The source offline dataset realpath must equal
`/home/coder/project/projects/weight-vae/workspace/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset`
and pass the existing source-parent audit. No text/audio target path may be
opened.

## Fresh names

These paths were absent when this design was frozen:

- external future contract:
  `docs/notes/global_context_latent_geometry_ablation_preexecution_contract_20260816.json`;
- runner output:
  `artifacts/crossmodal_united_structure/global_context_latent_geometry_ablation_run_20260816`;
- analyzer output:
  `artifacts/crossmodal_united_structure/global_context_latent_geometry_ablation_analysis_20260816`.

All three must be fresh at creation. The runner and analyzer outputs must be
disjoint, must not contain symlinks, and may not overwrite or amend the formal
prospective replication artifacts.

## Fixed population

Panels, in canonical order:

1. `source_vit_b_flickr`;
2. `beans`;
3. `trocr_sroie`.

Every panel contributes exactly 12 depths x 6 roles = 72 matrices. Frozen role
order is:

1. `attn_query`;
2. `attn_key`;
3. `attn_value`;
4. `attn_output`;
5. `ffn_up`;
6. `ffn_down`.

The four attention roles all have shape `768x768`; `ffn_up` is `768x3072`;
`ffn_down` is `3072x768`. All weights are pipeline-oriented FP32 `[d_in,d_out]`.

The old source seeds `26081601/26081602` and candidate seeds
`26081801/26081802` occupy disjoint tiling namespaces and are **not used for a
decision feature**. Their caches are provenance/audit references only.

Prospectively construct two new common tilings for every panel with seeds
`26081901` and `26081902`. Use one panel-, role-, and depth-independent identity
for each matrix shape:

`global_context_common_tiling_v2|shape=<d_in>x<d_out>`.

For each identity and seed, initialize a CPU `torch.Generator` with
`stable_seed(seed, identity)`, using the repository's pipe-joined SHA-256 rule.
Under bound PyTorch `2.10.0+cu128`, partition the `d_in/16` input
patch IDs with `torch.randperm` into groups of four, expand each group to its
four contiguous 16-variable patches, sort the resulting 64 row indices, and
independently partition all output indices with `torch.randperm` into sorted
groups of 64. For a fixed `(seed,shape)`, partitions must be bit-identical
across every panel, role, and depth cell with that shape. Thus all four
`768x768` attention roles at all 12 depths receive the same partition for a
given seed; `ffn_up` and `ffn_down` similarly share within their respective
shapes. Panel/checkpoint, role, and depth identity must not enter the generator.
Shape remains inherent to the matrix and is explicitly challenged by the
equal-shape attention-role probe and the predeclared raw/simple controls.

The two common seeds are technical repeats, not statistical replicates. Within
a panel and tiling, require exact coverage once and bit-exact split and
reassembly. Each panel/tiling has 20,736 tiles. Thus one representation has:

- 3 panels x 2 tilings x 72 matrix entries = 432 aggregate entries;
- 3 panels x 2 tilings x 20,736 = 124,416 tile codes.

## Dataflow lock

Before reading source data, install the existing source-only seal: offline
Hugging Face/Transformers modes, blocked socket connects and subprocess
launches, forbidden target path/module markers, and zero target/network/
subprocess events.

Input materialization happens before either model is constructed:

1. Verify each full input file and internal weight tensor manifest.
2. Recover the exact source records through the audited source manifest.
3. Copy only `(panel_id, depth, role, W, tiling indices, weight hashes)` into a
   W-only structure.
4. Destroy panel payloads and all `X_context`, `X_score`, and A/B activation
   references; run garbage collection.
5. Only then import/build the Weight-AE and expose the W-only structure to the
   encoding loop.

The encoding function may accept only `(model, W_tiles, fixed_template)`.
Panel, role, depth, dataset, activation, and label values may not be arguments
to a model method. On every primary `learned_global`, `learned_zero`, and
untrained-global path, panel/role/depth may be used outside the model only to
select a frozen W tensor, retrieve its already shape-only tiling, and write
metadata; they may not select a condition or otherwise alter the model call.
The sole exception is the segregated, non-gating `learned_cell` positive
reference, where `(role,depth)` selects the immutable source
`cell_mean[(role,depth)]` template. That exception must have distinct call
records and is ineligible for the primary global dataflow claim and for both
Level A and Level B decisions.

For learned global-C and all untrained-global seeds, every batch must receive
the same two tensor hashes above for both encoder condition components. For the
paired `learned_cell` reference, select only the 72 immutable source
`cell_mean[(role,depth)]` conditions and record their hashes. For learned
zero-C, both 256-vectors are exact FP32 zeros. No distribution encoder is run.
No decoder forward, reconstruction metric, operator score, role gain, or panel
activation is part of this experiment.

Store a static call-graph audit and runtime call records. For the primary
global paths, the runtime audit must show one global `c_var` hash, one global
`c_patch` hash, zero activation consumers, and zero metadata-label consumers.
Cell-C calls are segregated and ineligible for that primary dataflow claim.
Reading an activation-containing container during W extraction does not make
the result activation-conditioned; allowing any activation value to survive
into the model stage does.

## Model paths and representations

Global seed for Python, NumPy, learned-model Torch, and CUDA setup is
`26081931`. Deterministic latent sampling is off.

The learned model contract is fixed to checkpoint step 480,000, strict state
load, raw-integer 2-D RoPE, `patch_size=16`, `flat_lat_dim=512`, no encoder-mu
head, `disable_z_shortcut=True`, eval mode, CUDA:0, BF16 autocast, FP32 stored
codes, and no decoder call. Re-run the existing bit-exact known-source numeric
preflight before new codes.

Exactly these twelve representations are evaluated:

| ID | Definition | New AE forward? | Feature dimension |
|---|---|---:|---:|
| `learned_cell` | Learned AE, source cell-mean role/depth C, re-encoded on both new common tilings | yes | 2,560 |
| `learned_global` | Learned AE, one fixed global C | yes | 2,560 |
| `learned_zero` | Learned AE, exact zero C; secondary/non-gating | yes | 2,560 |
| `untrained_global_26081971` | Same architecture, fresh random parameters, global C | yes | 2,560 |
| `untrained_global_26081972` | Same architecture, fresh random parameters, global C | yes | 2,560 |
| `untrained_global_26081973` | Same architecture, fresh random parameters, global C | yes | 2,560 |
| `raw_simple` | Fixed 37-D full-matrix descriptor | no | 37 |
| `countsketch_26081941` | Fixed raw-tile CountSketch | no | 2,560 |
| `countsketch_26081942` | Fixed raw-tile CountSketch | no | 2,560 |
| `countsketch_26081943` | Fixed raw-tile CountSketch | no | 2,560 |
| `countsketch_26081944` | Fixed raw-tile CountSketch | no | 2,560 |
| `countsketch_26081945` | Fixed raw-tile CountSketch | no | 2,560 |

Zero-C is included prospectively because it adds only one encode pass and
directly distinguishes a useful fixed source chart from a completely absent
condition. It is secondary and cannot fail Level A or Level B.

The immutable old cell-code caches and old tiling-index cache are audited only.
They do not supply `learned_cell`, probe, RSA, PCA, or decision features. This
run re-encodes learned cell-C under the same two common partitions as every
other representation, making cell/global comparisons exactly paired.

### Same-architecture untrained control

Use the exact learned checkpoint-resolved model config and class, but load no
checkpoint tensor. Independently for seeds `26081971`, `26081972`, and
`26081973`, enter a CPU `torch.random.fork_rng` scope, set that seed,
instantiate the model with its default PyTorch initialization, set deterministic
latent/rope flags to the learned contract, record every parameter name, shape,
dtype, and tensor hash, save the fresh state-dict hash, then move to CUDA eval
mode. The trainable-parameter name/shape grid must exactly match the learned
model; no checkpoint tensor may be copied. Shared deterministic buffers are
allowed and recorded. All three untrained encoders receive the same global
template, W tiles, batching, AMP, and aggregation as `learned_global`.

Each untrained seed is evaluated independently. The declared untrained endpoint
is the arithmetic mean of the three seed-specific endpoint metrics. Never
select, concatenate, or average latent coordinates across untrained seeds.

## Exact feature definitions

### Latent and CountSketch aggregation

For an entry with `n_tiles x 512` values, cast to FP64 and compute each of the
512 coordinates across tiles:

1. arithmetic mean;
2. population standard deviation (`ddof=0`);
3. q10;
4. q50;
5. q90.

Quantiles use NumPy linear interpolation. Concatenate in the block order above
to obtain exactly 2,560 FP64 features. Codes must be finite, nonzero,
`[n_tiles,512]`, and keyed to exact row/column partition hashes.

### CountSketch raw baseline

Use five fixed seeds: `26081941` through `26081945`. For flattened local tile
coordinate `j in [0,4095]`, form UTF-8 bytes

`global_context_countsketch_v1|<seed>|<j>`

and compute SHA-256 digest `h`. Define:

- bucket `b_j = int.from_bytes(h[0:8], "big") mod 512`;
- sign `s_j = +1` when `(h[8] & 1) == 0`, otherwise `-1`.

For FP64 flattened tile `w`, CountSketch coordinate `k` is exactly
`sum_{j:b_j=k} s_j*w_j`; apply no additional normalization. Store the 4,096
bucket/sign pairs and their hashes for every seed. Apply the same five maps to
every panel and tiling, then use the exact 2,560-D aggregation above.

Each seed is evaluated independently. The declared CountSketch endpoint is the
arithmetic mean of the five seed-specific endpoint metrics. Do not select the
best seed, average incompatible sketch coordinates before fitting a probe, or
concatenate seeds.

### Simple raw baseline

Compute one FP64 descriptor from each full W matrix, independent of tiling, and
duplicate it for the two technical tiling rows. Its exact 37 features are:

- `log(numel)`, `log(d_in)`, `log(d_out)`;
- scalar `mean`, population `std`, RMS, Frobenius norm, mean absolute value,
  minimum, maximum;
- full-weight q01, q05, q10, q25, q50, q75, q90, q95, q99;
- nine summaries of the `d_in` row RMS values: mean, population std, minimum,
  q10, q25, q50, q75, q90, maximum;
- the same nine summaries of the `d_out` column RMS values.

All quantiles use FP64 NumPy linear interpolation. Because this representation
is tiling-invariant, it is ineligible for the tiling-reliability criterion.

### W-only zero-baseline validity sanity

`zero_weight_sanity.json` is a validity artifact only. It must be computed
while the 216 full FP32 CPU W matrices are available and must invoke no learned
or untrained model, encoder, decoder, distribution encoder, condition, or
activation. It cannot enter Level A, Level B, a feature, a plot, or any probe.

The JSON has exactly the top-level keys `schema_version`, `matrix_count`,
`model_forward_calls`, `entries`, and `summary`, with schema version
`global_context_zero_weight_sanity_v1`, matrix count `216`, and model forward
calls `0`. `entries` are in canonical panel, depth, role order and contain
exactly:

- `panel_id`, `depth`, `role`, `d_in`, `d_out`;
- `weight_shape_bytes_sha256`, `weight_tensor_sha256`;
- `numel`, `finite_count`, `nonzero_count`;
- FP64 `sum_squared`, `frobenius_norm`, and `rms`;
- `literal_zero_relative_squared_error`.

For each matrix, cast W to FP64, let
`denominator = sum(W**2, dtype=float64)`, and independently compute
`numerator = sum((zeros_like(W)-W)**2, dtype=float64)`. The reported relative
error is `numerator/denominator`. `summary` has exactly the keys
`all_finite`, `all_have_nonzero_entry`, `all_positive_sum_squared`,
`all_literal_zero_ratios_exactly_one`, and `pass`. The artifact passes iff the
canonical 216-entry grid is exact, every `finite_count == numel`, every
`nonzero_count > 0`, every denominator is finite and strictly positive, every
reported ratio is exactly FP64 `1.0`, `model_forward_calls == 0`, and all four
component summary booleans and `pass` are true.

## Technical-repeat handling

Tiling repeatability is measured before averaging. For every other diagnostic,
average the two 2,560-D (or 37-D) feature vectors elementwise within each exact
`(representation,panel,role,depth)` cell. Probes and RSA therefore see 72 cells
per checkpoint, not 144 pseudo-replicates. No bootstrap resamples tiling index.

Cell/global and global/zero tile-level comparisons are paired only within the
same panel, matrix, and exact tiling partition. Tiling index 1 across different
panels is not a paired stochastic draw.

## Scaling, PCA, and fixed probes

No hyperparameter is selected from these three panels.

### Decision-metric scaling

For each representation and each outer held-out checkpoint, fit a population
mean/std (`ddof=0`) feature scaler on the two training checkpoints only, after
tiling averaging. Drop training features with std `<1e-12`; apply the frozen
training scaler/mask to the held-out checkpoint. No held-out feature value may
affect the mask, scaler, PCA, classifier, or regression coefficients.
If a fixed probe fold has insufficient kept-feature count or matrix rank for
its frozen PCA dimension, the fold is invalid; do not reduce PCA dimension
adaptively.

RSA, tiling reliability, and exploratory PCA use source-only preprocessing;
they never fit a scaler on Beans or TrOCR. For `learned_global`, fit one
source-only averaged-cell scaler/mask on its 72 source role/depth cells for RSA,
and one source-only unaveraged scaler/mask on its 144 source tiling rows for
tiling reliability and exploratory PCA. Apply those learned-global transforms
unchanged to all panels. Because latent coordinates align, project
`learned_cell` and `learned_zero` through the same learned-global transforms.

Independent random encoders and CountSketch seeds have incompatible coordinate
bases; each therefore gets its own representation-specific scaler/mask fit on
the exact same 72 averaged or 144 unaveraged source cells, never on a target
panel. `raw_simple` analogously gets its own 37-D source-only scaler. This is
not a best-case transform: every control follows the identical source-only fit
rule, and no transform is shared across incompatible coordinates. RSA and
tiling distances use the full standardized kept-feature space, not PCA.

### Fixed attention-role probe

Use only `attn_query`, `attn_key`, `attn_value`, and `attn_output`; they have
identical shapes and balanced 12-depth support. After the training-only scaler,
fit `sklearn.decomposition.PCA(n_components=16, svd_solver="full",
whiten=False)` on the two training checkpoints. Then fit multinomial L2
logistic regression with:

- solver `lbfgs`;
- `C=1.0`;
- intercept enabled;
- no class weights;
- tolerance `1e-8`;
- maximum iterations `10,000`;
- random state `26081931` where accepted by the bound sklearn version.

Require convergence. Train on 96 cells and evaluate 48 cells in each of the
three fixed outer folds. Primary metric is balanced accuracy; chance is 0.25.
There is no nested tuning.

### Fixed depth probe

Fit a separate model for each of the six roles. After a role-specific scaler
fit on the 24 training cells, fit
`PCA(n_components=8, svd_solver="full", whiten=False)`, then
`Ridge(alpha=1.0, solver="svd", fit_intercept=True)` to target `depth/11`.
Evaluate the 12 held-out depths. Report Spearman rho and MAE per role and the
unweighted macro mean over six roles. Undefined Spearman is recorded as zero
and is a suspicious result. There is no tuning.

### Exploratory PCA

Exploratory plots do not enter either decision. Use the learned-global
source-only unaveraged scaler/mask above, then fit full-solver PCA with 10
components and no whitening on the 144 source `learned_global` tiling rows
only. Apply this frozen source scaler/PCA to Beans and TrOCR global-C rows.
Project `learned_cell` and `learned_zero` through the same source-global
scaler/loadings for the condition overlay. No target panel contributes to PCA
preprocessing.

## Distances, RSA, and resampling

All random-analysis streams use
`stable_seed(base,*labels) = int(first 16 hex digits of
SHA256("|".join(str(x) for x in (base,*labels)))),16) mod (2**63-1)` and NumPy
`Generator(PCG64(seed))`.

Fixed base seeds:

- structured QAP: `26081951`;
- depth-block stability resampling: `26081961`.

Every stream label includes metric, representation, panel or panel pair, and
comparator where applicable. Store every resolved seed.

### Tiling reliability

For a fixed panel, role, and depth `d`, matched distance is Euclidean distance
between repeat 1 and repeat 2. Wrong-depth distance is the mean of
`dist(repeat1[d],repeat2[(d+6) mod 12])` and
`dist(repeat2[d],repeat1[(d+6) mod 12])`. The panel statistic is
`median(matched)/median(wrong-depth)` over all 72 cells.

Generate 10,000 depth-block resamples: sample 12 depth indices with replacement
and include all six roles for every sampled depth. The 5th and 95th percentiles
use NumPy linear interpolation and are named `S05/S95`, not confidence bounds.
They measure sensitivity to the fixed depth composition only.

### Cross-checkpoint RSA

After tiling averaging and the frozen representation-appropriate source-only
scaling defined above, form one 72x72 Euclidean distance matrix per checkpoint
in canonical role/depth order. Observed RSA is Spearman correlation of
corresponding strict upper triangles.

For each of the three panel pairs generate 10,000 structured QAP mappings of
the second matrix. Draw a bijection `sigma` of the four equal-shape attention
roles and fix `ffn_up` and `ffn_down`. Independently for each original role
`r`, draw a depth bijection `tau_r` on `0..11`. In canonical cell indexing,
define exactly `pi(index(r,d)) = index(sigma(r),tau_r(d))`. Because `sigma` and
every `tau_r` are bijections, `pi` must contain each integer `0..71` exactly
once; assert this before use. Apply this same `pi` jointly to both rows and
columns: `D_perm = D[pi][:,pi]`. Recompute upper-triangle Spearman. Monte Carlo
one-sided p is
`(1 + count(null_rho >= observed_rho))/(10,001)`.

The three pairwise tests share checkpoints and are dependent. Report all three;
do not combine p-values or call them independent replications.

### Probe stability summaries

For each held-out checkpoint, generate 10,000 depth-block resamples using the
same sampled 12 depths across all roles and all representations. For the
attention probe, include all four attention-role predictions for each sampled
depth. For depth, include all six role trajectories. Recompute balanced
accuracy or macro Spearman. A degenerate bootstrap Spearman is zero. Report
`S05/S95` as fixed-depth stability summaries, not population intervals.

For Level B attention-role deltas, independently resample the 12 depths within
each fixed checkpoint, use the identical draw across representations, compute
checkpoint-macro balanced accuracy, and subtract each predeclared comparator.
Use 10,000 draws and report delta `S05/S95`. On every draw, form the untrained
comparator by averaging its three seed-specific resampled metrics and the
CountSketch comparator by averaging its five seed-specific resampled metrics;
never select a seed. Also store the three fixed per-checkpoint point deltas for
each comparator before computing the checkpoint macro.

## Variance decomposition

For each of `learned_cell`, `learned_global`, and `learned_zero`, report
balanced sums-of-squares fractions in:

- full standardized 2,560-D space;
- the first 10 exploratory global-C PCs;
- exploratory PC1/PC2.

The four components are shared role-by-depth cell, checkpoint main effect,
checkpoint-by-cell interaction, and tiling residual. This is descriptive and
non-gating.

## Frozen decision rules

### Level A: fixed-chart, no sample-specific activation-label injection

Level A passes only if every validity check and every A1-A4 cell passes for
`learned_global`:

1. **A1, technical-repeat reliability:** point tiling ratio `<0.75` and `S95<1`
   separately for all three checkpoints.
2. **A2, matched cross-checkpoint geometry:** observed RSA `>=0.20` and
   structured-QAP `p<=0.01` separately for all three dependent checkpoint
   pairs.
3. **A3, equal-shape role correspondence:** fixed attention-role balanced
   accuracy `>=0.40` and depth-block `S05>0.25` separately for every held-out
   checkpoint.
4. **A4, ordered depth correspondence:** macro six-role depth Spearman
   `>=0.30` and depth-block `S05>0` separately for every held-out checkpoint.

A pass supports only the narrow claim at the top of this note. A failure must
identify the exact failed cells; it does not license threshold or probe tuning.

### Level B: learned Weight-AE organization beyond frozen controls

Define checkpoint-macro attention balanced accuracy as the unweighted mean of
the three fixed outer-fold accuracies. Define three comparators in advance:

1. untrained-global endpoint mean across seeds `26081971/26081972/26081973`;
2. `raw_simple`;
3. CountSketch endpoint mean across all five fixed seeds.

Do not select a winning raw baseline. Level B passes only if:

1. Level A passes;
2. for each comparator and each of the three held-out checkpoints separately,
   learned-global minus comparator attention accuracy has point delta `>=0`;
3. for each comparator, learned-global minus comparator checkpoint-macro
   attention accuracy has point delta `>=0.05` and paired depth-block `S05>0`;
4. for each comparator and each held-out checkpoint separately,
   learned-global depth rho is no more than `0.05` below the comparator;
5. for each comparator and each of the three checkpoint pairs separately,
   learned-global RSA is no more than `0.05` below the comparator.

CountSketch comparisons always use the arithmetic mean of five seed-specific
metrics in the original metric scale, including inside every paired resample.
Untrained comparisons analogously use the arithmetic mean of the three
seed-specific metrics. No best-seed or best-baseline operation is allowed.

`learned_cell` is a paired positive reference, not a Level B competitor.
`learned_zero` is a mechanism diagnostic, not a gate. Report global-versus-cell
and global-versus-zero tile cosine, relative L2, and all geometry endpoints.

## Fixed plots

Write exactly these 11 PNGs and mechanically and visually inspect all of them:

1. `global_pc1_pc2_by_panel.png`;
2. `global_pc1_pc2_by_role.png`;
3. `global_pc1_pc2_by_depth.png`;
4. `global_role_facets_by_panel.png`;
5. `global_depth_facets_by_panel.png`;
6. `cell_global_zero_projected_pc1_pc2.png`;
7. `tiling_reliability_by_panel_and_representation.png`;
8. `rsa_by_panel_pair_and_representation.png`;
9. `attention_role_balanced_accuracy.png`;
10. `depth_spearman_by_panel_and_representation.png`;
11. `level_b_attention_paired_deltas.png`.

Panel/dataset/model colorings are not duplicated because they encode the same
three identities. Plots must show both tiling repeats where applicable,
predeclared thresholds on decision plots, readable legends, and no clipped or
nonfinite points.

## Expected runner artifacts and counts

The raw runner manifest is self-excluded and must declare exactly 21 files
(22 files including `artifact_manifest.json`):

1. `run.log`;
2. `resolved_config.json`;
3. `preexecution_contract.json`;
4. `preexecution_binding.json`;
5. `input_audit.json`;
6. `target_access_seal.json`;
7. `source_template_audit.json`;
8. `model_contract.json`;
9. `known_source_numeric_preflight.json`;
10. `dataflow_audit.json`;
11. `tiling_manifest.csv`;
12. `tiling_indices.pt`;
13. `code_manifest.csv`;
14. `latent_codes.pt`;
15. `raw_simple_features.csv`;
16. `countsketch_maps.json`;
17. `aggregate_feature_manifest.csv`;
18. `aggregate_features.npz`;
19. `zero_weight_sanity.json`;
20. `input_immutability_recheck.json`;
21. `runner_metadata.json`.

Required raw counts:

- `tiling_manifest.csv`: 432 rows;
- `tiling_indices.pt`: 432 exact row/column partition entries, with identical
  partition hashes across all panel/role/depth cells for every common
  `(seed,shape)`;
- new model/condition code entries: 6 x 432 = 2,592;
- new stored codes: 6 x 124,416 = 746,496 FP32 rows of width 512;
- `raw_simple_features.csv`: 432 rows x 37 feature columns plus keys;
- CountSketch maps: 5 x 4,096 = 20,480 bucket/sign entries;
- aggregate feature entries: 12 x 432 = 5,184 rows;
- aggregate arrays: eleven `432x2560` arrays and one `432x37` array.

The six new code representations are learned-cell, learned-global,
learned-zero, and three seed-specific untrained-global controls. Old
learned-cell caches remain immutable audit/reference inputs only.
Runner logging must follow repository visibility rules. Flush and close the log
before writing the manifest; do not mutate a manifested file afterward.

## Expected analyzer artifacts and counts

The analyzer manifest is self-excluded and must declare exactly 36 files
(37 including `artifact_manifest.json`): 25 top-level files plus the 11 fixed
plots.

Top-level files:

1. `analysis.log`;
2. `resolved_analysis_config.json`;
3. `input_manifest_audit.json`;
4. `feature_audit.json`;
5. `latent_pair_effects.csv`;
6. `tiling_reliability.csv`;
7. `tiling_bootstrap.csv`;
8. `rsa.csv`;
9. `rsa_qap_permutations.csv`;
10. `role_probe_predictions.csv`;
11. `role_probe_metrics.csv`;
12. `role_bootstrap.csv`;
13. `depth_probe_predictions.csv`;
14. `depth_probe_metrics.csv`;
15. `depth_bootstrap.csv`;
16. `level_b_role_deltas.csv`;
17. `level_b_role_bootstrap.csv`;
18. `level_b_secondary_noninferiority.csv`;
19. `variance_decomposition.csv`;
20. `pca_scores.csv`;
21. `pca_scaler_and_loadings.npz`;
22. `decision_cells.csv`;
23. `decision.json`;
24. `plot_audit.json`;
25. `README.md`.

Required analysis counts:

- `latent_pair_effects.csv`: 864 entry-level rows (cell/global and
  global/zero, 432 each);
- raw tiling-reliability endpoints: 11 eligible representations x 3 panels =
  33 rows, plus 3 CountSketch seed-mean and 3 untrained seed-mean rows;
  `raw_simple` contributes 3 separate ineligible/invariant rows, for 42 rows
  total in `tiling_reliability.csv`;
- Level A tiling resamples: 30,000 rows;
- raw observed RSA: 12 x 3 = 36 rows, plus 3 CountSketch seed-mean and 3
  untrained seed-mean rows, for 42 rows total;
- global-C QAP null: 3 x 10,000 = 30,000 rows;
- role predictions: 12 x 3 x 48 = 1,728 rows;
- raw role metrics: 12 x 3 = 36 rows, plus 3 CountSketch seed-mean and 3
  untrained seed-mean rows, for 42 rows total;
- Level A role resamples: 30,000 rows;
- Level B role-delta resamples: 3 x 10,000 = 30,000 rows;
- depth predictions: 12 x 3 x 6 x 12 = 2,592 rows;
- raw per-role depth metrics: 12 x 3 x 6 = 216 rows; add 36 raw panel-macro
  rows, 36 CountSketch/untrained seed-mean per-role rows, and 6 corresponding
  seed-mean panel macros, for 294 rows total in `depth_probe_metrics.csv`;
- Level A depth resamples: 30,000 rows;
- `level_b_role_deltas.csv`: 9 per-checkpoint point rows plus 3
  checkpoint-macro rows = 12 rows;
- `level_b_secondary_noninferiority.csv`: 9 checkpoint-specific depth rows
  plus 9 checkpoint-pair-specific RSA rows = 18 rows;
- variance fractions: 3 aligned learned latent representations x 3 spaces x
  4 components = 36 rows;
- exploratory PCA scores: 432 native global rows plus 864 projected
  cell/zero rows = 1,296 rows;
- decision cells: 12 Level A cells (3 each for A1, A3, A4 and 3 pairwise A2),
  31 Level B cells (Level A prerequisite, 9 per-checkpoint role deltas, 3
  role macro/stability cells, 9 per-checkpoint depth cells, and 9 pairwise RSA
  cells), plus exactly 8 validity summary cells, for 51 rows total in
  `decision_cells.csv`.

The eight validity cells are fixed as: immutable input/source seal; W grid and
hashes; common-tiling grid/same-shape identity across panels, roles, and depths;
learned-model contract and known-source preflight; three-untrained-model
initialization/fingerprint grid; primary global dataflow/template constancy;
code/feature/artifact grids; and analyzer isolation/probe convergence/plot
audit.

All intervals and p-values must be recomputed from stored prediction/null rows.
The analyzer must not import or execute either Weight-AE.

## Execution and post-run review gate

Before execution:

1. implement runner and analyzer without changing this note;
2. run compile/lint and corruption fixtures;
3. create the external contract binding this note SHA, all input hashes,
   dependency closure, runtime versions, script hashes, expected grids/counts,
   seeds, and fresh paths;
4. obtain independent review of dataflow, random-control fairness, CountSketch
   construction, train-only scaling, QAP row/column action, thresholds, and
   artifact grids;
5. only an explicit implementation-level GO authorizes the first AE forward.

Estimated execution budget is under 30 minutes of H100 model encoding for six
new code representations plus input audits and raw baselines; analysis should
remain comfortably within the overnight budget. The 746,496 FP32 512-D code
rows occupy about 1.53 GB before container overhead; expected total new runner
storage is below about 2 GB.

After execution, inspect metrics, logs, every plot, exact input/output
manifests, nonfinite values, missing/duplicate rows, convergence, accidental
held-out scaling, CountSketch seed spread, technical-tiling disagreement, and
random-model fingerprints. Report Level A and Level B independently. If a
threshold fails, preserve the result; any tuned follow-up is a new exploratory
experiment and cannot amend this one.
