# Cross-domain Weight-AE: decision log

Date: 2026-08-16 UTC

**Latest status:** the three-panel global-context latent-geometry run was
executed after its source fixed-conditioning gate had failed. It is invalid as
evidence for or against cross-domain transfer. New agents must read the final
"Latest canonical status" section before using any result from that branch.

**Current proposed central hypothesis (2026-08-17):** the reviewer-approved
global downstream-salience hypothesis is recorded in
`docs/notes/weight_ae_global_salience_hypothesis_20260817.md`. It is a proposed,
unestablished causal hypothesis and does not rehabilitate the invalid
global-context run. The broad "reusable structure" wording below is retained as
historical framing, not the current precise claim.

This is the persistent decision record for the current paper-facing experiment.
It records what is established, what has been ruled out, and which conclusions
are still prohibited. It is not a substitute for the artifact-level reports.

## Paper hypothesis under investigation

The ambitious hypothesis is that a Weight-AE trained on source vision
transformers learns reusable structure of trained linear operators that can be
detected under a controlled modality/domain shift. The intended finding is not
merely that a VAE can ingest weights, nor that nearest-neighbour retrieval works.
The paper-facing target is evidence that trained operators share structure
across tasks/domains and that the learned weight representation exposes that
structure.

The present source gates are necessary mechanism checks. They cannot establish
cross-domain transfer, a global manifold, interpretability, or useful weight
reconstruction by themselves.

## Canonical model and immutable semantics

- Checkpoint: deterministic AE stage 1, step 480,000.
- Path: `artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt`.
- SHA256: `d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00`.
- The later KL/VAE fine-tune is not used.
- Checkpoint-era 2-D RoPE coordinates are raw integer coordinates.
- Latent sampling is disabled; the decoder entry code is the post-`latent_norm`
  512-dimensional `z_dec`.

The original apparent contradiction with Comet is resolved. A newer loader
silently used `normalized_center` RoPE for a checkpoint whose serialized config
predated that field. On the exact historical source and a deterministic
reconstruction of the checkpoint-era CUDA slicing contract, changing only the
fallback changes behavioral direction loss from `0.999528` to `0.641098`,
closely matching the stored Comet-era value `0.642187`. The ephemeral April
sliced bytes were not persisted, so this must not be described as bit-exact
replay of the April batch.

Evidence:

- `artifacts/crossmodal_united_structure/historical_exact_replay_20260816/README.md`
- `artifacts/crossmodal_united_structure/historical_batch_provenance_audit_20260816/README.md`
- `artifacts/crossmodal_united_structure/step480_rope_contract_audit_20260816/README.md`

The first pre-fix smoke numbers are quarantined and must not be cited as AE
quality evidence.

## Metric contract

Training/Comet `behavioral_operator` is an absolute, input-width-normalized RMS
quantity. It is not numerically comparable to the full-matrix relative operator
error `E_X = ||X(W_hat-W)||^2 / ||XW||^2`, whose exact zero-weight baseline is
one. Behavioral direction and scale are separate components of the training
objective.

Paper claims must keep these levels separate:

1. causal use of a W-dependent latent code;
2. directional/operator-structure signal;
3. absolute reconstruction relative to meaningful baselines;
4. cross-domain transfer;
5. semantic latent geometry/interpretability.

Evidence for an earlier level does not imply a later one.

## Corrected source-only confirmatory result

The locked source panel contains 72 held-out ViT-B matrices: 12 depths by six
roles, with disjoint context-A and score-B activation records and two exact
tilings.

- G0 validity passes.
- The old preregistered G1 as a whole fails and cannot be retroactively passed.
- Fixed role-by-depth mean C has raw macro `E_X` about `1.062/1.070`, but raw
  micro `E_X` about `0.744/0.757`.
- A source-only fitted scalar gain gives macro `E_X` about `0.870/0.873`, but
  this does not repair weight error and cannot be fitted on target data.
- Native per-instance C is a partial positive control, not a faithful weight
  codec: operator reconstruction is better, while weight error remains above
  the zero-weight baseline.
- Role heterogeneity is large. In particular, FFN-up dominates micro operator
  energy, while late FFN-down layers are weak.

Evidence:

- `artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2`
- `artifacts/crossmodal_united_structure/source_confirmatory_postrun_diagnosis_20260816`
- `artifacts/crossmodal_united_structure/source_confirmatory_g1_mechanism_review_20260816/README.md`

Narrow surviving observation: fixed source-derived C retains weak but
reproducible activation-independent operator signal. This is not yet evidence
of domain transfer or globally good reconstruction.

## Closed branch: sparse estimation of fixed C

The conditioning-density experiment increased selected activation records from
1,608 to 11,028 and mean Kish-effective record support from about 20.3 to 90.6.
It compared two-dataset/all-dataset breadth and K=1/4/8 records per
source-by-dataset pair on the exact locked source panel.

Result:

- `all_k8` does not improve primary raw macro `E_X` over locked `d2_k1` on both
  tilings;
- raw micro `E_X` is slightly but significantly worse on both tilings;
- template movement saturates and is small (joint mean relative L2 about
  `0.0093` from `d2_k1` to `all_k8`);
- the sparse reference is reproduced bit-exactly.

Decision: archive the narrow hypothesis that the old G1 failure was caused by
sparse estimation of the hierarchical mean C and would be rescued by this
dense estimator family. Do not generalize this negative result to all fixed-C
estimators, cross-domain transfer, or the Weight-AE/operator hypothesis.

Evidence:

- `artifacts/crossmodal_united_structure/source_condition_density_gate_20260816`
- `artifacts/crossmodal_united_structure/source_condition_density_postrun_review_20260816/README.md`
- `artifacts/crossmodal_united_structure/source_condition_density_postrun_hostile_review_20260816/README.md`

## Historical preregistration: is the learned latent code load-bearing?

Before opening any target domain, the source-only factorial intervenes at
`z_dec = latent_norm(latent_slots.flatten(1))`.

The decisive comparator cyclically permutes `z_dec` among column tiles of the
same matrix and same row group, with no fixed points. It preserves the code
multiset, role, depth, tiling, row-specific activation context, and local
decoder coordinates. The decoder receives no global tile identity. Therefore
correct versus permuted tests whether the assignment of W-derived code to the
specific weight tile matters. Exact zero `z_dec` is a complementary, but
out-of-distribution, decoder/C-prior control. A depth-plus-six donor is
secondary.

Preregistered claim hierarchy:

1. **Primary fixed-C load-bearing gate:** under cell/cell context, correct code
   must beat both within-row permutation and zero code by a comparator/correct
   raw `E_X` ratio of at least `1.05`, with paired ratio `L95 > 1`, for macro and
   micro aggregation on both tilings (8/8 rows).
2. **Native-context robustness:** the analogous 8/8 native/native rows. This is
   not required to establish the fixed-C result.
3. **Context-robust strong screen:** all 16 rows pass.
4. **Directional/operator tier:** under fixed C, correct code must improve
   operator cosine over the within-row permutation with paired `L95(delta)>0`
   for macro and micro on both tilings. This is required before using
   directional/operator-structure wording.
5. **Absolute quality remains separate:** even a full factorial pass does not
   establish a competent codec because the known fixed-C raw macro `E_X` is
   above the zero-weight baseline.

Exact positive wording allowed by tiers 1 and 4:

> On held-out source ViT-B matrices, replacing instance-specific activation
> context with a frozen source-derived role-by-depth template does not remove
> the causal contribution of the encoded weights. Correct tile-to-latent
> alignment consistently outperforms both a matched-marginal within-row latent
> derangement and a zero latent, and improves operator direction. The output is
> therefore not explained by activation context and decoder priors alone; the
> latent transmits tile-specific, W-dependent information relevant to the
> reconstructed operator.

Use “without instance- or target-dataset-specific activations,” not
“activation-free”: fixed C is still derived from source activations.

If a pass criterion is missed, distinguish an excluded material effect from an
inconclusive result. A `ratio U95 < 1.05` excludes the preregistered 5% effect;
a wide interval crossing the threshold is inconclusive. Role-only,
micro-only, one-tiling, or sub-5% effects are exploratory.

Design and implementation:

- `docs/notes/source_latent_code_factorial_design_20260816.md`
- `experiments/source_latent_code_factorial.py`
- `experiments/analyze_source_latent_code_factorial.py`

## Completed latent-code factorial: mixed but mechanistically informative

The frozen source-only factorial completed after the section above was
preregistered.  All validity checks pass: the formal grid has 4,608 unique
rows and 28 unique decoder calls per matrix, all 288 parent predictions and
sufficient statistics reproduce exactly, native within-row conditioning is
bit-identical, all correct/control latent pairs differ, prediction collisions
are absent, and the frozen analyzer independently reproduces all aggregates
and bootstrap quantiles to at most `8.88e-16` absolute disagreement.

The exact preregistered result is mixed:

- primary fixed-C tier: `6/8`, **not passed**;
- native/native robustness tier: `8/8`, passed;
- fixed-C directional/operator-cosine tier: `4/4`, passed;
- context-robust all-16 tier: not passed;
- frozen status for the fixed-C >=5% composite: `INCONCLUSIVE_FOR_5_PERCENT_EFFECT`,
  because neither failed endpoint has ratio `U95 < 1.05`.

The two fixed-C failures are both macro-over-role comparisons against zero
`z_dec`.  Their zero/correct point ratios are `0.9485` and `0.9415`, so the
point estimate is about 5--6% worse than the zero-latent decoder prior.  The
micro comparisons against zero pass strongly (`1.3146` and `1.2937`) because
operator energy is highly role-imbalanced.  This does not permit the exact
positive paragraph preregistered above.

At the same time, the matched-marginal, same-matrix within-row intervention is
strongly positive under fixed C on both tilings:

- macro permutation/correct raw-E_X ratios: `1.3414`, `1.3443`;
- micro ratios: `1.1644`, `1.1631`;
- macro correct-minus-permuted operator-cosine effects: `0.2617`, `0.2644`;
- micro cosine effects: `0.1325`, `0.1355`;
- every one of the four directional lower bootstrap bounds is positive.

Thus the fixed-C latent is not empty or explained only by C and a decoder
prior: the assignment of a W-derived code to its actual column tile carries
reproducible directional information.  The stricter claim that this code is
uniformly beneficial over zero latent in raw role-balanced reconstruction is
not established.

The correct-code `Cenc x Cdec` diagnostic also shows a large, reproducible
matching interaction.  For raw E_X, `NN - NC - CN + CC` is about
`-0.263/-0.260` macro and `-0.234/-0.231` micro; for operator cosine it is
about `+0.193/+0.194` macro and `+0.198/+0.196` micro.  All corresponding
paired intervals exclude zero, and the sign repeats in every role.  The
leading mechanism is therefore a context-dependent or gauge-like code in
which encoder and decoder conditioning must match, not a demonstrated
standalone W-only manifold.  This remains a leading interpretation until a
discriminator distinguishes chart matching from other nonlinear context
interactions.

Evidence:

- `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816`
- `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/postrun_analysis/README.md`
- `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/postrun_analysis/mechanism_criterion_rows.csv`
- `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/postrun_analysis/directional_operator_criterion_rows.csv`
- `artifacts/crossmodal_united_structure/source_latent_code_factorial_20260816/paired_c_effects.csv`

Text/audio targets remain sealed while independent hostile review and the
source-only next-discriminator decision are completed.  A later prospective
target protocol may use the permutation/directional finding as a distinct
hypothesis, but it must be frozen explicitly and must not relabel the failed
8/8 fixed-C composite as passed.

## Post-hoc source-only scale-mechanism diagnostic

The raw preregistered fixed-C verdict remains `6/8` and non-passing.  A frozen,
source-only post-hoc diagnostic tested whether the two raw macro failures were
caused materially by role-dependent radial/scale miscalibration rather than an
empty latent code.  Six role gains were fitted on a 72-matrix source panel
disjoint from the 72-matrix held-out ViT-B/Flickr panel, then applied unchanged
to correct, within-row-permuted, and `zero_code` predictions.  No target asset
was accessed.

With the source-frozen role gains, all eight fixed-context comparisons pass the
same 5% point-effect and paired-bootstrap criteria:

- within-row permutation/correct ratios are `1.1920/1.1377` macro/micro on
  tiling 1 and `1.1936/1.1358` on tiling 2; their ratio `L05` values range from
  `1.1204` to `1.1854`;
- `zero_code`/correct ratios are `1.1463/1.3918` macro/micro on tiling 1 and
  `1.1432/1.3719` on tiling 2; their ratio `L05` values range from `1.1302` to
  `1.2805`;
- correct code improves operator cosine over both controls in all eight
  directional rows, with every paired lower bound above zero;
- the calibrated-versus-raw correct-minus-`zero_code` difference-in-differences
  has `U95 < 0` for macro and micro on both tilings;
- the correct role-to-gain assignment ranks first among all 720 role
  permutations for macro and micro on both tilings (`p=1/720` each);
- source role-matched gains beat a common source-median shrink and even
  descriptive heldout-fitted per-role oracle controls.

The intervention reduces the raw radial penalties in V, attention-output, and
FFN-down from roughly `0.26--0.47` to `0.001--0.005`.  This supports a narrow
mechanism: the source latent contains reproducible operator-direction signal,
while role-dependent radial miscalibration materially caused the raw
correct-versus-`zero_code` macro failures on this panel.  It does not show that
scale is the only limitation.  In particular, calibrated fixed-context weight
error remains poor (`E_W` about `1.013` macro and `1.059` micro), so this is not
evidence for a high-fidelity standalone weight codec.

An independent recomputation reproduced the formal statistics within
`7.3e-16`.  It also found a non-decision schema defect in formal
`aggregate_metrics.csv` and unreadable title overlap in five formal PNGs.  The
decision reads explicit `E_X` fields and is unaffected.  Corrected
publication-facing tables and all six visually inspected figures are stored in
the suffixed `_v2` directory; the sealed formal directory was not mutated.

This entire section is post-hoc evidence from one source panel.  It cannot
establish cross-domain transfer and cannot authorize target access.  A
prospective source replication on an untouched, geometry-matched trained model
is required next; two tilings of the same panel are not independent model or
domain replications.

Evidence:

- `docs/notes/source_role_gain_mechanism_diagnostic_design_20260816.md`
- `artifacts/crossmodal_united_structure/source_role_gain_mechanism_20260816/README.md`
- `artifacts/crossmodal_united_structure/source_role_gain_mechanism_independent_audit_20260816/README.md`
- `artifacts/crossmodal_united_structure/source_role_gain_mechanism_publication_plots_20260816_v2/ERRATA.md`

## Target seal and next decision

Text/audio target artifacts remain sealed. No cross-domain metric has been used
to design or revise these source gates.

- The preregistered raw fixed-C tier did not pass; it must remain reported as
  `6/8`.
- The post-hoc source diagnostic supports radial/scale miscalibration as a
  material explanation and yields an `8/8` calibrated result on the same
  held-out panel, but cannot turn that discovery into prospective evidence.
- Before target unsealing, freeze and run at least one untouched,
  geometry-matched source-model replication with a task-quality gate.  The
  conservative paper-facing standard is two independent source panels before
  treating the mechanism as sufficiently stable for a confirmatory target
  test.
- Do not inspect target outcomes to choose gains, metrics, activation sampling,
  baselines, or pass criteria.

Any later target result must separately test transfer against zero, simple
weight/statistical baselines, source-native and fixed-C controls, role/depth
heterogeneity, and absolute as well as directional metrics. Latent plots by
domain, dataset, model family, layer role, and depth are diagnostics; they do
not substitute for the transfer endpoint.

## Frozen prospective geometry-matched source replication

The next source-only protocol is frozen in
`docs/notes/prospective_geometry_matched_source_replication_design_20260816.md`
(SHA-256
`fc23d52d07d0004afd681116578abdba562dff1a415b059fd72ec261b1e47d70`).
It holds the 12-block ViT-style encoder geometry, six matrix roles, matrix
shapes, activation-row counts, tilings, WAE checkpoint, source-fitted role
gains, controls, statistics, bootstrap units, and decision rule fixed.  This
isolates replication across independently trained tasks/checkpoints from layer
shape and scale extrapolation; it is not yet a cross-architecture-family or
cross-domain result.

The two panels are:

- Beans classification:
  `nateraw/vit-base-beans@41f85ace09a4613c2c65495b3b8465c4ceee1d00`;
- printed OCR:
  `microsoft/trocr-base-printed@93450be3f1ed40a930690d951ef3932687cc1892`
  on the frozen SROIE no-padding crop protocol.

Both checkpoints have 72 compatible core matrices and zero exact
`shape || bytes` matches against the 471 compatible AE-training weights.
Their task-quality gates are evaluated before any WAE outcome is produced.
The exact outcome precedence is `INVALID > FULL > DIRECTIONAL_OR_MIXED >
FAIL`.  `FULL` requires, on every split and tiling, not only the frozen-gain
correct/control and directional criteria but also absolute macro and micro
operator error below the literal-zero baseline with a one-sided upper 95%
bound below one.  Two `FULL` panels authorize freezing a later target protocol;
they do not authorize unsealing or establish text/audio transfer by themselves.

An initially scouted Food-101 checkpoint was invalidated before WAE access:
on the authoritative frozen 512-example validation subset it achieved
top-1 `41/512 = 0.080078125`, top-5 `45/512 = 0.087890625`, and predicted only
7 of 101 classes.  The one-sided Clopper--Pearson top-1 upper bounds are
`0.1026493` (95%) and `0.1123400` (99%), far below the preregistered `0.80`
gate.  Beans was the predeclared backup and was promoted solely on this
pre-WAE quality failure.  This exclusion supplies no evidence about the WAE.

Current execution status is **NO-GO** until the panel builder, WAE runner, and
independent CPU analyzer pass an exact-SHA hostile preflight.  No prospective
candidate WAE outcome has been computed at the time of this entry, and the
text/audio target remains sealed.

Evidence:

- `docs/notes/prospective_geometry_matched_source_replication_design_20260816.md`
- `artifacts/crossmodal_united_structure/prospective_source_panel_scout_20260816`
- `artifacts/crossmodal_united_structure/prospective_panel_feasibility_scout_20260816_v3`

## Latest canonical status: forensic correction of the global-context run

This section supersedes the earlier execution-status statements for the
three-panel global-context latent-geometry branch. The source-only factorial
and role-gain results above retain their stated narrow status; this correction
does not invalidate them.

### Failure definition

The completed global-context experiment cannot identify domain transfer and
cannot refute the Weight-AE hypothesis. This is not merely a mismatch between
the paper claim and the chosen estimand. The experiment itself violated its
causal ladder and used confounded panels, a different representation object,
an inadequately estimated conditioning stub, and a probe whose result is
sensitive to an arbitrary dimensional bottleneck.

No new model or GPU experiment was launched after this forensic finding.

### Fatal protocol violation: the instrument failed before target execution

The source decision file was written at 04:13 UTC and records:

- `G0_pass_both_tilings=true` for the native-context aggregate check;
- `G1_primary_cell_mean_pass_both_tilings=false`;
- the literal action
  `DO_NOT_RUN_G2_OR_UNSEAL_TARGET; DIAGNOSE_RECORDED_SOURCE_FAILURE`.

For the primary fixed cell-mean condition, raw macro operator error was
`1.0618/1.0700` on the two tilings against the exact zero-weight baseline of
one. The stronger non-primary global-C condition was also above the raw macro
zero baseline (`1.0676/1.0775`). Nevertheless, the target/global geometry run
was executed and `learned_global` was made decision-eligible.

The later "known source numeric preflight" did not rescue the failed positive
control. It bit-exactly replayed one cached 512-D encoder code, with
`decoder_calls=0`, no reconstruction/operator metric, and a cell-context code
rather than validation of the global-C chart. It established code-path parity,
not that fixed conditioning was a working in-domain measurement instrument.

The resulting causal table is incomplete:

| panel | native context | fixed/global context |
|---|---|---|
| source | partial aggregate positive control | failed |
| target | not measured as the paired control | measured |

Therefore a poor target/fixed result cannot distinguish failure of the
conditioning chart from failure of domain transfer.

Primary evidence:

- `artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/decisions.json`
- `artifacts/crossmodal_united_structure/global_context_latent_geometry_ablation_run_20260816/known_source_numeric_preflight.json`
- `artifacts/crossmodal_united_structure/global_context_forensic_execution_audit_20260816/README.md`

### Confounded panel construction: three labels but two weight lineages

The exact raw-weight audit found:

- source versus Beans median cosine `0.999736134` and median relative L2
  `0.022971360`;
- source versus TrOCR median cosine `0.000170819` and median relative L2
  `1.171844273`.

Source and Beans are therefore near-duplicate checkpoint trajectories, while
TrOCR comes from a materially different pretrained lineage and parameter
gauge. In the held-out-TrOCR fold the classifier is trained on two near-copies
of one lineage, not two independent domains. The repeated role/depth rows are
pseudo-replication and do not create independent model populations. Without
multiple lineages per domain, same-start controls, or an explicit valid
alignment intervention, domain is confounded with initialization, pretraining
lineage, and parameter gauge.

The labels also overstate dataset provenance. `source_vit_b_flickr` is the
generic `google/vit-base-patch16-224` checkpoint evaluated on Flickr30k, not a
checkpoint trained on Flickr30k. `trocr_sroie` is
`microsoft/trocr-base-printed` evaluated on SROIE. Under global C, neither
Flickr30k nor SROIE activations enter the primary encoder dataflow.

Evidence:

- `artifacts/crossmodal_united_structure/global_context_weight_lineage_posthoc_20260816/README.md`
- `artifacts/crossmodal_united_structure/global_context_forensic_execution_audit_20260816/feature_pair_similarity.csv`

### The requested all-dataset activation mean was not implemented

The source bank has 1,114 source weights annotated with at least three
datasets. Template construction used 948 allow-listed core units, selected at
most two activation datasets per unit, and selected exactly one deterministic
record per chosen source-by-dataset pair. Of the 948 units, 228 had three or
four available datasets and were truncated; 804 contributed two selected
datasets and 144 only one.

The selected conditions were then pooled over patch and within-patch positions,
interpolated over depth, and averaged over sources, roles, depths, and model
families. One resulting 256-vector was broadcast to every one of the `4 x 16`
input positions. This is a narrow, heavily pooled surrogate, not the requested
empirical average over all available dataset-conditioned activation tensors in
their native shape. Only one such estimator was initially used, with no
record-count or all-dataset sensitivity before target execution.

The later source-only conditioning-density experiment is a separate valid
diagnostic. It showed that the tested denser hierarchical-mean family did not
rescue raw source macro error, but it cannot retroactively validate the target
run or exclude other fixed-conditioning constructions.

Evidence:

- `artifacts/crossmodal_united_structure/global_context_forensic_execution_audit_20260816/source_template_sampling_summary.csv`
- `artifacts/crossmodal_united_structure/source_condition_density_postrun_review_20260816/README.md`

### The analyzed object was changed after the Weight-AE

The original learned object is a 512-D code for an individual training-like
`64 x 64` weight tile. The global-context run exhaustively Cartesian-tiled each
full matrix into 144 attention or 576 FFN tiles and then collapsed all tile
codes coordinatewise into mean, standard deviation, q10, q50, and q90. This
created an untrained 2,560-D full-matrix descriptor that discards tile identity,
spatial correspondence, and global row/column position.

Arbitrary tiling accounts for `0.1585414704` of the total standardized
learned-global sum of squares, but only `0.0003008431` in the displayed PC1/PC2
plane. The plots therefore hide most of the measured tiling instability. The
raw-simple comparator is also not information-matched: it receives whole-matrix
shape, norm/quantile, and row/column RMS statistics after the AE descriptor has
discarded layout.

Consequently the run evaluates the geometry of an ad-hoc
`tiles -> coordinate moments` pooler, not directly the learned tile-level
Weight-AE geometry.

Evidence:

- `artifacts/crossmodal_united_structure/global_context_latent_geometry_ablation_analysis_20260816/variance_decomposition.csv`
- `projects/weight-vae/workspace/experiments/run_global_context_latent_geometry_ablation.py`
  (descriptor construction around lines 2191--2244)

### Why the multinomial linear probe was used, and why it is not justified as
### a primary endpoint

The analyzer standardizes the descriptor, fits PCA on the nominal training
panels, retains 16 components, and trains multinomial logistic regression to
predict one of six layer roles on the held-out panel. The intended rationale
was conventional: turn visual clustering into a number, use a low-capacity
readout, avoid a `p >> n` fit for the 2,560-D descriptor, and give learned and
raw representations a common nominal dimension budget.

This rationale supports only the narrow claim that a six-way role label is
linearly accessible after this particular global affine chart and bottleneck.
It does not support reconstruction, operator transfer, or the existence of a
domain-invariant Weight-AE manifold. The deterministic AE was not trained with
a contrastive, role-supervised, metric, or cross-model alignment objective, so
there is no reason to require layer roles to share one linearly aligned chart
across independently trained checkpoints. Reconstruction also permits latent
reparameterizations that can preserve decoding while changing linear
separability.

The fixed PCA16 choice is empirically outcome-determining:

| held-out panel | learned global PCA16 | PCA32 | PCA64 | raw simple PCA16 |
|---|---:|---:|---:|---:|
| source ViT/Flickr | 0.5417 | 0.8958 | 1.0000 | 0.9583 |
| Beans | 0.6042 | 0.8958 | 1.0000 | 0.9375 |
| TrOCR/SROIE | 0.2500 | 0.2917 | 0.2917 | 0.5833 |

PCA16 retains only `0.7362/0.7365/0.7433` of learned-global standardized
training variance across the three folds, versus
`0.99975/0.99975/0.99989` for raw-simple features. Thus much of the registered
raw-versus-learned deficit on source and Beans was manufactured by an
asymmetric bottleneck. TrOCR remains weak in this probe family, but because of
the failed source fixed-C gate and lineage/gauge confound, this does not become
evidence against transfer.

Evidence:

- `artifacts/crossmodal_united_structure/global_context_probe_pca_sensitivity_posthoc_20260816/README.md`
- `artifacts/crossmodal_united_structure/global_context_probe_pca_sensitivity_posthoc_20260816/role_probe_pca_sensitivity.csv`
- `artifacts/crossmodal_united_structure/global_context_probe_pca_sensitivity_posthoc_20260816/pca_variance_retention.csv`

### Validity checks that were passed

The forensic review did not find a wrong-checkpoint, orientation, tiling, or
direct preprocessing-leakage explanation:

- the intended deterministic AE stage-1 checkpoint at step 480,000 was loaded,
  not the later KL fine-tune;
- checkpoint-era raw 2-D RoPE and deterministic latent decoding were restored;
- PyTorch linear weights were transposed consistently to `[d_in, d_out]` and
  exact role shapes were checked;
- matrix split/reassembly and coordinate coverage were bit-exact;
- scalers and PCA were fitted on nominal training panels only;
- the formal PCA16 endpoints were reproduced exactly in the post-hoc audit.

These checks show that the problem is not a simple corrupted tensor or stale
metric. They do not repair the broken causal design.

### Narrow supported conclusion and prohibited interpretations

The only supported interpretation of the global-context run is:

> This sparse fixed-C estimator, ad-hoc tile-moment pooler, arbitrary PCA/probe,
> and effectively two-lineage panel produce the stored numbers and plots.

It is prohibited to use this run to claim any of the following:

- the Weight-AE does or does not transfer across domains;
- globally shared trained-operator structure exists or is absent;
- target-domain information is absent from the latent code;
- raw statistics are intrinsically better than the learned representation;
- role clustering demonstrates domain-invariant interpretability.

The valid earlier source-only matched-marginal intervention still establishes
the narrower result recorded above: correct tile-to-code assignment carries
reproducible source-domain directional/operator information under fixed C. It
does not establish cross-domain transfer.

### Constraints on the next experiment

Do not launch another target run until its exact protocol is reviewed with the
user. At minimum it must:

1. measure the full paired `source/target x native/fixed-C` table using the same
   tiles, metrics, and decoder path;
2. require the fixed-C source positive control to pass before any target result
   is opened; otherwise stop and repair the conditioning instrument;
3. separate domain from checkpoint lineage/gauge using multiple independent
   lineages per domain, a justified same-start construction, or an explicit
   alignment intervention;
4. use tile-level 512-D codes and decoder/operator outcomes as primary evidence;
   any full-matrix pooler, PCA visualization, classifier, or retrieval metric is
   secondary;
5. if probes are retained, report a predeclared capacity/dimension curve with
   matched information access and nested regularization rather than one
   arbitrary PCA dimension.

Consolidated forensic evidence:

- `artifacts/crossmodal_united_structure/global_context_forensic_execution_audit_20260816/README.md`
- `artifacts/crossmodal_united_structure/global_context_probe_pca_sensitivity_posthoc_20260816/README.md`
