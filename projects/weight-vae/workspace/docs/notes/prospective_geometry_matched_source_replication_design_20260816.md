# Prospective geometry-matched source replication

Date: 2026-08-16 UTC

Status: **FROZEN DESIGN-LEVEL GO; EXECUTION REMAINS NO-GO UNTIL EXACT-SHA CODE
PREFLIGHT; NO CANDIDATE WEIGHT-AE OUTCOME HAS BEEN COMPUTED**.

The final executable lock is created only after the independent reviewer signs
off on the implementation hashes.  Quality-only checks may invalidate a panel
before that lock, but no quality outcome may change Weight-AE metrics,
interventions, gains, or scientific thresholds.

This experiment prospectively tests whether the source-only latent-code and
role-scale mechanism discovered on the held-out ViT-B/Flickr panel replicates
on new, trained checkpoints.  It is deliberately run before opening any
data2vec text/audio target artifact.  A successful result is a stability gate
for a later cross-domain protocol, not cross-domain evidence itself.

## Hypotheses and competing explanations

The primary hypothesis is that, under a frozen source-derived role-by-depth
activation template, the correct Weight-AE latent retains tile-specific
operator information on unseen trained weights.  The source-frozen role gains
should correct a transferable radial bias without needing activations from the
replication dataset.

Competing explanations and their distinguishing predictions are:

1. **Reusable weight/operator code plus transferable radial calibration.**
   Correct tile-to-code alignment beats matched-marginal code permutation and
   decoder `zero_code`; operator cosine improves; frozen role gains beat a
   common shrink and retain the correct role mapping.
2. **Decoder/C prior dominance.** Correct, permuted, and `zero_code` predictions
   are similar after applying exactly the same gains.
3. **Source-panel overfit of the six gains.** Directional code effects may
   remain, but source role gains fail against common shrink, the identity role
   mapping does not rank highly, or the zero-code differential rescue fails.
4. **Energy-heavy-role artifact.** Micro endpoints pass while macro or several
   individual roles fail.
5. **Bad/OOD downstream checkpoint-panel pair.** The model fails its task
   quality gate before any Weight-AE outcome is computed.  Such a panel is
   invalid rather than negative evidence about the Weight-AE.

## Immutable Weight-AE inputs

- Deterministic AE checkpoint: step 480,000,
  `artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt`,
  SHA-256
  `d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00`.
- Checkpoint-era 2-D RoPE coordinates are raw integers; latent sampling is off.
- Source condition templates:
  `artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/source_condition_templates.pt`,
  SHA-256
  `8fc6c61bb6baae4e7b1d618133ec651a91386d90c66f540182faa7dfb1655f99`.
- The only Weight-AE context in the primary experiment is the frozen
  `cell_mean[(role, canonical_depth)]` template, used for both encoding and
  decoding.  No replication-panel activation is passed to the Weight-AE or its
  distribution encoder.
- Source-frozen zero-intercept role gains:
  `artifacts/crossmodal_united_structure/source_confirmatory_gate_20260816_clean2/source_fit_role_gains.csv`,
  SHA-256
  `e05f2339ba33eb65feee00df2f095655c9319ecf118cd81d7159f6e14f70c4f3`:

  | Role | Gain |
  |---|---:|
  | `attn_query` | 0.5270182885626962 |
  | `attn_key` | 0.5562907139098529 |
  | `attn_value` | 0.3990051254514762 |
  | `attn_output` | 0.3564476277035192 |
  | `ffn_up` | 1.0280206811266708 |
  | `ffn_down` | 0.3048084709039325 |

The later KL-finetuned VAE checkpoint, target-fitted gains, panel-fitted gains,
and native/panel-specific Weight-AE activation context are prohibited.

## Prospective panels and pre-Weight-AE quality amendment

Both executable panels have exactly 12 encoder blocks, hidden width 768,
intermediate width 3,072, separate Q/K/V matrices, and the same six core roles
and matrix shapes as the source held-out panel.  This holds layer size and role
geometry fixed while checkpoint, downstream task, and dataset change.  Both
tested cores remain ViT-style image encoders; this is
not an architecture-family replication.

### Invalidated candidate: Food101 classifier

The original P1 candidate was
`ashaduzzaman/vit-finetuned-food101@57f4382fcd34e48cdb21bb2b5a1d7a5e1c598ed0`
on the locally pinned `ethz/food101` validation cache at
`83488de741c1bd1ce27aa6a2b33e19c7bdf92ca9`.  Its exact core weights were
unseen by the AE, but a deterministic quality-only pilot performed before any
Weight-AE forward produced top-1 `41/512=0.080078125` and top-5
`45/512=0.087890625` on the 512 validation indices with smallest
`SHA256("260816|food101_validation|{global_index}")`, with predictions collapsed
to seven of 101 classes despite exact `id2label`/dataset-label agreement and a
strict load with no missing, unexpected, or mismatched keys.

This is far below the predeclared `0.80` quality threshold.  The pilot's exact
selection, predictions, model loading information, processor parity, and a
one-sided Clopper--Pearson upper bound are sealed at
`artifacts/crossmodal_united_structure/prospective_panel_feasibility_scout_20260816_v3`
(artifact-manifest SHA-256
`30557a4b57d489393627ee44368a81c0944702b505818972c7e6b24b049e75f1`).
The manifest has 26/26 matching files.  For top-1 `k=41,n=512`, independently
recomputed one-sided Clopper--Pearson bounds are `U95=0.1026493` and
`U99=0.1123400`, both far below `0.80`.  Food101 is
`INVALID_PANEL_PRE_WAE`; no
Weight-AE output will be computed for it, and it cannot later be reinstated
based on results from another panel.

### P1: Beans classifier (predeclared backup promoted after Food invalidation)

- Model: `nateraw/vit-base-beans` at revision
  `41f85ace09a4613c2c65495b3b8465c4ceee1d00`.
- Local model directory:
  `projects/shared/storage/data/models/vit_base_beans_nateraw_41f85ace`.
- Model weights: `pytorch_model.bin`, SHA-256
  `fc443b145fcc3a09cf07eb28b94e9a989ec3f0e8f7255e1fa08c0953ed4bae91`.
- Config SHA-256
  `366d2ad9bf48e94932bb83e0d2367e6f95ed18befb1a4ef6903934b7af737237`;
  preprocessor SHA-256
  `af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948`.
- Dataset: `beans` at revision
  `27aa014ce09b193e1a6f58112d4a66e0eddb69c5`, with immutable local
  `train`, `validation`, and `test` Parquet SHA-256 values
  `7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19`,
  `33f774593d8b31585457b70c224744e9409ffdee4e91a11822b1ebfe8242928f`,
  and
  `534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27`.
- The model card reports verified test accuracy `0.9453125`.  The frozen local
  gate evaluates all 128 test examples and requires top-1 `>=0.85`, exact
  agreement of the three dataset class names with `id2label`, 3/3 reference
  class coverage, 3/3 predicted class coverage, and finite logits.
- For activations, rank every train example separately within its reference
  class using canonical JSON with fields
  `{"class_id":int,"dataset_index":int,"namespace":"beans_activation_rank_v1","seed":26081831}`.
  Serialization is UTF-8 `json.dumps(..., sort_keys=True, separators=(",",":"),
  ensure_ascii=True)` and the rank key is the full SHA-256 digest interpreted
  lexicographically, with hash collisions prohibited.  Per class, ranks 0--41
  form A and ranks 42--83 form B: 126 disjoint images per split.  Test quality
  images and train activation images must also be disjoint by both stable ID
  and SHA-256 of decoded contiguous RGB bytes prefixed by height and width.

Promotion of Beans is a quality-only amendment explicitly allowed by the
pre-outcome backup rule.  It was fixed before any Food, Beans, or TrOCR
Weight-AE outcome.  Beans may replace invalid Food; it may not replace an
invalid P2, because doing so would remove the independently required OCR/model-
family replication.

### P2: SROIE OCR

- Model: `microsoft/trocr-base-printed` at revision
  `93450be3f1ed40a930690d951ef3932687cc1892`.
- Local snapshot:
  `projects/shared/storage/data/models/trocr_base_printed/models--microsoft--trocr-base-printed/snapshots/93450be3f1ed40a930690d951ef3932687cc1892`.
- Model weights: `model.safetensors`, 1,333,384,464 bytes, SHA-256
  `1cf4a6eedab26afaaf505f1c7f73d9634944924dbd1ed049d569db98039cd596`.
- Config SHA-256
  `5bda1deab455661feb3d91906656e5600e2ca520d5c00a2a03836614b850c93e`;
  preprocessor SHA-256
  `2fcc0da9466ee00be0403b26027373039e032820ebac409e207b32e52e52119d`.
- Generation config SHA-256
  `41149cdcffec4d657f32dfcddd9b208037f01286c9e07945c724908c58ed0193`;
  tokenizer config
  `5a1356884c6ae736a621841535264ba7c5bebd52f169258add2c48fcbb32d50a`,
  special-token map
  `c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00`,
  vocabulary
  `06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab`,
  and merges
  `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`.
- Dataset: `priyank-m/SROIE_2019_text_recognition` at revision
  `04f6537e418eeb88863d617eb27817cc496522d7`, whose test split contains
  18,704 SROIE text-line crops and ground-truth strings.  Local `test.zip`
  SHA-256 is
  `533dba4d017a70617943f857fe01a986d34da5095255112f88af03df4325484b`.
- Rank all test records by canonical JSON with fields
  `{"file_name":str,"namespace":"sroie_partition_v1","reference_text":str,
  "seed":26081821}`, serialized exactly as for Beans.  Full SHA-256 collisions
  are prohibited.  Ranks 0--1,023 form quality, 1,024--1,151 form A, and
  1,152--1,279 form B.  Require disjoint file names, raw ZIP-member byte hashes,
  and decoded shape-prefixed RGB hashes across all three sets.
- OCR generation is exact local-processor preprocessing followed by
  `model.generate(pixel_values, do_sample=False, num_beams=1,
  max_new_tokens=96, use_cache=False, return_dict_in_generate=False,
  output_scores=False)`.  Frozen IDs are decoder-start `2`, EOS `2`, and pad
  `1`; decoding stops at generated EOS per example or at the 96-new-token hard
  cap.  Every generated token-ID sequence is stored.
  Decode with `processor.batch_decode(..., skip_special_tokens=True,
  clean_up_tokenization_spaces=False)`.  No padding, crop expansion, beam
  search, or outcome-selected preprocessing is allowed.
- Normalize reference and prediction with Unicode NFKC, `casefold`, stripped
  and collapsed Unicode whitespace; punctuation remains.  Empty normalized
  references are a hard validity failure.  Character distance is Levenshtein
  distance over Unicode code points.  Corpus CER is
  `sum(edit_distance)/sum(len(normalized_reference))`; per-example CER is
  `edit_distance/len(normalized_reference)`.  The frozen quality gate requires
  nonempty normalized predictions `>=0.99`, exact match `>=0.50`, corpus CER
  `<=0.25`, and median per-example CER `<=0.20` on the 1,024 records.  Store
  every file name, hashes, raw/normalized reference and prediction, edit count,
  and denominator.
- Only the 12-layer 768/3,072 image encoder participates in the Weight-AE
  panel.  The OCR decoder is used solely for the pre-outcome model-quality
  gate.

There is no further substitute panel in this protocol.  A P1 or P2 Weight-AE
failure is a result, not a reason to search for another checkpoint.  A P2
quality/provenance failure leaves only one valid panel and therefore cannot
authorize a confirmatory target protocol.

## Unseen-weight and panel validity gate

At process start, before reading any panel, source, training-bank, or output
input, install the source-only audit seal.  Set Transformers/Hugging Face
offline modes; block every `socket.connect`; block every subprocess launch;
reject original and symlink-resolved filesystem paths containing `data2vec`;
reject any imported module containing `data2vec`; and reject target-related
path arguments before resolving them.  Formal research inputs are bound to the
exact realpaths and hashes in this document and the preexecution contract.  The
run must record the installed hook, zero target access events, zero network
connections, and zero subprocesses.

Only then, before a panel is decoded:

1. Extract and transpose the 72 exact core `Linear.weight` tensors to pipeline
   orientation `[d_in,d_out]`; require the exact six-role by 12-depth grid and
   shapes `(768,768)`, `(768,3072)`, and `(3072,768)`.
2. Hash `struct.pack("<QQ",d_in,d_out) + b"\0"` followed by contiguous
   little-endian FP32 bytes for every matrix; apply the same hash to every
   compatible-shape weight in the immutable AE stage-1 offline training bank.
   Require zero exact matches.  The scout's earlier byte-only comparison over
   471 already shape-filtered weights remains useful evidence, but the formal
   gate must recompute the stated shape-plus-bytes hash.  Model-name absence
   alone is not enough.
3. Non-gating audit: for every candidate matrix, report its nearest compatible
   AE-bank matrix by FP64 cosine and relative Frobenius distance.  This
   characterizes proximity to the training bank; no threshold is selected from
   it and it cannot rescue or fail a panel.
4. Require all resource hashes, model revisions, source-template/gain hashes,
   and the canonical AE hash above.
5. Run and pass the task-quality gate before loading the Weight-AE.  Record the
   quality artifacts even if it fails.

A quality/provenance failure marks the panel invalid.  It is not scientific
evidence for or against the latent mechanism.

## Activation-score collection

For each panel split, preprocess images with the exact local model processor
and register forward pre-hooks on the six core linears in all 12 encoder
blocks.  For each image retain eight input-token rows: CLS plus seven distinct
patch-token indices ranked by canonical JSON with fields
`{"image_id":str,"namespace":"activation_token_rank_v1","panel":str,
"score_split":str,"token_index":int}`, using the serialization and full-digest
collision rule above.  Candidates are indices 1--196 for Beans and 1--576 for
SROIE; index 0 (CLS) is always prepended.  Beans therefore stores exactly 1,008
rows per matrix per split and SROIE exactly 1,024.  Write and hash complete
sample and token manifests before loading the Weight-AE.  Require unique stable
sample IDs, unique selected token indices within an image, and A/B disjointness
by ID, raw/decoded image hash, and manifest row.  Hooks must validate
batch/image/token alignment, finite nonzero values, input width, and bit-exact
equality of Q/K/V input rows where the architecture implies equality.

These activations are used only to score `XW` versus `XW_hat`.  They are never
used to build C, fit gains, choose a latent, or alter a prediction.  Each
decoded prediction is computed once and reused for A/B scoring.

## Weight-AE interventions

For each of the 72 matrices, build two exhaustive random 64-by-64 tilings with
seeds `26081801` and `26081802`.  Tiling construction follows the locked source
gate: disjoint 64-row groups formed from four 16-variable patches and disjoint
64-column groups.  The matrix identity string is exactly
`{panel}|{checkpoint_sha256}|depth={depth:02d}|role={role}`.  Reuse the source
gate's `stable_seed(tiling_seed,matrix_identity,role,depth)`, whose byte input is
the UTF-8 pipe-join of those four `str(...)` values and whose integer is the
first 16 SHA-256 hex digits modulo `2**63-1`, with a CPU `torch.Generator` under
PyTorch `2.10.0+cu128`.  Persist all row/column index tensors so the result does
not depend on later RNG/library behavior.  Split/reassembly of both matrix
values and unique coordinate IDs must be bit exact with coverage exactly one.

At the deterministic decoder entry
`z_dec = latent_norm(latent_slots.flatten(1))`, evaluate:

- `correct`: the code encoded from the actual tile under frozen cell C;
- `permuted_within_row`: within each matrix and row group, cyclically derange
  codes across column tiles.  Serialize canonical JSON fields
  `{"depth":int,"namespace":"within_row_code_derangement_v1","panel":str,
  "role":str,"row_group":int,"tiling_seed":int}` as above, interpret the
  first eight digest bytes as unsigned big-endian `uint64`, and set
  `offset=1+value%(n_col_groups-1)`.  Target column `j` receives code
  `(j+offset)%n_col_groups`.  This preserves the exact same-row code multiset
  and admits no fixed point;
- `zero_code`: decode an exact all-zero `z_dec` under the same frozen C.

Reassemble every unscaled prediction in FP32 first; scaling is scalar FP64
arithmetic on full-matrix sufficient statistics and, when a concrete scaled
matrix is needed for hashing/plotting, an FP32 multiply after reassembly.  All
three arms receive the same frozen gain for the target role.  Also retain raw
`g=1` sufficient statistics for every arm.  For `correct`, analytically
evaluate the frozen source-median common gain
`0.46301170700708616`, all 720 assignments of the six source gains to roles,
and radial/angular decomposition.  Literal zero output (`E_X=1`) remains
separate from decoder `zero_code`.

Hard intervention checks include non-colliding correct/control prediction
hashes, exact preservation of the permuted code multiset, no permutation fixed
points and a bijection in every row group, exact A/B prediction SHA equality,
finite outputs/statistics, and parity of raw and analytic scaled
sufficient-statistic recomputation.

## Metrics and frozen statistical analysis

For every panel, split, tiling, depth, role, and arm, store FP64 sufficient
statistics `(T,P,D)=(||Y||^2,||Y_hat||^2,<Y,Y_hat>)` for weight space and
operator space.  For a scalar gain `g`, use
`E(g)=(T-2gD+g^2P)/T`, `r=sqrt(P/T)`, `c=D/sqrt(TP)`, and verify
`E(g)=1-c^2+(g*r-c)^2`.  Report:

- relative operator error `E_X=||X(W_hat-W)||^2/||XW||^2`;
- operator cosine;
- relative weight error `E_W`;
- prediction/target radial ratio, radial penalty, and angular floor;
- literal-zero, raw, source-role-gain, and common-shrink baselines.

Compute role-level values, role-balanced macro means, and energy-weighted micro
ratios.  First sum sufficient statistics over the 12 blocks within a role and
derive its radial/angular quantities.  Macro `E_X` is the arithmetic mean of
the six role `E_X` values; micro `E_X` is derived after summing sufficient
statistics over all 72 matrices.  Macro radial summaries are arithmetic means
of role summaries; micro radial summaries derive from globally pooled
sufficient statistics.  With role-specific gains and role-pooled raw values
`(T_r,P_r,D_r)`, micro uses `T_sum=sum_r T_r`,
`P_g=sum_r g_r^2 P_r`, and `D_g=sum_r g_r D_r`; derive its error, cosine,
radius, radial penalty, and angular floor from `(T_sum,P_g,D_g)`.  Require
`T>0` and `P>0` for every decoded arm/cell used in cosine or radial analysis;
otherwise the panel is invalid.  Literal-zero output has `P=0` by definition
and is used only for exact `E_X=1`, never for cosine/radial quantities.  Use
10,000 paired bootstrap draws over the 12 block indices, carrying all six
roles, intervention arms, and calibrations together.
Splits and tilings remain separate; they are never pooled as independent
samples.  Bootstrap seeds are `26081903` for Beans and `26081902` for SROIE.

The calibration difference-in-differences is exactly
`[E_correct(g_role)-E_zero_code(g_role)] -
 [E_correct(1)-E_zero_code(1)]`, recomputed inside every paired bootstrap draw.
The common gain is exactly the median of the six frozen gains,
`0.46301170700708616`.  For all 720 bijections assigning gains to roles, lower
`E_X` is better.  Sort ascending separately for macro/micro, panel, split, and
tiling.  Ties within absolute `1e-12` receive the conservative worst rank among
the tied assignments; the identity is in the best 5% only when rank `<=36`.

Intervals are conditional on one fixed checkpoint, fixed score images/tokens,
and one frozen source-gain fit.  A/B and tilings are robustness repeats, not
independent sample size.  The block bootstrap contains neither
activation-population uncertainty nor source-gain estimation uncertainty and
does not support a model-, dataset-, or domain-population confidence interval.
The success rule is a conjunction, so no multiplicity correction is used; the
claim remains restricted to exactly two unseen ViT-style vision encoders.

## Frozen success criteria

A panel is a **full prospective mechanism replication** only if all validity
and quality checks pass and all of the following hold under source-frozen role
gains:

1. For correct versus each of `permuted_within_row` and `zero_code`, on both
   A/B score splits, both tilings, and both macro/micro aggregations, the
   comparator/correct `E_X` point ratio is `>=1.05` and its paired-bootstrap
   lower 5% bound is `>1` (`16/16`).
2. The corresponding correct-minus-control operator-cosine effect has paired
   lower 5% bound `>0` in all 16 cells (`16/16`).
3. Calibrated correct has lower point `E_X` than calibrated `zero_code` in every
   role on both splits and tilings, and also has point `E_X<1` in every such
   role; equivalently correct is below `min(zero_code,literal_zero)` in all 24
   role/split/tiling cells per panel.  Effects below 5% must remain visible.
4. Relative to raw scaling, calibration differentially improves
   correct-minus-`zero_code`: the paired difference-in-differences upper 95%
   bound is `<0` for macro and micro on both splits and tilings.
5. In `attn_value`, `attn_output`, and `ffn_down`, the frozen role gain reduces
   correct-code radial penalty on every split and tiling.
6. Source role-matched correct beats the frozen source-median common shrink for
   macro and micro on both splits and tilings: the point difference
   `E_role-E_common<0` and paired-bootstrap `U95<0` in all eight cells per
   panel.  The identity role assignment has conservative rank `<=36/720` for
   both aggregations, splits, and tilings.
7. Absolute operator quality beats literal-zero output: source-gain correct has
   macro and micro point `E_X<1` and block-bootstrap `U95(E_X)<1` on both
   splits and tilings (eight cells per panel).  Without this condition, a
   relative win over worse decoder controls is only causal/directional
   evidence, not a full calibrated mechanism replication.

The locked raw source verdict remains non-passing regardless of this outcome.
Weight-space reconstruction quality is reported but is not a pass criterion;
no high-fidelity codec claim is allowed unless it independently beats weight
baselines.

Outcome labels use this strict precedence and admit no narrative override:

- `INVALID_PANEL`: any provenance, resource-hash, geometry, activation,
  downstream-quality, seal, completeness, or numerical-validity check fails.
- Otherwise `FULL_REPLICATION_PASS`: every condition 1--7 above passes.
- Otherwise `DIRECTIONAL_ONLY_OR_MIXED`: correct beats
  `permuted_within_row` in point `E_X` and has correct-minus-permutation
  operator-cosine `L05>0`, all under frozen source-role gains, for macro and
  micro on both splits and tilings
  (eight point and eight interval cells), but at least one full condition fails.
- Otherwise `MECHANISM_FAIL`.

Only two `FULL_REPLICATION_PASS` results on Beans P1 and TrOCR/SROIE P2 permit
freezing a later confirmatory target protocol.
This does not itself unseal target data.  Mixed or negative Weight-AE outcomes
cannot be replaced by another checkpoint and do not authorize target fishing; an
exploratory target run would require an explicit scope change.

## Diagnostic latent geometry

This section is explicitly exploratory and non-gating; no view may be selected
after seeing separation.  The population is fixed to correct, fixed-cell codes
from Beans, TrOCR/SROIE, and the original ViT-B/Flickr held-out panel.  The
original source caches are
`factorial_code_cache_seed_26081601.pt` (SHA-256
`a585c2ecfe4d1e560c79dca19518543dbdfb5f897bfa413b3ef5148f01995c8f`)
and `factorial_code_cache_seed_26081602.pt` (SHA-256
`c56f1ed3ccd0038f3f78ec767facd32cc0bba490f2e31229ec5d7b18a211bfd6`);
use only keys `enc=cell|code=correct`.  New-panel correct codes from both formal
tilings are persisted before outcomes are aggregated.

For each of 3 panels by 2 tilings by 72 matrices, summarize all tile codes with
feature-wise mean, population standard deviation, and FP64 linear-interpolation
quantiles `q=(0.10,0.50,0.90)`, concatenated in that order into 2,560 features.
This gives exactly 432 equally weighted matrix rows; no role/tile resampling is
performed.  Feature-standardize over all 432 rows with population variance,
drop features with standard deviation `<1e-12`, then fit scikit-learn `1.8.0`
PCA with `n_components=10`, `svd_solver="full"`.  Store inputs, scaler, loadings,
scores, and explained variance.

Render the same fixed PC1/PC2 coordinates (tiling as marker style) in exactly
these readable views: colour by panel/domain, dataset, checkpoint/model,
six-role layer type, and normalized depth `depth/11`; plus one six-role facet
coloured by panel and one four-bin depth facet coloured by panel.  No UMAP,
t-SNE, clustering score, retrieval, linear probe, or result-selected axis is
part of the formal run.  Visual separation cannot substitute for causal or
absolute operator endpoints.

## Required artifacts and review

Execution is prohibited until three separate programs exist: a quality/panel
builder that cannot import the Weight-AE, a formal Weight-AE runner, and a CPU
post-run analyzer that cannot import or execute the Weight-AE.  Freeze their
SHA-256 values plus this final design SHA and every transitive model/evaluator
dependency in a preexecution contract.  The runner must verify those constants
internally, strict-load the canonical AE, and pass decoder-entry parity checks
against the historical fixed-cell path before decoding a candidate.  The
analyzer must independently reconstruct every aggregate, bootstrap interval,
720-role permutation rank, criterion, and outcome from stored sufficient
statistics.

The formal output directory must be a fresh, absent path under
`artifacts/crossmodal_united_structure`; no overwrite or outcome-dependent
resume is allowed.  A quality/panel cache may be reused only after every file
and internal tensor-manifest hash matches and the reuse is logged.  Total new
formal/cache storage is budgeted below 10 GiB; do not duplicate full prediction
matrices by A/B split or analytic scale.

Expected formal counts for two valid panels are: 72 matrices per panel; 20,736
tiles per panel/tiling; 432 decoded full matrices per panel
(`72 x 2 tilings x 3 arms`); 864 unique weight sufficient-statistic rows; and
1,728 operator sufficient-statistic rows
(`2 panels x 2 splits x 2 tilings x 72 matrices x 3 arms`).  Require complete,
unique hard grids before analysis.  Primary calibrated error and directional
criterion tables each contain 32 rows across both panels; all row counts for
role, absolute, difference-in-differences, common-shrink, radial, and mapping
criteria must be derived in the preexecution contract and asserted, not
silently inferred after the run.

At minimum store resolved config, exact design/script/dependency hashes,
resource and training-bank overlap manifests, quality predictions/metrics,
activation selection and tensor manifests, matrix/tiling/code manifests,
per-matrix sufficient statistics, aggregates, bootstrap draws or quantiles,
criterion rows, decision JSON, latent tables, all plots, logs, and a complete
SHA-256 output manifest.  Runtime logs must expose config, device, dtype, seeds,
cache hits, stages, progress/rate, current panel/role/depth/arm, and final paths
and metrics.

Freeze and record the runtime stack: Python 3.12 environment, PyTorch
`2.10.0+cu128`, Transformers `5.1.0`, safetensors `0.7.0`, Pillow `12.0.0`,
datasets `3.6.0`, NumPy `2.3.5`, scikit-learn `1.8.0`, pandas `3.0.0`, and
pyarrow `23.0.0`.  Also bind the exact unseen-weight scout manifest at
`artifacts/crossmodal_united_structure/prospective_source_panel_scout_20260816/artifact_manifest.json`,
SHA-256
`623cbf54611bdba49a0007840b52457cbfa382935507cb13020ffef5e44e5254`.

Hard data-flow checks must show that replication activations are read only by
operator scoring: no panel `X` tensor may reach distribution-context encoding,
latent encoding, decoding, gain selection, or tiling.  Require exact transpose
and role-map parity, raw-stat algebra, analytic-scale parity, A/B prediction
hash equality, finite/nondegenerate targets, and a complete output SHA-256
manifest.

The output manifest lists every formal file except the manifest itself; its own
SHA-256 is reported externally by the independent audit rather than recursively
embedded.

An independent reviewer must audit the frozen design and implementation before
`--execute`, then recompute the decision from stored sufficient statistics and
visually inspect every cited plot after the run.
