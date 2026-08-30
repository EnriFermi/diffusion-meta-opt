# High-quality cross-domain Weight-AE transfer: two-tier plan

Date: 2026-08-16

Status: checkpoint and forward contract audited; deterministic AE and all
target AE metrics remain sealed. This plan supersedes Flickr8k-from-scratch as
the decisive primary.

## Pretarget conditioning amendment (2026-08-16)

No data2vec Weight-AE forward or target reconstruction/latent metric had been
run when this amendment was locked. Corrected source-only diagnostics showed
that one global exchangeable `Source-mean-C0` retains weak direction but does
not beat the exact-zero reconstruction broadly: full-matrix BEiT held-out
operator error was `1.0118` (matrix macro `1.0955`) versus Native-C `0.6896`.
That global stub erases role and depth in addition to erasing target-domain
activation identity, so it is a test of a stronger estimand than the requested
matched-cell domain intervention.

The Tier-A confirmatory primary is therefore amended, before target unsealing,
to `Source-cell-mean-C[role, normalized-depth]`: two exchangeable source-only
vectors (`C_var`, `C_patch`) on the fixed 12-point normalized-depth grid for
each of the six roles, shared with identical hashes across vision, text and
speech. `C_pooled` is derived from broadcast `C_var`; it is not fitted
separately. Within a matched role/depth cell, no target activation enters the
Weight AE, so between-domain variation enters through `W`. This only supports
a role/depth-conditional weight-path claim; role/depth organization is supplied
and cannot be claimed as discovered from weights in this arm.

The original global `Source-mean-C0` remains a mandatory stronger stress and
claim-upgrade arm, not a discarded result. A source-cell medoid is a locked
support sensitivity and cannot rescue a failed cell mean; Native-C remains a
joint `(W,C(X))` positive control. The full construction, matched-baseline
requirements, decision tree and reviewer-facing wording are frozen in
`artifacts/crossmodal_united_structure/conditioning_contract_review_20260816/README.md`.
Only a held-out Vit-B source G0--G2 pass for the cell mean authorizes
data2vec-vision unsealing.

The exploratory source-only BEiT micro-preflight that gave cosine near zero was
invalidated. The checkpoint was trained with raw integer 2-D RoPE coordinates,
but a post-training missing-config fallback silently selected
`normalized_center`; strict state loading could not detect this parameter-free
semantic change. On the same 144 paired source slices, restoring raw coordinates
changed operator cosine from `0.0080` to `0.6051` and relative operator error
from `1.5439` to `0.7686`. On the exact saved historical batch it restored
`behavioral_dir=0.6437` versus the stored April value `0.6422`. The loader is now
backward-compatible and guarded by tests. Text/speech metrics remain sealed
until a complete source full-matrix/PCA gate passes. Causal evidence is stored
in `artifacts/crossmodal_united_structure/step480_rope_contract_audit_20260816/`.

## 1. Scientific target and impossibility boundary

The intended empirical question is whether a deterministic Weight AE trained on
real pretrained vision operators retains representation-specific skill on
competently pretrained text and speech Transformer operators.

No currently identified experiment simultaneously provides all four properties
within one H100 night:

| Design | Exact shared graph | Independent learning | Source-like quality | Overnight |
|---|---:|---:|---:|---:|
| Flickr8k scratch | yes | yes | no/unestablished | yes |
| shared strong initialization + distillation | yes | no, shared basin | attainable locally | likely |
| official data2vec Base triplet | no, core roles/shapes only | yes | yes | evaluation yes |
| full controlled three-domain pretraining | yes | yes | potentially yes | no |

Therefore one benchmark must not pretend to prove everything. The evidence is
split into a high-quality independent ecological tier and a controlled
shared-basin tier. The tiers have different claims.

## 2. Frozen deterministic AE

Use only:

    /home/coder/project/projects/shared/storage/artifacts/training/checkpoints/
    weight_quantile_vae_gpu0_square/stage_1/latest.pt

- step 480000;
- SHA256
  `d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00`;
- its 1,339 model tensors are bit-identical to those in
  `step_0480000.pt` (whose container SHA256 is
  `0327871d26daac3833556a6c1a7212d07dcd639eb43f8c3c85adb083cb284006`);
- this exact `latest.pt` is recorded by the historical held-out latent/cluster
  evaluation;
- construct deterministic AE with `use_latent_sampling=false` and strict-load;
- because this legacy checkpoint predates `rope_2d_coord_kind`, resolve its
  missing field to `raw`; newer checkpoints with an explicit
  `normalized_center` value retain that value;
- do not report posterior/KL quantities.

Fit every gain, normalization, nuisance model, PCA/random codec, projection and
threshold on source vision data only and lock them before target metrics are
read.

## 3. Tier A: high-quality independent ecological test

### 3.1 Locked v1 family

Use the official pretrained checkpoints:

| Domain | Model | SHA256 of original PyTorch weights |
|---|---|---|
| vision | `facebook/data2vec-vision-base` | `e1942d45a5003da3053504a1d9a7e12fcf70c0e90f2ee9a2b9c7553ba643acfd` |
| text | `facebook/data2vec-text-base` | `65f2e9bc8b76d5bfe3be263ce6d3a05a248c2d1e954fbc4b26dc96f592d53df3` |
| speech | `facebook/data2vec-audio-base` | `4fd2317db95a7d67d3e328647a06f3f90c1b30e2044024a56ccec18827841225` |

Actual checkpoint bytes were loaded before AE evaluation. All three contain 12
layers of width 768, 12 heads, FFN 3072, and exactly 72 matched core matrices:
Q, K, V, O and FFN up/down at every layer. The schema audit is stored at:

    artifacts/crossmodal_united_structure/
      hf_data2vec_schema_audit_20260816/

These checkpoints are independently pretrained and paper-validated through
competitive downstream fine-tuning. They are not an exact graph intervention:
vision is BEiT-like pre-LN with LayerScale/relative position bias, while text
and speech use different normalization/position/frontends and every modality
has its own data, masking and schedule.

### 3.2 Extraction

- include only the 72 core Q/K/V/O/FFN matrices;
- exclude frontends, embeddings, positional modules, LayerNorm, LayerScale,
  pooler, predictor and heads;
- transpose PyTorch `[d_out,d_in]` to the project convention
  `W=[d_in,d_out]` before tiling;
- assemble all `64 x 64` tiles for every complete matrix;
- do not sample layers or blocks;
- preserve domain, model, role, depth and source key in metadata.

### 3.3 Primary endpoints: fixed conditioning, not crossed raw activations

The earlier name `Common-X` is retired. It incorrectly suggested that one raw
vision activation tensor would be treated as an activation of independently
trained text and speech models. Shape compatibility does not align their hidden
coordinate systems, so that construction is not a target-operator endpoint.

Distinguish two objects throughout the evaluation:

- `X_cond` is consumed **inside** the Weight AE;
- `X_score` is a held-out native activation used **outside** the Weight AE to
  score functional reconstruction error.

For a `64 x 64` weight tile the frozen distribution encoder maps
`X_cond=[n,64]` to

    C(X_cond) = (C_var [4,16,256], C_patch [4,256], C_pooled [4,256]).

`C_var` conditions the weight-patch tokenizer, and `C_patch` conditions every
encoder layer and the decoder queries. Therefore a valid fixed-conditioning
ablation must clamp both tensors; replacing only one "activation embedding" is
not sufficient.

The activation-independent primary is `Source-mean-C0`. Before any target AE
metric is read, compute `C` on a balanced locked subset of the existing source
vision cache, average over records and coordinate/patch positions, and broadcast
the resulting exchangeable source-derived template unchanged to **every**
vision, text and speech weight tile. Target activations are never used to form
`C0`, and no raw source activation is run through a target architecture. A
coordinate-preserving source mean, a source medoid with random coordinate
permutations, and role-specific source means are sensitivity arms, not the
primary, because they can attach arbitrary source-coordinate or role metadata
to target weights.

For `Source-mean-C0` report two endpoints:

1. raw normalized weight MSE, cosine, norm/scale and spectral error. This is the
   genuinely common activation-independent transfer endpoint;
2. native functional error, using each target layer's correctly hooked,
   disjoint held-out `X_score^B` only after reconstruction:

       e_native = ||X_score^B(W - g_r W_hat)||^2
                  / (||X_score^B W||^2 + eps).

The source-fit role gain `g_r` and all codec baselines remain locked on source
vision. `X_score^B` differs across models because it must live in the coordinate
system of its own W; it is never used as fixed conditioning. Aggregate
numerator and denominator before division.

Tier-A transfer is supported separately for text and speech only if:

1. source and data2vec-vision validity gates pass before targets are read;
2. under `Source-mean-C0`, the upper confidence bounds of raw normalized MSE
   and `e_native` are below their exact zero-reconstruction baselines;
3. under the same fixed C0, AE beats source-fit PCA and random projection by at
   least 5% in point estimate and the paired layer/role bootstrap upper ratio is
   below 1;
4. target excess error relative to data2vec vision is at most 1.5x;
5. the conclusion survives scale gauge, second tiling, role/depth adjustment,
   entry/singular-vector randomization and deterministic-code shuffle;
6. the AE advantage over PCA is larger on pretrained weights than on
   architecture-matched random initializations. This last comparison is a
   trained-structure falsification, not a paired recovery of the unknown true
   initial seeds.

The statistical population is the locked checkpoint family, not all neural
networks. Layers are repeated measurements, not independent model draws. A
positive result closes the exact-checkpoint existence claim; it does not close
population-level universality.

### 3.4 Mandatory conditioning ablations

Run the following without any raw cross-model activation pairing:

1. `Source-mean-C0`: one exchangeable source-only template for every W;
   zero-shot weight-path primary.
2. `Source-role-mean-C`: a source-only template per matrix role, unchanged over
   domain/model/dataset/depth; in-support sensitivity with explicit role
   metadata.
3. `Native-C`: each model's native context-A activations are encoded to C;
   ecological full `(W,X)` operator state.
4. `Global-mean-C`: one equal-domain mean of native vision/text/speech C,
   reused for every W; secondary because target conditioning is observed.
5. `Zero-C`, deterministic-code shuffle, and a source-embedding medoid are
   negative/stress controls.

Raw `Crossed-X` is prohibited as primary evidence: independently trained models
do not share a hidden basis. A shuffle is valid only within the same hooked
model/layer, where the activation coordinates remain aligned to W.

The source cache is not a universal activation bank. It is the offline Weight-AE
training dataset of paired `(W slice, matching hooked X)` records from 11 vision
models. Its manifest reports 348,049 accepted records and about 300 GiB. The
training config enabled 17 datasets, while 15 have recorded dataset statistics
in the final manifest; Mapillary Vistas and ReLAION-400M are configured but do
not appear in those recorded statistics. Every source X is meaningful only for
the W/model/layer from which it was hooked. The cache is used only to estimate
post-distribution-encoder source templates and source-only baselines.

The same checkpoint/layer W is paired with many activation records collected
on different image datasets. Consequently, any apparent *dataset* separation
under `Native-C` is a property of the conditioned code `z(W,C(X_dataset))`, not
evidence that W itself contains the dataset label. Under one fixed C0, duplicate
copies of the same W must map to the same deterministic code; disappearance of
dataset clusters is the expected negative control, not a failed transfer result.

For `Native-C`, lock disjoint context-A and scoring-B activation panels:

1. **Aligned Flickr8k panel.** Use the official 1,000-image validation split for
   context A and the disjoint 1,000-image test split for scoring B. For every
   image retain its five written captions and five corresponding human spoken
   captions, aggregate first per image, and then give every image equal weight.
2. **Native-domain robustness panel.** Use ImageNet-1K validation for vision
   (fixed class-stratified hash split A/B), WikiText-103 validation/test for
   text, and LibriSpeech dev-clean/test-clean for speech. If ImageNet access is
   unavailable, lock COCO-2017 validation as the vision fallback before any AE
   target is read.

Within every panel, sample the same number of valid activation rows per domain,
role and depth. Keep sequence masks and document/utterance/image grouping;
confidence intervals resample the top-level item, never individual tokens. Use
native tokenization and native sequence length inside each original model. Only
sample/pool correctly hooked activation rows to the Weight-AE context size.

Persistence of domain separation in `z(W,C0)` shows that domain-associated
information remains in the weight path after removing per-model activation
identity. It does **not** by itself establish semantics, causal universality or
a meaningful latent metric: domain is also aliased with model identity,
parameterization, role/depth mixture, scale and lineage. Quantify separation
with locked source PCA and a leave-one-model/family-out low-capacity probe; use
UMAP/t-SNE only as an exploratory appendix. Also analyze
`delta_z = z(W,Native-C) - z(W,C0)` to localize conditioning-specific structure.

## 4. Tier B: controlled shared-basin adaptation

This tier asks a narrower causal question and is not allowed to inherit Tier
A's broad wording.

### 4.1 Model and data

- one canonical 12-layer, width-768, FFN-3072 core whose complete state is
  cloned across vision, text and speech;
- initialize from one vision checkpoint inside source-AE support;
- add a paired random-initialization arm cloned across domains;
- use information-preserving frozen modality frontends and native sequence
  lengths; exclude every frontend parameter from Weight-AE analysis;
- use aligned Flickr image/caption/spoken-caption IDs;
- predict one frozen common semantic target per aligned item with one identical
  cosine-plus-InfoNCE objective, optimizer, batching and update budget.

Retrieval/probe performance is a competence certificate, not the paper's main
claim.

### 4.2 Sealed pre-AE quality gate

Do not load the Weight AE until every domain/seed has passed on held-out image
IDs:

    q_d = (M_shared - M_no_core) / (M_native_reference - M_no_core)

Require `q_d >= 0.80` and a paired/bootstrap lower bound at least 0.70. The
native reference must itself be clearly above chance. Also require:

- final performance plateau and acceptable train/validation gap;
- non-collapse/effective-rank gates;
- shuffled aligned IDs, padding/length-only and renderer-only baselines near
  chance;
- rewinding the trained core to `W0` while retaining frontend/head reduces
  normalized quality by at least 0.25;
- no-core or linear-core baseline remains below 0.60 normalized quality.

If frozen frontends solve the task or any modality fails the gate, the AE result
is not read and the controlled claim is INDETERMINATE. Do not keep only passing
seeds or match accuracy post hoc.

### 4.3 Full-weight and learned-update endpoints

Shared vision initialization makes final-weight reconstruction potentially
trivial. Therefore report both `E(W_T)` and reconstruction of the learned
update:

    E_delta = ||X[(W_T-W_0) - (W_hat_T-W_hat_0)]||^2
              / (||X(W_T-W_0)||^2 + eps)

Use identical `Source-mean-C0`, a source-fit gain, and
PCA/random/norm-spectrum-matched update baselines. Functional scoring uses each
arm's own correctly hooked held-out native `X_score`; the fixed C0 is only an
internal Weight-AE condition. Full-weight PASS with `E_delta` FAIL means
inherited source compatibility, not learned cross-domain organization.

The honest strongest wording from this tier is:

> Within a shared vision initialization basin, functionally necessary
> image/text/speech adaptation trajectories of an identical Transformer core
> remain representable by a vision-trained Weight AE.

Only a passing paired random-initialization arm upgrades this from shared-basin
adaptation toward independently learned controlled evidence.

## 5. Overnight decision and schedule

From the current repository state, a fully implemented, quality-reviewed dual
tier is not responsibly guaranteed in ten hours: the data2vec schema/cache is
ready, but the target Weight-AE harness, native activation capture and Flickr
audio/distillation cache are not preflighted, disk has roughly 24 GiB free, and
all plots/results require review.

If one night is mandatory, Tier A runs first and Tier B remains a pilot unless
its data/harness preflight is completed before the clock:

| Clock | Stage |
|---|---|
| 00:00-01:30 | source-only AE load/calibration and strict extraction tests; no target metrics |
| 01:30-03:30 | Source-mean-C0 full-matrix data2vec evaluation and random-init controls |
| 03:30-05:00 | PCA/random/SVD, gauge, tiling, shuffles and nuisance baselines |
| 05:00-06:30 | native activation capture; Native-C, Global-mean-C and fixed-C sensitivities |
| 06:30-08:00 | locked-PCA geometry, role/depth/domain plots and quantitative probes |
| 08:00-10:00 | statistics, plot/log inspection, suspicious-value audit, decision and report |

If source validity is not established by 01:30, stop target evaluation rather
than spend the night producing uninterpretable target numbers. Do not drop text,
speech, a bad role/layer, a baseline, or an activation ablation to meet ETA.

## 6. Interpretation matrix

| Tier A | Tier B | Supported conclusion |
|---|---|---|
| pass | pass including delta | strong ecological plus local-causal triangulation; still not arbitrary-model universality |
| pass | full W only, delta fail | independent high-quality transfer exists; controlled success is initialization inheritance |
| pass | quality gate fail | high-quality ecological transfer only; common overnight recipe unvalidated |
| fail | pass | local shared-basin structure, no evidence for independent strong checkpoints |
| fail | fail with valid quality | tested claims fail; narrow architecture/domain boundaries may remain |
| fail | weak-quality Flickr only | broad hypothesis not evaluated |
