# Controlled vision-text-speech AE transfer: preregistered 10-hour experiment

Date: 2026-08-16
Status: **SUPERSEDED AS A DECISIVE PRIMARY before any target result was read.**
The compute benchmark and the controlled-training design are retained as useful
engineering/mechanistic material, but this document does not authorize the
Flickr8k run as evidence for transfer among competently trained natural models.

The raw `Common-X`/`Crossed-X` construction below is also historical and
retired: independently trained models do not share a hidden coordinate basis.
The active protocol fixes post-distribution-encoder conditioning tensors as
`Source-mean-C0`; see `cross_domain_high_quality_two_tier_plan.md`.

## 0. Checkpoint-quality validity correction

The source AE was trained on weights and activations from real pretrained vision
checkpoints. The stored source manifest is dominated by named pretrained model
families including Mask2Former/Swin, SwinV2, SigLIP, DeiT, MAE, DINOv2, BEiT,
CLIP, and BLIP. Nine freshly trained 20k-update Flickr8k models would therefore
change two scientifically relevant properties at once:

1. observation domain (vision versus written text versus speech); and
2. the checkpoint population, from broad pretrained models to small-corpus,
   possibly memorizing or shortcut-solving models.

Weak downstream quality is not literally a pre-treatment confounder: under the
fixed recipe it is a post-treatment outcome of domain and optimization. It is
nevertheless a fatal **construct-validity failure** for the intended claim,
which concerns structure shared by competently trained neural operators. A
causal effect of renderer on AE error remains identifiable for this recipe, but
it does not answer whether the vision-trained AE transfers to trained NLP and
speech operators.

The current `64 x 256` interface adds a second validity problem. Transformer
core matrices do not depend on sequence length. There is no need to force
vision, text, and audio to the same `N=64` in order to obtain identical
`12 x 768`, FFN-3072 core weights. Nearest-neighbor byte resampling can duplicate
or discard categorical text positions, whole-utterance adaptive pooling can
smear speech structure, and the three renderers admit different prediction
shortcuts. A falling masked-regression loss or non-collapsed covariance does not
establish useful visual, linguistic, or acoustic competence.

Consequences fixed before target AE evaluation:

- the H100 result in Section 14 establishes compute feasibility only;
- the `S/D/C/I` thresholds below are suspended as confirmatory decisions;
- no target AE metric may be read merely because optimization/non-collapse
  gates pass;
- the proposed Flickr8k run, if reused, is a controlled mechanistic ablation or
  renderer stress test, not the main transfer experiment;
- the previously stated overnight/18-hour delivery estimate is withdrawn until
  an independently validated quality-preserving checkpoint family or training
  recipe is locked.

The replacement evidence program has two complementary arms:

1. **High-quality ecological arm.** Evaluate independently pretrained
   vision/text/speech checkpoints with the same core scale and published
   modality quality (the data2vec Base family is the leading candidate). Use
   native modality tokenization and sequence length; exclude frontend weights
   from the Weight-AE analysis; compress or sample only the activation context
   supplied to the Weight AE. This directly tests the useful empirical claim,
   but modality-specific frontends, masking, normalization, data and budgets
   prevent calling it a domain-only causal intervention.
2. **Controlled causal arm.** Train identical cloned cores behind
   information-preserving modality adapters, with native sequence lengths and a
   common semantic objective. Exclude adapters from Weight-AE analysis. Before
   loading the AE, require independently held-out downstream/semantic probes in
   every domain to recover a preregistered fraction of a strong modality-native
   reference, plus renderer-shortcut, memorization and collapse controls. This
   arm may take longer than one H100 night; it is not allowed to substitute toy
   quality for the intended population.

A convincing paper should triangulate both arms: the first supplies relevance
to strong natural models, while the second isolates mechanisms. A positive
result in the original Flickr design alone is insufficient, and a negative
result there does not reject the broad cross-domain hypothesis.

The replacement preregistration and the completed target-sealed checkpoint
schema audit are:

    projects/weight-vae/docs/cross_domain_high_quality_two_tier_plan.md
    artifacts/crossmodal_united_structure/
      hf_data2vec_schema_audit_20260816/

## 1. Historical controlled design

The old proposal using pretrained ViT, BERT/Wav2Vec, and Llama checkpoints does
not identify a domain effect. Modality was aliased with architecture, width,
depth, frontend, objective, optimizer, training budget, data volume, and
lineage. Matching only a 768 x 768 matrix or a 64 x 64 AE tile does not remove
those aliases.

The primary experiment changes one treatment:

> the input distribution/renderer, while the complete trainable graph, initial
> state, objective, optimizer, number of updates, processed-token budget, matrix
> shapes, tile counts, semantic corpus, and evaluation protocol are fixed.

Natural pretrained models remain an external-validity stress test. They cannot
enter the causal confidence interval, rescue a failed controlled result, or be
used to reject the controlled claim.

## 2. Decision the night run may make

The main falsifiable claim is:

> A deterministic weight AE trained only on visual-model operators retains
> non-trivial, representation-specific reconstruction skill for shared-core
> Transformer weights learned independently from matched image, written-text,
> and spoken-audio observations, when architecture, initialization, learning
> rule, semantic corpus, and compute budget are fixed.

The primary estimand is the additional reconstruction penalty caused by
replacing the vision training distribution with text or speech. It uses one
identical activation context for every domain so that only W changes. Native
and crossed activation contexts are separate mechanism tests.

A positive result establishes controlled cross-domain transfer on this
benchmark. It does not establish a universal manifold of all ViTs, LLMs and
audio models, cross-architecture or cross-scale transfer, preservation of
end-to-end quality after decoding all weights, or meaningful global Euclidean
latent distance.

A stronger interpretability result requires a readout of original-model lesion
responses fitted on vision and transferred unchanged to text and speech.

## 3. Frozen deterministic AE

Use only:

    /home/coder/project/projects/shared/storage/artifacts/training/checkpoints/
    weight_quantile_vae_gpu0_square/stage_1/step_0480000.pt

- step: 480000;
- SHA256: 0327871d26daac3833556a6c1a7212d07dcd639eb43f8c3c85adb083cb284006;
- state fingerprint: 9b452a6d51201d09e4fb370016f91efa4725449d18352e02ac73f028ae633c8a;
- posterior-head keys are absent.

The embedded config is stale. Construct the deterministic model, force
use_latent_sampling=false, and strict-load. Call the representation z, never a
posterior mean. Do not report KL, log-variance, posterior density, or posterior
uncertainty.

Calibration, normalization, PCA codecs, and thresholds are fitted or selected
using only the historical vision bank and locked before text/speech AE metrics
are read.

## 4. Matched natural triplets

### 4.1 Corpus

Use the official Flickr8k image/text corpus plus Flickr8k Audio Caption Corpus:
8,000 images, five written captions per image, and human recordings of those
exact 40,000 captions. Use the official image-ID split:

- train: 6,000 images / 30,000 aligned triplets;
- validation: 1,000 images / 5,000 triplets;
- test: 1,000 images / 5,000 triplets.

No image ID or caption crosses the split. Download, checksum, and index the
approximately 4.2 GB audio archive before the clock starts.

The aligned scientific row is:

    (image_i, written_caption_ij, spoken_caption_ij)

The same triplet IDs, order, batches, and mask coordinates are used in all
three treatment arms.

### 4.2 One non-trainable interface

Every input becomes exactly 64 x 256 before a trainable parameter:

1. Vision: resize RGB to 64 x 64; make 64 non-overlapping 8 x 8 x 3 patches
   of width 192 and append 64 zeros.
2. Speech: mono 16 kHz; fixed waveform normalization; 64-bin log-mel with
   25 ms window and 10 ms hop; deterministic temporal pooling to 256 frames;
   concatenate four adjacent frames into each of 64 vectors of width 256.
3. Text: NFC-normalized UTF-8 bytes; one-hot width 256; deterministic
   nearest-neighbor resampling of the complete caption to 64 positions.

Apply the same parameter-free per-token LayerNorm(256), then the same trainable
Linear(256, 768).

The renderer is the operational definition of the domain intervention. Spatial,
acoustic, and byte observations necessarily differ in entropy and correlation;
those are consequences of domain, not hidden trainable architecture changes.
Renderer code and constants are frozen before training.

### 4.3 Iso-content falsification control

Create a small paired synthetic corpus from one latent sequence of 64 symbols,
rendered as an 8 x 8 grid, deterministic tones/chirps, and text symbols. Sample
IDs and latent content are exactly matched. This is a pipeline sanity test, not
ecological evidence and not a replacement for Flickr8k.

## 5. One exact model and training recipe

### 5.1 Primary scale

Only a Base core enters the scientific gate:

- 12 pre-LN blocks;
- d_model 768, 12 heads;
- d_ff 3072;
- GELU;
- separate Q, K, V, and attention-output projections;
- dropout and stochastic depth zero;
- learned 1-D positions of shape 64 x 768;
- identical predictor in every arm.

These core dimensions and roles match canonical ViT-B bodies represented
heavily in the AE source bank. Micro models may be used before the clock for
smoke/timing only. They never enter a primary gate or pooled plot. There is no
scale factor in the primary experiment.

### 5.2 Paired randomization

For each seed in 0, 1, 2:

1. instantiate the complete model and optimizer once;
2. byte-clone their states into vision, text, and speech arms;
3. store hashes and require exact equality;
4. use the same aligned batches and masks at every update.

Initialization seed is the top-level paired inferential unit, n=3. Layers,
tiles, triplets, contexts, and checkpoints are nested repeated measurements.

### 5.3 Shared self-supervised learning

Use one data2vec-style masked contextual regression in every arm:

    target_t = mean of parameter-free LayerNorm outputs from
               the top six EMA-teacher blocks
    loss = mean SmoothL1(student_t, target_t; beta=2)
           over masked positions

- teacher sees the complete sequence;
- student has exactly 32 of 64 positions masked;
- aligned triplets use identical masks;
- identical EMA cosine schedule from 0.99 to 0.9999;
- AdamW, betas 0.9/0.95, weight decay 0.05;
- peak LR 5e-4, 1,000-update warmup, cosine decay;
- gradient clip 1.0, BF16, batch 256;
- exactly 20,000 updates per arm;
- checkpoints at 0, 2k, 5k, 10k, 20k.

No early stopping, per-domain tuning, target-specific normalization, or
different masking is allowed. A source-only vision pilot may tune one universal
recipe. After it passes, freeze everything before launching or reading targets.

## 6. Validity before interpretation

Every domain/seed must have:

- finite model, teacher, loss, gradients, activations, z, and reconstruction;
- non-collapsed teacher targets above source-pilot variance/effective-rank floors;
- held-out loss at least 10% below step 0 and below a constant target predictor;
- median selected-layer movement ||W20k-W0|| / ||W0|| at least 0.05;
- structured sequences beating a within-example position-shuffled seed-0
  training control by a predeclared held-out margin;
- identical step-0 common-context AE metrics across cloned arms.

A target that does not learn is INDETERMINATE, not a negative AE result. A
source-only preflight must first show usable final-vision AE dynamic range.
After that source gate is locked, a valid target failure rejects the benchmark
domain claim.

## 7. Fixed matrices and full assembly

For every block collect the same roles:

- attention Q, K, V, output: 768 x 768;
- FFN up and down: 768 x 3072 and 3072 x 768 in W=[d_in,d_out] orientation.

Primary reconstruction uses attention output and FFN down. Frontend, position
table, predictor, LayerNorm parameters, and output heads are excluded.

Assemble every selected matrix from all 64 x 64 tiles:

- attention output: 12 x 12 = 144 tiles;
- FFN down: 48 x 12 = 576 tiles.

No sampled-block approximation is used. Shapes, tile count, roles, depths, and
tilings are identical across domains.

## 8. Context conditions that isolate mechanisms

Use disjoint aligned validation triplets for context rows A and scoring rows B.
Sample equal valid rows per triplet/layer.

1. Common-X, the causal primary: one role/depth-specific empirical source-vision
   prototype, identical for every W and domain.
2. Global-mean-X, the user's literal fixed-activation ablation: form one
   role/depth prototype by giving vision, text, and speech equal weight after
   source-locked scalar normalization, then use that identical tensor for every
   W. It uses only disjoint A rows and cannot enter the primary zero-shot gate
   because target activations were observed.
3. Native-X: each model's own held-out activations.
4. Crossed-X: each final W paired with context pools from all three arms of the
   same seed/layer.
5. Gaussian-X: one fixed source-covariance-matched synthetic probe.

Score reconstructions on corresponding disjoint B rows and on common-B.
Common-X/common-B changes only W. Global-mean-X is the literal test of whether
weight-domain organization survives after removing per-domain activation
identity. Native-X measures the ecological full operator state. Crossed-X
estimates X main effects and W-by-X interaction.

## 9. Metric and estimands

Fit one zero-intercept gain per role on historical vision only and lock it:

    g_r = sum <X W_hat, X W> / sum ||X W_hat||^2

For a complete matrix:

    e_op = ||X_B (W - g_r W_hat)||^2 / (||X_B W||^2 + eps)
    skill = 1 - e_op

e_op=1 is the exact zero-decoder baseline. Aggregate numerator and denominator
before division. Report raw operator error, cosine, scale error, raw weight
error, gains, and finite rate next to the calibrated endpoint.

For d in text,speech and T=20k:

    theta_d = mean_seed,role,block [
        (log e[d,T] - log e[vision,T])
      - (log e[d,0] - log e[vision,0])
    ]

At step 0 the common-X term must be zero numerically. exp(theta_d)=1 means no
additional target reconstruction penalty from training-domain replacement.

Also estimate absolute target error, native-context benefit, W/X/interaction
terms from the crossed matrix, learned gain over PCA from step 0 to 20k, and
checkpoint trajectories without treating checkpoints as independent samples.

## 10. Baselines and falsifications

All codecs use the same tiles, assembly, and source-only gains:

1. zero decoder;
2. identical random AE;
3. fixed 512-dimensional random orthoprojector;
4. 512-component PCA fitted on historical vision only;
5. storage-budget-matched per-tile SVD;
6. X-aware SVD oracle;
7. direct W norm/quantile/rank/spectrum, X covariance, and XW features.

Required falsifications:

- shuffle trained z among same-role/same-shape tiles;
- seed-0 structured versus within-example position-shuffled training;
- entry shuffle and singular-vector randomization;
- exact scalar gauge W'=cW and X'=X/c, preserving XW;
- canonical and coherent half-tile-offset tilings;
- coherent coordinate permutation on one depth per role;
- synthetic iso-content renderer control.

If scale gauge or simple statistics remove the effect, do not call it shared
learned operator structure.

## 11. Preregistered decisions

### S: source metric validity

S=PASS requires final controlled vision:

- one-sided 95% upper bound E_vision below 0.90;
- finite rate at least 99.9%;
- zero decoder in [0.999, 1.001];
- AE beats random projection and PCA in paired point estimate;
- source gains in [0.25, 4.0].

Launch target evaluation only after a separate source-only pilot demonstrates
this protocol can satisfy the gate. Exclude that pilot from inference.

### D: controlled weight-domain transfer

After S=PASS and all learning gates, D=PASS only if both text and speech satisfy
under common-X/common-B:

1. one-sided 95% upper bound E_d below 1.00;
2. upper bound exp(theta_d) at most 1.50 and every seed estimate below 2.00;
3. AE/PCA point ratio at most 0.95 and paired upper bound below 1.00;
4. AE/random-projection paired upper ratio below 1.00;
5. conclusion survives scale gauge, second tiling, and nuisance adjustment.

Add D=EQUIVALENT if both exp(theta) upper bounds are at most 1.25.

D=FAIL if a valid target has lower bound E_d at least 1, lower bound exp(theta)
above 2, or PCA beats AE by at least 10% in at least two of three seeds. Failure
in either target rejects the three-domain claim but may leave a narrower
two-domain boundary.

### C: activation-conditioned transfer

C=PASS separately in text and speech only if:

- native context improves error by at least 5% over both common-X and realistic
  same-domain context deranged across depth;
- paired lower bounds are above zero;
- shuffled z worsens error by at least 10%;
- X-only statistics do not match the AE;
- crossed contexts show a native W-by-X benefit, not an X-only effect already
  present at cloned initialization.

D=PASS with C=FAIL supports a transferable weight prior only.

### I: shared causal organization

For each original undecoded model, attenuate one selected role/layer at two
fixed norm-matched severities and measure held-out masked-loss increase. Fit a
low-capacity ridge from vision z to the two-point lesion-response curve, grouped
by seed. Freeze it and apply unchanged to text and speech.

Compare with depth/role, W spectra, X covariance, XW, PCA, and random codes.
I=PROMISING on both targets only if:

- lesion-response Spearman at least 0.40;
- latent improves correlation by at least 0.10 over the best nuisance baseline;
- response-sign accuracy above 75%;
- stability across activation pools and second tiling.

Domain clusters do not enter S, D, C, or I.

## 12. Interpretation

| Result | Supported conclusion |
|---|---|
| D PASS, C PASS, I PROMISING | Controlled cross-domain compression plus shared functional organization evidence. |
| D PASS, C FAIL | Vision-trained weight prior transfers; activation conditioning is not load-bearing. |
| D PASS, I weak | Transferable compressor, not an interpretability result. |
| Text or speech fails | Three-domain claim rejected; retain only supported boundary. |
| Only synthetic control passes | Pipeline works; natural-domain transfer fails. |
| AE ties/loses to PCA | No evidence this nonlinear AE is needed. |
| Success exists only at step 0/shuffled training | Generic initialization compatibility, not learned structure. |

## 13. Morning dashboard

Use initialization seed as top-level unit; never tile-level confidence intervals.

1. treatment balance, data/state/optimizer hashes, and fixed factor audit;
2. training loss, target rank/variance, weight movement, shuffled controls;
3. common-X transfer forest for E, theta, AE/PCA/random;
4. full-layer domain by role/depth/checkpoint heatmap;
5. crossed W-domain by X-domain interaction matrix;
6. native/common/global-mean/deranged/Gaussian paired arrows;
7. fixed-X versus native-X latent-domain signal: identical vision-fitted PCA
   views plus a leave-one-seed-out low-capacity domain probe, so persistence of
   domain organization is measured without assuming a latent metric;
8. one vision-fitted PCA coordinate system recolored by domain, seed, role,
   depth, checkpoint, and context;
9. within-model role-depth trajectories;
10. vision-fit locked lesion-response predictions on text/speech;
11. tiling, gauge, nuisance, and iso-content audits;
12. final S/D/C/I decision panel.

UMAP/t-SNE is exploratory appendix only, with balanced layer aggregates,
multiple fixed settings, and no global-distance or cluster-significance claim.

## 14. Ten-hour H100 schedule

Download/checksum, rendering cache, implementation, and source-only smoke happen
before the clock.

A stored compute microbenchmark on the current H100 NVL measured the Base
student/EMA-teacher forward, top-six target construction, Smooth-L1, backward,
AdamW, and EMA update at 0.06012 seconds/step and 6.963 GiB peak allocated
memory for batch 256 x sequence 64. This projects to 20.04 minutes per 20k-step
Base arm and about 3.0 hours of pure GPU time for nine arms. It excludes real
decoding, dataloading, checkpoints, AE evaluation, and analysis, so the schedule
retains a large systems buffer. The timing surrogate used fused QKV while the
scientific model uses separate Q/K/V projections; parameter count and leading
compute are matched, but the pre-result timing smoke remains authoritative.

Stored timing evidence:

    /home/coder/project/experiments/crossmodal_h100_benchmark.py
    /home/coder/project/artifacts/crossmodal_united_structure/
      h100_shared_core_benchmark_20260816/README.md
    /home/coder/project/artifacts/crossmodal_united_structure/
      h100_shared_core_benchmark_20260816/metrics.csv
    /home/coder/project/artifacts/crossmodal_united_structure/
      h100_shared_core_benchmark_20260816/run.log

| Clock | Stage | Mandatory output |
|---|---|---|
| 00:00-00:20 | startup audit | redacted config, AE/data/renderer hashes, device/dtype/disk, exact cloned-state hashes |
| 00:20-05:40 | nine Base trainings | 3 domains x 3 paired seeds, 20k updates, matched checkpoints, progress/ETA |
| 05:40-06:20 | validity and X capture | non-collapse/loss/movement gates, disjoint A/B pools |
| 06:20-07:40 | frozen AE evaluation | full matrices, common/native/crossed/Gaussian X, step 0/final and bounded trajectories |
| 07:40-08:25 | codecs/falsifications | PCA/random/SVD, shuffled z, gauge, second tiling, shuffle/iso-content controls |
| 08:25-09:10 | causal response | lesion curves, vision-only fit, locked target predictions |
| 09:10-10:00 | statistics/review | paired bootstrap, inspect all plots/logs, decision.json and review.md |

The timing fallback is chosen only from a pre-result 256-step smoke: use 15k
updates for all nine models, with checkpoints 0/1.5k/4k/8k/15k. Never drop a
domain, seed, role, control, or bad result.

Every long stage logs config, device, dtype, seed, cache paths, output root,
domain/seed/checkpoint, step/loss/rate/ETA, written artifacts, and summary
metrics. Completion includes inspection, not merely a successful exit.

## 15. Implementation and artifacts

Build one standalone controlled-domain harness with no modality-specific
trainable branch. It must support deterministic triplet caches; exact model and
optimizer cloning; strict AE load; targeted activation reservoirs; complete
tile assembly; source-only calibration/PCA; contexts, gauges, lesions,
statistics, plots; resumable stages; and verbose runtime visibility.

Artifact root:

    artifacts/big_vae/eval/controlled_flickr8k_domain_transfer/<run_id>/

Required contents:

    config_resolved.redacted.yaml
    environment.json
    checkpoint_audit.json
    dataset_manifest.json
    renderer_manifest.json
    randomization_and_state_hashes.json
    logs/*.log
    training/*/metrics.csv
    training/*/selected_core_weights.safetensors
    activations/index.parquet
    metrics/layer_metrics.parquet
    metrics/domain_estimands.csv
    metrics/baseline_comparisons.csv
    metrics/context_cross.csv
    metrics/lesion_responses.csv
    metrics/bootstrap.csv
    latents/layer_aggregates.pt
    latents/layer_metadata.parquet
    figures/*
    decision.json
    review.md

Store full resumable states only where required; selected matrices at analysis
checkpoints. Cap new artifacts at 10 GB. Redact credentials from configs and
manifests. Never copy credentials found in historical manifests.

## 16. Residual limitations that remain fixed

This design removes trainable and optimization aliases, but cannot remove facts
that define the observation domains:

- a caption contains only a selected description of its image;
- each image is repeated for five distinct captions, while text/audio examples
  differ across those five rows;
- speech is a human reading of text, so text-speech similarity is built in;
- spatial patches, acoustic frames, and byte symbols have different entropy,
  correlation, and effective information after rendering;
- one 6,000-image training corpus and three initialization seeds provide pilot,
  not population-level, inference;
- identical shared cores say nothing about cross-architecture robustness;
- training from scratch for 20k updates is not equivalent to web-scale
  pretraining of a natural LLM or speech encoder.

These are scope boundaries, not reasons to reintroduce confounded pretrained
models into the primary causal interval.

## 17. Correct positive wording

If all gates pass:

> With architecture, initialization, learning objective, optimizer, semantic
> corpus, processed-token budget, matrix scale, and evaluation fixed, a
> vision-trained deterministic weight AE retains non-trivial compression skill
> for independently learned vision, written-language, and spoken-language
> Transformer cores. A shared functional-organization statement additionally
> requires I=PROMISING.

Architecture and scale are deliberately fixed here and require separate later
factorial studies.
