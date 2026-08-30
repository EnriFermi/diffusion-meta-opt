# WeightCLIP-scale Weight-AE + conditional flow: implementation plan

Status: **IMPLEMENTATION COMPLETE; DATA PREPARATION IN PROGRESS; PRODUCTION AE
NOT APPROVED (2026-08-18)**. This document is the
execution contract for code, tests, zoo construction and data preparation. It
is deliberately **not** approval to launch the approximately 700M production
AE: the user must separately choose and approve its exact architecture
fingerprint after reviewing the scaling ledger. The canonical project
decisions in `weight_operator_diffusion_workshop_bible_20260817.md`, especially
Section 19, take precedence.

Date: 2026-08-18.

The source-by-source fidelity and deviation ledger is
`weightclip_protocol_audit_20260818.md`. Every launcher must emit a resolved
diff against that ledger; a value absent from the paper/released code cannot be
presented as an official WeightCLIP choice.

## 1. Outcome and non-goals

The first campaign must answer one practical question with the smallest honest
amount of compute:

> On a newly reconstructed WeightCLIP ResNet18-Slim population, does an
> activation-conditioned, fixed-cardinality-per-tile Weight-AE followed
> by a dataset-conditioned flow provide better held-out initializations and
> early fine-tuning curves than released WeightCLIP and ordinary baselines?

The production outputs are:

1. one approximately WeightCLIP-sized deterministic Weight-AE;
2. task-fitted latent targets for the reconstructed source population;
3. matched anchor-free and oracle anchor-conditioned flows in both our latent
   space and the released WeightCLIP latent space;
4. one common downstream harness and a reviewed result table/curve bundle.

This campaign does **not** attempt to establish canonical operator learning,
train a graph encoder, reproduce the unpublished original WeightCLIP zoo, train
the HF-Hub foundation model, or prove full-matrix size generalization. Those are
later experiments.

## 2. Frozen scientific and fairness contract

### 2.1 Population

- Architecture: official `ResNet18Slim(width_mult=0.5)`, roughly 2.8M
  parameters, from the released WeightCLIP implementation.
- Source datasets: the ten WeightCLIP ResNet training datasets named in Bible
  Section 17.
- Zoo recipe: follow the released ResNet data configuration rather than the
  inconsistent population sentence in the paper: 50 independent final runs
  per dataset and two terminal checkpoints per run (released zero-based indices
  43 and 44, i.e. one-based epochs 44 and 45). This gives 500 independent
  lineages and the same 1,000 primary checkpoint files.
- Epoch language in this document is one-based training epoch. The released
  ResNet config selects zero-based indices `[43,44]`, corresponding to
  one-based epochs 44 and 45 and files `checkpoint_000043` and
  `checkpoint_000044`. The manifest stores both fields and tests the mapping to
  prevent an off-by-one population change.
- Split by `(dataset, training_seed)` lineage before any tiles are made. Every
  snapshot from one run, including both primary terminal snapshots and the
  archival trajectory, always remains in the same split.
- Apply any WeightCLIP-style channel-permutation augmentation only after this
  split. Permuted views never count as independent checkpoints, lineages or
  bootstrap units.
- The previously proposed `16/2/2` split was not taken from WeightCLIP. The
  released repository uses random `70/15/15` checkpoint-level splitting and
  explicitly warns that this can leak epochs of one trajectory across splits.
  Preserve its sample-count ratios without the leakage: within each dataset,
  use 35 train, 7 validation and 8 internal-test lineages. This gives exactly
  700/140/160 primary checkpoints after selecting the two terminal epochs.
- Do **not** select only two source datasets as the flow meta-validation set.
  Dataset-to-dataset weight variation is large, so a decision based on two
  datasets has unacceptably high variance and can be dominated by an
  idiosyncratic pair. Select every AE/flow hyperparameter on the pooled
  lineage-safe validation partition containing 7 lineages from **each of all
  ten** source datasets (70 lineages total), and report per-dataset values as
  well as the macro average and worst-dataset value. The separate 8-lineage
  internal-test partition from every source dataset is opened only after those
  choices are frozen. The actual dataset-generalization test remains the six
  sealed OOD datasets; no claim of unseen-dataset generalization is made from
  the pooled source validation alone.
- OOD targets: the six datasets locked in Bible Section 17. They remain sealed
  until source/meta-validation choices are frozen.

The original WeightCLIP ResNet zoo was not released. Released WeightCLIP on our
new zoo is a new-checkpoint transfer evaluation, not an exact reproduction.
Paper numbers are shown only in a separate `reported by authors` table.

There are therefore two comparison stages:

1. **fast screening:** ours trained on the reconstructed zoo versus released
   WeightCLIP transferred to this new checkpoint population;
2. **baseline calibration:** before interpreting our win, run the released
   WeightCLIP checkpoint on the reconstructed zoo and the same six OOD datasets
   using the complete published epoch-0/1/10 protocol. Compare every
   per-dataset cell and aggregate, not only the grand mean, against the paper.

If the released checkpoint reproduces the paper-level metrics within its
measured seed/candidate variation, the reconstructed population is treated as
an exchangeable new draw from the same zoo recipe and released WeightCLIP is
accepted as the primary baseline. Exact checkpoint identity is then not
required. If calibration materially fails, the comparison remains a transfer
screen only and a matched WeightCLIP retrain on our train lineages is required
before an architecture-superiority claim. The matched retrain launcher is
implemented up front but is not launched unless this calibration gate fails.

### 2.2 Method access

- Every conditionable method receives the same frozen released WeightCLIP
  dataset encoder, the same number of target training images, and the same
  image sampling/normalization.
- Anchor-free flow receives no target checkpoint.
- Oracle anchor-conditioned flow receives one real fully trained target
  checkpoint and starts at its clean encoder code. The untouched checkpoint is
  always reported beside the transported result.
- The classifier head is freshly default-initialized for every arm.
- Only body convolutional weights are generated by our codec. All BatchNorm
  keys are outside its representation. A materialized model keeps PyTorch's
  architecture defaults: affine `gamma=1`, `beta=0`, running mean `0`, running
  variance `1`, and `num_batches_tracked=0`; it then receives the same
  calibration/downstream schedule as every other arm. WeightCLIP's native arm
  preserves the complete released decoded head/BN policy. Its controlled arm
  instead keeps decoded BN affine, installs a fresh paired head, resets BN
  running state and receives the common calibration schedule. These are
  separate tables, not a post-hoc diagnostic switch.
- Single sample is primary. A fixed matched best-of-K table is secondary and
  uses one common validation selector and generation budget.
- At generation time, the decoder activation context uses one fixed,
  manifest-recorded pool of at most 512 target-training images. Deeper contexts
  are obtained only by forwarding that pool through the already generated
  prefix. The dataset encoder samples its official 10-image sets from the same
  allowed training pool. No target validation/test image enters conditioning.

### 2.3 Parameter and compute ledgers

WeightCLIP's released configuration uses a 12-layer, `d_model=1600`, symmetric
weight AE with a 512-token window. Its ResNet checkpoint object is 9.14 GB,
while its released dataset-encoder checkpoint is only 3.76 MB. Exact trainable
parameter counts will be derived from instantiated models/state dicts, not file
size.

- Match our complete Weight-AE encoder + decoder + activation-conditioning
  modules to the WeightCLIP weight backbone within 1% if an exact count is
  recoverable, otherwise to the documented approximately 700M target.
- Report dataset encoder and flow parameters separately and also report the
  total deployed system.
- Report optimizer steps, tile examples, independent checkpoint lineages seen,
  GPU-hours, peak memory, and generation NFE. A step count alone is not a fair
  compute comparison.
- Match effective representation rate as well as parameter count: calculate
  valid input scalars per latent scalar for the official sparse WeightCLIP
  tokenizer and choose the nearest supported `(num_latents, d_lat)` for our
  selected tile geometry. Record both raw and padding-adjusted rates.

## 3. Code organization to implement

Keep the official WeightCLIP repository unmodified and pinned by commit. Add a
thin compatibility layer in this repository rather than copying its training
logic into unrelated files.

```text
big_vae/weightclip_benchmark/
  contract.py             # immutable dataset/architecture/head/BN contract
  official_bridge.py      # released encoder, WeightCLIP model and tokenizer
  resnet18slim.py          # topology, parameter adapters, functional assembly
  parameter_adapters.py   # conv-im2col, tile maps and inverse maps
  metadata.py             # fixed hard-coded architecture/layer/tile features
  manifests.py            # lineage splits, hashes and schema validation
  activation_capture.py   # native train-split hooks and deterministic sampling
  task_latent_fit.py      # joint full-checkpoint z_task optimization
  evaluation.py           # common fine-tuning/AULC/BN/head harness
  reporting.py            # tables, curves, compute and provenance ledgers

big_vae/flow_matching/
  model.py                # conditional velocity transformer
  paths.py                # Gaussian and paired-anchor interpolation paths
  objective.py            # conditional flow-matching loss
  solvers.py              # fixed-step Euler and Heun
  dataset.py              # grouped z_enc/z_task records and latent stats
  train.py                # EMA, checkpoint/resume, validation and logging
  sample.py               # topological checkpoint materialization

training/weightclip_benchmark/
  build_zoo.py
  build_operator_dataset.py
  fit_task_latents.py
  train_flow.py
  evaluate.py
  review.py

conf/weightclip_benchmark/
  zoo.yaml
  operator_dataset.yaml
  ae_700m.yaml
  task_latent_fit.yaml
  flow_ours_anchor_free.yaml
  flow_ours_oracle_anchor.yaml
  flow_weightclip_anchor_free.yaml
  flow_weightclip_oracle_anchor.yaml
  evaluation.yaml

tests/weightclip_benchmark/
  ...
```

Existing generic BigVAE code remains the implementation source for model
blocks, losses, optimizer, checkpointing, background prefetch, telemetry and
crash forensics. New code should extend those interfaces rather than create a
second incompatible trainer.

## 4. Stage A: pin WeightCLIP and reconstruct the zoo

### A1. Official bridge

Pin the official repository commit and Hugging Face revision in an immutable
manifest. The bridge must:

- load the released ResNet WeightCLIP checkpoint and sibling dataset encoder;
- reconstruct its exact `ResNet18Slim` topology and sparse tokenizer;
- print trainable parameter counts by component;
- expose dataset prompt encoding with the official 10-image, 32x32 RGB path;
- expose released `direct_decode`, `memory_bank`, and neighbour modes without
  silently changing candidate selection;
- save all missing or ambiguous official choices as explicit fields.

No 9 GB checkpoint is loaded in every worker. One process extracts the required
frozen state or serves it from a local content-addressed cache.

### A2. Zoo builder

Adapt the released trainer but add the guarantees its scripts lack:

- deterministic global seed and per-lineage seed;
- exact dataset file hashes and split sizes;
- reproduce the released shell's full per-dataset LR sweep: two seeds for each
  of `{5e-2, 7e-2, 1e-1, 2e-1, 3e-1, 5e-1, 7e-1}`, cached once per dataset;
- SGD + momentum, OneCycle schedule, width 0.5 and the released cached-image
  input path; do not invent an image augmentation that is absent from the
  rebuilt `dataset.pt` trainer;
- train exactly 50 lineages per dataset, matching the released rebuild script
  and increasing the number of statistically independent trained models;
- retain model-only FP32 snapshots at initialization and after every one of the
  45 epochs. Only epochs 44--45 are the primary WeightCLIP/AE population,
  matching the released ResNet config's zero-based indices 43/44. Epochs
  41--45 are a pre-registered late-stationarity diagnostic, while the denser
  history is an archival trajectory bank for later projects and is not silently
  added to the main training set;
- keep optimizer/scheduler state only in one rolling resume checkpoint per
  active lineage rather than duplicating it in every archival snapshot;
- write one row per lineage to `zoo_manifest.parquet` with config, metrics,
  checkpoint hashes, dataset hash and failure state;
- validate every checkpoint by loading it, running one fixed batch, and
  checking finite logits and expected keys/shapes.

Run the 500 independent trainings on the local H100 NVL only. The released
trainer's process-level concurrency was benchmarked on this exact machine with
the official 2.8M-parameter model, batch 256 and cached 32x32 tensors. Two and
four concurrent model processes both processed 16 synthetic lineages of 8,192
training images plus validation/test forwards in about 17 seconds; eight
processes was slower (about 19 seconds). Production therefore starts with four
workers and may fall back to two if real-data memory/I/O profiling is better.

The ten published source datasets contain 124,817 training images in total.
Fifty lineages, 45 epochs therefore require 280,838,250 training examples. A
real official `land-cover-class_0_10` measurement (14,399 training images)
completed four concurrent 45-epoch lineages in about 127 seconds; the final
two-lineage batch took 69 seconds. The full 14-run sweep took 6m54s. Scaling by
the known per-dataset example counts gives roughly 4--5 hours of pure final zoo
plus sweep compute. Allowing for task-size imbalance, process startup, full
every-epoch FP32 checkpoint writes, dataset materialization, retries and final
integrity review, the measured production ETA is **6--8 hours** end to end on
the current H100 NVL. Replace it again with online telemetry after the first
four production lineages of each dataset; do not silently reduce the sweep or
population if throughput is worse.

An FP32 ResNet18-Slim state dict is about 11.21 MB. Initialization plus all 45
epoch states gives 46 snapshots per lineage and about 257.8 GB decimal (240.1
GiB) for 500 lineages, before roughly 2 GB of materialized image tensors and
manifests. This intentionally supersedes the earlier 50 GiB cap: retain the raw
FP32 states first and compress/archive later. A measured exact XOR-delta + zlib
archive compressed one trajectory only `1.281x`, so no storage estimate assumes
a large lossless compression win. Dataset tensors are loaded once per worker
and reused across seeds.

Read-only retention audit at `2026-08-19T01:05:31Z` measured 400 complete
lineages: epoch states 188.600 GiB, initializations 4.191 GiB, completed-lineage
resumes 1.050 GiB, plus 1.787 GiB source and 0.847 GiB OOD data. The measured
500-lineage projection is 240.948 GiB for all 46 model states and about
243.589 GiB for the root without stale resumes. Current-paper dependencies are
all 1,000 zero-based `{43,44}` terminal checkpoints (700 train/140 val/160
test), while initialization and full configs/metrics/manifests/hashes are also
preserved by user decision.

No pruning policy is approved. The later decision ledger contains: (A) a
three-state floor, approximately 18.356 GiB root; (B) the preferred raw-FP32
nine-state grid, initialization plus one-based
`{1,4,8,16,24,32,44,45}`, approximately 49.784 GiB root; and (C) an
unapproved ten-state compression study, approximately 51.322 GiB root under a
weakly measured 1.076x full-state zlib ratio. Semantic/tensor dedup and XOR
delta remain research options, not assumed savings. Finish/audit the 500 zoo,
produce a content-addressed dry-run retention manifest, obtain explicit user
approval of its SHA, materialize and verify recovery, and only then propose
quarantine/pruning. No files are moved, compressed or deleted by this plan.

Safe tooling status: `training.weightclip_benchmark.zoo_retention` implements
only the read-only `plan` and `verify` portions of this contract. The frozen
config is `conf/weightclip_benchmark/zoo_retention.yaml`; default invocation is
plan-only. Planning requires the literal reviewed final-manifest SHA, fails if
`build_zoo` is live, validates exact 500/23,000/1,000 counts, `35/7/8` splits,
all referenced manifest/dataset/checkpoint hashes and the complete live file
inventory before writing an immutable content-addressed artifact outside the
zoo. The plan source-seals `zoo_retention.py`, `build_zoo.py`, `manifests.py`
and `metadata.py`. Verification rejects any added, removed, resized or rehashed
file. Plan and verify both repeat the live-PID, final-manifest, referenced
inventory-manifest and exact path-set checks after their long scan. The
plan stores each file's device/inode/size/`mtime_ns`/`ctime_ns`; every content
hash is bracketed by equal before/after snapshots, and the post-scan gate
compares all snapshots again. Thus same-size in-place writes, metadata-only
drift and transient writers fail closed without a second content pass. The
post-scan gate does not double-hash approximately 240 GiB, so a narrow residual
TOCTOU window remains after the last snapshot; it must be closed immediately
inside any future atomic apply. Current apply is disabled, so this residual
cannot cause a mutation.

The v1 grid keeps initialization and one-based epochs
`{1,4,8,16,24,32,44,45}` plus all non-candidate provenance/data artifacts.
Rolling resumes are conservatively kept and separately accounted; therefore
the 49.784 GiB projection is the grid9 root estimate without resume overhead,
not a promised post-tool size. The `apply` stage checks for an exact approval
but always refuses because atomic quarantine, recovery drill and separately
approved reclaim are not implemented. This is intentional. Do not interpret
the CLI's existence as authorization to prune.

```bash
python -m training.weightclip_benchmark.zoo_retention plan \
  --config conf/weightclip_benchmark/zoo_retention.yaml \
  --final-zoo-manifest-sha256 <EXACT_REVIEWED_FINAL_MANIFEST_SHA256>
python -m training.weightclip_benchmark.zoo_retention verify \
  --plan <retention-plan-PLAN_SHA_PREFIX.json>
```

Only synthetic/mocked retention tests were run during implementation. No real
zoo plan, verification, quarantine, move, compression or deletion was run.

### A3. Split and quality gate

Create train/meta-validation/internal-test splits at lineage level before AE
training. The gate requires:

- exactly 500 complete lineages, 1,000 readable primary checkpoints at
  one-based epochs 44--45 (file indices 43--44), and the declared 23,000
  model-only snapshots
  including initialization, or a manifest
  that explicitly identifies retries/exclusions;
- no seed or checkpoint hash in more than one split;
- no unexplained collapsed/near-random source models;
- distributions of final accuracy and weight norms reviewed by dataset.

Poor source checkpoints are not silently discarded after seeing downstream
results. Any quality threshold is fixed from source validation and recorded.

## 5. Stage B: build the activation-conditioned operator dataset

### B1. Parameter representations

- Convolution: use the ordinary im2col operator matrix
  `[in_channels * k_h * k_w, out_channels]`. This is exact for the convolution
  and does not build a giant constrained Toeplitz matrix.
- Linear body weight: `[d_in, d_out]`.
- BatchNorm affine, running statistics and counters: excluded from the codec
  and reset to the architecture defaults declared in Section 2.2.
- Classifier head: excluded from generation and recorded as `default_init`.

All transforms need exact round-trip tests and a coverage report listing every
state-dict key as generated, defaulted or excluded. Unknown keys are fatal.

### B2. Activation capture

For every checkpoint, capture activations only from its native dataset training
split. For each supported layer:

- linear: sample input rows;
- convolution: sample im2col receptive-field rows before the convolution;
- cap at 512 deterministic rows, with the sampling seed derived from lineage,
  epoch and layer key;
- store dtype/source statistics and reject NaNs/infs.

Implement the released WeightCLIP five-permutation augmentation as an exact
graph-gauge transform, not as independently labeled data. A channel permutation
must be propagated through every affected convolution/residual consumer, and
the corresponding activation-context channels must be permuted consistently.
Verify bitwise/close functional equivalence on a fixed batch. Materialize these
views on the fly or by compact permutation indices; do not duplicate full
checkpoint/context storage five times. This gives both codec pipelines the
official symmetry augmentation without creating pseudo-independent evidence.

The two terminal epochs of one lineage may provide training diversity, but are
weighted so one trajectory does not count as two independent models.

### B3. No-duplication storage format

Do not reuse the current format's repeated activation tensors. Build two
content-addressed banks:

1. `context_bank`: one unique `[n, d_in_tile]` activation row-tile per
   checkpoint,
   layer and input-tile, plus masks, quantiles, mean/log-scale and covariance;
2. `weight_tile_bank`: contiguous fixed-shape weight tiles, masks, operation metadata
   and a compact `context_id` reference.

Multiple output-column tiles share one context entry. Shards are large enough
to amortize file opens, memory-mappable, checksummed and immutable. Metadata is
stored as integer indices, not repeated Python dictionaries.

Before creating the bank root or hashing 700 large checkpoint files, the
builder must pass the frozen exact-selection preflight. It requires the ten
declared source datasets, exactly 35 train lineages per dataset, exactly
zero-based checkpoint indices `{43,44}` per lineage, and therefore exactly 700
unique `(dataset,lineage,index)` identities and 700 unique canonical paths.
Every row must be `split=train`, `is_primary=true`, carry a strict lowercase
64-hex declared SHA, point to an existing file, and match its recomputed SHA.
Cheap structural failures occur before any checkpoint hash or bank write.

`OperatorDatasetProtocol` also freezes the scientific/storage contract rather
than treating these fields as runtime tuning knobs: `checkpoint_splits=[train]`,
`primary_checkpoint_indices=[43,44]`, 16 quantiles, five permutation views,
raw/bank dtype `float32`, context shards of 256 records and weight shards of
512 records. The shard sizes are frozen layout/provenance choices; changing
them requires a new reviewed protocol and content-addressed artifact. This
preflight is implementation evidence only; it does not mean the final bank has
been built or scientifically validated.

Precomputed distribution summaries bypass repeated quantile/covariance work in
the AE forward. Raw sampled `X` remains available for the exact historical
behavioral operator/direction/scale losses. A required equivalence test compares
raw-stat and precomputed-stat model outputs/losses in FP32 and BF16.

### B4. Data-loader throughput

- lineage-balanced sampler, then balanced dataset/layer/depth/operation type;
- configurable mixture of canonical and symmetry-permuted views, with total
  sampling weight one per original lineage;
- persistent workers, pinned-memory ring buffer and nonblocking H2D on a
  dedicated CUDA stream;
- context-ID-aware batching for cache reuse without allowing one model to
  dominate a batch;
- bounded RAM cache for hot context shards;
- log disk read, decode, queue wait, pin, H2D and GPU step time separately.

The exact production-shaped profiler gate is: after warmup, GPU active time at
least 90%, input wait at most 5% of wall time, nonempty prefetch queue for at
least 95% of steps, and stable throughput over at least 500 steps. If this gate
fails, production training does not start.

## 6. Stage C: train the approximately 700M deterministic Weight-AE

### C1. Architecture derivation

Use a script, not hand arithmetic, to search `d_model`, `d_lat`, latent slots
and layer counts while preserving the existing successful topology:

- activation distribution encoder with empirical quantiles and covariance;
- conditioned patch tokenizer with near-identity residual path;
- zero-initialized conditioning outputs and alpha initialized at `1e-3`;
- per-encoder-layer gated activation adapters;
- local attention + Perceiver resampling encoder;
- coordinate/RoPE cross-attention decoder;
- direction/log-scale output parameterization with bounded log scale;
- disabled/frozen z shortcut.

The primary model has no posterior/log-variance heads and no sampling. Startup
asserts this from actual module/state keys. It also asserts the resolved latent
rate and trainable parameter budget. A stale YAML is never accepted as evidence
of model identity.

### C2. Loss and optimizer

Start from the actual square-AE recipe, not the current typo-prone default YAML:

```text
L = L_behavior + L_structure
L_behavior = 50 * L_operator + 1 * L_output_direction
             + 10 * L_output_log_scale
L_structure = 1 * L_weight_direction + 10 * L_weight_log_scale
```

Normalized reconstruction and relation-Gram terms stay zero initially. Keep
mask-aware normalization, Huber log-scale, magnitude weighting and all
non-finite diagnostics. There are no BatchNorm tiles.

Primary optimizer: copy the proven square-AE recipe rather than selecting LR
from a short run: fused AdamW, BF16 autocast, TF32, LR `5e-5` from the
serialized config inside
`weight_quantile_vae_gpu0_square/stage_1/latest.pt` at step 480k, betas `(0.9, 0.999)`,
weight decay `0.01`, global clip `5.0`, cosine decay, and exact resume of
optimizer/scheduler/scaler/step. There is **no 2k-step LR triage**. Historical
training has a fast initial fall, long noisy plateaus and delayed further loss
drops; a short trial measures only entry into the first bad plateau and can
prefer an LR that is worse over the actual horizon. Changing this LR requires
a full-horizon experiment, not a smoke run.

No KL is present. After the deterministic AE is sealed, a separate optional
copy may be fine-tuned with a predeclared KL/gate schedule.

### C3. Speed configuration

- Primary proposed geometry is square static `128x128` with 512 activation
  rows. On the exact 20 generated ResNet body matrices it uses 193 tiles at
  88.24% valid fill. The alternatives are `64x64`: 694/98.16%, `96x96`:
  379/79.88%, and `128x64`: 354/96.22%. `96x96` is rejected because it has both
  more codes and more padding than `128x128`; only 2/20 matrices fit in one
  `96x96` tile, versus 3/20 for `128x128`.
- Before freezing the config, run a production-shaped throughput/memory profile
  for `128x128`. Do not use a 2k loss curve as a quality selector. `128x64` is
  an emergency technical fallback only if square `128x128` is infeasible or
  materially misses the throughput gate; any fallback requires an explicit
  recorded decision rather than a silent config change.
- native BF16; no FP16 scaler unless BF16 is unsupported;
- fused AdamW, `zero_grad(set_to_none=True)`, TF32 and Flash/SDPA;
- `torch.compile` only after eager/compiled numerical equivalence;
- cudagraph output reuse remains disabled until its known bug is explicitly
  cleared;
- activation checkpointing is disabled unless the largest throughput-optimal
  batch does not fit;
- batch size is auto-profiled from powers of two and chosen by tile/sec, not by
  maximum occupancy alone;
- gradient accumulation is used only if needed for the fixed effective batch.

The current compute-only 737.4M benchmark is an optimistic floor. The exact
model gets a 500-step production-shaped benchmark before ETA is updated.

### C4. Logging without recreating the 30 GB failure

Preserve the existing semantic visibility:

- total, behavioral and structural components;
- LR, grad norm, clip ratio, steps/sec, tiles/sec and lineages seen;
- data-build/wait/H2D timing and prefetch depth;
- source diversity/perplexity;
- activation-adapter and tokenizer alpha statistics;
- latent mean/std/effective rank and decoder scale bounds;
- GPU memory/utilization and checkpoint duration;
- crash report plus full offending batch statistics for nonfinite values.

Change only the pathological storage policy: cheap global/part gradient stats
can log every 10--100 steps; a full per-layer snapshot is stored at step 1 and
then every 2,000 steps, plus on anomaly. Store top/bottom layers and quantiles
in compressed Parquet/Zstd, not a wide CSV every five steps.

At startup write a secret-redacted resolved config, git commit, dependency
versions, model/state-derived architecture signature, parameter/rate ledger,
dataset manifest hash, device/dtype, seed and output directory. Historical
artifacts contain plaintext external-service credentials; production code must
reject secret-looking serialized config fields.

### C5. Validation and stop policy

- deterministic fixed meta-validation tiles every 20k steps; validation wall
  time must remain a small fraction of training wall time;
- checkpoint plus complete resume state every 10k;
- run the full 500k optimizer steps;
- no early stopping, plateau rule, LR-on-plateau reaction, or short-run model
  selection. The user reviews the known multi-stage loss landscape; the agent
  still stops on invalidity such as NaNs, wrong data/config, or a broken run;
- use the 500k checkpoint as primary unless it is technically invalid, then
  seal its hash.

Validation reports native correct context, no context and shuffled context. It
also reports zero-weight-normalized operator distortion, structural components
and latent health. These are validity diagnostics, not an early-stop selector
or a downstream success claim; the technically valid step-500k state is the
primary checkpoint.

## 7. Stage D: build `z_enc` and task-fitted `z_task`

### D1. Encoding

Freeze and hash both codec checkpoints. Encode every supported tile with its native
context, retaining checkpoint/layer/tile order and hard metadata. Store latent
normalization statistics from the training split only.

Build the analogous WeightCLIP record from the same checkpoint/split manifest:
`z_enc_wc` is produced by the frozen released WeightCLIP encoder. This is
required for a matched flow in their latent space; their released mapper alone
is not an adequate control for the choice of generative substrate.

### D2. Joint checkpoint latent fitting

For one checkpoint at a time, run task-latent construction separately through
each frozen decoder (`ours` and `WeightCLIP`) with the same task data, head/BN
policy, optimizer-step budget and validation rule:

1. initialize all body tile codes from `z_enc`;
2. freeze the decoder and every model parameter except latent codes;
3. assemble the generated ResNet in topological order;
4. for our decoder, recompute activation context from the current generated
   prefix and stop gradient through that context path into earlier layers;
5. for WeightCLIP, use its native context-free decoder; do not fabricate an
   activation input it was never trained to consume;
6. optimize all checkpoint codes jointly on native task cross-entropy;
7. use a fixed source train/validation split, save best validation codes and
   the complete trajectory diagnostics;
8. store the full code collection as one grouped checkpoint record.

WeightCLIP-specific controlled semantics are stricter than this generic list.
Use the pinned official sparse/full-model `WindowedDataset` with complete,
padded, consecutive, non-overlapping windows no longer than 512 tokens. Derive
the body mask from its ordered layer IDs. Optimize
`z_task_wc = z_enc_wc + body_mask * delta`, where body means convolution and BN
affine. Head, padding and all other rows remain bitwise `z_enc_wc` with zero
gradient. All windows still participate in encoder/decoder attention and flow
records; they are not dropped to improve the effective rate. Store both native
full-window and controlled body-effective rate ledgers.

Our decoder/context path and the WeightCLIP decoder path are separately
microbatched/vectorized; full checkpoint graphs are not rebuilt in Python every
step. A functional model adapter and cached index perform scatter/gather.

First benchmark fitting cost on 8--16 lineages. Then fit one terminal checkpoint
per independent lineage (500 targets) in parallel on the available compute
queue. The second terminal checkpoint is added only if the measured flow data
requirement justifies the compute; if added, it retains lineage grouping and
total lineage weight one. This avoids paying twice for near-duplicate task fits
by default.

Each fit must beat its own `z_enc` start on source validation task loss without
nonfinite codes. Failed fits remain in the manifest and are not silently
filtered by final performance.

## 8. Stage E: train the matched conditional flow-matching priors

### E1. Shared conditioning

All flows consume the same available conditioning fields:

- frozen WeightCLIP dataset embedding;
- operation type;
- input/output sizes and convolution kernel/stride when applicable;
- normalized topological depth;
- residual branch/role;
- row/column tile indices and tile-grid sizes.

No graph encoder and no activation context enter the first flow. Activation
context is decoder-side only during sequential materialization.

Reuse the tested FiLM/cross-attention transformer blocks from the existing
latent diffusion prior, replacing its noise schedule/objective with a velocity
field. Normalize latent dimensions using training-only robust statistics and
store the transform with the flow checkpoint.

Train the same flow family separately in the two incompatible latent spaces:

1. ours: `N(0,I) -> z_task_ours`;
2. WeightCLIP: `N(0,I) -> z_task_wc`;
3. ours oracle: `z_enc_ours -> z_task_ours`;
4. WeightCLIP oracle: `z_enc_wc -> z_task_wc`.

The core flow parameter budget, conditioning access, number of target records,
optimizer steps and NFE are matched. Input/output adapters necessarily differ
with latent shape and are counted explicitly. This comparison tests the latent
substrate under the same learned sampler rather than comparing our flow to
WeightCLIP's simpler released mapper.

### E2. Anchor-free flow

For target code `z1 = z_task` and base `z0 ~ N(0,I)`:

```text
z_t = (1-t) z0 + t z1
v_target = z1 - z0
L_FM = E ||v_theta(z_t, t, condition) - v_target||^2
```

Train tile records while batching/splitting by parent checkpoint lineage. The
independent tile prior is conditioned on exact position/role metadata, so it is
not an unconditional bag-of-tiles prior.

### E3. Oracle anchor-conditioned flow

Use paired codes from the same checkpoint/tile:

```text
z0 = z_enc
z1 = z_task
```

No artificial noise is added to `z_enc`. A checkpoint-level pairing assertion
is fatal if dataset, lineage, layer or tile IDs differ.

For paired WeightCLIP transport, mask the learned velocity to the controlled
body rows and project the sampled endpoint to
`anchor + body_mask * (endpoint - anchor)`. Assert bitwise equality to the clean
anchor on every non-body row. Anchor-free WeightCLIP flow still generates the
complete valid official token sequence.

### E4. Training and sampling

Implementation handoff (2026-08-18, flow/E4 code-complete but not scientifically
executed): flow training preserves an effective batch of 256 for all four arms;
codec-specific microbatch/gradient-accumulation choices are permitted only when
their product remains exactly 256, and resume is at optimizer-step boundaries.
The E1 condition schema is semantic rather than codec-native (operator type,
kernel/stride, canonical graph depth/residual role, and conceptual tile
coordinates/grid). Prompt-candidate banks and their hashes must match across
ours/WeightCLIP before training.

E4 is an explicit three-stage pre-OOD gate. First, each of the four unsealed EMA
flows runs the complete pooled source-validation sweep Euler/Heun x
`{4,8,16,32}`, with decoded health at `t={0,.25,.5,.75,1}`; the held-out
inventory must exactly match the sealed balanced 10-dataset x 7-lineage
manifest. Second, `training.weightclip_benchmark.select_flow_e4` chooses **one
common** solver and step count for all four arms using mean endpoint RMSE after
a four-arm finite decode gate (and records actual NFE, including two evaluations
per Heun step). Third, each flow is immutably sealed against its provisional
sweep and the same global selection. Runtime reload re-hashes both reports.
These are protocol guards only; no benchmark result has yet been established.

- BF16, fused AdamW, EMA, exact resume and the same runtime telemetry standard;
- validation by held-out lineage, not held-out tiles;
- log velocity MSE, endpoint error, per-role error, latent norms, decoded
  intermediate health, tiles/sec and generation NFE;
- fixed-step Euler/Heun sweep `{4, 8, 16, 32}` on the pooled validation
  lineages from all ten source datasets;
- choose one solver/NFE before OOD unsealing;
- inspect decoded points at `t={0,.25,.5,.75,1}`. If straight paths traverse
  decoder-dead regions, enable the already-implemented alternative
  rectified/OT coupling as a planned fallback, not post-hoc target tuning.

Do not add decoder/task loss to the first flow objective. It would make compute
and target access harder to compare. Decoder-aware flow loss is a later
ablation if plain flow matching fails despite healthy `z_task` targets.

## 9. Stage F: sequential model materialization

Build the known ResNet graph in topological order. At each body operation:

1. the selected codec's flow samples the latent from dataset + hard
   architectural metadata;
2. our decoder receives that latent and activation context produced by the
   already generated prefix; WeightCLIP remains context-free and uses our
   multi-window extension over the official tokenizer/512-token codec;
3. decoded tiles are assembled into the operation tensor;
4. execute the operation and continue through the graph.

Residual branches follow graph topological dependencies, not state-dict list
order. Independent tile flow samples are allowed in the first implementation,
but all tiles of one layer are decoded in one vectorized call. The classifier
head is default-random; all BN state is default for our method.

WeightCLIP has two non-interchangeable materializers. The controlled
materializer transfers decoded conv and decoded BN affine, installs a fresh
paired random head, resets BN running statistics and optionally updates those
statistics with the common calibration batches. Its `z_task` critic uses the
source head and source BN running buffers but leaves decoded BN affine active.
The native materializer loads the complete state decoded by this extension,
including head and BN. Native and controlled results must never be merged into
one arm. “Native” here describes key policy, not a released multi-window mapper.

The released direct/memory/retrieval script maps only one 512-token window.
Therefore the exact released path is retained as a clearly labeled single-window
diagnostic, and an exact-paper full-checkpoint comparison is unavailable. Our
honest full-checkpoint extension stores a grouped source-code bank
`Y=[N, 25*512, 192]`: ridge/memory fitting is global over the whole flattened
checkpoint code and retrieval returns a whole checkpoint code, never 25
independent reuses of the released first-window mapper. The native-window codec
then preserves window order, decodes each <=512-token window, concatenates token
windows, and calls `tokenizer.detokenize` once. This is a **WeightCLIP
multi-window extension**, not the released baseline; its full/native and
body-effective rate ledger must be reported.

## 10. Stage G: common evaluation and baselines

### G1. Primary arms

1. scratch/default initialization;
2. exact released WeightCLIP single-window direct/memory/retrieval diagnostics
   (never presented as full-checkpoint baselines);
3. our WeightCLIP multi-window direct extension;
4. our global full-checkpoint memory-bank/retrieval extensions over grouped
   `25*512*192` codes, followed by native-window decoding and one final ordered
   detokenization;
5. matched-retrained WeightCLIP versions of the selected released modes only
   if released-checkpoint calibration fails;
6. WeightCLIP + matched anchor-free flow to `z_task_wc`;
7. AE reconstruction `decode(z_enc)` diagnostics for both codecs;
8. ours anchor-free flow to `z_task_ours`;
9. oracle target-trained anchor untouched;
10. WeightCLIP oracle flow `z_enc_wc -> z_task_wc`;
11. ours oracle flow `z_enc_ours -> z_task_ours`.

The WeightCLIP-native candidate-selected result is separate from the controlled
matched-candidate table. In the native table only, preserve the released
decoder's classifier-row adaptation and test-selected top-5 behavior. Label it
`native released protocol / test-selected oracle`; do not use it as the only
architecture-superiority comparison.

Additionally, run a matched paper-style oracle table for every generative arm:
generate 100 candidates, rank them by target-test accuracy, and report the top
five before and during fine-tuning. This table is mandatory because sample
quality is high-variance, but it is explicitly labeled as test-selected and is
kept separate from both the single-sample primary and validation-selected
controlled table.

For the oracle arms, train target anchors with the same ResNet18-Slim zoo recipe
using only the target training split. Use one independently trained anchor per
paired evaluation seed, keep every anchor completely out of AE/flow training,
and report its untouched curve. These checkpoints are inputs to the oracle arm,
not extra candidates for the anchor-free arm.

### G2. Minimal causal ablations

Only after the primary path runs:

- correct decoder activation context vs zero/no context vs shuffled context;
- `z_task` flow target vs plain `z_enc` target;
- flow matching vs matched-budget diffusion, only if time permits;

Do not train a weight-only 700M AE, graph encoder, full-matrix AE or joint
full-checkpoint prior before the primary result. Those are high-cost secondary
experiments.

### G3. Downstream protocol

- official target train/test data preprocessing and released frozen dataset
  encoder;
- common random-head initialization seed paired across methods;
- common 200-training-batch BN calibration schedule; our BN affine/state starts
  from architecture defaults; controlled WeightCLIP keeps decoded BN affine but
  resets/calibrates running state; native WeightCLIP retains the complete
  released decoded BN/head policy in its separate audit table;
- common optimizer, LR, augmentation, batches and epoch schedule;
- epoch 0 diagnostic, every-step/epoch curve, early AULC primary, epoch 1 and
  epoch 10 reported for WeightCLIP comparability;
- at least paired seeds sufficient for uncertainty, with bootstrap unit equal
  to target dataset/run rather than tile;
- primary single random sample; mean-K and validation-selected best-K secondary;
- report generation and selection compute.

The review script checks missing rows, duplicate seeds, mismatched heads/BN,
unequal candidate counts, stale caches, NaNs, impossible accuracies and unreadable
plots before producing a summary.

## 11. Tests required before any expensive run

Unit/integration tests must cover:

- official WeightCLIP load and deterministic dataset embedding;
- ResNet18Slim state-key and parameter-count parity;
- conv im2col numerical equivalence to `Conv2d`;
- BN exclusion: no BN key enters our tile bank, and materialization reproduces
  exact declared default affine/running state;
- tile split/assemble exact round trip and full state-key coverage;
- lineage split leakage and weighted sampling;
- raw-X versus precomputed-distribution-stat equivalence;
- old versus refactored AE losses on a pinned batch;
- AE eager versus compiled forward/backward tolerance;
- actual AE identity: no posterior heads, no stochastic sampling;
- checkpoint/resume bitwise next-batch identity where deterministic;
- task-fit gradient stops through activation context but reaches every latent;
- train-only context provenance and the invariant that later train/validation/
  test images cannot trigger weight re-decoding;
- WeightCLIP multi-window-extension ResNet tokenizer round trip, complete window
  ordering, global whole-checkpoint mapper/retrieval semantics, one final
  detokenization, and no codec sequence longer than 512;
- WeightCLIP body-mask exactness: non-body `z_task` and paired-flow endpoint are
  bitwise their corresponding clean anchors with zero optimization update;
- controlled/native WeightCLIP key policy and full/native versus body-effective
  rate ledger;
- same-checkpoint anchor pairing assertions;
- flow objective on analytic Gaussian/translation problems;
- Euler/Heun endpoint convergence;
- random classifier head and identical BN policy across arms;
- complete run artifact/provenance schema with secrets redacted.

Smoke and overfit tests validate code paths only. They are never reported as a
scientific baseline.

## 12. Launch DAG and stop/go gates

```text
pin official code + build common contract
                 |
          reconstruct zoo (local H100, 2--4 workers)
                 |
      build operator/context banks
                 |
 exact 700M profile -> train AE (local H100)
                 |
         seal AE checkpoint hash
                 |
 encode/fix z targets in both codec spaces
             /                 \
   our matched flows      WeightCLIP matched flows
             \                 /
       sealed source/meta-validation review
                         |
               unseal six OOD datasets
                         |
          fast screening + artifact review
                         |
      released WeightCLIP paper-metric calibration
                         |
        calibration passes -> confirmatory benchmark
        calibration fails  -> matched WeightCLIP retrain
                                      |
                            confirmatory common benchmark
```

Gates:

1. **Contract gate:** official bridge, head/BN policy and lineage split tests
   pass.
2. **Data gate:** manifest integrity and no-duplication/equivalence tests pass.
3. **Throughput gate:** production-shaped H100 loader/profile thresholds pass.
4. **AE gate:** stable source/meta-validation losses and healthy latents; user
   approves the known loss-landscape trajectory.
5. **Task-fit gate:** `z_task` improves its own source validation loss over
   `z_enc` for a meaningful cross-dataset set; otherwise flow has no useful
   target and must not be blamed.
6. **Flow gate:** held-out-source endpoint/decoder health and solver choice are
   frozen before OOD.
7. **Evaluation gate:** paired head/BN/candidate/compute checks pass before any
   performance claim.
8. **Baseline calibration gate:** a released-model transfer win becomes the
   primary baseline comparison only if the full per-dataset epoch-0/1/10 table
   reproduces the paper within measured seed/candidate variation. Otherwise it
   stays screening evidence and the claim waits for matched retraining.

## 13. Compute-minimization choices

- Reconstruct the zoo once and use it for ours and released WeightCLIP.
- Store activations by shared context ID rather than per weight tile.
- Fit one terminal checkpoint per independent trajectory first; add nearby
  epochs only if flow data volume is the measured bottleneck.
- Train one production AE, not a grid of 700M models. Do not run a short LR
  sweep that cannot see the delayed loss transitions.
- Train the matched anchor-free/oracle flows for both codec spaces; do not add graph/joint/diffusion
  variants until the primary source gate passes.
- Implement but do not immediately launch the matched 700M WeightCLIP retrain;
  trigger it only if released-checkpoint paper-metric calibration fails.
- Use pooled, dataset-balanced source validation across all ten datasets for
  every tunable choice, then freeze before the internal source test and six OOD
  datasets.
- Cache dataset embeddings, encoded latents, task-fitted latents and evaluation
  data by content hash; every cache hit is logged.

## 14. Expected order after approval

Implementation is split across agents only after user approval, but merged in
dependency order:

1. official WeightCLIP bridge + benchmark contract + evaluation harness;
2. zoo/activation/operator dataset and storage pipeline;
3. 700M AE config derivation, trainer validation and throughput telemetry;
4. functional ResNet + task-latent fitting;
5. flow matching, sequential materializer and launchers;
6. independent integration/fairness review.

All code is implemented and tested before the first long production launch.
Long runs are still launched in the DAG above: implementing everything up front
does not justify spending downstream compute before upstream gates pass.

## 15. Protocol audit status

The paper/repository audit found several nontrivial conflicts rather than an
exact executable paper protocol: 20 lineages with five terminal snapshots
versus 50 lineages with two terminal snapshots (all lineages still train for 45
epochs); paper
Lego Bricks versus repository ASL; paper classification annealing 600 versus
ResNet config 0; random checkpoint split with lineage leakage; and test-label
selection of the top five generated candidates. The exact evidence and chosen
resolution for each are frozen in `weightclip_protocol_audit_20260818.md`.

The most important consequence is that results are never collapsed into one
ambiguous table. The native-fidelity table follows released behavior, including
test-selected top-5 and native head/BN handling. The controlled architecture
table uses a random full head, paired access, and validation-only selection.

## 16. Approval points

The plan makes three remaining concrete choices that require explicit user
acceptance:

1. train the primary AE on the reconstructed WeightCLIP ResNet population,
   rather than the old heterogeneous 300 GB vision bank;
2. freeze square `128x128` after its production-shaped throughput profile;
   use `128x64` only as an explicitly approved infeasibility fallback;
3. fit `z_task` first for one terminal checkpoint per 500 independent lineage,
   rather than paying immediately for the second nearby checkpoint.

The estimator policy is no longer an approval point: the controlled table uses
single-sample expectation as primary, and the paper-style test-selected
`100 -> top 5` experiment is mandatory in its own oracle-labeled table.

Everything else above implements decisions already fixed in Bible Section 19
or validity/throughput requirements needed to execute them honestly.

## 17. Executable handoff and current status

This section is an operational checklist, not a result section. No production
AE, task-fit, flow or downstream benchmark has been run by this implementation
handoff.

### 17.1 Code status

- Official bridge, contract/evaluation harness, zoo reconstruction, operator
  bank, exact AE profiler/launcher gate, functional ResNet, grouped `z_task`,
  four flows, sealed EMA loading and controlled/native materializers are
  implemented.
- Operator-bank training is deterministic by committed logical sample index.
  On resume, the sampler starts from the last committed optimizer batch, so
  prefetched but uncommitted worker items are regenerated rather than skipped.
  Full optimizer/scheduler/scaler/RNG/step restoration is enforced.
- Decoder activation context is a fixed manifest-recorded train-only pool.
  Materialization and validation use static cached weights; evaluation inputs
  cannot silently recondition the decoder.
- The production launcher remains approval-gated. The current unselected,
  content-addressed meta-profile is
  `candidate_profiles-3c18deaa8d067dd1/`: artifact-set SHA
  `3c18deaa8d067dd159aa410ff5d234781eed83f79968b12977ed1cb799379589`,
  report SHA
  `f30bef12e0815cc8b27e8ba7c4996cca9320ae055a7cb96c1514500baf91e5d9`,
  artifact-index SHA
  `cde4ac76d706bc21802da7a452dddc6fe7bc4bdf902c3e8b410bcdea73469528`,
  summary-CSV SHA
  `9c30a57e4fde2654ee31acdf113312833745b4c2ac925c52c03530bc4d997d0b`,
  and 158-file source seal
  `7a109801bd0a5b2bcb004b69fa297bfb0fb100bdcb1bc39ae3eccd20e0e34152`
  (seal-artifact SHA
  `464ec0cd65ac1793e9cac45fe1cd2b1e848244cb158f56b9c742ee71cea9349e`).
  Supersession-ledger SHA is
  `c6c07caddbcb6e6603e2c7df2d13e5b71d52952e96ff50ff0d9cf0e81a587560`.
  Two consecutive profiler invocations returned exactly these hashes. Legacy
  root artifacts and earlier bundles are superseded with per-artifact reason
  codes. The previous `85e1...` bundle is source-seal stale. This includes
  `candidate_profiles-133b41aa341dc5e4`, whose old broad secret matcher
  redacted scientific `patch_tokenizer` fields. The corrected stored report
  contains no `<redacted>` scientific fields and recomputes exactly to its
  artifact-set SHA; launch-time index validation repeats that recomputation,
  and the runtime profiler validates the complete worker summary before it can
  print completion. This is not a candidate selection, user approval, bounded
  GPU profile, or scientific result.

  The preceding `647a...` bundle is source-seal stale: the bounded legacy run
  revealed that `worker.py` referenced `_prepared_batch_prefetch_blockers`
  through a wildcard aggregator whose `__all__` omitted it. The worker now has
  an explicit import, and the regression traverses real operator-bank
  background prefetch plus one CPU optimizer step.

  The subsequent `b76b...` bundle is also source-seal stale. A real launcher
  proved that runtime artifact setup rewrites `logging`, `training_artifacts`,
  and eleven enumerated output-path interpolations, while immutable JSON stores
  `hf.token` redacted. Active/stored comparison now symmetrically applies the
  shared credential redactor and removes only those operational paths; model,
  loss, LR, optimizer, scheduler, and data changes remain strict. A mismatch
  produces a secret-safe path/type/value-hash report.

  The following candidate count/fingerprint pairs describe the superseded
  first runtime panel: `legacy_ratio...` 701,967,641 /
  `a2314118877304c10b11d32c8a707bd8081ac6d0cc459ad81ea5711c88cdc503`;
  `deep_640...` 700,925,473 /
  `e9e913bf71daad3f36e7a6073933528a5d3ea891b2718b6814c995183de4d9d9`;
  `decoder_heavy...` 705,873,871 /
  `6f00b4848889d78ab167d92ab2c58ffe30d509440a735a6c9e3f9b7eca3b9fd7`;
  and `latent_800...` 701,172,373 /
  `fdee8e4a29dd35faf7a1c00c82469e78e93c8e004931957141c1d4fce0058144`.

  The replacement grid holds latent capacity fixed at `32 x 400 = 12,800`
  scalars for all four variants. Using the actual 193-tile operator-mask
  distribution, its effective rate is `1.1294689`, versus WeightCLIP's
  padding-adjusted `1.1383923` (relative difference `-0.78%`). Exact trainable
  counts are 698,749,881 (balanced), 702,148,225 (deep), 704,397,551
  (decoder-heavy), and 700,959,053 (FFN-heavy). Old runtime profiles are
  historical only; the replacement grid is not approved.

### 17.2 Current data stage

At the read-only `2026-08-19T01:05:31Z` snapshot, all ten source `dataset.pt`
files were present and 400/500 lineages (eight complete datasets) had all 45
epochs. The exact final 500-lineage audit, immutable final checkpoint manifest
and final operator bank were still incomplete. The OOD materializer has run and
written its verified `SEALED_NOT_EVALUATED` data artifacts; this is data
preparation, not an OOD evaluation or scientific result. Process IDs and live
progress are intentionally not treated as durable completion evidence.

### 17.3 Commands in dependency order

Run and review each stage before starting the next:

```bash
# 1. Official archive -> ten dataset.pt files (complete at file level; inspect manifest).
python -m training.weightclip_benchmark.build_zoo materialize \
  --config conf/weightclip_benchmark/zoo.yaml

# 1b. Data preparation only: verified raw_m_test stream + torchvision CIFAR10.
# This writes SEALED_NOT_EVALUATED metadata and does not run OOD evaluation.
python -m training.weightclip_benchmark.build_zoo materialize-ood \
  --config conf/weightclip_benchmark/zoo.yaml

# 2. Published LR sweep for all source datasets.
python -m training.weightclip_benchmark.build_zoo sweep \
  --config conf/weightclip_benchmark/zoo.yaml

# 3. Fifty 45-epoch lineages per dataset with rolling resume/full FP32 archive.
python -m training.weightclip_benchmark.build_zoo train \
  --config conf/weightclip_benchmark/zoo.yaml

# 4. Immutable lineage-safe manifest and forward audit.
python -m training.weightclip_benchmark.build_zoo manifest \
  --config conf/weightclip_benchmark/zoo.yaml

# 5. Train-only activation/operator banks, then reader-only I/O profile.
python -m training.weightclip_benchmark.build_operator_dataset build \
  --config conf/weightclip_benchmark/operator_dataset.yaml
python -m training.weightclip_benchmark.build_operator_dataset profile \
  --config conf/weightclip_benchmark/operator_dataset.yaml --steps 500

# 6. Recompute exact current AE candidates/fingerprints (meta device only).
python -m training.weightclip_benchmark.profile_ae \
  --config conf/weightclip_benchmark/ae_700m.yaml
```

Before requesting production approval, run the bounded *real worker-path*
profiler only after the final bank exists and the H100 is idle. It retains the
500,000-step scheduler horizon but is hard-capped to at most 32 optimizer steps,
cannot resume/write checkpoints/use external tracking, and consumes the exact
bundle-worker -> stratum-preserving mixer -> batch32 -> pin/prefetch -> five
tensor H2D -> production loss path:

```bash
python -m training.weightclip_benchmark.profile_ae_runtime \
  --config conf/weightclip_benchmark/ae_700m.yaml \
  --candidate <CANDIDATE> \
  --mode production_exact \
  --candidate-artifact-index <candidate_profiles-.../artifact_index.json> \
  --pair-manifest <final-operator-bank-pair.json> \
  --warmup-steps 4 --measured-steps 28 --execute
```

The 32 consumed optimizer steps are a hard cap, not the producer request cap.
With the exact production queue size 24, profile mode alone requests through
step 84 (`32 + 24 + 28`) so the measured phase must observe concurrent refill.
Only steps 0--31 may commit; queued tail batches are discarded on close and do
not alter the deterministic consumed prefix or resume state. The summary must
show an advancing measured producer cursor and final produced cursor beyond
the committed cursor, plus explicit consumed/produced/discarded, queue,
input-wait, and producer/consumer rate ledgers. This bounded window is a stall
diagnostic and does not automatically establish no-starvation or a sustained
1.5x ingress margin.

The old real legacy artifact
`legacy_ratio_768_e24_d8_ffn4_l32_20260819T074551Z_pid684058` is diagnostic
only. It completed the former 4+16 worker window, but its producer cursor was
already 640 throughout the measured phase; its preserved summary and timings
therefore do not establish concurrent refill. Its post-worker launcher failure
was caused solely by comparing stored redacted `hf.token` with the raw empty
runtime token. The parity validator now redacts credentials symmetrically and
still rejects model/loss/LR/optimizer/scheduler/data changes.

Review its immutable `source_implementation_seal.json`,
`resolved_profile_config.json`, `profile_contract.json`,
`step_timings.{jsonl,csv}`, `nvml_samples.{jsonl,csv}` and `summary.json`, plus
the immutable operator locality audit beside the pair manifest. The audit
preserves the exact old global dataset/op/depth/role sequence and tile multiset,
but deliberately does not preserve lineage/checkpoint temporal order inside a
stratum. Diversity numbers are evidence for explicit approval, not an automatic
threshold. No production AE launch is allowed from reader-only throughput or
from an unreviewed bounded profile.

The canonical `ae_700m.yaml` deliberately keeps `data_contract.pair_manifest:
null`; neither profiling nor production mutates that pending scientific config.
Both CLIs require the finalized pair explicitly. The bounded profile contract
and worker summary bind the pair SHA together with the content-addressed
candidate artifact-set, report, index and current source seal. A stale candidate
bundle or builder-source seal fails before checkpoint/model work.

After step 6, stop and obtain explicit user approval for exactly one new
fingerprint. Never edit a candidate pending JSON or the canonical
`ae_700m.yaml` by hand. First create and validate the non-authorizing pending
request described in section 17.5. After the user supplies an explicit approval
statement, the sole transition is the source-sealed materializer below. It
writes an immutable approved JSON config and approval record *beside* the
canonical YAML so every relative path retains exactly the same meaning. The
approved config may differ from the request-bound canonical mapping only in
`status`, `production_launch_enabled_after_approval`, `selected_candidate`, and
`approval_file`.

```bash
python -m training.weightclip_benchmark.approve_ae \
  --approval-request <PENDING_REQUEST.json> \
  --canonical-config conf/weightclip_benchmark/ae_700m.yaml \
  --approved-config conf/weightclip_benchmark/ae_700m.approved-<REQUEST_SHA_PREFIX16>.json \
  --approval-file conf/weightclip_benchmark/ae_700m.user-approval-<REQUEST_SHA_PREFIX16>.json \
  --user-statement 'I explicitly approve WeightCLIP AE production candidate <CANDIDATE> for approval request <FULL_64_HEX_REQUEST_FINGERPRINT>.' \
  --approved-at-utc '<CANONICAL_UTC_TIMESTAMP_ENDING_IN_Z>' \
  --confirm-explicit-user-approval
```

The launcher independently validates the candidate index/report/source seal,
the runtime summary/contract/resolved config, exact parameter count, pair SHA,
and production scientific-config SHA. Missing or mismatched evidence fails
closed. Trailing Hydra arguments are limited to checkpoint output and Comet/W&B
run names; LR, optimizer, scheduler, AMP, accumulation, loss, steps, model and
data overrides are forbidden. Populate the immutable pair-manifest path and
approval fields.
The production launcher must first be exercised without `--execute`; do not run
either form while the config approval flag remains false:

```bash
python -m training.weightclip_benchmark.train_ae \
  --config conf/weightclip_benchmark/ae_700m.approved-<REQUEST_SHA_PREFIX16>.json \
  --candidate <APPROVED_CANDIDATE> \
  --approval-request <PENDING_REQUEST.json> \
  --candidate-artifact-index <candidate_profiles-.../artifact_index.json> \
  --pair-manifest <final-operator-bank-pair.json> \
  --runtime-profile-summary <bounded-profile/summary.json> \
  --producer-stress-summary <producer-stress/summary.json> \
  --approval-file conf/weightclip_benchmark/ae_700m.user-approval-<REQUEST_SHA_PREFIX16>.json \
  --launch-mode fresh \
  --checkpoint-root <UNIQUE_NONEXISTENT_CHECKPOINT_ROOT>

# Add --execute only after the dry-run contract, throughput gate and explicit
# user approval have all been reviewed.
```

After the approved 500k run completes, seal that exact checkpoint.  The seal
command rejects a checkpoint whose embedded resolved config/preflight/source
seal or exact state-dict key/shape/dtype schema differs from the approved
candidate; launcher-created `logging` and `training_artifacts` paths are the
only normalized runtime fields:

```bash
python -m training.weightclip_benchmark.seal_ae \
  --config conf/weightclip_benchmark/ae_700m.approved-<REQUEST_SHA_PREFIX16>.json \
  --candidate <APPROVED_CANDIDATE> \
  --approval-request <PENDING_REQUEST.json> \
  --approval-file conf/weightclip_benchmark/ae_700m.user-approval-<REQUEST_SHA_PREFIX16>.json \
  --checkpoint <COMPLETED_500K_CHECKPOINT.pt> \
  --resolved-model-config <PRODUCTION_RESOLVED_CONFIG.json> \
  --output <IMMUTABLE_AE_SEAL.json>
```

Post-AE stages require prepared, content-addressed bundles and sealed
checkpoints; placeholders below are intentionally explicit:

```bash
python -m training.weightclip_benchmark.fit_task_latents \
  --config conf/weightclip_benchmark/task_latent_fit.yaml \
  --bundle <OURS_GROUPED_CHECKPOINT_BUNDLE.pt>

python -m training.weightclip_benchmark.fit_weightclip_task_latents \
  --config conf/weightclip_benchmark/task_latent_fit_weightclip.yaml \
  --bundle <WEIGHTCLIP_GROUPED_CHECKPOINT_BUNDLE.pt>

for cfg in \
  conf/weightclip_benchmark/flow_ours_anchor_free.yaml \
  conf/weightclip_benchmark/flow_ours_oracle_anchor.yaml \
  conf/weightclip_benchmark/flow_weightclip_anchor_free.yaml \
  conf/weightclip_benchmark/flow_weightclip_oracle_anchor.yaml; do
  python -m training.weightclip_benchmark.train_flow --config "$cfg"
done

python -m training.weightclip_benchmark.evaluate \
  --config conf/weightclip_benchmark/evaluation.yaml \
  --candidate-manifest <SEALED_CANDIDATES.jsonl> \
  --output-dir <EVALUATION_OUTPUT_DIR>
python -m training.weightclip_benchmark.review \
  --metrics <EVALUATION_OUTPUT_DIR/metrics.jsonl> \
  --output-dir <REVIEW_OUTPUT_DIR>
```

### 17.4 Runtime stop checklist

- [ ] Ten materialized datasets exist and hashes/preprocessing were inspected.
- [ ] Six OOD `dataset.pt` files have exact official slugs/class counts and a
      `SEALED_NOT_EVALUATED` manifest; their test data remain unevaluated.
- [ ] Zoo contains the expected 500 lineages, 1000 primary checkpoints and no
      cross-split lineage/hash duplication.
- [ ] Operator/context pair manifest is immutable, train-only and checksum-valid.
- [ ] Actual AE loader profile meets the input-wait/GPU-utilization gate.
- [ ] Current profiler was rerun; old synthetic fingerprints are not reused.
- [ ] User approved one exact candidate/fingerprint; dry-run resolved contract
      matches it byte-for-byte.
- [ ] Official WeightCLIP codec (~9 GB), dataset encoder and pinned checkout are
      locally available and hash-verified.
- [ ] Task bundles contain train-only context provenance and one position-aware
      architecture row per tile/window.
- [ ] Task-fit runtime decoder checkpoint and resolved codec-config hashes match
      the hashes stored when each bundle was built; ours and WeightCLIP each use
      one identically sized deterministic task minibatch per optimizer step.
- [ ] Each `z_task` improves its own validation loss before any flow is blamed.
- [ ] Flow EMA seal, normalizer, codec/path kind, anchor contract and solver/NFE
      gate pass before OOD evaluation.
- [ ] Grouped records, normalizer, flow checkpoint, v2 seal and loaded decoder
      all carry/recompute the same canonical codec fingerprint; an A→flow→B
      negative test rejects before sampling.
- [ ] Native and controlled WeightCLIP rows remain separate; no scientific claim
      is made until reviewed downstream artifacts exist.

### 17.5 Frozen two-mode AE runtime confirmation (2026-08-19)

The selected AE candidate now requires two separately immutable bounded worker
profiles before an approval request can be formed. Both consume exactly four
warm-up plus 28 measured optimizer steps and retain the production 500,000-step
scheduler horizon. `production_exact` preserves the production prefetch queue
of 24. `producer_stress` changes only that operational queue to four and must
pass all frozen ingress gates: measured producer advance at least 832 tiles,
producer/consumer span ratio at least `26/27`, input-wait fraction at most 1%,
and input-wait p95 at most 5% of median host-step time. Both modes retain exact
candidate, pair, artifact-set/report/index, source, config, launcher-result and
redacted immutable worker-log bindings.

```bash
python -m training.weightclip_benchmark.profile_ae_runtime \
  --config conf/weightclip_benchmark/ae_700m.yaml \
  --candidate <CANDIDATE> --mode production_exact \
  --candidate-artifact-index <CURRENT_INDEX> --pair-manifest <CURRENT_PAIR> \
  --execute
python -m training.weightclip_benchmark.profile_ae_runtime \
  --config conf/weightclip_benchmark/ae_700m.yaml \
  --candidate <CANDIDATE> --mode producer_stress \
  --candidate-artifact-index <CURRENT_INDEX> --pair-manifest <CURRENT_PAIR> \
  --execute
```

The historical q24 panel is preserved, not promoted to current approval
evidence, in content-addressed report
`comparisons/comparison-f03faa46a939b1c4/comparison.json` (SHA-256
`9dd055009c682955de5292a50ac415b4ec09e75ce4548484c16c0dd43096594a`).
It records legacy 1.137018855 step/s at 78,039 MiB NVML peak, deep
0.939292929 step/s at 93,071 MiB, decoder OOM in the first forward, and latent
OOM in the first backward. The exact config/crash/launcher evidence and success
chains are copied byte-for-byte into read-only snapshots. This yields a
recommendation for legacy (21.05% faster; deep/legacy rate ratio about 0.8261),
but records no user approval and authorizes no production launch. Only legacy
meets the frozen `rate >= 0.95 * fastest` rule for the current q4 confirmation.

After both *current-source* legacy summaries exist, create the still-pending,
non-authorizing request with:

```bash
python -m training.weightclip_benchmark.prepare_ae_approval_request \
  --config conf/weightclip_benchmark/ae_700m.yaml \
  --candidate legacy_ratio_768_e24_d8_ffn4_l32 \
  --candidate-artifact-index <CURRENT_INDEX> --pair-manifest <CURRENT_PAIR> \
  --production-exact-summary <Q24_SUMMARY> \
  --producer-stress-summary <Q4_SUMMARY> \
  --historical-comparison <COMPARISON_JSON> --output-root <REQUEST_ROOT>
```

The request validator reconstructs the Hydra production config, candidate and
parameter count, current source/index/report seal, pair, both runtime modes and
historical comparison. It cannot flip `approved_by_user` or authorize launch;
production still requires a separate exact user-approved JSON. Both new chain
CLIs are included in the 158-file source seal of the current
`candidate_profiles-3c18deaa8d067dd1` bundle above; two meta-only invocations
were idempotent. No GPU run, bank build or user approval was performed by this
implementation pass.

### 17.6 Fail-closed user-approval transition and disk gate (2026-08-19)

`prepare_ae_approval_request` remains non-authorizing. The source-sealed
`approve_ae` CLI is the only supported pending-to-approved transition. It
requires a non-empty caller-supplied user statement, an explicit confirmation
flag, and a timezone-bearing timestamp. It fully recomputes the pending request,
then emits a separate immutable approved config and approval record; it never
mutates `conf/weightclip_benchmark/ae_700m.yaml`.

`train_ae` now requires `--approval-request` and rejects any `--config` other
than the approved config bound by that transition. Candidate, approval file,
pair, candidate index, q24 summary and q4 summary paths must be the exact paths
bound by the request. Launch preflight and therefore every production checkpoint
bind request SHA/fingerprint, canonical and approved config SHAs, approval-file
SHA, candidate-index SHA, both profile SHAs, pair/source/model/scientific hashes,
and a disk-preflight report SHA. `seal_ae` revalidates the same external chain,
the embedded checkpoint preflight, and the disk report before sealing.

The disk report assumes FP32 model tensors, two FP32 Adam moments, all 50
numbered 10k checkpoints, `latest.pt`, one rolling resume state, the worst atomic
temporary copy, 5% serialization overhead, and extra free-space headroom
`max(20%, 20 GiB)`. Dry-run records insufficiency; `--execute` fails closed when
observed free bytes are below the conservative bound. It never deletes or
reclaims storage.

Fresh launches require `--launch-mode fresh` and a unique nonexistent
`--checkpoint-root`; dry-run does not create it, while execute atomically claims
the root, creates the exact nested resume root, acquires a nonblocking process
lease before any mutable-root validation, and under that lease rechecks
filesystem identity/free space, the exact resume SHA, every archive SHA/stat,
and the complete numbered-checkpoint path inventory. The lease is held through
the worker subprocess. Directory-scanning
auto-resume is forbidden. A continuation instead uses:

```bash
python -m training.weightclip_benchmark.train_ae \
  <THE SAME APPROVAL/REQUEST/INDEX/PAIR/Q24/Q4 ARGUMENTS> \
  --launch-mode resume \
  --checkpoint-root <EXACT_EXISTING_CHECKPOINT_ROOT> \
  --resume-state-checkpoint \
    <CHECKPOINT_ROOT>/stage_1/resume_state/step_<ALIGNED_STEP>.pt
```

The resume artifact must be immutable, stable across hashing and loading,
below 500k, aligned to the 10k save cadence, and contain the exact finite
meta-model key/shape/dtype schema, a complete name-bound AdamW moment state for
every trainable parameter, exact LambdaLR/disabled-BF16-GradScaler state,
restorable Python/NumPy/CPU/CUDA RNG, and exact operator-stream contract/cursor.
CUDA RNG state is captured only for the configured active CUDA device and is
bound to its logical device, canonical `GPU-*` UUID/name, and the child
`CUDA_VISIBLE_DEVICES=<GPU-UUID>` mask. The UUID is resolved through Torch's
logical-to-NVML index plus `nvidia-smi`; opaque Torch 2.10 `_CUuuid` object
representations are never serialized. Resume validates the CUDA RNG bytes in a
short-lived helper process masked to that UUID; the launcher itself must remain
CUDA-uninitialized and never enumerates or creates contexts on other GPUs. The former all-visible-
device list schema is not a production-resume compatibility path. Both bounded
profiles must name the same physical UUID. Profile and production children are
masked to it; execute holds a per-UUID nonblocking lease and rechecks foreign
compute processes plus profiled peak VRAM and explicit 5%/4096-MiB headroom.
Its embedded prior launch must match candidate/model/scientific/source/pair/
request/config and both runtime-profile bindings. Preflight also hashes and
validates every immutable numbered model checkpoint from 10k through `K` plus
`latest.pt`; their config/provenance/schema must be exact, every tensor finite,
and the final numbered/latest/resume model states bitwise identical. There is
no permissive optimizer reset or directory-scanning fallback. Disk accounting
then charges only remaining numbered checkpoints plus atomic overwrite
transients; retained files are already reflected in observed free space.

This approval-transition change invalidates the previously current 158-file
AE source seal and `candidate_profiles-3c18deaa8d067dd1`; they are historical
until a post-review content-addressed bundle is regenerated. No approval,
production run, GPU profile, or operator-bank build was performed while adding
the transition.
