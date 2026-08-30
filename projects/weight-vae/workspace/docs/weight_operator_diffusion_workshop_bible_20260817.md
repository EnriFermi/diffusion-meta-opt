# Activation-Conditioned Weight Generation Workshop Plan

**Date:** 2026-08-17; canonical decisions updated 2026-08-18.  
**Status:** canonical project direction. Section 19 is the latest decision
ledger and overrides any older wording that was not yet mechanically rewritten.
**Audience:** every agent or researcher who works on the Weight-AE workshop
submission. Read this document before proposing or running experiments.

## Current executable handoff (2026-08-18; no scientific result yet)

The WeightCLIP full-checkpoint route is our explicit **multi-window extension**,
not the released single-window baseline: each official <=512-token window runs
through the learned codec independently, all windows remain in the grouped
record, and one ordered detokenization reconstructs the state. Controlled
generation transfers convolution and BN-affine body values, uses a fresh paired
head, and resets/common-calibrates BN running statistics; native generation
retains the full decoded state. Paired flow preserves every non-body code exactly
at the clean anchor. The E4 validator loads the large learned WeightCLIP codec
once and only retargets the tokenizer per validation group.

Executable finalization order:

1. Run each `decoded_flow_validation_*.yaml` for all four unsealed flow arms;
   each config must name the sealed 10-dataset x 7-lineage validation inventory.
2. Run `python -m training.weightclip_benchmark.select_flow_e4` with the four
   provisional reports to produce one immutable global solver/NFE selection.
3. Run `python -m training.weightclip_benchmark.seal_flow` per arm with both its
   provisional report and the same global selection.
4. Only after all four seals exist and agree may the pre-OOD unseal gate run.

Flow training uses matched effective batch 256; codec-specific microbatches are
ours 64x4 accumulation and WeightCLIP 8x32 unless an explicitly matched config
replaces them. Resume, normalizer, record content hashes, codec fingerprint,
prompt-bank parity and the semantic E1 feature contract are fail-closed. AE
production profiles remain approval-gated. No scientific metric is claimed by
this implementation status.

## 1. Executive decision

The paper's primary contribution is an empirical/system result on conditional
weight generation and initialization: the proposed architecture should beat
WeightCLIP on a controlled ResNet-18 Slim benchmark and later scale to a broad
foundation-style model population. Conditional flow matching is now the primary
generative prior; diffusion is a secondary matched-prior ablation. The
contribution is not a claim that every individual component is unprecedented.

HyperLDM and D2NWG already cover latent diffusion over weights, and D2NWG
already combines a weight VAE, a dataset encoder, and dataset-conditioned latent
diffusion. They are related work and baselines, not reasons to discard a system
that obtains a decisive common-benchmark result.

The primary candidate paper thesis is:

> Our activation-conditioned weight autoencoder and dataset-conditioned latent
> prior provide a better architecture for conditional weight generation than
> existing SANE/WeightCLIP-style systems: under matched data, compute, code-rate,
> and candidate-selection budgets, the system achieves stronger held-out
> initializations and early-training curves.

The architecture is motivated by two design choices whose contributions are
measured through ablations:

1. one fixed-cardinality code for a variable-shape weight matrix instead of a
   token/chunk count that grows with matrix size;
2. activation context in weight encoding and decoding, plus functional
   reconstruction losses.

Neither design choice is the headline by itself. The headline survives only if
the end-to-end metrics are strong. Size generalization, latent organization,
and emergent functional structure are secondary findings.

The first empirical target is to beat released WeightCLIP on a reconstructed
version of its ResNet-18 Slim zoo and downstream initialization/fine-tuning
protocol. The HF Model Hub foundation-scale experiment is a later scale-up,
not the first go/no-go. MultiZoo-SANE is not a primary benchmark. A
leaderboard result is primary, but minimal controlled comparisons are still
required so the paper can attribute the win to its architecture rather than an
unmatched sampler, target-data budget, or best-of-many selection rule. In
particular, the same flow class must be evaluated with the activation ablations,
and any optional SANE comparison must use the same prior and information budget.

## 2. User's intended system

The intended pipeline has three learned components:

1. A Weight-AE/VAE maps a weight object and an activation-derived context to a
   latent code and reconstructs the weight object from the code plus context.
2. The released WeightCLIP dataset encoder maps a small sample from the target
   dataset to a dataset embedding. It is frozen and shared across the primary
   compared systems. Training a new dataset encoder is an ablation, not the
   primary comparison.
3. A conditional flow-matching prior maps either Gaussian base samples or
   encoded oracle anchors to task-fitted weight latents. The latents are decoded
   into a known target architecture and used as an initialization; Epoch-0
   quality and early fine-tuning measure the result. Diffusion remains an
   optional matched-prior ablation, not the default system.

The user's resource and timing constraints are:

- the local NVIDIA H100 NVL 96 GB for the first approximately 700M-parameter
  representation training;
- eight A100 GPUs for later parallel priors, ablations, and evaluation;
- approximately one month total;
- a decisive main-result go/no-go in one week if possible, and no later than
  roughly two weeks;
- avoid retraining the expensive representation repeatedly; perform one
  production representation training after short gates, and parallelize priors
  and ablations around the frozen representation.

## 3. Terminology and exact scope

### 3.1 This paper does not claim operator learning

A matrix defines a linear map and activations define the input measure under
which its reconstruction matters. A useful functional distortion is, for
example,

\[
d_{P_X}(W,\widehat W)
= \mathbb E_{X\sim P_X}\lVert X(W-\widehat W)\rVert_2^2.
\]

However, the current objective combines structural reconstruction of weight
values with functional reconstruction. It does not explicitly impose
invariance/equivariance to function-preserving reparameterizations, contrastive
closeness of functionally similar matrices, or a quotient geometry over weight
symmetries. Therefore the paper must not claim that it learns operators in the
strong representation-learning sense.

Use **activation-conditioned weight autoencoder**, **distribution-aware weight
codec**, or **functional reconstruction loss**. Avoid `operator learning` and
the unqualified term `neural operator`. If functionally meaningful organization
emerges despite the mixed objective, report it as a secondary finding. A true
operator-learning objective with appropriate augmentations is reserved for a
separate project/paper.

### 3.2 Fixed architecture interface versus the current 64x64 training regime

The user's intended architectural claim is now resolved as follows:

- one complete linear weight matrix/operator, of variable rectangular shape,
  is mapped to the same `K x d_lat` latent cardinality;
- the decoder expands that code using dynamic coordinate queries plus declared
  shape/role/context side information;
- `64x64` is the current in-distribution training/quality operating point, not
  an asserted architectural maximum;
- feeding one larger complete matrix through the same encoder/decoder is an OOD
  size-scaling experiment, with graceful rather than invariant quality expected;
- a full model still consists of a variable number of operator codes when its
  layer count varies. The claim is fixed-cardinality **per operator**, not one
  global code for every possible model.

The current *pipeline and stored corpus* nevertheless decompose large matrices
into multiple `64x64` slices. This establishes fixed cardinality per processed
slice, but it has not yet empirically established the stronger one-code-per-full-
matrix claim. The architecture may support dynamic shapes; the evidence must be
obtained by actually encoding complete matrices without post-hoc tiling.

This distinction is central:

- `architectural capability`: dynamic token/query counts with a fixed latent
  resampler can in principle process a full matrix of a new size;
- `current training support`: the model was optimized mainly on `64x64` slices;
- `paper evidence`: must include full-matrix in-support sizes plus larger unseen
  size buckets, using one bitwise fixed-cardinality code per matrix.

No finite code can uniformly reconstruct unconstrained matrices of unbounded
dimension. The honest claim is therefore variable shapes within a declared
support plus a measured rate-distortion degradation curve outside that support,
not size-independent fidelity for arbitrary matrices.

### 3.3 A useful generative substrate must be operationalized

Fixed cardinality is convenient but neither necessary nor sufficient for flow
matching or diffusion. A representation is a better generative substrate only
if, under a matched prior and budget, it improves measurable outcomes:

- held-out latent likelihood or denoising error, used only as a diagnostic;
- sample validity and diversity without nearest-checkpoint memorization;
- decoded functional distortion at a declared code rate;
- Epoch-0 downstream quality;
- early fine-tuning area under the learning curve (AULC);
- sample efficiency as the number of generated candidates varies.

The scientific claim is an empirical advantage on this frontier, not the mere
fact that a flow or diffusion process can be launched.

### 3.4 Primary flow-matching formulation

Two modes are distinct and must never be conflated in reporting.

**Anchor-free flow.** Sample `z_0 ~ N(0, I)` and transport it to the
task-fitted latent distribution conditioned on the frozen WeightCLIP dataset
embedding and architecture/layer metadata. This is the true generation mode:
no trained target checkpoint is available at test time.

**Oracle anchor-conditioned flow.** For a held-out trained checkpoint
`W_anchor` from the target dataset, construct

\[
z_0 = E(W_{anchor}, C_{anchor}), \qquad z_1 = z_{task}.
\]

The flow is trained on paired paths, for example
`z_t = (1-t) z_0 + t z_1`, with velocity target `z_1-z_0`. At test time the
true trained target-dataset checkpoint is deliberately exposed to the method.
This is explicitly an oracle/with-peeking refinement experiment, not
anchor-free generation. The untouched anchor curve must be reported beside the
flow output so improvement cannot be attributed merely to receiving a trained
model.

The paired coupling is essential. Independently sampling the source and target
marginals would learn distribution transport but would not preserve the
identity or useful information of a particular anchor. If unpaired transport
is later studied, it requires an explicit OT or other declared coupling and is
an ablation.

Start with separate anchor-free and oracle-anchor flow models so failures are
identifiable. A single vector field with a source-type flag is a later
compression/engineering ablation. The first prior is conditioned on dataset and
hard-coded architecture/layer features, not generated-prefix activations;
activation context is initially supplied only to the frozen decoder.

## 4. Closest prior work and honest novelty boundary

The three works provisionally inferred from voice transcription are:

1. SANE, *Towards Scalable and Versatile Weight Space Learning*
   (arXiv:2406.09997).
2. WeightCLIP, *Aligning Datasets and Models for Weight Space Learning*
   (arXiv:2607.03551).
3. *Learning Model Representations Using Publicly Available Model Hubs*
   (arXiv:2510.02096).

These identities must be confirmed by the user. In particular, "ViteClip" is
currently interpreted as **WeightCLIP**, whose released dataset encoder is a
DeepSets-style set encoder, not a pretrained ViT-CLIP image encoder.

Mandatory related work also includes:

- Hyper-Representations (arXiv:2209.14733);
- HyperLDM / classifier-free diffusion guidance for meta-learning;
- D2NWG, *Diffusion-Based Neural Network Weights Generation*
  (arXiv:2402.18153);
- *Neural Network Diffusion* (arXiv:2402.13144);
- RPG (arXiv:2501.11587);
- *Structure Is Not Enough* (arXiv:2503.17138);
- *Generative Modeling of Weights: Generalization or Memorization?*
  (arXiv:2506.07998);
- NNiT (arXiv:2603.00180);
- DeepWeightFlow (arXiv:2601.05052).

Consequences for novelty:

- `weight VAE + dataset encoder + conditional diffusion` is occupied by D2NWG;
- latent diffusion over weights is occupied by HyperLDM and several later works;
- variable-size/architecture generation is not uniquely enabled by our method;
- dataset-weight alignment is occupied by WeightCLIP;
- behavioral losses for weight AEs already exist;
- activation-aware weight compression is a broad existing area.

What may remain novel is the conjunction:

1. activations are direct encoder/decoder conditions, rather than only a loss
   signal or a dataset-level prompt;
2. the codec exposes one common fixed-cardinality weight-matrix interface
   across variable shapes, rather than spending more latent tokens/chunks as a
   matrix becomes larger;
3. controlled experiments show that this interface is a better substrate for
   conditional generation than weight-only token representations;
4. activation interventions causally explain part of the downstream gain;
5. improvements hold on held-out datasets and architectures and survive
   anti-memorization tests.

This conjunction supports a strong empirical architecture paper. It does not
support a claim that the latent has learned a canonical space of operators.

## 5. What the paper must not claim without new evidence

Do not claim any of the following from the current implementation or evidence:

- first dataset-conditioned weight diffusion;
- first latent diffusion over neural weights;
- size-independent reconstruction fidelity for unrestricted matrices, or one
  global fixed-size code for any full model;
- one-code-per-full-layer success until it has been evaluated without tiling;
- generation of arbitrary architectures: the current decoder needs an external
  target architecture, shapes, masks, and coordinates;
- that the latent is a canonical operator representation, is invariant under
  functional equivalences, or necessarily encodes the pair \((W,P_X)\);
- native-training-distribution semantics for arbitrary HF models whose original
  dataset is unknown;
- that diffusion is inherently superior to a simpler prior;
- that visual clusters, retrieval, reconstruction MSE, or successful process
  completion alone establish the paper claim;
- that independent slice sampling preserves full-network compatibility.

## 6. Decisive technical risks

### 6.1 Resolved generation semantics: sequential prefix conditioning

The selected primary protocol is sequential generation through a known target
architecture:

1. compute the dataset embedding from a fixed target-training prompt;
2. begin with target inputs and the declared architecture template;
3. for layer `l`, summarize the current pre-layer activations produced by the
   already generated prefix;
4. sample the layer latent conditioned on the dataset embedding, activation
   summary, normalized depth, layer role, shape, and architecture metadata;
5. decode and insert the layer weights;
6. execute the new prefix to obtain context for layer `l+1`.

This is a sequential, execution-conditioned initializer. Target architecture
and target-training examples are given; true target weights or activations from
a hidden trained checkpoint are not used.

The main new risk is teacher-forcing/rollout mismatch. Training contexts come
from real trained prefixes, whereas inference contexts come from imperfect
generated prefixes. Sequential conditioning may correct accumulated errors, but
it may also compound them. Required diagnostics are:

- teacher-forced real-prefix versus free-running generated-prefix evaluation;
- activation drift and downstream utility versus depth;
- context corruption/noise augmentation during training;
- if needed, scheduled generated-prefix rollouts or rollout finetuning;
- free-running versus fixed/mean-context ablation.

Architecture conditions should use generalizable descriptors (normalized
depth, role, shape, family features) rather than unique model/layer IDs that
permit memorization.

#### Architecture-graph conditioning decision path

Closest methods do not provide a complete reusable solution:

- D2NWG uses a categorical ID for each known `dataset x architecture` pair in
  its mixed-architecture ablation and explicitly names a graph encoder as an
  unimplemented alternative. For LLM chunks it conditions on chunk indices,
  manually orders selected `q/k/v/o/up/down/gate` layers, trains separate models
  for groups, and performs validation-guided sequential candidate replacement.
- The HF-Hub SANE variant encodes flattened state-dict order using a sinusoidal
  3-D position `[global token index, layer index, token-in-layer index]`; this is
  scalable but not a computation-graph representation.
- RPG uses 2-D `[layer index, token index]` positions plus a Mamba recurrent
  model whose prototypes condition token diffusion. Its ablation reports
  independent token generation without the recurrent state collapsing to
  random-chance performance, so global coherence is a serious risk even with a
  small number of full checkpoints.

Recommended staged implementation:

1. Trace the known architecture template with `torch.fx` or an equivalent
   adapter and form a DAG of parameterized operations.
2. Generate/expose nodes in topological frontiers. Parallel operations such as
   Q/K/V share the same available input activation and may be generated in one
   frontier; residual merges occur only after all predecessor branches exist.
3. MVP condition per-layer diffusion on explicit structural metadata: operation
   type, input/output shape, normalized topological depth, fan-in/fan-out,
   branch/merge indicators, and parameter-sharing/tied-group ID.
4. If metadata-only conditioning fails to generalize, encode the given DAG with
   a small GraphSAGE/Graph Transformer/Graph Metanetwork encoder and use the
   resulting node embedding `h_l` as the diffusion condition.

Because the first flow does not use generated-prefix activations, all
layer latents can be sampled from `(dataset embedding, graph/node embedding)`
up front and decoded on demand when the topological executor makes each node's
input activations available. A later activation-conditioned flow can sample at
each frontier instead.

### 6.2 Decoder utility probe, not a logical ceiling

Poor reconstruction of an individual trained checkpoint does not logically rule
out a useful generated initialization. A decoder can preserve coarse statistics
or optimization-friendly structure while losing the exact function, and a
prior can sample latents outside posterior locations used by direct
reconstruction. Encoded/decoded checkpoints are therefore a high-value risk
probe, not a strict upper bound or automatic NO-GO.

Before training the expensive prior, still run the codec utility probe:

- encode and decode real held-out checkpoints;
- compare to the original weights, zero weights, raw low-rank/rate-matched
  codecs, and SANE reconstruction;
- measure functional distortion, Epoch-0 quality, and early AULC;
- measure early optimization utility even when Epoch-0 fidelity is weak.

If the probe is near scratch, run a small prior/generation rescue test rather
than immediately rejecting the paper. If both direct decoded checkpoints and
small-prior samples have no early-training advantage, scaling the prior is a
high-risk use of compute.

The absence of a train/validation overfitting gap suggests possible capacity
underfitting. Before the single production run, perform a short capacity sweep
on the same subset and compare loss/utility versus model size and A100-hours.

### 6.3 Full-network coherence

The selected first flow samples layer codes independently given the dataset and
layer/architecture metadata. This maximizes the effective number of training
records and is the lowest-risk choice when verified whole-checkpoint pairs are
scarce. A full joint checkpoint prior is deferred unless evidence shows
that cross-layer dependence is load-bearing.

Before training a joint prior, run a no-training discriminator:

- decode/resample complete observed code sets from one checkpoint;
- independently shuffle layer codes across good checkpoints within the same
  dataset, architecture, role, and shape buckets;
- compare downstream utility and activation drift;
- compare to the independent learned prior.

If within-checkpoint code sets work but matched layer shuffling collapses, the
independence assumption is causally implicated. Then prefer a data-efficient
hierarchical repair (one small global checkpoint code conditioning shared
layer-wise priors) before a monolithic full-checkpoint diffusion. If shuffling
does not materially hurt, an independent prior is sufficient and easier to
scale across architectures.

### 6.4 HF corpus and activation availability

The HF Model Hub paper uses roughly 2,000 training models and 200 validation
models, with about 171B individual parameters. Original training data is absent
or unreliable for many checkpoints. Therefore one cannot straightforwardly add
native activations to the exact corpus.

The actual published corpus is restricted to HF computer-vision tags
(`image-classification`, `image-segmentation`, `depth-estimation`, and
`object-detection`), not NLP/LLM checkpoints. Its reported architecture makeup
is 42.0% transformers, 21.8% ConvNets, 5% hybrids, and 31.4% unknown by model
name; about half appear to use ImageNet variants and the other half have no
dataset information. GPT-2/OpenWebText is an out-of-domain downstream test of a
vision-trained backbone, not part of foundation training. This is directly
aligned with using NLP as a transfer-domain evaluation.

The selected design separates the autoencoder and prior corpora:

- **Autoencoder:** use the native training dataset whenever it can be verified
  and accessed. Otherwise use a compatible declared probe dataset with the
  correct modality, preprocessing, shape, and input contract. Store
  `context_provenance = native | proxy` for every record and never describe a
  proxy context as the checkpoint's training distribution.
- **Conditional diffusion:** use only checkpoints with verified dataset–weight
  pairs. If this paired subset is too small, create a controlled paired zoo by
  training/fine-tuning a declared set of architectures on known datasets.

Proxy contexts may improve robustness, but may also teach the autoencoder to
ignore activation context. Native/proxy cohorts must therefore be measured
separately, and context-use ablations must be stratified by provenance.

### 6.5 Pseudoreplication and leakage

Many tiles from one matrix are not independent training examples. Splits and
confidence intervals must be grouped by checkpoint lineage, dataset, and model,
not by tile. The current diffusion corpus has many local slices but far fewer
unique source matrices/checkpoints, so random tile splits would exaggerate
generalization.

### 6.6 Full-checkpoint parameter coverage and convolutions

The current autoencoder primarily targets two-dimensional linear weights. A
generated checkpoint also contains embeddings, convolutions, biases,
normalization parameters, classifier heads, and possibly BatchNorm running
statistics. Before calling the output a generated model, define for every
parameter class whether it is generated, reshaped with a declared activation
convention, copied from an anchor, freshly initialized by a common policy, or
excluded.

The selected policy is to generate all supported linear and convolutional
weight tensors and BatchNorm affine parameters. The final classifier head is
always freshly initialized with the architecture default for every compared
method; it is neither copied from an anchor nor generated in the primary
benchmark. BatchNorm running statistics are not generated: initialize them by
the architecture default and apply exactly the same training/calibration
protocol in every arm. Apply the same declared default initialization to all
other unsupported parameters. A dense convolution is represented by the unconstrained kernel
matrix

\[
W_{conv}\in\mathbb R^{C_{out}\times C_{in}\times k_h\times k_w}
\longleftrightarrow
W_{flat}\in\mathbb R^{(C_{in}k_hk_w)\times C_{out}},
\]

with activation rows constructed using `unfold/im2col`, so
`Y_col = X_col W_flat`. This is not the huge structured Toeplitz matrix; every
kernel maps bijectively to one unconstrained matrix of the declared shape.

Grouped convolution is a set of independent unconstrained group matrices of
shape `((C_in/groups)*k_h*k_w, C_out/groups)`, or an equivalent masked block
representation. Do not decode forbidden cross-group entries. Depthwise,
transposed, 1-D, and 3-D convolutions need explicit adapters and tests.

Stride, padding, and dilation affect `im2col` activations and metadata, not the
kernel-matrix parameterization. Sample activation patches rather than
materializing the entire potentially enormous `im2col` tensor.

Report generated parameter coverage as both count and percentage of checkpoint
scalars for every architecture. The same default policy for biases, unsupported
norm parameters, heads, embeddings, and running statistics must be applied
across methods.

The selected initial NLP policy is to exclude token/position embeddings from AE
training and initialize them with the declared architecture default. This is a
reasonable distribution-shift boundary: embeddings are categorical lookup
tables with sparse/tied semantics rather than dense hidden-to-hidden maps. The
Muon optimizer analogy supports expecting a different optimization geometry,
but does not prove that embeddings contain no learnable structure. Phrase this
as an engineering/domain decision, not a theorem. Preserve tied embedding/LM
head constraints and report the resulting generated-parameter coverage.

## 7. Fair comparison contract

There are two different comparisons and both must be reported.

### 7.1 Codec ledger

Compare only the representation modules:

- encoder and decoder trainable parameters;
- code cardinality and true quantized bits, including per-layer metadata;
- training weight scalars/tokens and activation samples seen;
- training FLOPs, GPU-hours, and peak memory;
- encode/decode FLOPs and latency;
- oracle reconstruction and functional rate-distortion.

### 7.2 End-to-end system ledger

Compare everything needed to make an initialization:

- codec;
- dataset encoder;
- mapper, diffusion, flow, KDE, or other prior;
- all trainable and frozen/pretrained parameter counts separately;
- total training FLOPs/GPU-hours;
- target-data queries;
- number of sampled candidates and selection metric;
- sampling/decoding time;
- total downstream fine-tuning budget.

The main table must count flow matching and any diffusion baseline. They are
learned components, not free samplers. A frozen dataset encoder may be excluded from *trainable* parameters
only if every compared method receives the same encoder and access; its frozen
parameter count, pretraining source, and inference compute still must be shown.

Parameter count and optimizer steps alone are inadequate. A fair protocol also
matches or reports raw data exposure and compute because a SANE token, a full
local code, and an activation-conditioned record have different costs.

Recommended presentation:

- a **core-capacity-matched** experiment to isolate representation quality;
- a **full-system-compute-matched** experiment to establish practical SOTA;
- a rate/compute Pareto curve rather than a single conveniently chosen match.

## 8. Minimum experiment matrix

Every generative method must use the same train/validation/test split, target
architecture information, frozen dataset encoder where applicable, candidate
count, candidate-selection rule, downstream optimizer, and fine-tuning budget.

| ID | Representation | Activation context | Generative prior | Purpose |
|---|---|---|---|---|
| B0 | scratch / untouched oracle anchor | none | none | task floor / anchor control |
| B1 | released WeightCLIP | frozen released dataset encoder | released mapper/sampler | primary published-system baseline on reconstructed zoo |
| B2 | optional SANE | same allowed conditions | matched flow | representation control if added later |
| O0 | fixed-cardinality weight AE | no activations | matched flow | activation ablation |
| O1 | fixed-cardinality weight AE | correct activations | anchor-free conditional flow | proposed generation system |
| O2 | fixed-cardinality weight AE | correct activations | oracle anchor-conditioned paired flow | proposed refinement system |
| O3 | fixed-cardinality weight AE | shuffled/foreign/mean activations | matched flow | causal context control |
| O4 | fixed-cardinality weight AE | correct activations | simple equal-budget prior | test learned-flow necessity |
| O5 | fixed-cardinality weight AE | correct activations | matched diffusion | sampler-class ablation |

Additional required controls:

- decoded-real-checkpoint oracle for every codec;
- nearest-neighbor/anchor sampling;
- interpolation and noise baselines;
- untrained codec where affordable;
- permutation/canonicalization control;
- independent versus joint/sequential slice prior.

## 9. Evaluation protocol

Primary downstream endpoints:

1. Epoch-0 test/validation quality from one generated initialization.
2. AULC over the exact early fine-tuning schedule used by the target benchmark,
   including the published Epoch 1/5/10 or 1--5 epoch checkpoints.
3. Mean and confidence interval over independent held-out dataset/architecture
   units and seeds.
4. Quality as a function of candidate count. Report single-sample expectation
   as primary and top-\(k\) only with an identical selection budget.

Representation endpoints:

- structural and held-out functional distortion;
- performance after decode of real held-out weights;
- true rate-distortion curve over at least three code budgets;
- shape, layer-role, and architecture held-outs.

Generative validity endpoints:

- nearest training checkpoint in aligned weight and functional distance;
- coverage/diversity versus quality;
- train-versus-held-out dataset/architecture gaps;
- duplicate/interpolation detection;
- samples from intentionally wrong dataset conditions.

Retrieval and latent plots are diagnostics, not headline endpoints.

## 10. Minimum ablations for architecture attribution

The primary claim is the end-to-end metric result. The following ablations are
the minimum needed to explain why the proposed architecture wins; they are not
independent headline claims.

### 10.1 Activation-conditioning contribution

Use the same architecture, latent rate, prior class, and training records while
varying only context:

- correct native context;
- no context;
- global/mean context;
- context shuffled within layer role and shape;
- foreign-dataset context;
- crossed encoder-context and decoder-context where feasible.

Activation conditioning is load-bearing only if correct context improves
functional reconstruction or downstream generation over matched controls on
held-out checkpoints. No claim about canonical operator geometry follows from
this contrast.

### 10.2 Fixed-cardinality architecture contribution

Train the same prior class and approximately the same prior capacity on SANE
codes and our codes. Match true code rate, model/data split, condition encoder,
optimizer exposure, and candidate selection. A win over SANE's native sampler
alone cannot distinguish a better prior from a better representation.

The decisive representation test must use one latent code per complete matrix
for our method. If both methods split every large matrix into proportional
numbers of chunks, that experiment can establish a metric win but not the
fixed-cardinality-operator explanation.

### 10.3 Generative-prior mechanism

Flow matching is primary. Compare it to at least one simple empirical prior;
add a matched diffusion ablation when resources permit. If the simple prior
matches flow, the representation may still be useful, but a special generative
geometry is not the right explanation.

### 10.4 Latent targets for prior training

The first flow is conditioned on the frozen WeightCLIP dataset embedding plus
architecture/layer metadata. Generated-prefix activations are supplied to the
frozen decoder, not to the first flow implementation. Activation-conditioned
flow is a later ablation if the base prior is insufficient.

Do not conflate three different ways to construct prior targets:

1. **Amortized encoder code**
   \[
   z_{enc}=E(W,C).
   \]
   This is cheapest but inherits encoder amortization and decoder reconstruction
   error.
2. **Decoder inversion code**
   \[
   z_{inv}=\arg\min_z d(W,D(z,C)).
   \]
   Here `d` may combine structural weight error and held-out functional error.
   This removes encoder amortization error while still targeting the source
   checkpoint.
3. **Task-fitted code**
   \[
   z_{task}=\arg\min_z L_{task}(D(z,C(z)))+\lambda R(z,z_{enc}).
   \]
   All per-layer codes are optimized through the assembled frozen-decoder model.
   This creates a new task-optimized decoded model rather than an encoding of
   the original checkpoint.

Primary recommendation: initialize inversion/task fitting from `z_enc`, freeze
the decoder, use a deterministic optimizer protocol, optimize only on source
training data, early-stop on source validation data, and keep held-out target
datasets sealed. Save the whole-checkpoint grouping and jointly fitted code set;
do not turn correlated layers into independent examples.

The production choice is `z_task`. Start every fit from `z_enc`, freeze the
decoder, and jointly optimize all supported layer codes of one checkpoint under
the downstream task loss. Recompute decoder activation context through the
current decoded/generated prefix, but stop gradients through activation context
into preceding layers. Store the jointly optimized whole-checkpoint code set as
one grouped record. Per-layer independent fitting, fixed context from the
original checkpoint, and fully differentiating through the activation chain are
diagnostic variants, not the default dataset-construction protocol.

Regularize optimized codes toward the encoder code or a declared latent prior.
Otherwise decoder non-identifiability can assign arbitrary, highly scattered
codes to functionally similar weights, making the flow target harder to
learn. Compare `z_enc`, `z_inv`, and `z_task` on decoded utility and prior sample
quality before selecting the production target.

If optimized codes become the main diffusion dataset, the encoder remains
essential to the selected training method: amortized reconstruction over the
large heterogeneous corpus jointly learns the latent coordinate system and
decoder and provides a canonical warm start for per-checkpoint fitting. It need
not run at generation time. Optimized-code ablations measure the encoder's
amortization gap inside that learned space; they do not turn the main method
into an encoder-free claim.

## 11. Recommended benchmark staging

No single published setup covers all desired claims.

### Tier A: first WeightCLIP-scale go/no-go

The selected first track is the published half-width ResNet-18 WeightCLIP
setting. The exact ResNet checkpoint zoo used to train released WeightCLIP was
not published. Therefore exact reproduction of the paper's metrics is
impossible from public artifacts. Do not claim otherwise.

Reconstruct the zoo from the published architecture, datasets, optimizer, and
checkpoint schedule using new random seeds. Train the proposed approximately
700M-parameter activation-conditioned autoencoder on this reconstructed
population. Evaluate released WeightCLIP on the same reconstructed population.
This is an approximate protocol reconstruction and, for released WeightCLIP, a
test of transfer to a new checkpoint population; published paper numbers appear
only as contextual `reported by authors` rows.

The first decisive comparison is:

- scratch and untouched oracle-anchor controls;
- released WeightCLIP with its frozen dataset encoder and released mapper;
- our weight-only/context-control codec plus matched flow;
- our correct-activation codec plus anchor-free flow;
- our correct-activation codec plus oracle anchor-conditioned paired flow;
- a simple prior, with diffusion as a later sampler ablation.

Evaluate independently sampled models/seeds as the primary endpoint. Any
top-`K` result must use the same candidate and target-validation budget in every
arm and remain secondary.

### Tier B: HF-Hub foundation-scale extension

After the WeightCLIP-scale architecture has been debugged and shown useful,
train the heterogeneous foundation-style model on the declared HF population.
Every source checkpoint must have a compatible native or explicitly labeled
proxy dataset for activation collection. This stage establishes breadth; it is
not required before the first architectural go/no-go.

### Tier C: breadth after the main result

Only after Tier A is interpretable, add held-out architecture families, larger
HF models, NLP transfer, permutation tests, full-matrix one-code size scaling,
and extensive rate/compute curves. MultiZoo-SANE is not part of the primary
plan; it may be added only if a later reviewer-facing comparison needs it.

## 12. One-week execution plan on eight A100s

This is a decision schedule, not a promise that every production model can be
trained in seven days.

### Day 0--1: freeze the contract and reconstruct

- Clone and pin WeightCLIP code, model, and dataset encoder artifacts.
- Freeze dataset/model splits, candidate selection, metrics, and resource
  ledgers before inspecting held-out results.
- Reconstruct the missing ResNet-18 Slim zoo with new seeds. Exact reproduction
  of a released row is not a gate because the original checkpoint zoo is absent.
- Run the released WeightCLIP implementation end to end on the reconstructed
  zoo and label this result as checkpoint-population transfer/approximate
  reconstruction, not exact reproduction.
- Measure the true current local code rate and the number of codes per model.
- Implement grouped lineage-aware manifests and leak checks.

### Day 1--2: codec risk and capacity gate

- Encode/decode real held-out models with correct, missing, mean, and shuffled
  contexts.
- Run Epoch-0 and short AULC comparisons against original, zero/scratch,
  rate-matched linear/SVD, and SANE reconstruction.
- Measure context dependence and test whether the encoder actually uses
  activations.

**Interpretation:** weak reconstruction alone is not an automatic stop. Continue
to a small prior rescue test, but do not commit the complete eight-GPU campaign
if both decoded-checkpoint utility and generated-sample early utility are at
scratch level. Use the capacity pilot to select the production AE size.

### Day 2--4+: production representation and matched priors

- After short correctness and throughput checks, launch one production
  approximately 700M-parameter activation-conditioned AE on the local H100 NVL.
  The user will monitor the known multi-stage loss landscape; the agent remains
  responsible for config/checkpoint/data correctness and runtime visibility.
- Launch the weight-only/context controls with shared initialization where
  feasible.
- Build grouped `z_task` records and train separate Gaussian-to-task and
  oracle-anchor-to-task conditional flows; train a simple-prior control.
- Track total records, unique checkpoints/matrices, scalar exposure, FLOPs,
  GPU-hours, and cache provenance.

### Day 4--6: held-out downstream evaluation

- Seal checkpoints before target evaluation.
- Generate with identical sample counts and validation selection.
- Evaluate Epoch-0 and the exact early AULC schedule.
- Run nearest-train/interpolation tests and inspect every plot.
- Use grouped bootstrap over held-out datasets/models, not tiles.

### Day 6--7: decision review

The headline is alive only if all of these hold:

1. at least one early codec/prior utility probe is non-degenerate;
2. ours + flow beats released WeightCLIP under the agreed controlled protocol;
3. correct activations beat weight-only and shuffled/mean activations;
4. the gain appears on held-out dataset units and in early AULC, not only best
   of many samples;
5. generated weights are not nearest-neighbor memorization;
6. full-system cost is competitive enough to support the practical claim.

If (2) fails, the central performance thesis is rejected for this setup. If (3)
fails, the complete architecture may still win, but activation conditioning
cannot be presented as the cause. If all early utility probes fail, the
autoencoder/prior interface should be repaired before the full campaign.

## 13. Resource allocation principle

Do not allocate all eight GPUs to one opaque run before the ceiling gate.
After the gate:

- 2--3 GPUs: production activation-conditioned codec or data-parallel run;
- 1 GPU: weight-only/context ablation;
- 1 GPU: anchor-free flow;
- 1 GPU: oracle anchor-conditioned flow;
- 1 GPU: simple/diffusion prior or WeightCLIP baseline;
- 1 GPU: downstream evaluation, artifact review, and contingency.

Exact allocation should be based on measured throughput. Every long run must log
resolved config, commit, dataset manifest hash, checkpoint origin, device,
dtype, seed, cache files, stages, progress, output paths, and summary metrics as
required by `AGENTS.md`.

## 14. Remaining implementation decisions

The paper identity, first benchmark, primary prior, dataset encoder, latent
target, head policy, and oracle-anchor semantics are resolved in Section 19.
They must not be reopened silently.

Before the expensive launch, agents must still derive from the released code or
escalate the genuinely missing low-level facts: exact WeightCLIP input sampling
and normalization, the reconstructable subset of its zoo training recipe,
candidate-selection details, and BatchNorm calibration behavior. Missing
published details must be declared and assigned one common policy; they are not
permission to invent method-specific advantages.

## 15. Current repository-specific warnings

- The architecture is intended to expose one fixed-cardinality code per
  variable-shape matrix, but current stored training/diffusion data primarily
  proves the `64x64` sliced operating point. Full-matrix evidence is pending.
- Existing source absolute reconstruction evidence is weak; treat the codec
  ceiling as unproven.
- Existing diffusion data has many tiles but a much smaller number of unique
  source matrices/checkpoints; split by lineage.
- Current downstream artifacts do not yet establish a matched advantage over
  raw optimization or the required published baselines.
- Architecture and activation side information must be declared and made equal
  across methods.
- An audit reported plaintext external-service credentials inside stored
  experiment/config artifacts. Do not publish or copy those artifacts. Locate,
  revoke/rotate, and sanitize them before any repository or artifact release;
  never put credential values in logs or reports.

## 16. Paper success criterion

The workshop paper succeeds primarily if the proposed full system obtains a
clear, reproducible common-benchmark advantage in held-out initialization and
early-training metrics. No numerical win count or effect threshold is declared
before the exploratory pipeline is known to be valid: the first runs are
expected to expose implementation and protocol errors. This does not relax the
validity contract or permit changing the held-out evaluation repeatedly after
seeing results. Once the pipeline is validated on source/meta-validation data,
freeze the final OOD protocol before the confirmatory evaluation.

To make a positive result publishable rather than a tuning anecdote, it should
also demonstrate that:

1. at least one preregistered codec/prior utility probe is non-degenerate and
   generated initializations improve held-out early training;
2. released WeightCLIP and simple/matched-prior controls do not explain away the
   win;
3. correct activation context contributes under a minimal no/shuffled-context
   ablation, if activation conditioning is advertised as a design contribution;
4. the result does not rely on unmatched compute, target-data access, or
   best-of-many selection;
5. generated initializations improve held-out early training without relying on
   excessive candidate selection or memorization;
6. fixed-cardinality full-matrix and size-OOD behavior are characterized as
   supporting architectural analysis, not elevated into an unsupported main
   claim.

No claim about learning a canonical operator space is required for this paper.

## 17. Concrete first-stage WeightCLIP protocol

The primary first-stage benchmark is the published half-width
ResNet-18 WeightCLIP zoo (about 2.8M parameters, channels
`{32, 64, 128, 256}`). It separates an architecture effect from the broader
pretraining population used in the HF-Hub foundation track.

The original 1,000 ResNet checkpoints used by WeightCLIP are not publicly
available. Reconstruct new runs from the published recipe. Consequently:

- do not claim exact reproduction of the paper's WeightCLIP metrics;
- evaluate the released WeightCLIP model on the reconstructed checkpoint
  population and describe this as transfer to new seeds/checkpoints;
- keep original paper metrics in a clearly separated reported-results table;
- use the same reconstructed population, splits, target examples, and
  downstream harness for our method and every runnable baseline.

Published training datasets:

- Artworks;
- Blood Cells;
- Breast Cancer Tissues;
- Aerial Cactus;
- Cassava Leaf;
- CT Images;
- Land Use;
- Lego Bricks;
- Real/Fake Legos;
- Casting.

Locked OOD datasets:

- Colorectal Histology;
- COVID-19;
- Speed Limit Signs;
- Honeybee Pollen;
- Real or Drawing;
- CIFAR-10.

The paper says 20 independent runs per training dataset with five checkpoints,
but the released ResNet configuration uses 50 independent runs and the two
terminal zero-based checkpoint indices 43/44. For this campaign the runnable
repository is treated as authoritative: use 50 lineages/dataset and one-based
epochs 44/45, for the same 1,000 primary checkpoint files but 500 independent
trainings. The two terminal snapshots from one trajectory must never cross a
split boundary.
Group all splits, sampling, and uncertainty estimates by independent lineage.

Before unsealing the six OOD datasets, tune on pooled lineage-safe validation
covering all ten source datasets: 7 validation lineages per dataset, with macro
and worst-dataset reporting. Do not base the decision on one fixed pair of
source datasets; between-dataset variance makes that estimate unstable. Open a
separate 8-lineage-per-dataset internal source test only after choices are
frozen. Use only training images to collect source activations or build target
prompts.

### Classifier head and BatchNorm policy

OOD datasets have different class counts. In the controlled table the
classifier head is always freshly initialized using the architecture default
for every method; it is not generated, decoded, or copied. Therefore Epoch-0
full classification accuracy is diagnostic and early AULC is primary. Our
codec excludes all BatchNorm keys and uses default affine/running state before
the common calibration schedule. WeightCLIP retains its released generated
affine values and BN conditioning. This intentional system-level difference is
reported, not hidden; no BN-reset arm is required in the first campaign. A
separate native-fidelity table preserves WeightCLIP's released decoded-head and
test-selected top-5 behavior.

### Candidate selection

Published methods use incompatible oracle budgets: WeightCLIP reports selected
samples from a large candidate set, while other protocols use different anchor
and best-of-many rules. The controlled table should use a common candidate count
(initial recommendation `K=8`) and report:

- a random single sample;
- the mean over all `K` samples;
- best-of-`K` selected only using an identical fixed target-train validation
  budget;
- all generation and selection compute.

Native-paper protocols may appear in a separate reproduction table, but their
numbers must not be mixed with the controlled table.

### Interpretation policy

Do not predeclare an arbitrary number such as wins on five of six datasets
before the first valid run. Inspect the complete metric table, uncertainty,
learning curves, generation validity, and failure modes. Validity constraints
remain fixed: equal candidate count, head policy, BatchNorm handling, prompt
budget, splits, and downstream compute. A gain visible only in unmatched
best-of-many selection, only after long fine-tuning, or only under substantially
greater hidden compute is not evidence for the intended architectural claim.

## 18. Repository evidence supporting the current risk assessment

These are evidence locations, not claims that every historical experiment was a
valid confirmatory test:

- canonical 64x64-tile AE training:
  `projects/shared/storage/artifacts/training/runs/`
  `train_big_vae_20260413_110353_pid1451259_e88349/logs/train_rank0.log`;
- local-code construction for multiple tiles:
  `projects/weight-vae/workspace/big_vae/eval/`
  `vit_tiny_latent_optimization/latent_store.py`;
- model-size scaling summaries:
  `projects/weight-vae/workspace/post_train_research/vit_latent_scaling/`
  `artifacts/cifar10/50k/ae_diffusion_prior/summary.json`;
- decoder dependence on target shape, masks, and activation condition:
  `projects/weight-vae/workspace/big_vae/models/`
  `big_weight_vae_parts/decoding_mixin.py` and `forward_mixin.py`;
- prior source-only and cross-domain gate history:
  `projects/weight-vae/workspace/docs/notes/`
  `crossmodal_united_structure_decision_log_20260816.md`;
- unique-source versus local-record counts:
  `projects/shared/storage/artifacts/training/checkpoints/weight_quantile_vae/`
  `stage_1/offline_dataset/manifest.json` and
  `projects/shared/storage/artifacts/training/datasets/`
  `big_vae_latent_diffusion_AE/stage_1/manifest.json`;
- short existing latent-prior versus raw summaries:
  `projects/weight-vae/workspace/post_train_research/vit_latent_scaling/`
  `artifacts/cifar10/50k/{ae_diffusion_prior,raw}/summary.json`.

The credential warning in Section 15 specifically concerns the latent-diffusion
manifest above and
`projects/shared/storage/artifacts/training/checkpoints/`
`weight_quantile_vae_gpu0_square/stage_1/latest.pt`. Do not expose their values.
Credential revocation/rotation is an external action and artifact sanitization
is a separate maintenance task; neither is silently performed by this planning
document.

## 19. Latest canonical decision ledger (2026-08-18)

This section records the decisions made in the long-form user dialogue. It is
the fastest handoff for a new agent and takes precedence over stale language
elsewhere in this file.

### 19.1 Paper and benchmark scope

- The headline is an empirical architecture/system win on initialization and
  early fine-tuning metrics. Strong operator-learning or canonical latent-space
  claims are explicitly out of scope for this paper.
- The first benchmark is WeightCLIP's half-width ResNet-18 setting, not
  MultiZoo-SANE and not the large HF-Hub corpus.
- MultiZoo-SANE was considered and rejected as the primary first benchmark.
  SANE may return only as an optional matched-prior representation baseline.
- The HF-Hub experiment remains the later foundation-scale extension after the
  architecture works on the controlled WeightCLIP-scale setting.
- The original ResNet checkpoint zoo used to train WeightCLIP was not released.
  Therefore an exact metric reproduction is impossible. Reconstruct a new zoo
  with the published datasets/architecture/training schedule and new seeds.
  Running released WeightCLIP on it is a transfer test to a new checkpoint
  population, not an exact reproduction. Never describe it otherwise.
- Calibrate that transfer explicitly: run the released checkpoint on the
  reconstructed zoo and the same six OOD datasets under the full published
  epoch-0/1/10 protocol. If the full per-dataset table, not merely its grand
  mean, matches the paper within measured seed/candidate variation, treat the
  reconstructed zoo as an exchangeable new draw and accept released WeightCLIP
  as the primary baseline. If it does not, the released run is screening only
  and a matched WeightCLIP retrain on our train lineages is required for the
  architecture-superiority claim.
- The reconstructed source population follows the ten published WeightCLIP
  training datasets and groups the five stored epochs from one run as one
  lineage. No trajectory may cross train/validation/test boundaries.

### 19.2 Representation training

- The production representation is an activation-conditioned autoencoder,
  initially without KL. A KL-regularized fine-tune is optional only after the
  AE is trained and stable.
- The old `64x64` operating point is not frozen for the scaled run. For the
  exact ResNet18-Slim body (BN and classifier excluded), `64x64` produces 694
  tiles at 98.16% valid fill, `96x96` produces 379 at 79.88%, `128x64`
  produces 354 at 96.22%, and `128x128` produces 193 at 88.24%. Square
  `128x128` is primary. `96x96` is dominated on both code count and padding;
  `128x64` is only an explicit fallback if the production-shaped square profile
  is infeasible. A short loss run must not be used to choose tile geometry
  because it cannot see delayed loss transitions. One
  code for a full variable-size matrix and the size-OOD degradation curve are
  later architectural experiments, not a condition for the first leaderboard
  run.
- Target encoder+decoder capacity is approximately the released WeightCLIP
  backbone scale, around 700M parameters. The flow and dataset encoder are
  reported separately in the full-system ledger rather than hidden in the codec
  count.
- Train this first approximately 700M representation on the local H100 NVL
  96 GB rather than occupying all eight A100s. A synthetic 737.4M-parameter
  compute-only benchmark and the planning range are stored in
  `projects/weight-vae/workspace/docs/`
  `h100_700m_training_time_benchmark_20260818.md`.
- Run the deterministic AE for the full 500k optimizer steps. Validate only
  every 20k and keep validation a small fraction of wall time; save complete
  resume state every 10k. There is no early stopping or plateau rule. Historical
  training shows a fast first loss decrease followed by long noisy plateaus and
  delayed additional drops; higher LR can lengthen the bad plateau, and a 2k
  LR trial is non-informative. Use the proven square-AE LR `5e-5` from the
  serialized config in
  `weight_quantile_vae_gpu0_square/stage_1/latest.pt` (step 480k), not the stale
  `3e-5` currently present in a `v2` YAML. A genuine full-horizon experiment is
  required to justify changing it. The user monitors the multi-stage landscape;
  agents remain responsible for wrong config/data/checkpoint, NaNs, stalls, and
  other technical invalidity.

### 19.3 Dataset conditioning

- Use the released WeightCLIP dataset encoder and freeze it in the primary
  comparison. Give the same embedding and target-example budget to every method
  that can consume it.
- A newly trained or jointly trained dataset encoder is an ablation. It cannot
  silently replace the released frozen encoder in the main comparison.
- The initial flow is conditioned on dataset embedding and hard-coded
  architecture/layer metadata: operation type, input/output shape, normalized
  depth, branch/role features, and tile location. A graph encoder is a later
  ablation.
- Generated-prefix activation context is initially used by the decoder only,
  not the flow. The known architecture is executed in topological order so the
  decoder sees activations from the generated prefix.

### 19.4 Flow matching replaces diffusion as primary

- Conditional flow matching is the primary generative prior everywhere.
  Diffusion is retained as a secondary matched-budget sampler ablation and is
  not the paper's claimed novelty.
- Train matched flows separately in both latent spaces, not only for our AE:
  1. ours anchor-free `N(0,I) -> z_task_ours`;
  2. WeightCLIP anchor-free `N(0,I) -> z_task_wc`;
  3. ours paired oracle-anchor `z_enc_ours -> z_task_ours`;
  4. WeightCLIP paired oracle-anchor `z_enc_wc -> z_task_wc`.
  The flow core, conditioning budget, target records, optimizer budget and NFE
  are matched; shape-specific adapters are counted. The released WeightCLIP
  mapper is still reported, but it cannot replace this matched-flow control.
- Do not add noise to the clean anchor latent merely to imitate diffusion. The
  anchor-conditioned ODE starts directly at the encoder-induced latent.
- For anchor transport, source and target codes must be paired for the same
  trained checkpoint. Independently pairing marginal samples would not test
  preservation/refinement of that anchor.
- Inspect decoded intermediate flow times. Straight latent paths may cross
  decoder-dead regions; if this occurs, test rectified/OT coupling rather than
  declaring flow matching invalid after one run.

### 19.5 Exact oracle-anchor semantics

- The anchor-conditioned experiment deliberately permits peeking: at test time
  the method receives a real, fully trained, held-out checkpoint for the target
  dataset. Encode that true checkpoint and start the flow from its clean
  `z_enc`.
- This arm is labeled `oracle anchor-conditioned` or `with target-trained
  anchor`; it is never described as unconditional or anchor-free generation.
- Report the untouched trained anchor itself alongside the transported output.
  Otherwise receiving a trained target solution is an unaccounted advantage.
- The anchor-free arm remains the operational generation experiment and never
  receives target weights.

### 19.6 Task-fitted target construction

- `z_task` is the production flow target.
- Initialize from `z_enc`, freeze the decoder, and jointly optimize all
  supported layer codes of one checkpoint through the assembled model's task
  loss.
- Recompute decoder context through the current decoded prefix but stop
  gradients through the activation-context path into previous layers.
- Store each jointly optimized full-checkpoint code collection as one grouped
  record. Splitting correlated layers into independent statistical units or
  across dataset partitions is forbidden.
- Decoder inversion, independent layer fitting, original-checkpoint fixed
  context, and full cross-layer activation backpropagation are ablations or
  diagnostics.
- The released WeightCLIP sparse tokenizer can emit complete non-overlapping
  windows containing convolution, BatchNorm-affine and classifier tokens, but
  the released mapper script consumes only one 512-token window. Therefore an
  exact-paper full-checkpoint mapper comparison is unavailable. Our complete-
  checkpoint path is explicitly a **multi-window extension**, never the
  released end-to-end baseline. Its source bank is grouped as
  `Y=[N,25*512,192]`; ridge/memory fitting is global over a whole flattened
  checkpoint code and retrieval returns a whole checkpoint code. Reusing the
  released first-window mapper independently 25 times is invalid. The codec
  decodes native windows in order, concatenates them, and performs one final
  `tokenizer.detokenize`. In
  the controlled arm those windows are never truncated: derive a body-token
  mask from the official ordered tokenizer layer IDs, parameterize
  `z_task = z_enc + body_mask * delta`, and permit `delta` only for convolution
  and BN-affine rows. Classifier/padding/non-body rows remain bitwise equal to
  `z_enc` and have zero gradient. This supersedes any older one-window or
  unconstrained-WeightCLIP-latent wording.

### 19.7 Parameter coverage

- Generate supported body linear weights and convolution weights via the
  declared matrix representation. Our codec does not encode any BatchNorm key.
- The classifier head is always freshly initialized using the architecture
  default in the primary benchmark. It is not copied, preserved, or generated.
- For our method all BatchNorm state starts from architecture defaults:
  affine `gamma=1`, `beta=0`, running mean `0`, running variance `1`, counter
  `0`. Apply the same downstream calibration schedule to every arm.
  WeightCLIP has two explicitly separate policies. The native audit preserves
  the complete released decoded state, head and BN. The controlled arm
  transfers decoded convolutions and decoded BN affine, uses a fresh paired
  random classifier head, resets BN running statistics, and applies the same
  declared calibration batches as every controlled candidate. During
  WeightCLIP `z_task` fitting, the original source head and BN running buffers
  are frozen critics, while decoded BN affine remains task-active. Never mix
  rows from these two policies in one table.
- For later NLP transfer, token/position embeddings and the LM head use the
  declared default initialization while preserving required weight tying; body
  attention and MLP matrices are generated.

### 19.8 Evaluation and interpretation

- Early fine-tuning AULC is primary because the classifier head is random.
  Epoch-0 metrics are diagnostic.
- Do not declare a numerical success rate such as `5/6` before trustworthy
  exploratory results exist. The first runs may expose implementation errors.
- This freedom applies to the scientific effect threshold only. It does not
  permit changing data splits, head/BN policy, candidate count, target-example
  budget, optimizer, or evaluation harness after inspecting held-out outcomes.
- Report anchor-free and oracle-anchor-conditioned results separately. Report
  single-sample expectation as primary; matched best-of-K is secondary.
- Also run the paper-style `100 candidates -> top 5 by target test accuracy`
  experiment for WeightCLIP and our generative methods. It is mandatory because
  candidate quality can have high variance, but it is labeled
  `test-selected oracle` and kept separate from the validation-selected
  controlled table.
- Every completed experiment includes artifact inspection, curve review, bug
  checks, and interpretation. Process completion alone is not a result.

### 19.9 Approved implementation blueprint

The full file-level, data-layout, training, flow-matching, evaluation and launch
plan is stored in:

`projects/weight-vae/workspace/docs/`
`weightclip_flow_implementation_plan_20260818.md`.

Its blueprint was **APPROVED FOR IMPLEMENTATION (2026-08-18)**. The code is
implemented through an artifact-driven Stage-D/G handoff, but it is not a
production GO until the remaining gates in Section 20 are satisfied. Its dependency DAG
and stop/go gates remain the execution contract. Approval covered
implementation, tests, zoo construction and data preparation. It does not authorize production
training of the approximately 700M AE: before that launch, present the exact
architecture/capacity alternatives and obtain explicit user approval for one
content-addressed model fingerprint.

### 19.10 Zoo construction, split provenance, storage and H100 budget

- The zoo also runs only on the current H100 NVL, not on eight A100s. A local
  official-trainer benchmark found 2--4 concurrent model processes optimal;
  eight was slower. The released script's full two-seed/seven-LR per-dataset
  sweep is mandatory. The completed real `land-cover-class_0_10` probe measured
  about 127 seconds per four concurrent 45-epoch lineages, 6m54s for its full
  14-run sweep and 69 seconds for two final runs. Scaling to all known task
  sizes and adding full checkpoint I/O/materialization/review gives a current
  production ETA of **6--8 hours** for the 500-lineage zoo plus sweep. Refresh
  it from online telemetry after each dataset starts.
- `16/2/2` was an internal proposal, not a WeightCLIP fact. The official repo
  uses random `70/15/15` checkpoint splits and warns about trajectory leakage.
  Our corrected split is lineage-safe `35/7/8` per dataset: 700/140/160 primary
  checkpoints.
- Split before the released five-permutation checkpoint augmentation. Apply
  channel permutations consistently to the entire ResNet graph and our
  activation-context channels, verify functional equivalence, and never count
  permuted views as independent lineages or uncertainty units.
- Train 50 independent lineages per dataset for the full 45 epochs and use the
  released ResNet config's zero-based checkpoint indices 43/44. This
  intentionally chooses the runnable repository's 50-lineage/two-terminal-
  snapshot population over the paper's contradictory 20-lineage/five-terminal-
  snapshot wording and increases independent model diversity. It does **not**
  mean that a lineage is trained for only two epochs.
- Primary inputs are one-based epochs 44--45. For future projects also archive FP32
  model-only states at initialization and every epoch: 46 snapshots/lineage,
  about 257.8 GB decimal (240.12 GiB) for 500 lineages.
  These are one-based training epochs; the released trainer's zero-based files
  for the primary two are indices 43--44. Store both indices in the manifest.
  Keep only one rolling optimizer/scheduler resume state per active lineage.
  The earlier 50 GiB target is superseded: preserve raw FP32 first and compress
  later. An exact XOR-delta + zlib test on a 45-epoch trajectory achieved only
  `1.281x`, so do not assume a large lossless compression win.
- Full paper/repository protocol conflicts and the native-versus-controlled
  evaluation split are recorded in
  `projects/weight-vae/workspace/docs/weightclip_protocol_audit_20260818.md`.

### 19.11 Real-data lineage probe result (2026-08-18)

A real official `land-cover-class_0_10` probe was completed on the H100 using
the released two-seed/seven-LR sweep and two final 45-epoch ResNet18Slim
lineages. The sweep selected its upper boundary, LR 0.7, using the released
test-based rule. Final test accuracy was 91.176% and 91.121%. Epoch 44 -> 45
changed test accuracy by -0.111/+0.499 percentage points, but changed weights
by 4.488%/4.380% of norm. Across epochs 41 -> 45, test accuracy improved by
2.553/4.495 points and endpoint weight displacement was 21.485%/23.729%. The
OneCycle LR was still 0.03442 after epoch 45 because the released scheduler
horizon is 50.

Decision: retain the runnable repo's 50 independent lineages x two terminal
snapshots. This is justified by the executable contract and greater lineage
diversity, not by convergence after two epochs or complete stationarity of the
late window. Group terminal snapshots by lineage for splitting and uncertainty.
The LR=0.7 boundary choice is not a scientifically frozen optimum: controlled
production tuning must use lineage-safe validation and a boundary diagnostic.

Reviewed evidence and exact reproduction commands are in
`projects/shared/storage/artifacts/weightclip_protocol_probe_20260818/README.md`.

### 19.12 Production AE architecture approval gate (2026-08-18)

The parameter target can be spent on materially different architectural axes,
so matching approximately 700M parameters does not determine a unique model.
The implementation must profile several equal-budget alternatives rather than
silently choosing one. The current required alternatives emphasize balanced
width/depth, depth, width/FFN capacity, and a larger fixed latent payload.

The canonical profiler/config/launcher locations are:

- `big_vae/weightclip_benchmark/ae_scaling.py`;
- `training/weightclip_benchmark/profile_ae.py`;
- `training/weightclip_benchmark/train_ae.py`;
- `conf/weightclip_benchmark/ae_700m.yaml`.

Every candidate is composed through the exact production Hydra config and then
instantiated on the meta device. It gets an exact trainable,
frozen and component-wise parameter ledger, representation-rate ledger, memory
lower bound and SHA-256 of the fully resolved model config. The production
launcher requires both an explicit enable flag and a separate user-approved
JSON whose candidate name, model fingerprint, tile geometry, 500k-step horizon,
scientific-config SHA, candidate artifact-set/report SHA, exact parameter count,
operator-bank pair SHA, successful bounded-runtime summary SHA and transitive
source-implementation seal all match exactly. Scientific overrides after
approval are forbidden. This gate
exists specifically so a future agent cannot start the run merely because the
total parameter count is close to 700M.

The first bounded legacy launch exposed a pre-step import-ownership bug:
`worker.py` called `_prepared_batch_prefetch_blockers`, but the helper lived in
`presliced.py` and was omitted from `data.py::__all__`, so the wildcard import
did not bind it. The worker now imports that helper explicitly. A CPU regression
imports it through the worker namespace and traverses a real operator-bank
background-prefetch batch, masks, forward/backward, and optimizer step. The
older `647a...` candidate bundle is therefore source-seal stale.

The next bounded attempt exposed a separate parity issue before step 0. The
launcher legitimately rewrites `logging`, `training_artifacts`, and eleven
explicit output-path interpolations, while immutable JSON redacts `hf.token`.
The parity gate now removes only that enumerated operational path allowlist and
applies the shared credential redactor symmetrically to active and stored
configs. Model, loss, LR, optimizer, scheduler, and data-selection mutations
still fail closed. On failure it writes a secret-safe structural diff containing
field paths, types, and value hashes rather than credential values. Full Hydra
compose/configure/write/read regressions cover empty and fake HF tokens. The
intermediate `b76b...` bundle is source-seal stale.

The meta-device profiler was re-run from the exact Hydra-resolved contract on
2026-08-19. Its current unselected immutable outputs are stored under
`projects/shared/storage/artifacts/weightclip_benchmark/ae_scaling_profile/`
`candidate_profiles-3c18deaa8d067dd1/`. Two consecutive invocations produced
the same content-addressed directory and hashes. Artifact-set SHA is
`3c18deaa8d067dd159aa410ff5d234781eed83f79968b12977ed1cb799379589`,
report SHA is
`f30bef12e0815cc8b27e8ba7c4996cca9320ae055a7cb96c1514500baf91e5d9`,
artifact-index SHA is
`cde4ac76d706bc21802da7a452dddc6fe7bc4bdf902c3e8b410bcdea73469528`,
summary-CSV SHA is
`9c30a57e4fde2654ee31acdf113312833745b4c2ac925c52c03530bc4d997d0b`,
and the 158-file transitive source seal is
`7a109801bd0a5b2bcb004b69fa297bfb0fb100bdcb1bc39ae3eccd20e0e34152`
(seal-artifact SHA
`464ec0cd65ac1793e9cac45fe1cd2b1e848244cb158f56b9c742ee71cea9349e`).
The supersession ledger SHA is
`c6c07caddbcb6e6603e2c7df2d13e5b71d52952e96ff50ff0d9cf0e81a587560`.
Root-level legacy artifacts and earlier content-addressed bundles are explicitly
superseded with per-artifact reason codes; the prior `85e1...` bundle is stale
because its 156-file implementation seal predates the final profile/bank
provenance guards. In particular, `candidate_profiles-133b41aa341dc5e4` is stale: its
generic secret redactor incorrectly replaced scientific fields containing the
substring `token`. The credential matcher now preserves those scientific
fields, and the stored report independently recomputes to the artifact-set SHA
above. The candidate-index validator now performs this same canonical
recomputation, and the bounded launcher validates the complete worker summary
before printing completion. The following counts/fingerprints are retained as
the historical first profile panel. They were superseded after review showed
that their latent capacity was not matched to WeightCLIP:

- `legacy_ratio_768_e24_d8_ffn4_l32`: 701,967,641,
  `a2314118877304c10b11d32c8a707bd8081ac6d0cc459ad81ea5711c88cdc503`;
- `deep_640_e32_d16_ffn4p6_l32`: 700,925,473,
  `e9e913bf71daad3f36e7a6073933528a5d3ea891b2718b6814c995183de4d9d9`;
- `decoder_heavy_864_e14_d20_ffn4_l32`: 705,873,871,
  `6f00b4848889d78ab167d92ab2c58ffe30d509440a735a6c9e3f9b7eca3b9fd7`;
- `latent_800_e20_d10_ffn5p2_l64`: 701,172,373,
  `fdee8e4a29dd35faf7a1c00c82469e78e93c8e004931957141c1d4fce0058144`.

These historical artifacts are **not approvals**. No candidate is
selected and `production_launch_enabled_after_approval` remains false. The
operator-bank *builder* has now been measured (§20.5), but the actual 700M AE
worker step/GPU-utilization profile is still missing. Present the candidates
and that production-shaped AE throughput to the user before creating the
separate approval JSON or launching the 500k run.

The replacement configuration fixes every candidate to `32 x 400 = 12,800`
latent scalars per 128x128 operator tile. On the actual operator-bank tile-mask
distribution this gives `1.1294689` valid weight scalars per latent scalar,
within `0.78%` of WeightCLIP's padding-adjusted sparse-codec rate `1.1383923`.
The four replacement trainable counts are 698,749,881; 702,148,225;
704,397,551; and 700,959,053. They require new runtime profiles; the historical
speed/OOM panel cannot select among the replacement models.

## 20. Implementation and execution handoff (2026-08-18, no scientific result)

This section records code state only. Passing tests and materializing data do
not establish a benchmark win, reconstruction quality, useful `z_task`, or
flow quality.

### 20.1 Implemented contracts

- The reconstructed-zoo launcher, immutable lineage/checkpoint manifests,
  official LR sweep, complete FP32 checkpoint archive, rolling resume state,
  and lineage-safe `35/7/8` split are implemented in
  `training/weightclip_benchmark/build_zoo.py`.
- The activation/operator bank, content-addressed shards, train-only native
  activation provenance, graph-consistent five-view gauge augmentation and
  mmap reader are implemented in `build_operator_dataset.py`,
  `big_vae/weightclip_benchmark/manifests.py`, and
  `big_vae/datasets/operator_bank.py`.
- AE consumption uses a step-addressed logical operator-bank stream. Resume is
  derived from the last committed optimizer batch, so DataLoader prefetch does
  not advance the restored stream. Model, optimizer, scheduler, scaler, RNG and
  step restoration remain mandatory. This deterministic committed-index
  guarantee applies to the operator-bank mode; do not generalize it to old data
  modes whose checkpoint metadata still says the stream is not exactly
  restorable.
- The production AE launcher resolves the complete data/loss/device/resume
  contract and is double-gated by `production_launch_enabled_after_approval`
  plus an exact user-approved candidate/fingerprint JSON. Protected model,
  data and loss overrides are forbidden after approval.
- Functional ResNet assembly, generated-prefix decoder context, grouped
  checkpoint `z_task` fitting, four conditional flows, EMA/hash-sealed loading,
  Euler/Heun solvers and common materialization hooks are implemented. Decoder
  context is built only from a manifest-recorded train-image pool. A task or
  validation image cannot recondition weights during forward evaluation.
- Task-fit bundles bind the exact decoder checkpoint and resolved codec config.
  Launchers reject a reconstructed decoder whose checkpoint/config hashes differ
  from bundle-build provenance. Ours and WeightCLIP use the same configured
  task batch size, one minibatch per optimizer step, and a separate full
  validation pass; WeightCLIP must not optimize on its entire train set per step.
- Every grouped `z_task` record carries one canonical codec fingerprint. A flow
  dataset must have exactly one fingerprint, and the same value is bound into
  its normalizer, checkpoint and v2 seal. Runtime recomputes the fingerprint
  from the actually loaded decoder and rejects codec-A-flow/codec-B-decoder
  composition before sampling. WeightCLIP fingerprints bind learned checkpoint,
  dataset encoder, pinned code/contract and learned model/tokenizer config, but
  deliberately exclude reference checkpoint, class count, device and runtime
  data/override paths so different datasets under the same codec remain compatible.
- The WeightCLIP extension uses the official `WindowedDataset` substrate: sparse full-model,
  `num_windows_per_model=auto`, padded, consecutive, non-overlapping windows of
  at most 512 tokens. Encoder and decoder attention stay within each official
  window; decoded windows are reassembled in exact order before one
  `tokenizer.detokenize` call. The obsolete giant full-sequence path is
  forbidden. The exact released single-window mapper is diagnostic only; no
  multi-window direct/memory/retrieval or flow row may be labeled “released
  official baseline.”
- Controlled WeightCLIP `z_task` uses the official body-token mask and has no
  classifier/padding scratch degrees of freedom. For **both** Gaussian and
  paired WeightCLIP flows the same controlled-body mask drives normalization,
  transformer attention, velocity/loss, endpoint metrics and sampling. Before
  decoding, non-body rows are restored bitwise from a hash-bound non-trained
  architecture template for Gaussian flow and from clean `z_enc` for paired
  flow. The complete 25 x 512 official sequence is retained only as grouped
  decoder context and for the native/full-window versus body-effective rate
  ledger; it is not extra controlled-flow capacity.
- Controlled and native WeightCLIP materializers are separate functions. The
  controlled key policy is decoded conv + decoded BN affine, fresh paired head,
  reset/common-calibrated BN running state. The native table retains the decoded
  released head and BN affine, while BN running statistics are deliberately reset
  and recalibrated for up to 200 batches, as fixed by the protocol audit.

### 20.2 Current data stage

As of 2026-08-18 18:37 UTC, all ten source `dataset.pt` files and all six OOD
`dataset.pt` files existed with their manifests. The current H100 process was
executing:

```bash
python -m training.weightclip_benchmark.build_zoo all \
  --config conf/weightclip_benchmark/zoo.yaml
```

At the latest reviewed observation (2026-08-19 00:40 UTC), Artworks, Blood
Cells, Breast Cancer Tissues, Aerial Cactus, Cassava Leaf and CT Images were
complete at 50 lineages each; Land Use also completed and passed its full audit
(350/500 total), while Lego Bricks completed its released sweep and entered
final-lineage training. The first seven completed datasets were checked for the
expected checkpoint/metric/init inventories and no NaN/inf metrics. Aerial
Cactus has 2,250 epoch states,
2,250 metric rows and 50 initializations; final test accuracy mean is 99.8633%,
population std 0.0717%, range 99.6281--100%, and the exact recomputed released
sweep selects LR 0.2. Its legacy live-process `lr_sweep.json` predates the new
dataset/protocol-hash fields, but every lineage resolved config binds the
dataset SHA and full training knobs. Breast Cancer Tissues had final test
accuracy mean 69.346%, population std 1.472%, range 65.786--72.327%; the
released test-label selection rule chose LR 0.7. This is a live progress
milestone, not the final ten-dataset integrity audit.
Cassava Leaf likewise has exactly 50 initializations, 2,250 FP32 epoch states
and 2,250 metric rows; the exact 7-LR x 2-seed sweep uniquely selects LR 0.5.
Its final test accuracy is mean 65.26483%, population std 0.88869%, range
63.00794--67.21158%, with zero non-finite values across 22,500 numeric metric
fields. All lineage configs bind dataset SHA
`9908da4125bd24ac1fb6e4ed91037774772683b0db4a741109ba88cfefdb5654`
and the complete frozen training protocol. Its only anomaly is the same known
legacy omission of dataset/protocol SHA fields in the old-code `lr_sweep.json`;
the recomputed sweep and downstream lineage records are internally exact.
The CT Images released 7-LR x 2-seed sweep reached exactly 100% for every
candidate and selected LR 0.05 by the frozen lower-LR tie break. Its complete
inventory is 50 initializations, 2,250 FP32 epoch states and 2,250 metric rows;
all 50 final checkpoints are exactly 100%, and all 22,500 numeric metric fields
are finite. This surprising saturation
was checked rather than treated as evidence of a healthy dataset: an immutable
exact-content audit found zero repeated float32 images within train,
validation, or test and zero exact pixel-hash overlap between any pair of
splits. Across all 50 lineages, epoch-1 test accuracy is exactly 46.9925%; the
epoch-3 mean is 93.10%, first-ever 100% occurs during epochs 4--10, and the
first permanently-100% epoch ranges from 5 to 39. Thus saturation occurs after
learning rather than at initialization. The supported conclusion is only that
exact-image leakage and an immediate initialization-level leakage signature
are excluded. Perceptual near-duplicates and patient/subject-level leakage
remain not established because the source artifacts expose no patient/group
IDs; the perfect terminal score therefore remains suspicious but currently
has no demonstrated contamination mechanism.
Evidence:
`projects/shared/storage/artifacts/weightclip_zoo_audits/ct_images_exact_content_leakage_audit.json`
(SHA-256
`66bce15a86155c0fd594d6d2e9ecc3d104b74981f5b103ff03559640aeabb7ce`).
Land Use has exactly 50 initializations, 2,250 FP32 epoch states and 2,250
metric rows, with zero non-finite values across 22,500 numeric fields. Its
independently recomputed 7-LR x 2-seed sweep uniquely selects LR 0.5. Final
test accuracy is mean 90.73807%, population std 0.63963%, range
88.79023--91.84240%; final validation is 91.96446% +/- 0.63308%. Every
lineage/config/completion/checkpoint identity matches dataset SHA
`1efabd43ce21206a60e9fd9abab4f4504a0c8412988c8a0a098e340a8e8f9af2`
and the frozen protocol. Immutable evidence:
`projects/shared/storage/artifacts/weightclip_zoo_audits/land_use_zoo_audit_summary.json`
(SHA-256
`a1495a859067130ed436449d295978c34080358a183921109807a5df57eda396`).
The Lego Bricks sweep is also exact 7 x 2: LR 0.3 and 0.7 tie on the released
selection score and the frozen lower-LR rule correctly selects 0.3; terminal
candidate scores span 97.3354--98.4326% without 100% saturation. This is a
live sweep milestone, not yet a completed-dataset audit.
Therefore source and OOD materialization are complete at the file level, but
zoo training, final immutable zoo manifests and operator banks remain **not
complete** until all 500 outputs and hashes are inspected. Process IDs and
elapsed time are volatile and are not evidence of completion.

Read-only retention audit snapshot (`2026-08-19T01:05:31Z`): 400/500 complete
lineages (eight datasets x 50) contain 18,000 epoch files totaling
202,507,398,000 bytes (188.600 GiB), 400 initializations totaling
4,500,446,000 bytes (4.191 GiB), and 50 completed-lineage resume files totaling
1,127,369,150 bytes (1.050 GiB). Metrics/config/completion metadata total less
than 7 MiB; the source and OOD dataset trees occupy 1.787 and 0.847 GiB. At the
same measured sizes, the completed 500-lineage archive projects to 5.238 GiB
of initializations plus 235.710 GiB of 22,500 epoch states, or 240.948 GiB for
all 46 model states and about 243.589 GiB for the root including source/OOD
data but excluding stale resumes. Retaining a completed resume for every
lineage would add about 10.499 GiB.

The exact current-paper checkpoint requirement is all 1,000 primary states at
zero-based indices `{43,44}` (one-based epochs 44/45): 700 train, 140
validation and 160 test. The operator bank consumes exactly the 700 train
states. By user decision, initialization, complete configs, all 45 metric rows,
completion records, LR-sweep evidence, dataset/zoo manifests and hashes also
remain retained. Completed-lineage `resume.pt` files are not downstream
inputs, but nothing may be deleted before the final 500-lineage manifest/audit
and explicit user approval.

Three recoverable retention candidates are recorded for later approval; **no
pruning decision has been made**:

- floor: initialization plus one-based epochs `{44,45}` (three states), about
  15.714 GiB of states and 18.356 GiB for the projected root;
- recommended raw-FP32 candidate: initialization plus
  `{1,4,8,16,24,32,44,45}` (nine states), about 47.142 GiB of states and
  49.784 GiB for the root, with no codec dependency;
- compression study only: initialization plus
  `{1,4,8,16,24,32,41,44,45}` (ten states), 52.380 GiB raw. A single-trajectory
  full-state zlib measurement was only 1.076x, projecting about 51.322 GiB for
  the root; this estimate requires a stratified preflight and must fall back to
  the raw nine-state policy if it misses budget. A richer 12-state XOR-delta
  option measured 1.281x on only one trajectory and is too weakly established
  to recommend.

Raw SHA dedup currently saves nothing across the 400 initializations (400
distinct serialized-file hashes). One cross-dataset seed-1 pair was tensor-wise
bit-identical despite different `.pt` bytes, so semantic dedup may exist, but it
is not budgeted: it needs complete tensor digests and changes byte-level
recovery provenance. A future immutable retention manifest must bind every
original relative path/size/SHA, logical dataset/lineage/split/seed/epoch,
primary flag, semantic state SHA, keep/drop reason, stored object SHA/path,
codec/base dependency, final zoo-manifest/config/dataset/source hashes, exact
counts/bytes, tool source SHA and the user's approval. Required sequence is:
finish all 500 -> build and audit the forward zoo manifest -> dry-run the
retention inventory -> obtain approval of the exact policy SHA -> materialize
and verify the content-addressed object store -> verify recovery -> only then
quarantine/prune against an unchanged live inventory. Shell globs are not an
acceptable substitute.

Implementation update: a standalone, manifest-driven v1 CLI now exists at
`training/weightclip_benchmark/zoo_retention.py` with frozen config
`conf/weightclip_benchmark/zoo_retention.yaml`. Its default stage is `plan`.
`plan` requires the caller to supply the exact final zoo-manifest SHA, rejects a
live `build_zoo` process, requires the immutable 500-lineage/23,000-state
manifest with every-checkpoint forward audit, validates its lineage/checkpoint
JSONL hashes, exact `35/7/8` splits, dataset artifacts and every checkpoint
size/SHA, then writes a content-addressed plan outside the zoo. `verify`
requires an unchanged file set and rehashes every planned file. Both stages are
read-only with respect to zoo data. The plan binds a source inventory for the
retention CLI and its local producer/manifest dependencies:
`zoo_retention.py`, `build_zoo.py`, `manifests.py` and `metadata.py`; changing
any one changes the plan fingerprint and invalidates later verification.

Both stages check for a live builder before the expensive scan and again after
it, then recheck the final-manifest SHA, referenced lineage/checkpoint manifest
SHAs and the exact inventory path set. `verify` hashes every planned file once;
each hash is enclosed by identical before/after regular-file stat snapshots
containing device, inode, size, `mtime_ns` and `ctime_ns`. Those snapshots are
persisted per file, compared for the full inventory after the scan, and checked
the same way during verification. This detects same-path/same-size replacement,
metadata drift and a writer active during hashing. The post-scan barrier
deliberately does not hash the approximately 240 GiB a second time. A narrow
filesystem race remains between its final stat/SHA checks and any future
consumer. This is recorded rather than hidden and is harmless in v1 because
destructive `apply` is disabled; a future apply implementation must perform its
own immediate atomic precondition check.

The frozen policy is the raw grid9: initialization plus one-based epochs
`{1,4,8,16,24,32,44,45}`. It preserves all 1,000 primary epochs 44/45 and every
config, metric, completion, LR sweep, manifest, dataset, OOD dataset,
operator-bank/unclassified auxiliary artifact. Existing `resume.pt` files are
also preserved under a separate `rolling_resume_preserved_pending_separate_decision`
ledger; the approximately 49.784 GiB projection excludes those resumes, so the
actual v1 retained bytes may be higher until the user separately decides their
policy. `apply` is intentionally non-operative in this version: it validates
an exact approval binding and then refuses. No move, compression, quarantine,
deletion or reclaim implementation has been approved. A future destructive
version must use atomic quarantine, verify it, perform a recovery drill, and
obtain a separate reclaim approval before releasing bytes.

Only after the live zoo finishes and the final manifest SHA is independently
reviewed may the read-only stages be invoked:

```bash
python -m training.weightclip_benchmark.zoo_retention plan \
  --config conf/weightclip_benchmark/zoo_retention.yaml \
  --final-zoo-manifest-sha256 <EXACT_REVIEWED_FINAL_MANIFEST_SHA256>

python -m training.weightclip_benchmark.zoo_retention verify \
  --plan <retention-plan-PLAN_SHA_PREFIX.json>
```

Do not invoke `apply`: it is intentionally disabled, and a plan/verification
artifact is not a pruning approval. This implementation handoff ran only
synthetic/mocked filesystem tests; it did not scan/hash the live zoo into a
real plan artifact.

The `materialize-ood` stage has completed. It verified `raw_m_test.tar.gz`
against SHA-256
`38bb8b7b03fad9b2291ffc521d708525d01c9297d668d63d3619f605df210eb3`,
streams without extraction, and selects exactly:
`colorectal-histology` (8), `covid19` (3), `speed-limit-signs` (4),
`honeybee-pollen` (2), and `real-or-drawing` (10). `cifar10` (10) is loaded
through `torchvision.datasets.CIFAR10`, with the official 50k train split
partitioned deterministically 45k/5k at seed 0 and the official 10k test split.
The OOD manifest records dataset hashes, raw provenance, train/validation/test
sizes and `SEALED_NOT_EVALUATED`. These files are sufficient to derive a
train-only dataset embedding/context for anchor-free OOD conditioning; no
target checkpoint is required or permitted during OOD conditioning.

### 20.3 Stage-D/G gates and remaining runtime blockers

Stage D/G now has an executable artifact-driven resolver at
`training/weightclip_benchmark/resolve_stage_dg_dag.py` with frozen configs in
`conf/weightclip_benchmark/stage_dg_pipeline.yaml`. It discovers immutable
upstream indices and emits commands for source bundles, matched `z_task` fits,
the four flows, four decoded E4 validation jobs over the exact matched 10 x 7
validation inventory, one global E4 selector, four flow seals, train-only OOD
conditioning, train-only target anchors,
codec-specific anchor `z_enc`, the WeightCLIP common-zoo full-window bank,
ridge/memory/nearest-code producers, the exact expected evaluation grid,
candidate manifest, explicit OOD unseal, and evaluation. Missing upstream
artifacts block a stage; there is no synthetic fallback. The only intentional
human gates are the approved AE seal and the later explicit OOD-test unseal.
The resolved graph currently has 33 stages, and its reachability check requires
every candidate producer to depend on all four flow seals.

OOD target anchors are trained from the target train split only, with a
precommitted source-policy LR. Their cache contract binds dataset, full frozen
training protocol, LR, seeds and checkpoint bytes. Both ours and WeightCLIP
anchor-codec bundles are then built from the same sealed anchor. Ours captures
operator activations on the fly from only the recorded train context; it does
not falsely query the source operator bank for an OOD checkpoint.

The candidate spec carries an exact method x dataset x seed x protocol grid.
Candidate generation, unseal and reporting all reject missing or extra cells.
The final paper grid uses the same precommitted seeds `[0, 42, 777]` for every
arm. Every generative flow arm has controlled single-sample,
validation-selected 100 -> top-5 and separately labeled test-selected 100 ->
top-5 estimates. Candidate IDs and immutable selection commits include the
protocol, preventing cross-table collisions. Group results are committed
immutably before advancing, so an interrupted evaluation resumes without
reopening already completed test groups and refuses config/manifest/unseal drift.
Unseal parses four distinct schema-v4 flow seals, requires the exact
ours/WeightCLIP x Gaussian/paired set with one shared solver/NFE selection, and
checks every flow candidate against the bound flow checkpoint, normalizer and
AE seal. Controlled secondary and native-oracle estimates both retain their
frozen 100 -> top-5 budgets; controlled test access begins only after an
immutable validation-selection commit.

For rapid week-one diagnostics, the resolver also emits a separate
`exploratory_single_seed0` candidate configuration and experiment label. It
contains only controlled single-sample seed-0 cells. Its expected-grid kind is
different, and the OOD unseal/evaluator hard-reject it as a final paper
benchmark; it cannot silently substitute for the three-seed/100-candidate grid.

The common-zoo WeightCLIP memory mapper must use microbatch 1. The released
implementation allocates full `[batch, 12800, source_count]` logits before any
token subsampling, so its default batch 320 is forbidden by a hard runtime
gate. Every ridge/memory/nearest-code artifact records prompt indices,
embedding/code hashes and the fitted prior/source-bank provenance.

1. Inspect the completed ten-source-dataset materializer manifest; then finish
   the official sweeps, 50 lineages/dataset, and the manifest audit.
2. Build and inspect the immutable train operator/context banks and fill
   `data_contract.pair_manifest` with their exact content-addressed path.
3. Use the current AE meta-profile recorded in section 19.12, add the missing
   production-shaped throughput measurements, and obtain explicit user approval
   for exactly one current fingerprint. Production AE remains forbidden before
   this; any later protected-contract change invalidates the profile and forces
   a re-profile.
4. Run the production-shaped loader/memory/throughput gate. Only then may the
   approved 500k AE be launched.
5. The official approximately 9 GB WeightCLIP checkpoint and released dataset
   encoder are present at their pinned hashes; the real tokenizer-only
   WeightCLIP `z_task` CLI load path has passed. Keep these exact hashes pinned.
6. Build manifest-recorded train-only decoder-context pools and grouped
   position-aware architecture features for every task-fit record.
7. Execute and inspect every ready Stage-D/G node. The resolver being
   code-complete is not evidence that the produced artifacts or metrics exist.
8. No `z_task`, flow, downstream curve or scientific comparison has yet been
   established by this implementation handoff.

### 20.4 Frozen implementation verdict and residual production work

After the final hostile pass, the implementation verdict is **GO** and the
production verdict remains **NO-GO**. After the final integrity-complete
operator-bank changes, root verification is `115 passed, 1 skipped`, Ruff
clean, compileall clean; an independent hostile re-audit gives the configured
operator-bank builder **GO** with no remaining P0/P1. The independently
resolved DAG has 33 stages, 44 uniquely produced artifacts, no dangling
generated inputs and no unresolved placeholders. The final canonical grid is
6 OOD datasets x 3 seeds x 24 method/protocol cells = 432 cells and 25,380
candidates. The resource-gated real official WeightCLIP checkpoint test passed
separately at checkpoint SHA-256
`73b2a5de1a9a167145ecdfe3684725c9094ed766156589d4a00647e21128d95a`.

Production is blocked only on real artifacts/runtime gates, not missing core
code: finish and audit 500 zoo lineages; build the final checkpoint manifest and
operator bank; profile the production operator-bank loader on the H100; obtain
the exact AE candidate approval; then execute and review `z_task`, four flows,
E4/seals and Stage G. Remaining non-blocking engineering risks are repeated
large-codec loads in per-bundle `z_task` subprocesses, mapper resume after a
mid-run crash, accurate generation-time accounting, and safe sharding/timing of
the 25,380-candidate Stage-G evaluation. Full-window WeightCLIP rows must remain
labeled as our `WeightCLIP codec + common-zoo full-window prior` extension, not
as the unavailable exact released multi-window mapper baseline.

Data-preparation command (it does not unseal OOD evaluation):

```bash
python -m training.weightclip_benchmark.build_zoo materialize-ood \
  --config conf/weightclip_benchmark/zoo.yaml
```

### 20.5 Operator-bank numeric gate and measured builder throughput

The first real one-checkpoint build initially failed the five-view gauge gate
under default CUDA arithmetic with maximum logit drift `2.043724e-3`. This was
not a graph/permutation defect. The discriminating audit found:

- gauge then inverse-gauge restores all 122 state keys bitwise;
- CPU FP64 drift is `7.105e-15`, CPU FP32 drift is `5.245e-6`, and CUDA with
  both cuDNN and matmul TF32 disabled is `4.262e-6`;
- the first discrepancy is `layer1.0.conv1`, the first convolution whose input
  channel reduction order changes, and it compounds through later layers;
- deliberately incorrect BN affine/running-state, shortcut, classifier and
  inverse-axis variants produce errors of approximately 8--49.

Therefore the supported mechanism is TF32 finite-precision reduction-order
non-equivariance, while BN/head/shortcut/orientation defects are excluded by
intervention. The verifier now disables both TF32 modes only inside a scoped
strict FP32 gate, restores the original backend flags in `finally`, retains the
original `2e-5` tolerance, checks the inverse-gauge state roundtrip bitwise and
persists both strict-verification and original-runtime drift. Evidence:
`artifacts/weightclip_benchmark/gauge_falsification_20260818/` and
`projects/shared/storage/artifacts/weightclip_benchmark/operator_bank_profile_one_checkpoint/gauge_numeric_diagnostic.json`.

Context summary was separately measured as the original builder bottleneck:
the per-checkpoint CPU loop took about 129 s. Batched float32 summaries reduced
that work, but the host default of 32 intra-op threads still caused severe
small-kernel oversubscription. A thread sweep identified four threads as the
measured operating point. The v4 build contract now hash-binds summary device,
Torch/CUDA/GPU identity and effective intra-op thread count, and restores the
caller's thread setting after a library invocation.

The final integrity-complete real Artworks checkpoint profile (`129` context
records, `193` weight tiles) took `13.1038 s` inside the immutable builder
versus the original `137.7 s` (`10.51x` faster; `14.8371 s` including caller
setup and input hashing). Activation capture took `2.637 s`, strict gauge
verification `1.882 s`, context summary `1.612 s`, and serialization `0.982 s`.
At this deliberately integrity-heavy observed builder rate, 700 train
checkpoints project to about `2.55 h` sequentially. The final contract
also binds the resolved activation-dataset root and SHA for every used
`dataset.pt`, and independently rehashes every selected checkpoint against the
SHA declared by the checkpoint manifest before either a fresh build or cache
reuse. It also binds all gauge thresholds/sample counts, strict-verifier schema,
and the actual TF32/matmul/determinism policy. A matching cache hit rehashes the
pair manifest, both embedded bank contracts, every shard, permutation evidence
and coverage file; corruption fails closed. This verified cache path took
`2.021 s` for the one-checkpoint bank. The immutable contract SHA is
`66574d0cdf94d3d47e172f7c7cc618eadf0210472e2bb18920d0ae8bf49be672`;
the complete reports are
`projects/shared/storage/artifacts/weightclip_benchmark/operator_bank_profile_one_checkpoint/profile_result_final_integrity_v6.json`
and `verified_cache_hit_final_integrity_v6.json` in the same directory.
This establishes builder feasibility only. It does **not** establish the
required input-wait/GPU-active gate for a 700M AE optimization step.

The later `reader_capacity_profile_replacement_v2.json` must not be used to
close that missing gate. It is only a direct-tile `MMapOperatorDataset`
microbenchmark: 588.63 records/s at reader batch 8, with 4.94 ms mean iterator
wait and 1.21 ms synchronized W+x H2D per batch. It bypasses the production
full-operator reconstruction, graph-gauge permutation, re-tiling, hierarchical
committed sampler, IPC-batch-4 flattening, batch-32 assembly, three masks,
background pin/prefetch and five-tensor H2D path. It also fails its own
reader-only <=5% wait criterion (`36.38%`, false); its reported
`gpu_active_fraction=0.5475` is merely the wall fraction occupied by a trivial
square/mean section, not device utilization. The only supported interpretation
is narrow mmap-reader capacity on the one-checkpoint diagnostic bank.

The approval gate instead requires the exact resolved production worker path
and the final immutable bank. The bounded profiler must preserve the 500k
optimizer/scheduler horizon while separately limiting execution, disable all
checkpoint/tracker writes, and measure real AE microsteps together with exact
production data assembly. It must record pair/resolved-config hashes, committed
indices, all five tensor shapes/dtypes/bytes, CPU build and prefetch-wait tails,
queue depth, H2D latency, real step throughput, GPU utilization/VRAM and finite
losses. Input capacity is sufficient only if a sufficiently long
production-shaped measurement establishes a precommitted margin over AE
consumption. A bounded 32-step run can diagnose stalls and concurrent refill,
but cannot by itself establish a long-run 1.5x producer margin; a no-model
reader number also cannot establish this.

### 20.6 Bounded AE profiler and operator-stream locality gate

A dedicated bounded profiler is implemented in
`training/weightclip_benchmark/profile_ae_runtime.py` with minimal timing hooks
inside the actual `training/big_vae/worker.py` loop. It does not substitute a
short scheduler: `train.max_steps` remains exactly 500,000, while a separate
schema-checked stop condition permits exactly 32 optimizer steps (4 warmup +
28 measured). It requires single `cuda:0`, the final 700-checkpoint
train-only bank, exact pair SHA and candidate fingerprint; forbids resume,
auto-resume, fixed/synthetic/legacy data, Comet/W&B and every model/resume
checkpoint write; and kills the process group at a bounded timeout. The
profile traverses the real operator-bank/DataLoader/prefetch/five-tensor H2D
path and executes the production forward, loss, backward, optimizer and
scheduler. It writes immutable resolved config/contract, per-step timings,
NVML samples and summary. Focused verification after the first hard freeze was
23 passed. A legacy 700M/GPU attempt exists below, but it does not satisfy the
current bounded-profile evidence contract and cannot unblock approval.

The canonical AE YAML intentionally leaves `data_contract.pair_manifest` null.
The bounded profiler and approval-gated production launcher therefore require
an explicit finalized `--pair-manifest`; the profiler additionally requires the
exact content-addressed `--candidate-artifact-index`. Pair SHA, candidate
artifact-set/report/index SHAs and the current source seal are persisted in both
the immutable profile contract and worker summary, then rechecked by the
production launcher. This avoids mutating the pending canonical config and
prevents a runtime summary from one candidate bundle being reused with another.

A real bounded legacy attempt at
`projects/shared/storage/artifacts/weightclip_benchmark/ae_scaling_profile/runtime/`
`legacy_ratio_768_e24_d8_ffn4_l32_20260819T074551Z_pid684058/` completed its
old 20-step worker window and preserved `summary.json` (SHA-256
`e180564f5df1d53a4691a0d309da681f890e6b772d47f6d1efbfb79a361dab5d`)
and `step_timings.jsonl` (SHA-256
`6799d0da8960073b4941039c31f7bf7e7651498306d8a44d7b1969d56752c0f5`),
then correctly failed launcher post-validation. The exact mismatch was the
stored redacted `hf.token` versus the raw empty runtime token. Runtime parity
now applies the shared credential redactor symmetrically after removing only
the enumerated launcher-owned paths; scientific mutations remain fail-closed,
and future post-validation failures preserve an immutable failure artifact.

That old run is **not** evidence of sustained ingress capacity: its producer
cursor was already 640 at every measured step because the queue had prepared
the entire 20-step run during cold compilation. The frozen bounded protocol
therefore keeps the production queue size 24 and consumes exactly 32 steps,
but gives the profile-only producer a request horizon through step 84
(`32 + queue24 + measured28`). Only the first 32 steps may commit; prefetched
tail batches are discarded on close and cannot affect ordering, resume state,
or checkpoints. The summary must prove that the producer cursor advances in
the measured phase and ends strictly beyond the committed cursor, and must
report consumed, produced, discarded, queue, input-wait, and producer/consumer
rate ledgers. These are diagnostics, not an automatic no-starvation claim or
proof of a long-run 1.5x margin.

Before accepting the runtime profiler as an unblock, a locality audit found a
separate production input-path defect. On the one-checkpoint bank, the
hierarchical cycle has 20 operator groups/strata and 193 emitted tile records.
Strict stratum round-robin gives reuse distance 20 while each DataLoader worker
keeps only two materialized groups, producing 193/193 misses. It reconstructs
1,902 weight records and 1,263 context records instead of the ideal 193/129,
about 9.8x amplification. Warm-cache CPU measurement was 24.617 s (7.84
records/s) for the current stream versus 2.887 s (66.85 records/s) for an
exact-multiset group-contiguous upper bound, an 8.53x difference. With eight
workers and IPC batch 4, cache-size-only remains ineffective because caches are
worker-local. Evidence:
`projects/shared/storage/artifacts/weightclip_benchmark/operator_bank_profile_one_checkpoint/operator_bank_cycle_locality_audit_v1.json`
(SHA-256
`dd55278ae43171b571860c045c39bd177c47ee026741ac14078e914225da3183`).

Naively grouping tiles is not accepted because it destroys the intended
short-window dataset/op/depth/role balance. The required fix is bundle-level
full-operator materialization in workers plus a deterministic main-process
balanced tile mixer. The mixer must retain the exact exposure targets,
checkpoint-wide gauge consistency, bounded buffering and committed-index resume
suffix. Production input and 700M profiles remain **NO-GO** until this fix has
materialization-count, cross-cycle, worker/IPC invariance, batch-diversity and
resume-suffix tests and is re-profiled on the final bank.

Implementation status after the locality hardening pass: the worker now
materializes one full operator/gauge bundle, while a deterministic main-process
mixer preserves the *exact old global dataset/op/depth/role stratum sequence*
and only groups occurrences within each stratum by checkpoint/layer key. Thus
the exact tile multiset and checkpoint-wide canonical/five-view gauge are
preserved. Lineage/checkpoint temporal order inside a stratum is intentionally
changed and is **not** a preserved invariant. Each launch writes an immutable
old-vs-new per-32 diversity audit (unique lineage/checkpoint p5/mean, maximum
same-lineage run and adjacent-lineage autocorrelation); these are recorded for
approval, with no invented pass threshold. Resume is fail-closed on a
hash-bound stream contract and reconstructs the exact committed tile suffix.
The IPC path is bounded to one bundle per item and a two-item worker prefetch;
the immutable/runtime ledgers contain active-stratum, in-flight and total byte
bounds plus observed mixer maxima.

The bounded profiler now binds the active worker config to the immutable
resolved config, binds the profile contract SHA, actual model config and actual
parameter counts, and records cursor values from the live locality mixer rather
than arithmetic estimates. Consumed prepared-batch logical ranges and the
prefetch producer-ahead cursor are separate provenance fields. CPU producer timing is split into fetch/reconstruct,
pinning and total producer time; H2D is explicitly enqueue time, CUDA events
measure the active stream interval, and NVML is started during warmup so its
startup overhead is excluded and reported separately. The final-bank gate
requires ten datasets, 35 lineages per dataset and exact `{43,44}` checkpoints
per lineage, 700 unique logical identities and canonical paths, with checkpoint
path/SHA and coverage SHA validation. Every manifest-declared array-bank shard
payload is size/SHA verified; undeclared shard directories or payload files are
rejected and mmap opens only the manifest-derived inventory. Process tensor
memory includes active/in-flight bundles and a conservative
`prepared_queue_size + 3` batch-equivalent bound covering fetch/stack, pin-copy
and consumer residency. Torch `cuda:0` is bound to the physical NVML UUID, and
the live trainable parameter count must be within 1% of the approved 700M
target. Focused CPU
verification for locality, resume, workers 0/8, IPC 1/4, profiler safety and
legacy prepared-batch compatibility is 35 passed. A separate root-run full
`tests/weightclip_benchmark` verification after the main hard freeze is 146
passed, 1 skipped; after the final provenance hardening, both affected files
pass 38/38 independently. Python compilation, strict Ruff on the non-legacy
touched scope and `git diff --check` pass. A read-only hostile re-audit reports
no residual P0/P1 and gives **GO** to the locality and bounded-profiler
implementations. This is implementation evidence only: the final-bank 700M
H100 profile has not run, so the production throughput gate remains **NO-GO**
until that bounded profile is reviewed.

The exact-700 contract is enforced before the builder creates its output root
and, for malformed structure, before any expensive checkpoint hashing: ten
declared source datasets x 35 train lineages x exact zero-based indices
`{43,44}`; unique logical triples and canonical paths; `split=train`;
`is_primary=true`; existing paths; strict lowercase 64-hex declared SHAs; and
actual file/SHA equality. The protocol itself fail-closes unless the same
`[train]`/`[43,44]` selection, 16 quantiles, five permutation views, float32 raw
and bank dtypes, and frozen 256-context/512-weight shard layouts are present.
Shard counts are provenance-bearing storage layout, not a free runtime knob.
Adversarial tests cover wrong dataset/lineage counts, split/primary flags,
indices, logical/path duplicates, missing per-lineage terminal pairs, malformed
and mismatched hashes, and every frozen P2 field. This gate and the reviewer's
final implementation **GO** do not assert that the 700-checkpoint bank exists,
that a 700M GPU profile passed, or that any scientific metric improved.

The operator-bank build is also fail-closed against concurrent or drifting
inputs. A process-wide nonblocking advisory lock covers preflight through the
atomic pair commit, so a second builder fails before staging. The pair contract
binds a deterministic source-inventory seal for the bank builder and its local
Python dependencies; both the runtime loader and bounded-profile preflight
recompute that seal before consuming the pair. Checkpoints, dataset tensors,
checkpoint manifests, the builder config, and source files carry stable
device/inode/size/mtime/ctime snapshots around hashing and loading, followed by
a global stability barrier before either bank is finalized. Drift aborts the
build and removes only incomplete staging directories. These are implementation
guards; no real 700-checkpoint bank was built while adding them.

## 2026-08-19: AE architecture selection and confirmation contract

The bounded AE profiler has two frozen, approval-relevant modes. Both execute
the exact production worker/model/data/loss path for 4 warm-up + 28 measured
steps while leaving the scheduler horizon at 500,000 and forbidding resume,
tracking and checkpoints. `production_exact` uses the production prefetch queue
24. `producer_stress` uses queue 4 only as an ingress stress intervention and
must pass advance >=832 logical tiles, producer/consumer span >=26/27,
input-wait/wall <=1%, and input-wait p95 <=5% of median host-step time. The
worker log is redacted before both live display and immutable storage; success
is written only after summary validation.

Historical q24 evidence is immutably snapshotted at
`artifacts/weightclip_benchmark/ae_scaling_profile/comparisons/comparison-f03faa46a939b1c4/comparison.json`
(SHA-256 `9dd055009c682955de5292a50ac415b4ec09e75ce4548484c16c0dd43096594a`).
It supports only a recommendation: legacy 1.137018855 step/s / 78,039 MiB,
deep 0.939292929 step/s / 93,071 MiB, decoder first-forward OOM, latent
first-backward OOM. Legacy is 21.05% faster, so only legacy qualifies under the
rule `rate >= 0.95 * fastest` for current-source q4 confirmation. The report is
explicitly non-authorizing and the historical runs cannot satisfy current
approval because their source seal is stale.

After current legacy q24 and q4 pass, the
`training.weightclip_benchmark.prepare_ae_approval_request` CLI reconstructs
and binds config/base/pair/index/report/source, exact parameter count, both
summary+launcher+log chains, and the historical comparison into an immutable
`pending` request. It never records user approval. A separate exact approved
JSON remains mandatory for production. Both comparison and approval-request
CLIs are part of the current 158-file AE implementation source seal; the
content-addressed `candidate_profiles-3c18deaa8d067dd1` bundle above was
regenerated twice idempotently after the code freeze. No GPU, bank or
production run was launched while adding this contract.

## 2026-08-19: exact pending-to-approved AE transition

The canonical `conf/weightclip_benchmark/ae_700m.yaml` remains immutable and
pending. A pending approval request is not an approval, and editing either file
by hand is forbidden. After an explicit user statement, the only valid
transition is:

```bash
python -m training.weightclip_benchmark.approve_ae \
  --approval-request <PENDING_REQUEST.json> \
  --canonical-config conf/weightclip_benchmark/ae_700m.yaml \
  --approved-config conf/weightclip_benchmark/ae_700m.approved-<REQUEST_PREFIX16>.json \
  --approval-file conf/weightclip_benchmark/ae_700m.user-approval-<REQUEST_PREFIX16>.json \
  --user-statement 'I explicitly approve WeightCLIP AE production candidate <CANDIDATE> for approval request <FULL_REQUEST_FINGERPRINT>.' \
  --approved-at-utc '<CANONICAL_UTC_TIMESTAMP_ENDING_IN_Z>' \
  --confirm-explicit-user-approval
```

The approved config and record must be beside the canonical config. The config
diff allowlist is exactly `status`,
`production_launch_enabled_after_approval`, `selected_candidate`, and
`approval_file`; any scientific or operational extra change is rejected.
The approval record binds the pending request path/SHA/fingerprint, canonical
and approved config paths/SHAs, candidate/model/parameter/scientific hashes,
pair, candidate index/report/artifact set/source seal, both q24/q4 summaries,
historical comparison, exact user statement, and timezone-bearing timestamp.

Production `train_ae` requires this request/config/record chain and propagates
its hashes into launch preflight and checkpoints. `seal_ae` independently
recomputes the chain and checks the checkpoint before producing an AE seal. A
conservative immutable disk report additionally accounts for the frozen 500k
run, 10k checkpoint cadence, FP32 model checkpoints, rolling Adam resume state,
atomic-write transient, 5% serialization allowance, and
`max(20%, 20 GiB)` free-space headroom; execute mode fails closed if it does not
fit.

Output names are derived from the full request fingerprint and immutable writes
are absent-or-byte-identical. Fresh production uses an explicit unique,
nonexistent checkpoint root; execute atomically claims it and the exact nested
resume root. The process lease is acquired before mutable-root checks and held
through the worker; only while holding it does the launcher revalidate
filesystem/free space, exact resume SHA, every archive SHA/stat, and the exact
numbered-checkpoint path set. Auto-scanning resume directories is forbidden. Resume mode requires the
exact immutable `step_<10k-aligned>.pt` full-resume artifact inside that root,
stable stat+double hash around loading, a committed operator-stream cursor, and
embedded candidate/model/scientific/source/pair/request/profile bindings equal
to the current approved chain. Its disk report counts only checkpoints still to
be written while reserving atomic overwrite transients.

Resume is also structurally fail-closed before execute: the model state must
match the approved meta-model key/shape/dtype schema and be finite; AdamW must
have an exact parameter-name order and finite `exp_avg`/`exp_avg_sq` for every
trainable parameter at step `K`; LambdaLR, disabled BF16 GradScaler,
Python/NumPy/Torch RNG, and operator stream state must all match `K`. The full
CUDA RNG payload contains only the active training device and binds its logical
index, canonical `GPU-*` UUID/name, and child `CUDA_VISIBLE_DEVICES=<GPU-UUID>`.
Canonical identity comes from Torch logical-to-NVML mapping plus `nvidia-smi`;
the opaque Torch 2.10 `_CUuuid` repr is invalid provenance. Resume validates
that state in a short-lived helper child masked to the UUID; the launcher must
remain CUDA-uninitialized and never enumerates/touches other GPU contexts. Legacy all-device RNG-list
checkpoints fail closed. The q24 and q4 evidence must share this UUID; both
profile and production children are UUID-masked. Execute holds a private
per-UUID flock while rejecting foreign compute processes and insufficient free
VRAM relative to the q24 NVML peak plus max(5% total, 4096 MiB) headroom.
The full
immutable numbered 10k model archive through `K` and `latest.pt` are separately
SHA/config/provenance/schema/finite validated, and their final states must be
bitwise identical to the resume model. Empty or partially loadable states are
rejected instead of silently resetting optimizer or scheduler state.

Adding this source-sealed transition makes the prior 158-file candidate bundle
stale. It must be regenerated only after code review. No approval, GPU run,
operator-bank build, or production launch occurred in this implementation pass.
