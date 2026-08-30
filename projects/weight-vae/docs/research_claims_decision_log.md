# Weight-VAE research claims: persistent decision log

Last updated: 2026-08-16

## Purpose

This file records the ongoing paper-framing discussion for BigVAE so that future
work does not restart from attractive but already rejected ideas. It is a
decision log, not a polished manuscript and not a verbatim chat transcript.

The target is an idea-level experimental finding about trained neural networks.
The user does not want the paper to be centered on reconstruction, generation,
optimization, retrieval, model ranking, or a visualization application. BigVAE
may be the measuring instrument, but the headline must say something important
about learning or about populations of trained models.

## User's standing preferences and project constraints

- Prefer experimental interpretability/science findings over a method-only paper.
- The interesting distinction from SANE is a single representation that is not
  trained for one fixed architecture/model zoo and may cross domains/modalities.
- Decoder reconstruction is currently too weak to claim that encode/decode
  preserves downstream model quality.
- Latent optimization is not competitive with raw Adam/AdamW.
- Therefore decoded-weight generation, latent optimization, model repair, and
  transplantation through decoded weights cannot be the paper's evidential core.
- A claim must answer "what does this teach the community?", not merely produce a
  downstream task for the latent.
- A complicated VAE must be load-bearing. If the finding is equally available
  from direct `X W^T`, CKA, SVD, weight/activation moments, or a small random
  sketch, reviewers can reasonably ask why BigVAE exists.
- There are two separate justification questions:
  1. Why any learned encoder instead of direct statistics?
  2. If only the encoder is used, why a VAE rather than a deterministic AE or
     another self-supervised encoder?
- Unless posterior density/uncertainty, KL, sampling, or the decoder adds verified
  value, position the object as an activation-conditioned operator encoder learned
  through variational autoencoding, not as a faithful generative weight manifold.

## Verified facts about the current checkpoint/evaluation

### What is positive

- The internal report describes visible held-out organization by source model and
  source dataset, but only as a representational result. It explicitly reports a
  negative latent-optimization result:
  `workspace/docs/report/SOLUTION.md`.
- The current held-out plot contains 256 slice-level points with a 512-dimensional
  dumped latent. PCA PC1/PC2 explain approximately 0.1264 and 0.0229 of variance:
  `workspace/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset/eval/stage_1_latest/latent_plots/latent_plot_summary.json`.
- The full evaluator processed 14,336 records and 182,488 slices, but the plotting
  dump defaults to at most one slice per source and the plot is slice-level, not a
  whole-layer/model representation:
  `workspace/post_train_research/big_vae_heldout_eval/run_evaluate_big_vae_heldout.sh`
  and `workspace/post_train_research/big_vae_heldout_eval/evaluate_parts/runner.py`.
- Stage-1 targets are `patch_size=16`, `max_T_patches=4`, and `max_d_out=64`, so a
  plotted point describes at most a local 64-by-64 slice, not a complete large
  layer:
  `workspace/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset/eval/stage_1_latest/metrics_summary.json`.

### Critical limitation of the current dataset-colored plots

- The held-out source index reuses an exact weight source with multiple runtime
  datasets. Examples include one `vit_large_p16_224` layer with
  `dataset_names=[chexpert, food101, sun397]` and one `clip_vit_l14` layer with
  four datasets:
  `workspace/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset/sources.json`.
- Consequently, current dataset separation primarily establishes sensitivity to
  activation context/sampling for largely fixed pretrained weights. It does **not**
  establish that weights trained independently on different datasets cluster.
- A new factorial corpus with independently fine-tuned/trained `W_d` and crossed
  activation contexts `X_e` is required for any training-dataset imprint claim.

### Current objective does not establish a semantic latent metric

- The active stage config has `contrastive_coef=0.0`, so the model was not trained
  with positive/negative pairs or retrieval/ranking supervision:
  `workspace/conf/big_vae_experiment/trainer_parameters_specific_to_big_vae_stage.yaml`.
- The same config has `struct_loss.lambda_rec=0.0` and `lambda_rel=0.0`. The held-out
  evaluator reports `struct_rec` diagnostically, but this term was not active in
  the training objective.
- KL/prior regularization does not imply that Euclidean or cosine distance between
  posterior codes measures functional similarity.
- Any words such as "functional neighborhoods", "nearest operator", "semantic
  interpolation", or "shared metric manifold" require an explicit metric audit.

## Terminology precision

- For a PyTorch linear layer, `Y = X W^T + b`.
- A 64-by-64 weight tile represents only a partial contribution from an input
  coordinate subset to an output coordinate subset. It is not automatically an
  autonomous neural operator or a semantically replaceable module.
- Safe terms before whole-layer validation:
  - activation-conditioned weight block;
  - contextualized local linear map;
  - local operator contribution.
- "Neural operator" must not be confused with the FNO/DeepONet literature.
- "Operator atlas" requires whole-layer/tile aggregation, tiling robustness,
  control of parameterization symmetries, and independent functional validation.

## Claims discussed so far and current disposition

| Candidate | Why it was attractive | Current decision | What would reopen it |
|---|---|---|---|
| Generic dataset/model clusters in latent space | Immediate positive visualization | Insufficient and confounded. Current dataset plot changes `X` while often fixing `W`. Clustering alone is descriptive. | Independently trained/fine-tuned weights, fully crossed `W x X`, quantitative held-out-family tests, simple baselines. |
| Fixed/mean activation stub while varying dataset-trained weights | Tests whether provenance is in `W` without dataset-specific `X` | Keep as a necessary control, not a headline. | Strong architecture-held-out weight effect beyond spectra/norms and a larger conceptual claim. |
| Shared vocabulary/atlas of local operators across models | Potentially fundamental: independent networks may reuse recurring computations | Still an appealing hypothesis, but not established. Tile semantics, gauge, aggregation, metric, and simple baselines are unresolved. | A metric-free or causally validated law that recurs across held-out architectures/domains. |
| Contextual semantics: the meaning of weights is `W x X` | Native to activation conditioning | Reject as main claim. For a linear map, context dependence can be studied directly with `X W^T`; risk of tautology. | Only if a single frozen representation exposes a surprising cross-architecture law unavailable to direct summaries. |
| Fine-tuning as rerouting versus rewriting | Mechanistic and causally testable through a 2-by-2 base/FT factorial | Good independent science question, but BigVAE is not necessary inside one architecture. Reject as BigVAE paper spine. | Cross-architecture universal law for which one shared encoder is essential and direct per-model decomposition cannot substitute. |
| Layer selection / LoRA placement / model compatibility | Practical outcome | Reject as headline: crowded and not idea-level. BigVAE risks being decorative. | At most a secondary consequence of a stronger finding. |
| Mergeability / task interference prediction | Practical and intervention-linked | Reject as headline: crowded by task-vector, gradient, Fisher, activation, and 2026 mergeability work. | Only a cross-architecture law of task composition, not another score. |
| Functional retrieval / cross-model donor search | Could operationalize a shared atlas | User explicitly rejects retrieval as the main claim. Also no native latent metric has been established. | Optional validation only after a stronger idea-level claim and a passed metric/causal audit. |
| MoE expert ecology | Strong domain shift and natural `W x routed-X` setting | High-risk extension, not current spine. Router/activation statistics and pruning literature are strong alternatives. | A surprising universal ecological law that transfers across MoE families and beats router/output baselines. |
| Functional quotient/gauge invariance | Fundamental test of function rather than coordinates | Keep as a sanity/validity gate, not a headline. Exact direct functional metrics exist and symmetry-aware metanetworks are prior art. | Unexpected emergent invariance of the frozen model, verified broadly. |
| Universal phases of training | Idea-level claim about how networks learn | Still open, but early checkpoints may be OOD and phase-transition/training-dynamics literature is crowded. | Leave-architecture-and-dataset-out universal ordering not explained by time/loss/LR/norms. |
| Task composition and hysteresis | Idea-level law about joint/sequential learning | Still open and potentially strong. Task-vector/continual-learning prior art is crowded, so the cross-architecture law must be precise. | Same composition defect/order effect under a single frozen readout across held-out architectures, beyond gradients/task vectors/CKA. |

## Explicitly rejected loops

Do not re-propose any of the following as the main paper contribution without new
evidence that resolves the recorded objection:

1. "The latents cluster by dataset/model."
2. "The same weights move when the input activations change."
3. "Fine-tuning mostly reroutes activations."
4. "Use the latents to choose layers/LoRA placement."
5. "Use the latents to predict model merging."
6. "Use nearest neighbors for cross-model retrieval."
7. "A VAE latent is automatically a meaningful metric/manifold."
8. "A 64-by-64 tile is automatically a semantically autonomous operator."

These may be controls, secondary applications, or prerequisites. They are not the
idea-level headline currently sought by the user.

## Standard for a surviving main claim

A candidate main claim should satisfy all of the following:

1. **Idea-level:** changes how the community thinks about trained networks,
   learning dynamics, or task/domain composition.
2. **Falsifiable:** has a narrow statement and a result that would clearly kill it.
3. **Cross-architecture/domain:** the final evidence holds on entire unseen model
   families or modalities, not random held-out tiles.
4. **BigVAE load-bearing:** one frozen `E(W,X)` exposes a shared law that direct
   `W`, `X`, `XW`, spectra, CKA, random sketches, and per-architecture analyses do
   not expose with comparable capacity.
5. **No decoder dependence:** until reconstruction improves, evidence is based on
   original models/weights and independent causal/functional measurements.
6. **No assumed latent metric:** use shared low-capacity readouts, distributional
   tests, topology/order, or independently audited distances.
7. **Causal or external validation:** the latent pattern predicts an independent
   intervention or functional event; it is not only a projection.
8. **Confound control:** balance/match architecture, layer role, depth, shape,
   scale, lineage, dataset, and activation statistics.

## Metric audit status

The proposed controlled metric audit remains useful as a prerequisite, not a main
claim:

- Compare exact function-preserving reparameterizations, low-activation-support
  perturbations, and norm-matched active perturbations.
- Gold local distance is based on `||X (W1-W2)^T||`, with severity sweeps.
- Test Euclidean/cosine, calibrated Mahalanobis, and posterior-distribution
  distances on held-out model/dataset/architecture groups.
- If only a supervised/whitened readout works, say that the representation contains
  transferable information; do not claim intrinsic neighborhoods.
- If simple summaries match the VAE, drop the geometry/retrieval story.

## Current research round: requested 2026-08-15

The user rejected retrieval as a main claim and requested a fresh multi-agent
search for more idea-level hypotheses. Agents are explicitly instructed not to
return retrieval, ranking, layer selection, model archaeology, generic clustering,
or within-architecture rerouting as the paper spine.

Target hypothesis families for the new search:

- universality classes of trained operators;
- conserved quantities or distributions during learning/adaptation;
- cross-architecture laws of task composition;
- path dependence and hysteresis;
- convergent evolution of independently trained networks;
- conditional phase structure under runtime domains;
- other experimentally falsifiable laws for populations of learned models.

The next update to this file should record:

1. agent-proposed claims;
2. primary-source prior-art audit;
3. which candidates survive the load-bearing test;
4. the user's selection/rejection;
5. the minimal decisive experiment for the selected candidate.

## Research-round result: 2026-08-15

Three agents searched independently under different framings: cross-domain
universality and causal response laws; metric-free interpretability; and a
hostile ICLR-reviewer audit. Two agents were then asked to compare the leading
spines head-to-head. Retrieval was excluded from every prompt.

### Primary-source novelty audit that materially changed the search

- `Convergent Evolution in Algorithmic Space` (arXiv:2608.05985, submitted
  2026-08-06) already reports task-specific structural attraction and coordinated
  early weight drift for independently initialized MLPs. Generic same-task
  convergence is therefore occupied; cross-architecture scope alone is not
  enough to make it the headline.
- `Universality and individuality in neural dynamics across large populations
  of recurrent networks` (NeurIPS 2019) already separates architecture-sensitive
  representational geometry from a frequently universal computational scaffold.
  “Architecture-specific microstates, shared task macrostate” is a useful framing
  but is too close to existing conclusions without a new intervention or law.
- `Delta Activations` (arXiv:2509.04442) already reports domain clustering and
  additive behavior under mixed fine-tuning data. Endpoint organization or
  additivity alone is not sufficient.
- `Learning Model Representations Using Publicly Available Model Hubs`
  (arXiv:2510.02096) already trains heterogeneous weight-space representations
  from arbitrary hub models and reports unseen-modality generalization. “Unlike
  SANE, one encoder handles heterogeneous models” remains method novelty but is
  not by itself a safe experimental headline.
- `When Do Task Vectors Interfere?` (arXiv:2608.09490, submitted 2026-08-10)
  already studies input-conditioned functional non-additivity across task pairs,
  adaptation methods, scales, and another model family. A task-interference
  algebra is now especially crowded.
- `Toward Understanding In-context vs. In-weight Learning` (ICLR 2025), `IA2`
  (ICLR 2026), and `Weight Updates as Activation Shifts` (arXiv:2603.00425)
  occupy much of the ICL-versus-fine-tuning duality. A conditional
  elicitation/acquisition boundary remains intellectually interesting, but the
  present vision-trained encoder would face a large LLM-domain shift and direct
  activation/logit baselines are unusually strong.
- Heavy-tailed weight spectra, Fisher/Hessian/NTK diagnostics, activation
  similarity, and direct `X W` statistics are mandatory baselines rather than
  optional ablations.

### Candidates after the hostile audit

| Candidate | Idea-level statement | Status after this round | Main reason |
|---|---|---|---|
| **Universal compensatory dynamics after local damage** | Functionally matched local lesions elicit a transferable, depth-normalized spatiotemporal redistribution of computation; the response law predicts which intact regions are causally required for recovery in an unseen architecture. | **Leading recommendation; 2/3 initial agents and the final hostile comparison selected it.** | It is interventional rather than descriptive, has an independent causal validation, does not need the decoder or a latent metric, and gives the shared encoder a plausible load-bearing role. |
| **Behaviorally matched models can be different learners** | Different randomized curricula produce behaviorally matched endpoints whose target-conditioned operator macrostates predict future plasticity and forgetting across architectures. | Strong runner-up. | More consequential than mere history decoding, but collision with plasticity, warm-start, continual-learning, curvature, and NTK literature is high. BigVAE matters only if it beats these direct diagnostics. |
| Context-gated expression of weight memory / conditional susceptibility phases | Robustness or plasticity is a state of `(W,X)`, not of `W` alone, and a joint-only interaction transfers across architectures. | Supporting discriminator, not current headline. | The joint interaction is native to BigVAE and relatively unoccupied, but without a causal consequence it risks sounding tautological and may reduce to covariance, `XW`, Fisher, or Hessian. |
| Multitask population mosaic versus hybrid operators | A mixed-task model changes the proportions of task-like local populations rather than averaging every block. | Separate paper or secondary high-risk experiment. | Scientifically interpretable, but Task Arithmetic, TALL masks, WARP, Delta Activations, and new interference work make composition crowded; arbitrary tiles also complicate operator semantics. |
| Task-universal macrostates or developmental trajectories | Microscopic parameters are architecture-specific, while tasks induce shared coarse-grained states or trajectories. | Supporting language inside the repair story, not a standalone headline. | Generic convergence, RNN computational universality, prediction manifolds, and behavioral phases already occupy the broad claim. |
| ICL and in-weight learning conditional duality | ICL and fine-tuning share a functional state for elicitation but diverge for genuine acquisition. | Defer. | High conceptual upside, but extensive adjacent prior art, difficult operational definition of elicitation versus acquisition, strong activation-only baselines, and severe OOD risk for the current checkpoint. |
| Composition algebra, RG flow, or conserved latent charge | Learned operators obey a common algebra, depth-renormalization flow, or conservation law. | High-risk theory pilots only. | Each requires stronger semantics/geometry than the current representation has established; several directions have very recent direct prior art. |

### Recommended headline hypothesis

Working title: **Universal Compensatory Dynamics of Trained Networks**.

Narrow falsifiable statement:

> After functionally matched local lesions, networks with different seeds and
> architectures recover through a common normalized spatiotemporal response of
> activation-conditioned weight states. A response law learned on source
> architectures predicts which intact regions are causally necessary for
> recovery in a completely held-out architecture.

Do not claim that recovery itself is novel. Damage, pruning, retraining, and
representational recovery have prior art, including `Efficient rescue of damaged
neural networks`. The proposed novelty is the **architecture-independent,
predictive, causally validated response law**.

### Minimal decisive pilot for the leading hypothesis

1. Use two genuinely different architecture families, initially ResNet and ViT,
   with three seeds trained on the same task.
2. Apply a small rank-controlled lesion to a middle-depth linear operator. Match
   lesion severity per model by immediate output KL or accuracy drop, not by raw
   parameter norm. Keep the lesioned block frozen during recovery.
3. Continue training for 100--300 identical steps, storing 10--15 checkpoints and
   activations on one fixed probe set.
4. Encode every eligible layer with the frozen posterior mean. Decompose the
   response using `(W_t,X_0)`, `(W_0,X_t)`, and `(W_t,X_t)` so weight rewriting,
   activation rerouting, and their interaction can be audited separately.
5. Fit the lowest-capacity response kernel possible on one family as a function
   of relative depth from the lesion, normalized time, and matched lesion
   severity. Predict the layerwise response and recovery curve of the held-out
   family without refitting.
6. For causal validation, freeze the intact layer or depth band predicted to be
   the principal compensation carrier in a fresh held-out-family recovery run.
   Compare against depth-, parameter-count-, gradient-, and activity-matched
   control freezes.

The pilot counts as positive only if all of the following hold:

- the response transfers to the held-out family and predicts an independently
  measured recovery curve;
- freezing the predicted carrier impairs recovery more than matched controls;
- the joint frozen encoder materially outperforms lesion magnitude, layer
  metadata, norms/spectra, activation drift, `X delta-W`, gradients,
  Fisher/Hessian/NTK, W-only and X-only encoders, random sketches, and a random
  encoder;
- inference and uncertainty are computed at checkpoint/model level, not by
  treating thousands of tiles as independent samples.

Fatal outcomes:

- recovery is fully predicted by ordinary gradient dynamics or layer identity;
- no response law transfers across an entire architecture family;
- the causal carrier freeze is no worse than matched control freezes;
- BigVAE features add no held-out-family information beyond direct statistics.

### Current decision state

The leading recommendation has **not yet been selected by the user**. Until that
happens, do not start the repair experiment or silently replace the paper goal.
Retrieval remains rejected as a headline and may only appear later as an optional
sanity check.

## Cross-modal universality reframing: 2026-08-15

The user prefers a broader version of the original operator-structure idea over
the repair experiment as the current paper spine. The motivating question is:

> Does a representation learned only from visual models expose a genuinely
> shared organization of learned computations in language and audio models, or
> does it only recognize familiar Transformer parameter statistics?

This supersedes neither the repair proposal nor its evidence standard. It records
a new candidate spine that must pass a cheap go/no-go before the larger corpus is
built.

### What “inter-domain united structure” must mean

The phrase has several non-equivalent levels. Do not collapse them:

1. **Numerical compatibility:** a frozen vision-trained encoder accepts LLM or
   audio `nn.Linear` weights and activations without NaNs or saturation. This is
   an engineering prerequisite, not a scientific result.
2. **OOD informativeness:** a target-domain probe can recover layer role, depth,
   task, or model family from the codes. This is model archaeology and can still
   be caused by shape, scale, or architecture shortcuts.
3. **Cross-domain predictive semantics:** one low-capacity readout fitted only on
   vision maps a frozen code to an independently measured functional quantity in
   language and audio, with no target refit or calibration.
4. **A shared computational vocabulary:** a low-dimensional basis of causal
   response profiles learned only on vision also spans language and audio
   response profiles, while a vision-only map from code to basis coefficients
   transfers to both target domains. Domains may change the prevalence and
   depth-wise arrangement of the states, but not their causal meaning.

Only levels 3--4 support a strong version of the desired claim. Latent cloud
overlap, domain classification, PCA, and clustering do not establish them.

### Closest prior art and the seams it leaves

| Prior | What it already establishes | Remaining seam relevant here |
|---|---|---|
| SANE / Hyper-Representations | Weight autoencoding, property prediction, and generation in curated, mostly homogeneous model zoos. | No vision-only to language/audio functional transfer; original SANE normalization assumes matching architectures. |
| SNE (arXiv:2305.16625), GMN (arXiv:2312.04501), UNF (arXiv:2402.05232) | Weight processing and property prediction across varying architectures; principled handling of weight-space symmetries. | Not a frozen activation-conditioned representation with shared zero-shot causal semantics across modalities. |
| Falk et al., *Learning Model Representations Using Publicly Available Model Hubs* (arXiv:2510.02096) | One SANE-like weight-only backbone trained on 2,000 heterogeneous vision models; it generates an initialization for GPT-2 that trains faster than a random start. | This directly occupies “vision-trained weight AE transfers to language.” It does not test audio, shared functional coordinates, causal response semantics, or whether activation conditioning exposes the transferable part. |
| *Universal Weight Subspace Hypothesis* (arXiv:2512.05117) | Large-scale evidence for architecture-specific shared low-rank weight subspaces across many tasks and separately across ViT, Mistral/LLaMA/GPT/T5 collections. | The authors explicitly restrict subspace comparison to shared architectures and leave cross-architecture comparison open. Similar spectral decay across modalities is not a common cross-modal coordinate system. |
| Yunis et al., *Approaching Deep Learning through the Spectral Dynamics of Weights* (arXiv:2408.11804) | Direct SVD reveals declining effective rank and disproportionate leading-singular-value growth in vision CNN/UNet, speech LSTM, and language Transformer training. | This is a mandatory simple baseline and a common optimization bias, not aligned cross-domain subspaces or common activation-conditioned causal phenotypes. |
| Rosetta Neurons (ICCV 2023) and Rosetta-neuron scaling (arXiv:2606.03990) | Recurring activation-selective neurons across heterogeneous vision models and, in separate 2026 analyses, recurring populations/scaling laws within language and within vision. | The analyses compare matched activations within a modality, not weight-plus-context states with one causal definition shared between vision, language, and audio. |
| *Do LLMs and VLMs Share Neurons for Inference?* (arXiv:2602.19058) | More than half of selected inference neurons can overlap between tested LLM/LVLM pairs, and amplification/deactivation gives causal evidence for shared reasoning units. | The comparison is text LLM versus LLM-centric VLM, often with a shared/corresponding language backbone and matched neuron indices. It does not cover independently trained unimodal vision, language, and audio models, arbitrary shapes, or a source-only weight-context encoder. |
| Platonic Representation Hypothesis (ICML 2024) | Activation representations from capable vision and language models increasingly agree on paired-data relationships. | It concerns representations of datapoints, not learned local weight operators. The 2026 Aristotelian re-audit shows global similarity trends are width/depth-confounded; only calibrated local-neighborhood agreement survives. |
| *Structure Is Not Enough* (arXiv:2503.17138) | Adding model-output behavioral loss improves reconstruction/generation in conventional model zoos. | Behavior supervises the autoencoder and remains dataset/query dependent; it does not show zero-shot cross-modal functional organization in a frozen vision-only encoder. |
| *Cross-Architecture Model Diffing with Crosscoders* (arXiv:2602.11729) | Pair-specific crosscoders trained on 100M aligned activation pairs discover and causally steer shared/exclusive features between different LLM architectures. | This is strong evidence that cross-architecture functional alignment is possible, but it trains a new aligner for every model pair on target activations. It is neither target-free nor cross-modal, and it does not process weights. |
| WeightCLIP (arXiv:2607.03551), WARP (arXiv:2607.01686), Delta Activations (arXiv:2509.04442) | Dataset-model alignment, recovery of LLM training mixtures from weights, and task/domain organization of fine-tuned LLM activation shifts. | These establish provenance/task signals and useful embeddings, not a modality-independent causal law for local computations. |

The literature is therefore fragmented but not empty. A paper cannot claim only
heterogeneous encoding, cross-modal numerical transfer, low-dimensionality, or
recurring features. The defensible seam is a **single cross-modal causal meaning
for activation-conditioned local computations**, verified with locked source-only
predictions and original-model interventions.

### Recommended headline and extra finding

Working hypothesis: **Conditional Cross-Modal Universality**.

> Across vision, language, and audio Transformers, trained linear projections
> reuse a small shared set of activation-conditioned causal response states. A
> frozen BigVAE trained only on vision, followed by a readout fitted only on
> vision, predicts the response state of unseen language and audio layers without
> adaptation. Modalities primarily change the frequency and arrangement of
> states, rather than their causal definition.

A sharper additional discovery, if supported, is:

> Universality is conditional rather than purely parametric: raw-weight
> structure remains arcтhitecture-specific, while conditioning on the activation
> distribution exposes the shared causal core. The main boundary is
> computational role (for example, residual-writing projections versus Q/K/V or
> gates), not modality.

This makes the current VAE load-bearing only if native `E(W,X)` transfers while
neutral/stub context, shuffled context, W-only, X-only, direct `XW`, covariance,
spectral, and equal-capacity learned baselines do not.

Terminology remains deliberately narrow. The current collector hooks
`nn.Linear`, and a stage-1 point is a 64-by-64 local tile. The supported object is
an **activation-conditioned learned linear computation**. “All neural operators”
or a whole-module operator vocabulary would require whole-layer aggregation,
nonlinear/attention context, convolutional or recurrent support, and a broader
experiment.

At the model-API level the first Transformer-domain test is feasible without
retraining BigVAE: the forward pass accepts generic rank-2/3 `W` and `X` with
matching input dimension, and the hook flattens any `(..., D)` activation into
rows. At the data-pipeline level it is not ready: the active training profile and
all registered runners collect visual inputs/vision encoders, so language/audio
collectors, masks, canonical weight orientation, and perturbation/evaluation
harnesses still have to be implemented and audited.

### Independent causal response label

For each whole linear layer `b`, use many BigVAE tiles only as repeated
within-layer measurements and aggregate them. Define an external response vector
`R_b` by intervening on the original network, never on decoded weights:

- intervention families: attenuation, norm-preserving rotation/noise, and an
  activation-aligned low-rank perturbation;
- severity is calibrated to the same immediate block-exit RMS damage, with a
  sweep such as 2%, 5%, and 10%;
- record normalized propagation at the next block, quarter-depth, half-depth,
  and final output, plus endpoint JS/KL or task-loss change as a secondary
  measure;
- use disjoint examples for `X` passed to BigVAE and for measuring `R`, so the
  encoder cannot memorize the label probe batch.

The resulting response curve has a common coordinate definition across domains.
It also avoids the current lack of a trusted latent metric: the paper never needs
nearest-neighbor retrieval or Euclidean distance in `z`.

Call the vocabulary shared only if both conditions hold:

1. a small response basis fitted on vision explains language and audio curves
   almost as well as target-specific bases, and the target domains do not require
   new stable response directions;
2. one regularized source-only map `h(E(W,X)) -> R` or to the response-basis
   coefficients transfers without target refitting and beats all direct and
   learned baselines on complete held-out model families.

### Cheap go/no-go before a full paper corpus

Use several independently trained/fine-tuned checkpoints per family, with
matched hidden width where possible to remove the easiest shape shortcut:

- source: a small group of held-out ViT-B/16-family vision checkpoints rather
  than one model;
- unseen language: BERT-base-family checkpoints, with RoBERTa as an immediate
  family challenge;
- unseen audio: wav2vec2-base-family checkpoints, with HuBERT as an immediate
  family challenge;
- roles: attention output and MLP down projections at four depth quartiles;
- 64 posterior-mean tiles per whole layer, repeated with a second tiling;
- one perturbation family (attenuation), two severities (approximately 3% and
  10% immediate local damage), and four propagation horizons;
- fit ridge/readout hyperparameters only on grouped vision splits; lock the model
  and evaluate BERT and wav2vec2;
- compare metadata, W statistics/spectra, X statistics/covariance, `W Sigma_X
  W^T` / `XW`, random sketches, and a random encoder using equal readout capacity.

Do not build the full vocabulary/repair pipeline unless BigVAE beats the best
baseline on **both** target domains under model/layer-level inference. A run that
only produces finite LLM/audio codes is a smoke test.

### Full factorial evidence and kill criteria

The full study should include several unrelated model families per modality,
same-role comparisons, context pairs for the same fixed weights, trained versus
randomly initialized controls, and architecture-by-modality-by-objective coverage.
AST is useful as a lineage-positive control but is not decisive audio evidence
because it is a ViT-like architecture and is often initialized from vision.

The statistical unit is an independently trained model/checkpoint and whole
layer. Tiles and activation rows are nested observations, not independent
samples. Splits and uncertainty must be grouped by model family or independent
pretraining lineage.

The headline is killed if any of the following occurs:

- transfer works only for ViT-like vision/audio/LLM Transformer blocks and fails
  when architecture and modality change together;
- target response curves require new domain-specific basis directions;
- layer role, normalized depth, shape, scale, W/X moments, spectra, covariance,
  `XW`, Fisher/Jacobian/Hessian, or a random sketch matches BigVAE;
- W-only or neutral-context encoding transfers equally well, so activation
  conditioning is unnecessary;
- X-only or shuffled-context features explain the result, indicating a domain
  shortcut rather than a joint computation;
- predictions are unstable across tilings or function-preserving coherent
  permutations/rescalings;
- trained and randomly initialized networks show the same effect;
- target-domain refitting or calibration is required.

### Current decision state

This is now the preferred general hypothesis to evaluate, but it is **not an
established project result**. The current checkpoint and plots contain no LLM or
audio evaluation and do not show a cross-modal causal vocabulary. The next safe
action is the three-model go/no-go pilot above, after an implementation review of
the new language/audio collectors and perturbation harness.

## Preregistered cross-domain AE reconstruction gate: 2026-08-15

The user selected the deterministic AE before KL fine-tuning for the first
cross-domain experiment. The exact frozen checkpoint is
`weight_quantile_vae_gpu0_square/stage_1/step_0480000.pt`, step 480000, SHA256
`0327871d26daac3833556a6c1a7212d07dcd639eb43f8c3c85adb083cb284006`.
Its state has no `to_mu.*` or `to_logvar.*` keys. The embedded config is stale,
so the run must force deterministic AE construction and strict state loading.

The complete ten-hour preregistration is recorded in
`projects/weight-vae/docs/cross_domain_ae_10h_preregistered_plan.md`. Its main
binary gate tests calibrated local-operator reconstruction transfer to three
language and three audio families against source-trained PCA, a random linear
codec, rank-matched SVD, and X-aware SVD. Native, fixed-stub, mismatched, and
diagonal-Gaussian activation contexts; deterministic-code shuffling; two
tilings; a gauge stress test; and direct W/X/XW nuisance features are included
in the same run so that the main transfer result is not deferred to a later
confounder audit.

The plan separates three decisions:

- `T`: cross-domain AE reconstruction transfer;
- `C`: whether activation conditioning is load-bearing;
- `L`: whether source-only role/depth organization transfers and merits a later
  interpretability/causal study.

A positive `T` closes only the preregistered existence-of-transfer claim on the
tested benchmark. It does not establish a universal causal vocabulary or
downstream-preserving reconstruction. If `T` passes but `C` fails, the honest
result is weight-prior transfer, not conditional universality.

## Domain-only correction: 2026-08-16

The 2026-08-15 off-the-shelf target matrix is superseded **before any target
result was produced**. Matching full-layer width is necessary but insufficient:
ViT versus BERT/Wav2Vec/Llama still aliases modality with architecture details,
frontend/tokenizer, objective, optimizer, budget, data scale, and lineage.
Those checkpoints are now external-validity stress tests only.

The causal primary is a paired renderer-distribution intervention using the
Flickr8k image/text corpus and Flickr8k Audio Caption Corpus. Each natural row
contains an image, a written caption, and a human recording of that exact
caption. All domains are deterministically rendered to `64 x 256`; after that
boundary the trainable graph is identical. The primary model is fixed at one
ViT-B-like scale: 12 blocks, width 768, FFN 3072. For each of three seeds the
entire initial model and optimizer states are cloned across vision, text, and
speech, and aligned triplet batches, mask positions, objective, schedule,
updates, and processed-token budget are identical.

The frozen checkpoint remains the deterministic pre-KL AE at step 480000. The
primary weight-domain estimand uses one identical source-vision activation
prototype for all domains plus a step-zero difference-in-differences, so only
the learned W trajectory changes. Native and crossed activation contexts are
reported separately to distinguish W, X, and W-by-X effects. Full attention
output and FFN-down matrices are assembled from all `64 x 64` tiles; Llama-scale
sampled-block reconstruction is absent from the causal gate.

The user's original fixed-activation proposal is retained explicitly as
`global-mean-X`: a single role/depth activation prototype averaged with equal
weight over vision, text, and speech and then reused for every W. Because it
observes target activations it is an ablation, not the zero-shot primary. Domain
organization is compared quantitatively under native-X, source-only common-X,
global-mean-X, and Gaussian-X using identical source-fitted PCA coordinates and
a leave-one-seed-out low-capacity probe; clusters alone remain descriptive.

The updated preregistration is:
`projects/weight-vae/docs/cross_domain_ae_10h_preregistered_plan.md`.

The main decisions are now:

- `S`: controlled vision source/metric validity;
- `D`: weight-domain transfer to **both** written text and speech under common X;
- `C`: whether native activation conditioning adds transferable information;
- `I`: whether a lesion-response readout fitted only on vision transfers to
  text and speech beyond depth/norm/spectrum/X/XW baselines.

Clusters by domain never close a claim. A positive controlled result supports
shared structure only for independently trained, identically configured cores
under these paired renderers. It does not by itself establish transfer to
natural LLM/audio architectures. Conversely, a failure of a confounded
off-the-shelf stress model cannot refute the controlled domain effect.

A stored H100 compute benchmark (student plus EMA teacher, top-six target,
Smooth-L1, backward, AdamW, and EMA) measured the Base core at approximately
0.06012 seconds/step and 6.963 GiB for batch 256 and sequence 64. Evidence is in
`artifacts/crossmodal_united_structure/h100_shared_core_benchmark_20260816/`;
real rendering, I/O, checkpointing, AE evaluation, and analysis remain
unbenchmarked and retain explicit schedule buffer.

## Checkpoint-quality and native-interface correction: 2026-08-16

The user identified a decisive construct-validity problem before the Flickr8k
training was launched or any target AE result was read. The AE source corpus is
made from real pretrained vision checkpoints (the stored manifest is dominated
by Mask2Former/Swin, SwinV2, SigLIP, DeiT, MAE, DINOv2, BEiT, CLIP, and BLIP).
Fresh 20k-update models on 6,000 Flickr8k training images are not automatically
members of the same "competently trained checkpoint" population. If text or
speech learns a renderer shortcut, memorizes the small corpus, or remains a weak
encoder, its AE failure cannot reject transfer to trained NLP/audio operators.

This is not described as an ordinary pre-treatment confounder. With a fixed
recipe, downstream competence is partly a post-treatment mediator/outcome of
the domain. The narrower causal effect of renderer on the resulting weight
trajectory would still be identified. The problem is that this narrower
estimand is not the paper's intended claim. Passing loss, weight-movement and
non-collapse gates establishes optimization health, not source-like learned
competence.

The exact common `64 x 256` renderer is also demoted. Identical Transformer core
weights do not require identical sequence length: attention and FFN parameter
shapes depend on hidden width, not token count. Forcing bytes and whole speech
utterances through `N=64` risks irreversible information loss and
interpolation/padding shortcuts. The quality-preserving controlled design uses
native/frozen modality tokenizers or adapters, native sequence lengths, one
identical trainable core, and excludes adapter weights from Weight-AE analysis.
Only the activation context presented to the Weight AE is pooled/sampled to its
required fixed shape.

Decision update:

- `cross_domain_ae_10h_preregistered_plan.md` is superseded as a decisive
  primary; its compute and mechanistic components remain reusable;
- the H100 benchmark proves timing feasibility only and says nothing about
  checkpoint quality;
- the prior overnight/18-hour ETA is withdrawn;
- the leading one-night **ecological** experiment is a same-scale, high-quality,
  independently pretrained vision/text/speech family such as data2vec Base,
  with honest acknowledgement of modality-specific recipes;
- the **domain-only causal** experiment requires an independently passed
  per-modality competence gate with information-preserving interfaces and may
  exceed one night;
- the fixed-activation ablations remain mandatory: source common-X as zero-shot
  primary, literal equal-domain global-mean-X, native-X, crossed-X, and
  Gaussian-X, with identical weight-only comparisons wherever applicable.

The intended paper claim is now evaluated by triangulation rather than by
pretending one weak controlled benchmark does everything: high-quality natural
checkpoint transfer supplies ecological validity, and the controlled arm tests
whether the effect survives removal of architecture/scale/initialization
aliases. Neither arm may silently inherit the other's conclusion.

The actual bytes of the official Hugging Face data2vec v1 Base trio were then
downloaded and loaded without loading the Weight AE. Vision, text and speech
each contain 12 layers, width 768, FFN 3072 and exactly 72 role-matched core
matrices (Q/K/V/O plus FFN up/down); the text pooler is the only extra square
matrix and is excluded. This establishes implementation feasibility, not
transfer. Evidence and hashes are stored in
`artifacts/crossmodal_united_structure/hf_data2vec_schema_audit_20260816/`.

The revised two-tier preregistration is
`projects/weight-vae/docs/cross_domain_high_quality_two_tier_plan.md`. Tier A is
the independently pretrained, high-quality data2vec ecological test. Tier B is
a controlled shared-vision-basin adaptation test and must reconstruct learned
`delta W` in addition to final W; otherwise a positive result can be inherited
from the source initialization. A paired random-initialization arm can upgrade
the controlled evidence only if it passes the same sealed competence gate.

## Encoded-conditioning correction: 2026-08-16

The user caught a conceptual error in the `Common-X` wording before target AE
metrics were read. Equal hidden widths make raw matrix multiplication possible,
but do not align the learned hidden coordinate systems of independent
vision/text/speech models. A source activation multiplied by a target weight is
therefore not a target-operator evaluation. Raw cross-model `Crossed-X` and
`Common-X/Common-B` are retired as primary evidence.

The earlier phrase "source vision bank" referred only to the offline Weight-AE
training cache, not to a universal activation tensor. Its manifest reports
348,049 accepted paired records, about 300 GiB, and 11 vision models. Every
hooked X is matched to its own model/layer/W. Seventeen datasets were enabled in
the collector config, but only 15 appear in the manifest's recorded dataset
statistics; the cache must not be coordinate-wise interpreted as one shared
activation space.

The cache also repeats a checkpoint/layer W across many activation records and
image datasets. Therefore dataset clustering in the original native-conditioned
plots attributes structure to `(W,X_dataset)`, not to W alone. With fixed C0,
identical W copies must have identical deterministic codes; this is a necessary
negative control on the earlier interpretation.

The exact deterministic checkpoint interface was audited. For a `64 x 64`
weight tile, raw `X=[n,64]` is used only by the frozen distribution encoder,
which returns `C_var=[4,16,256]`, `C_patch=[4,256]`, and
`C_pooled=[4,256]`. `C_var` conditions the patch tokenizer;
`C_patch` conditions all 15 encoder layers and the decoder queries. Thus the
user's fixed-embedding ablation must clamp both C tensors, not one raw X or one
summary vector.

The replacement zero-shot arm is `Source-mean-C0`: estimate an exchangeable
post-distribution-encoder template from a balanced source-only subset, freeze it
before target evaluation, and broadcast it unchanged to every weight tile.
No source X is passed through a target model. Raw normalized weight error is the
activation-independent endpoint. Functional error is computed separately with
each model/layer's correctly hooked held-out native `X_score`, used only outside
the Weight AE. `Native-C` uses a disjoint native context split inside the AE;
`Global-mean-C` is secondary because it observes target conditioning.

Latent analyses now compare `z_fixed=z(W,C0)`,
`z_native=z(W,C(X_native))`, and `delta_z=z_native-z_fixed`. Domain separation
in `z_fixed` is evidence of domain-associated information in the weight path,
but it is not by itself evidence of semantic or causal universality; model
identity, parameterization, role/depth, scale and lineage remain alternative
explanations. The corrected protocol is in
`projects/weight-vae/docs/cross_domain_high_quality_two_tier_plan.md`.

## Checkpoint-provenance and smoke correction: 2026-08-16

The claim that the source smoke failure suggested a wrong or direction-only
checkpoint was withdrawn after a byte/payload/lineage audit. The canonical AE is
`weight_quantile_vae_gpu0_square/stage_1/latest.pt`, step 480000, file SHA256
`d4203bf9dfa76a474be511b5b97e4b6c3ebcda0d2b7afae257c3357b38c8ba00`.
Its 1,339 model tensors are bit-identical to `step_0480000.pt`; it has no
`to_mu`/`to_logvar` posterior heads. The historical held-out metrics and latent
plots explicitly record this exact `latest.pt` path.

The KL fine-tune is the separate
`weight_quantile_vae_gpu0_square_VAE/stage_1/step_0570000.pt`; its four posterior
head tensors are present. Its training log loads step 480000 and explicitly says
"Loaded AE checkpoint" before initializing the missing posterior heads and
starting the KL/sampling ramps. This establishes the AE-to-VAE lineage.

The earlier attribution of the current mutable training YAML to the AE payload
was also wrong. The embedded step-480k loss config has behavioral
`lambda_operator=50`, `lambda_dir=1`, `lambda_scale=10`, with both behavioral and
structural coefficients equal to one. The new 72-tile smoke used a different
relative full-output metric and a new slicing path. It was initially quarantined
because it conflicted with the stored historical evaluation over 182,488
slices. The exact replay below subsequently identified and repaired the
forward-contract cause.

## 2026-08-16: source-smoke discrepancy resolved as RoPE contract drift

- The falling Comet behavioral curve is genuine: the square AE's final 10k-step
  medians are `behavioral=1.6600`, `b_op=0.02075`, `b_dir=0.51076`, and
  `b_scale=0.01108`; the stored full held-out evaluation agrees.
- The checkpoint was trained before `rope_2d_coord_kind` existed and therefore
  used raw integer coordinates. Commit `0b1be4e` later made
  `normalized_center` the missing-key default without changing parameter shapes.
- On identical checkpoint tensors and paired W/X slices, the buggy fallback gave
  cosine `0.0080` and relative error `1.5439`; forcing the trained raw contract
  gave cosine `0.6051` and relative error `0.7686`.
- On the exact saved April batch, the fixed loader gives
  `b_op/b_dir/b_scale = 0.006871/0.643684/0.011722`, versus the stored
  `0.006843/0.642187/0.012026`.
- Decision: invalidate all AE-quality or transfer interpretations produced with
  the buggy missing-key fallback. Restart the source gate with the corrected
  loader; keep all target metrics sealed until that gate passes.
- Evidence:
  `artifacts/crossmodal_united_structure/step480_rope_contract_audit_20260816/`.
