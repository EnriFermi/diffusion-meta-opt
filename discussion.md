# Project Discussion Log

This is an append-only summary of material user-agent and agent-agent
discussions. It records decisions and reasoning, not a verbatim transcript.

## 2026-08-27 — Weight representation and adaptive-branch reset

### User direction

- Explanations must move from simple observations to conclusions. Avoid raw
  LaTeX and unexplained project jargon.
- Do not continue elaborating the V10/V11 carrier, orthogonal-complement, or
  multi-mechanism adaptive-branch architecture. The user considers that path an
  overcomplicated Frankenstein and wants a unified mechanism.
- Reconsider the problem from the representation of weights themselves. Raw
  weights are small, noisy, vary in amplitude, and may lose their signal through
  a deep encoder. Image-style local smoothness must not be assumed.
- Explore whether a serious model-quantization representation (for example AWQ
  or a method with calibration/error compensation), rather than naive scalar
  rounding, could be the native data representation learned by the model.
- Separately investigate why deep encoders preserve or lose weight information:
  ordinary encoder topology, residual/short gradient paths, normalization,
  bottlenecks, and conditioning on activations.

### Established experimental evidence

- V10 completed but the adaptive branch improved over the fixed floor by only
  about 0.00098 mean directional loss. Artifact root:
  `/mnt/shared/weightclip_benchmark/ae_v10_exact64_orthogonal_complement_v2_1984step`.
- V11 stopped at its precommitted step-512 scientific gate. Its learned
  residual shrank toward zero and complement normalized MSE stayed near 1.0.
  Artifact root:
  `/mnt/shared/weightclip_benchmark/ae_v11_exact64_four_trunk_complement_v1_1984step`.
- The deterministic V3 postmortem excluded structural-vs-complement gradient
  antagonism and severe cross-batch gradient cancellation as primary causes.
  It found head-dominated gradients and no held-out *linear* complement
  information in the learned adaptive code. Artifact root:
  `/mnt/shared/weightclip_benchmark/ae_v11_postmortem_bundle_v3_20260827T043130Z`.
- On the exact64 weights, FP32-to-BF16 conversion produced no zeros or sign
  flips; therefore numerical underflow alone is not the established signal-loss
  mechanism. The current flattened layout also showed little useful adjacency.

### Current hypotheses, not conclusions

- A gain-plus-shape representation may stop entire low-amplitude blocks from
  becoming low-energy hidden states: store an explicit block scale and encode a
  normalized signed shape. Discretizing the shape could make its hidden token
  amplitude independent of the original raw weight amplitude.
- The more task-natural coordinate may be activation-aware: transform weight
  errors according to how they change `XW`, rather than treating every raw
  coordinate equally.
- Advanced quantizers such as AWQ/GPTQ/QuIP/AQLM may be useful as learned
  coordinate systems or tokenizers, but this is not yet established. Their
  correction metadata, calibration dependence, invertibility, and suitability
  as an encoder input need separate analysis.
- Representation quality and deep-encoder trainability are independent axes.
  A better codec does not by itself repair a long encoder with weak identity
  paths or destructive normalization.

### Agreed next analysis

- Compare the user's proposal—learning over an advanced quantizer's complete
  representation, including its scales/codebooks/correction state—with simpler
  normalized continuous and discrete controls.
- Review primary literature on weight encoders and learned weight compression,
  focusing on exact information and gradient paths rather than stacking more
  heterogeneous modules.
- Keep representation, encoder, and objective as separate hypotheses until a
  controlled comparison shows which one matters.

## 2026-08-27 — Communication and documentation contract

- User reported that raw LaTeX does not render and makes formulas hard to read.
  Future user-facing formulas should use plain text plus an immediate verbal
  explanation.
- `AGENTS.md` and the Weight-VAE-local `AGENTS.md` now require all agents to
  maintain this file with key discussion decisions, evidence, disagreements,
  and open questions.

## 2026-08-27 — Quantizer-as-tokenizer and unified-encoder literature review

### Correction to the earlier interpretation

- The user's proposal is stronger than rescaling raw weights before the same
  encoder. The quantizer itself should be the tokenizer, and the learned model
  should never receive dequantized or raw floating-point weights as its normal
  input.
- The runtime representation must be the smallest self-contained payload that
  deterministically decodes to an approximate weight tensor: discrete indices,
  explicit scale tokens, positions/layout, and any required shared transform or
  codebook identity. Calibration activations, Hessians, Cholesky factors,
  search states, and temporary compensation residuals are encoding-time state,
  not tokens.
- A lossy quantizer reconstructs `W_hat`, not the original arbitrary FP32 `W`.
  Exact reconstruction would require storing `W - W_hat`, which would recreate
  the prohibited raw/residual side channel.

### What the quantization literature actually provides

- AWQ is activation-aware preconditioning before ordinary scalar rounding. It
  rescales important input channels based on activation statistics but performs
  no reconstruction optimization or residual error compensation. It is a good
  conditioning control, not by itself a new semantic token language.
- GPTQ uses activation/Hessian information while choosing ordinary quantized
  codes and compensates each committed rounding error by adjusting weights that
  have not yet been quantized. The final payload is still integer codes plus
  scales and zero points; the Hessian and error trajectory are not stored.
- AQLM and GPTVQ use vector-code indices and learned codebooks, so they are much
  closer to a token representation. Standard per-layer codebooks make token
  meanings layer-relative; a unified model would need shared codebooks or an
  explicit codebook identifier and must count the codebook payload.
- QuIP# uses a fixed global lattice vocabulary after a reversible randomized
  transform. It gives globally meaningful vector tokens but requires transform
  metadata and can scramble individual coordinate meaning.
- NWC is the closest direct precedent for a learned native weight codec. It
  normalizes columns, chunks them into groups of 16, uses four-block residual
  MLP analysis/synthesis transforms, quantizes the latent, and optimizes a
  task-aware rate-distortion objective. It also uses compensation during
  encoding, so codec quality and compensation must be ablated separately.
- SpQR is an explicit negative example for the desired clean design because it
  stores a dense low-bit stream plus a separate sparse high-precision outlier
  stream.

### Proposed unified representation, stated without a raw-weight carrier

For a block of weights, an AQLM-like token tuple could be:

```text
[block position, scale token, codebook-1 index, codebook-2 index]
```

The deterministic codec decodes it as:

```text
decoded block = decoded scale *
                (codebook-1 vector + codebook-2 vector)
```

The learned model receives embeddings of those tokens, not the small decoded
floating-point values. This makes hidden-state amplitude independent of the
absolute raw weight amplitude while preserving amplitude explicitly in the
scale token. All information travels through one counted token stream and one
bottleneck; there is no `fixed dequantization + learned residual` output.

The simplest first tokenizer should be ordinary shared-grammar GPTQ codes plus
quantized log-scale tokens. RTN and GPTQ can use the same payload grammar, so
their comparison isolates error-compensated code selection. A shared-codebook
AQLM/GPTVQ or fixed-lattice QuIP# tokenizer is a second step if discrete vector
motifs are beneficial. AWQ without rounding is retained as a preconditioning
control.

### Encoder literature and the deep-signal question

- Successful weight encoders do not normally pass tiny raw floats through a
  long heterogeneous chain. NWC uses four residual MLP blocks; SANE uses
  standardized weight tokens and moderate-depth Transformer encoder/decoder
  stacks; IGPG first builds VQ-VAE weight tokens and then trains a conventional
  autoregressive Transformer on their indices.
- The clean deep architecture is one evolving state and one repeated pre-norm
  residual block:

```text
state = state + attention(norm(state))
state = state + mlp(norm(state))
```

  The identity term is an internal gradient path, not a second semantic model
  and not an output carrier: every value still passes through the one
  bottleneck and decoder. If even residual updates are disallowed, the honest
  alternative is a shallow encoder, not a deep plain stack.
- The tokenizer and encoder failures remain separate. A discrete input can
  prevent whole low-amplitude blocks from entering the model as tiny vectors,
  but it cannot repair a bad product of encoder Jacobians. Conversely, a
  residual encoder cannot recover distinctions already collided by a lossy
  tokenizer.

### Precommitted simplest comparison before another architecture build

Use identical data, bit accounting, decoder budget, objective, starts, and
depth for these inputs:

1. normalized continuous weights plus explicit scale;
2. continuous AWQ-rescaled weights without rounding;
3. dequantized GPTQ values;
4. actual GPTQ integer codes plus scale tokens;
5. later, one shared vector-code tokenizer such as AQLM/GPTVQ or QuIP#.

Always include the fixed dequantizer-only result. Measure held-out layer-action
error, downstream quality, token/code stability across disjoint calibration
sets, full metadata bitrate, token occupancy, and at every encoder depth the
activation signal, reconstruction probe, gradient, predicted optimizer update,
and block-ablation effect.

Interpretation is precommitted as follows:

- If continuous AWQ helps, the main issue is input conditioning/coordinates.
- If code tokens outperform their own dequantized continuous values under the
  same model, discreteness is helping signal transport.
- If all representations show the same depth-wise decay, the encoder path is an
  independent failure.
- If the fixed dequantizer is already good but the learned model loses quality,
  the learned encoder/bottleneck/decoder is at fault.
- If the quantizer fails on held-out activations, it is not a valid task-space
  representation regardless of training convenience.

### Current decision status

- No new architecture or experiment is authorized by this discussion alone.
- The leading clean direction is a frozen quantizer/tokenizer followed by one
  homogeneous pre-norm residual token model and one explicit bottleneck.
- Whether GPTQ-style scalar tokens or shared vector tokens are the better native
  representation is not established; the controlled panel above is the next
  discriminator.
- Reviewer disagreement is recorded rather than hidden: the simplicity-first
  view prefers GPTQ because RTN and GPTQ share the same scalar payload and thus
  isolate compensation cleanly. The global-vocabulary view prefers QuIP#-NoFT
  4-bit because its fixed 8-D lattice gives each vector token the same meaning
  across operators. QuIP# also adds a randomized transform, so adopting it
  directly would bundle representation, normalization, and compensation. The
  planned panel must separate those effects before either becomes the default.

## 2026-08-27 — Exact residual block proposed for the token encoder

- The proposed processor is a standard four-layer bidirectional pre-norm
  Transformer, not a new weight-specific block and not a parallel carrier.
- Each layer has exactly two residual updates:

```text
u      = h + self_attention(RMSNorm(h), valid_token_mask)
h_next = u + linear_2(GELU(linear_1(RMSNorm(u))))
```

- Initial concrete size for a controlled baseline: hidden width 512, eight
  attention heads, MLP width 2048, four encoder layers, dropout zero. Decoder
  uses the same block type. Layers share the architecture but not weights.
- Use ordinary unit residual coefficients and ordinary nonzero initialization
  first. Do not add ReZero gates, DeepNorm, auxiliary branch losses, or
  per-depth outputs before a depth comparison establishes a need; exact-zero
  residual gates would also give branch weights zero gradient on step 1.
- Token input is the sum of code/value, field-type, codebook/group-position,
  and absolute/relative position embeddings. Scale is represented by typed
  scale-byte or log-scale tokens, never solely by multiplying an embedding by
  the scalar scale, because pre-norm would largely remove that amplitude.
- Padding has an explicit token and attention mask. A code index is always
  bound to its codebook and quantization-group position; otherwise equal
  integer indices from different vocabularies would be falsely conflated.
- The identity skips preserve hidden state and provide direct additive gradient
  paths inside the one encoder. They do not bypass the explicit bottleneck and
  therefore are not the previously rejected output carrier.
- Intermediate reconstruction probes, gradient/JVP telemetry, and block
  ablations are diagnostics only. They do not introduce auxiliary training
  paths or losses.

## 2026-08-27 — Authorized minimal three-arm representation experiment

### User decision

- Run the experiment now, with an expected turnaround of about one hour.
- Remove the previously proposed separate quantizer-floor stage and the depth
  sweep. Do not compare RTN, AWQ, NF4, VQ, or multiple encoder families.
- Compare only three input representations under one identical
  encoder–bottleneck–decoder architecture:
  1. normalized floating-point weights;
  2. GPTQ integer codes cast to continuous values;
  3. the same GPTQ integer codes consumed as categorical tokens.

### Frozen representation contract for this bounded experiment

- The natural record is an existing exact64 tile: `W[128,128]` with its real
  activation context `X[512,128]`. Operator train/test splitting remains grouped
  over all nine tiles of an operator.
- Quantization scale is local, not one global matrix scalar and not one scalar
  per 16-value token. For each tile and each output channel, one symmetric scale
  covers the corresponding 128 input weights. Each group is then serialized as
  eight ordered chunks of 16 values. The same standardized `log2(scale)` feature
  is attached to all eight chunks from that group.
- Scale is an input feature that must pass through the same bottleneck as the
  content. It is not passed directly to the output and is not a fixed
  dequantization carrier. Continuous scale is deliberately shared by all three
  arms in this first experiment so that the only intervention is the content
  representation.
- Normalized-float content is the exact signed `W / scale`. GPTQ-continuous
  content is the compensated integer code cast to float. GPTQ-token content is
  the identical integer code represented categorically. GPTQ calibration uses
  only the training half of each tile's activation rows; evaluation uses the
  disjoint held-out half.

### Unified bottleneck architecture

- Each tile becomes 1024 content tokens: 128 output-channel groups times eight
  ordered 16-value chunks. There is no assumption that neighboring chunks are
  image patches.
- The encoder prepends 32 learned latent slots, applies four ordinary
  bidirectional pre-RMSNorm Transformer blocks, discards the content states, and
  projects the retained slots to the sole deterministic bottleneck
  `z[32,384]` (12,288 scalars per tile).
- The decoder receives only `z`, projects it back to hidden width, appends 1024
  learned output queries, applies four blocks of the same kind, and predicts 16
  signed weights from each query. There is no protected floor, raw-weight
  bypass, fixed dequantized output, learned residual side path, KL, sampling, or
  second model of different complexity.
- Training targets the original floating-point weights through a primary
  activation-action loss. Raw-weight reconstruction, direction, scale, encoder
  depth gradients, and actual parameter changes are diagnostics.

### Precommitted interpretation

- Normalized float versus GPTQ-continuous isolates information lost or improved
  by GPTQ compensation and discretization.
- GPTQ-continuous versus GPTQ-token isolates whether categorical ingestion
  transports the same discrete information better through the encoder.
- A shared depth-wise gradient collapse in all arms implicates the encoder
  independently of the representation. Healthy gradients with poor GPTQ arms
  implicate the quantized representation; healthy GPTQ tokens with a weak
  continuous-code arm support the categorical-token hypothesis.
- This is a bounded diagnostic on exact64, not a final production baseline or a
  universal claim about weight-space compression.

### Added activation-metric arm and prelaunch correction

- The user added one fourth continuous representation: feed weights transformed
  as `A_train^(1/2) @ W`, where `A_train` is the input-activation second moment
  computed from the calibration half of the activation rows. In plain terms,
  this rotates and scales weight directions according to how strongly the real
  inputs excite them. It uses the identical unified model and bottleneck.
- The transform is computed over the complete 1152-input operator, not
  independently per tile. A thin SVD of `X_train[256,1152]` applies the square
  root without constructing a dense 1152-by-1152 matrix. The transformed matrix
  is then split into the same nine tiles and uses its own local per-output-group
  scale feature, which still must pass through the bottleneck.
- Independent prelaunch review found one real blocker in the first
  implementation: training and evaluation originally measured each of the nine
  tiles separately. That drops cross-terms between tile errors and therefore is
  not the functional error of the complete operator.
- The implementation was corrected before the real run. A batch now samples
  complete operators, processes all nine tiles, sums all nine `X_tile @ W_tile`
  contributions, and only then computes activation-action loss and NRMSE. GPTQ
  fixed-codec diagnostics use the same full-operator action. A corrected smoke
  run completed at
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_smoke_20260827T0530Z`.

## 2026-08-27 — Result of the authorized unified-bottleneck experiment

### Completed run

- Production artifact root:
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v1_20260827T151252Z`.
- Four arms were trained for 512 identical optimizer steps on 48 complete
  operators and evaluated at nine checkpoints on 16 held-out operators and the
  held-out half of their activation rows.
- The run completed without nonfinite values. `COMPLETE.json` matches the
  summary hash; all shared model starts match. The comparison plot was inspected
  and is readable.

### Experimental evidence

- Fixed GPTQ dequantization itself is good: full-operator held-out action NRMSE
  is 0.07997. Therefore failure of the learned GPTQ arms is not caused by the
  quantizer having already destroyed the task-relevant signal.
- Final held-out global action NRMSE is 1.02013 normalized-float, 1.04983
  GPTQ-continuous, 0.99039 GPTQ-token, and 1.03482 `A_train^(1/2) @ W`. A zero
  prediction has NRMSE 1.0. The best point anywhere on each curve is only
  0.99141/0.99505/0.99039/0.99512, respectively. No arm achieved useful weight
  reconstruction.
- GPTQ-token is relatively better at the final step: it beats GPTQ-continuous
  on 14/16 held-out operators, with paired mean NRMSE delta -0.03394 and a
  bootstrap 95% interval [-0.05817,-0.01105]. It beats normalized-float on
  14/16 as well, mean delta -0.02322, interval [-0.04311,-0.00452]. This is a
  reproducible relative signal in this run, not an architecture success.
- The activation-square-root representation is worse than normalized float on
  10/16 operators, mean delta +0.00761, bootstrap interval
  [+0.00286,+0.01273]. It improves train fitting but not held-out behavior.
- All four latent spaces collapse severely: final held-out participation rank
  is only 1.10--2.31 despite a 12,288-coordinate bottleneck. Raw prediction
  cosine with true weights is only 0.007--0.012.
- A saved-model intervention in `bottleneck_usage.json` shows GPTQ-token is the
  only arm with net useful latent information on the primary global metric:
  matched 0.99039 versus shuffled 0.99666 and zero-latent 0.99707. The gain is
  real but tiny. Other arms' matched latents are operator-specific but net
  harmful relative to zero on the energy-weighted metric.

### Narrow conclusion

- The hypothesis that raw small amplitudes alone make the signal disappear is
  contradicted: exact local normalization and `A_train^(1/2) @ W` do not solve
  the failure.
- Literal gradient death is also contradicted: every encoder block has a
  nonzero gradient. However gradients are heavily concentrated at the output
  head and usually attenuate through the four encoder blocks.
- The strongest supported mechanism is a unified-model failure, not primarily
  an input-coordinate failure: attention pooling into the 32 latent slots
  collapses to a nearly one-dimensional summary, while the decoder/head learns
  the easy near-zero/template output. GPTQ categorical tokens regularize this
  failure slightly but do not repair it.
- It is not yet established whether the root cause is slot-pooling collapse or
  decoder/head credit starvation. The full causal review is stored at
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v1_20260827T151252Z/analysis_review.md`.

### Loss clarification

- The only training objective was full-operator relative activation-action MSE.
  For each operator, the nine tile contributions were first summed into
  `Y_pred = sum_t X_t W_pred_t` and `Y_true = sum_t X_t W_t`; the batch loss was
  `sum((Y_pred-Y_true)^2) / sum(Y_true^2)`. No raw-weight, cosine, scale, KL, or
  auxiliary reconstruction loss was added.
- Final evaluation reports the square root of that ratio (action NRMSE). On all
  48 train operators the final loss/NMRSE pairs were: normalized float
  0.83950/0.91624, GPTQ-continuous 0.82912/0.91056, GPTQ-token
  0.92388/0.96119, and activation-square-root 0.82382/0.90765.
- For the activation-square-root arm, only the encoder input was changed to
  `A_train^(1/2) @ W`. The decoder still directly predicted ordinary `W_pred`,
  and the same `X @ W_pred` versus `X @ W` loss was used. There was no explicit
  inverse `A^(-1/2)` mapping and `A` itself was not provided to the decoder.
  Consequently this arm tests whether `A_train^(1/2) @ W` alone is a
  decode-sufficient representation for the original operator, not a pure
  invertible coordinate reparameterization. Since 256 calibration rows make
  `A_train` rank at most 256 in a 1152-dimensional input space, the transform
  discards a large calibration-null subspace; its negative result cannot by
  itself reject an A-aware codec that also transmits the transform or predicts
  directly in action space.
- A readable two-panel plot of the full-train loss curves (full range plus a
  post-warmup zoom) is stored at
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v1_20260827T151252Z/train_loss_curves.png`.

## 2026-08-27 — Authorized 10,000-step continuation experiment

- The user requested a fresh 10,000-step optimization run and a combined loss
  graph.
- The implementation and scientific comparison remain unchanged: four arms,
  identical shared starts, batches of six complete operators, nine-tile summed
  activation-action loss, 48/16 grouped operator split, and held-out activation
  rows. This is a fresh run rather than a resume from step 512.
- Full train/test evaluation is scheduled every 250 steps, producing 41 points
  while avoiding unnecessary full-set evaluation overhead. The cosine learning-
  rate schedule is stretched consistently over the new 10,000-step horizon.
- User correction: this 10,000-step run is purely an optimization-stability and
  train-fit experiment. With only 64 operators, held-out generalization is not a
  decision criterion. Test telemetry may remain in the artifacts but must not
  drive the verdict or the requested graph.
- The post-run review must compare full-train loss trajectories, late-step
  stability/oscillation, attainable train loss, depth gradients, and whether
  bottleneck rank recovers or remains collapsed. It must not call train/test
  divergence a failure for this experiment.

## 2026-08-27 — Revised five-arm 10,000-step stability experiment

### User decisions

- The obsolete four-arm 10k run was stopped shortly after launch and must not be
  interpreted; its incomplete root is
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_10k_v1_20260827T152930Z`.
- Add a fifth `raw_float` control: feed raw signed weight chunks without any
  normalization. To keep this a literal raw baseline, its scale-token input is
  fixed to zero rather than leaking an explicit local-scale feature.
- Replace the activation-action training objective with the shared structural
  loss used by the current Big VAE: patch direction loss plus 0.1 times patch
  log-scale loss, with patch size 16, direction weighting exponent 0.5 and
  scale Huber delta 0.1. Raw reconstruction, relative reconstruction,
  behavioral, KL, and the V11-topology-specific complement auxiliary are not
  part of this cross-representation objective.
- The comparison is now purely about optimization on the 48 training operators
  over 10,000 identical steps. Held-out metrics may remain as diagnostics, but
  the requested plot and verdict must use full-train structural loss only.

### Implemented comparison

- The five arms are `raw_float`, `normalized_float`, `gptq_continuous`,
  `gptq_token`, and `activation_sqrt`. They share the same depth-4 encoder,
  `[32,384]` bottleneck, depth-4 decoder, optimizer schedule, operator batches,
  and all non-input-specific initial parameters.
- A focused 4-step GPU smoke completed at
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v2_smoke_20260827T1536Z`.
  All five arms were finite, every parameter tensor had a gradient, all four
  encoder blocks had nonzero gradient RMS, and the shared-start digest matched.
  This smoke is only a contract check, not evidence about which representation
  learns best.
- One targeted independent review of the revised source and smoke is pending
  before the expensive 10,000-step launch.

### Executed outcome through step 4000

- Independent review issued FORMAL GO for the explicit 10,000-step command.
  The user then requested an early stop once the result was already clear. The
  run was stopped after the complete step-4000 full-train evaluation; its root
  is
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_10k_v2_20260827T154000Z`.
  It is intentionally marked `STOPPED_BY_USER.json`, not `COMPLETE.json`.
- The full-train curve has 17 common points at steps 0,250,...,4000 for all five
  arms. The combined linear/log plot is
  `train_structural_loss_curves.png`; detailed review is `analysis_review.md`.
- Step-4000 Big-VAE structural loss: raw float 0.557387, normalized float
  0.013084, GPTQ continuous 0.012395, GPTQ categorical token 0.112454, and
  `A_train^(1/2) W` 0.027696. Best observed losses are respectively
  0.475976/0.012438/0.012370/0.109716/0.027696.
- Evidence strongly supports a raw-amplitude conditioning failure: the same
  unified architecture learns normalized continuous weights approximately 43
  times better than raw weights by step 4000. This is not literal gradient
  death because every encoder block has finite nonzero gradient telemetry.
- GPTQ-continuous and normalized-float are effectively tied, so no benefit from
  GPTQ error compensation over ordinary normalization is established.
  Categorical weight tokens learn but remain about 9 times worse than the same
  GPTQ representation supplied as continuous values. Activation-metric input
  is viable but about 2 times worse than simple normalized float.
- The separation is almost entirely directional: final weighted scale terms
  are all around 0.0022--0.0030, while direction loss ranges from 0.0101 to
  0.5544. The current supported conclusion is about optimization/fit only, not
  held-out generalization or downstream generation.

### Cancelled latent-usage probe

- The user asked how strongly each decoder uses the encoder latent. A causal
  matched/zero/operator-shuffle/tile-shuffle/mean/scale-sweep probe was prepared.
- The interrupted step-4000 run had no model checkpoint because the original
  launcher saved `final_models.pt` only after normal completion. Loss curves
  and gradient telemetry cannot substitute for decoder interventions.
- A deterministic replay was started solely to recover step-4000 weights, then
  cancelled at the user's request around training step 180. Its incomplete root
  is
  `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_replay4k_latent_v1_20260827T162900Z`.
  No conclusion about decoder latent use was drawn.

## 2026-08-27 — Raw versus normalized weights with production Distribution Encoder

### User decision and narrow question

- The next experiment compares exactly two representations: `raw_float` and
  `normalized_float`. GPTQ, categorical tokens and `A_train^(1/2) W` are out of
  scope for this run.
- The purpose is a cheap transferability test for the production Weight AE:
  does simple weight normalization still provide the large optimization
  advantage once activation conditioning is introduced in the same semantic
  locations as production?
- The run is fixed to 3000 optimizer updates, batch size six complete
  operators, evaluation every 250 updates, and the same Big-VAE structural
  objective as the prior five-arm run. The decision remains based on full-train
  optimization stability and fit, not 16-operator test generalization.

### Conditioning contract

- The experiment instantiates the production `InputDistributionEncodingModule`
  with the exact resolved V11 dimensions: `k_s=64`, `Kq=128`, `d_var=256`,
  `d_dist=256`, six variable-attention layers with four heads, three DCN cross
  layers, three 128-wide deep layers, covariance enabled and patch size 16.
- Each tile activation tensor `X[512,128]` is processed into eight context
  vectors of width 256 exactly as in production. These context vectors are
  added only to encoder content keys at every encoder block. They are not
  inserted into weight values.
- On the decoder side, the same eight context vectors are repeated across the
  128 output positions, concatenated with the existing positional/output query,
  and passed through the decoder query MLP. They are not added directly to
  decoded output values. This copies production's encoder-key and decoder-query
  conditioning semantics while retaining the deliberately small depth-4 test
  backbone.
- Both arms have byte-identical starts for every shared parameter and consume
  the same operator batches. The normalized arm uses the existing reversible
  local representation: each output-group's 128 signed weights are divided by
  its max-absolute-derived scale, and standardized log scale is supplied to the
  same input path; the raw arm receives raw signed values and a zero scale
  feature. Both decoders predict ordinary weights and are trained against the
  same ordinary-weight structural target.

### Focused smoke evidence

- A four-update GPU smoke completed at
  `/mnt/shared/weightclip_benchmark/raw_vs_normalized_prod_dist_smoke_20260827T164033Z`.
- Both arms were finite, all parameter tensors had gradients, every encoder and
  decoder block was live, and shared-start hashes matched. The new paths were
  demonstrably active at step 1: Distribution Encoder gradient RMS was about
  `1.79e-4` and decoder query-conditioner gradient RMS about `4.67e-4` in both
  arms.
- The smoke is only a shape/memory/gradient contract check. It is not evidence
  for raw versus normalized training quality.

### Next action

- Obtain one independent review of the final two-arm 3000-step command and
  smoke artifact. If GO, launch a fresh real root, monitor through completion,
  inspect all train-loss points and the final plot, and report whether the
  normalization advantage survives production-style activation conditioning.

### Completed 3000-step result

- Independent review returned FORMAL GO. The completed run root is
  `/mnt/shared/weightclip_benchmark/raw_vs_normalized_prod_dist_3k_v1_20260827T1648Z`.
  It finished all 3000 updates in 718 seconds, wrote a valid completion marker,
  and saved both final models.
- There are 13 common full-train evaluations at steps 0,250,...,3000. Both
  curves are finite and strictly decreasing at every recorded interval.
- Final Big-VAE structural loss is `0.327115` for raw versus `0.022274` for
  normalized: normalized is 14.69 times lower. The difference is directional,
  not scale-only: direction is `0.324309` versus `0.019520`, while the weighted
  scale contribution is essentially tied (`0.002806` versus `0.002753`).
- The internal geometry separates sharply. Final train latent participation
  rank is `5.54` raw versus `50.17` normalized. Raw latent-state RMS grows over
  four encoder blocks as `1.80,3.18,4.16,4.94`; normalized remains controlled at
  `0.089,0.157,0.216,0.267`. All blocks remain gradient-live, so the supported
  mechanism is poor signal conditioning/low-diversity state rather than literal
  dead gradients.
- A zero-update context-shuffle intervention on the saved models confirms that
  both decoders use activation conditioning: shuffled context raises raw loss
  `0.3271→0.9826` and normalized `0.02227→0.79848`, with about 1.37 relative RMS
  output change in each. Raw latent itself changes only 0.002 relative RMS,
  whereas normalized latent changes 0.459, so raw uses context mainly through
  the decoder query path while normalized also integrates it into the weight
  latent.
- Narrow decision: production-style activation conditioning does not rescue raw
  input. Local weight-value normalization with explicit scale remains the
  supported next production-AE intervention. This is strong train-fit evidence,
  not a guarantee for the full V11 topology or held-out generalization.
- Detailed review:
  `/mnt/shared/weightclip_benchmark/raw_vs_normalized_prod_dist_3k_v1_20260827T1648Z/analysis_review.md`.

## 2026-08-27 — Transfer normalized input into the original 725M V4 AE

### User correction and frozen scope

- The requested production-scale model is not V11. It is the original
  approximately 725M `latent_feedback_perirms_qknorm_v4` model from the 50k
  lineage. All V9/V10/V11 complement/carrier architectures are out of scope.
- The experiment uses the same exact64 operator selection and schedule as the
  later bounded runs: 64 `layer3.0.conv2.weight` operators, nine 128x128 tiles
  per operator, 1984 optimizer updates, physical batch 6 with accumulation 3,
  seed 42, and evaluation every 256 updates.
- The objective is unchanged from the V4 structural run: patch direction loss
  plus 0.1 times patch log-scale loss. Behavioral, reconstruction, relative and
  KL terms remain zero. Targets and decoder outputs remain ordinary raw W.

### Single intervention

- Only the V4 encoder input representation changes. For each output column of
  each 128x128 tile, one local scale is computed as `max_abs / 7`. The 128
  signed values are divided by that scale, so the normalized values are in
  approximately [-7,7].
- The standardized `log2(scale)` is repeated over that column's eight 16-value
  patch tokens and added through a small MLP to the ordinary patch-token
  embedding. The frozen constants are copied from the successful small
  conditioned experiment: mean -6.328672409 and standard deviation
  0.595952213.
- There is no decoder-side multiplication, raw-W bypass or special residual
  reconstruction path. Both normalized values and scale must pass through the
  original patch tokenizer, ten-layer latent-feedback encoder, 32x384 latent
  bottleneck and original eight-layer decoder.

### Implementation and preflight evidence

- Launcher:
  `projects/weight-vae/workspace/training/weightclip_benchmark/run_ae_v4_normalized_operator_set.py`.
- Config:
  `projects/weight-vae/workspace/conf/weightclip_benchmark/ae_v4_normalized_exact64_1984step.yaml`.
- Exact parameter contract is 725,333,585 total and 706,305,089 trainable; the
  representation adds only 4,288 scale-embedding parameters.
- The exact64 dry-run passed with the established selection hash
  `6fb8ad70...` and schedule hash `7826db6a...`. Full V4 focused tests pass
  11/11. A CUDA BF16 synthetic forward/backward produced finite nonzero
  gradients in the scale MLP, patch tokenizer, encoder layers 0 and 9 and
  decoder layers 0 and 7. This is a liveness/contract check, not training
  evidence.
- Next action: one independent focused review, then launch the single normalized
  V4 exact64 run and inspect the full step0..1984 loss/gradient/latent evidence.

### Launch preflight correction

- Independent review first found and closed a fairness bug: constructing the
  new scale MLP advanced the global CPU RNG and changed later shared V4
  parameters. The module is now initialized inside a local RNG context. A
  regression test proves every shared raw/normalized state tensor is byte-equal
  at the same seed and that exactly four scale-MLP tensors are additional.
- The first launch root
  `/mnt/shared/weightclip_benchmark/ae_v4_normalized_exact64_v1_1984step`
  stopped before model construction or training because a legacy worker guard
  allowed the exact64 evaluator only for V9--V11. This is operational preflight
  evidence, not a training result. The guard is being extended narrowly to V4
  only when `per_output_maxabs_q7` is enabled; the real run will use a fresh v2
  root.

## 2026-08-27 — Normalized V4 horizon extended to 5k

- User asked to extend the original 725M V4 normalized-input experiment to
  5,000 updates. The partial v2 run was intentionally interrupted at step 660:
  its final-only checkpoint policy stored no optimizer/RNG state, so a model-only
  continuation at step 1984 would not have been an honest continuous trajectory.
- A fresh run was launched from the same seed/start/data with only the horizon
  changed to 5,000 updates. The data manifest explicitly records the repeated
  1,984-match base cycle: two full repetitions plus a 1,032-match tail, 90,000
  training tiles total. Other operator-set architectures retain their 1,984-step
  contract. Independent targeted review gave formal GO.
- Active run root:
  `/mnt/shared/weightclip_benchmark/ae_v4_normalized_exact64_v3_5000step`.
  Early eval@256 reproduces the prior failure pattern: mean direction loss
  `0.996088 -> 0.962440`, median NRMSE `9.31 -> 1.246`, while latent-shuffle
  median direction effect is only `1.25e-6` and output-relative change `0.217%`.
  This is preliminary evidence that the model first suppresses random output and
  largely ignores the encoded W; the 5k trajectory is needed to test late escape.
- User explicitly asked to stop live monitoring/token use and will wake the agent
  after the run finishes. Leave the process untouched. On return, review final
  checkpoint, all eval rows, train-loss curve, causal latent-use, and depthwise
  gradient telemetry before concluding.

### Live status after user wake-up

- The 5k process remains healthy at roughly step 1,100 with no logged errors.
  Five full64 eval rows are present through step 1,024.
- Direction loss improves only slowly after the initial retreat:
  `0.996088` (step0), `0.962440` (256), `0.962104` (512), `0.961693` (768),
  `0.961457` (1024). Latent use remains effectively absent: median z-shuffle
  direction deltas are around zero and z-shuffle output-relative delta remains
  about `0.0021`.
- Late encoder routing gradients continue to collapse: layer-9 local q/k gradient
  RMS is about `2.3e-12` at step1024, versus live values near `1e-8` at step1.
  This strengthens the preliminary conclusion that input normalization alone does
  not fix V4's deep credit/signal path, while the full 5k result is still pending.

## 2026-08-27 — Why the small normalized AE learns but normalized V4 does not

### User question

- Compare the successful small normalized model with the original 725M V4 after
  giving both the same normalized-weight representation. Identify the remaining
  fundamental architectural difference rather than blaming raw amplitude again.

### Source-level comparison

- The successful model is a 42M single-stream depth-4 encoder: direct linear
  projection of each signed 16-value weight chunk, explicit scale/position
  embeddings, then 32 latent tokens and 1,024 content tokens pass through the
  same four pre-norm residual self-attention blocks. Activation context changes
  encoder keys but is never added to content values. Its depth-4 decoder uses the
  same repeated block style and a direct 16-value output head.
- V4 is a 725M dual-stream recurrent graph. A nonlinear activation-conditioned
  patch tokenizer compresses each 16-value patch to 64 dimensions and applies a
  LayerNorm summary. Each of ten encoder stages then performs per-output local
  token attention, a second activation-conditioning adapter, latent cross-attention
  over all tokens, and from stage 2 onward a shared latent-to-weight feedback write.
  Thus W-specific tokens are repeatedly mixed with common latent/context signals.
- Both models serialize the same 32x384 latent bottleneck, but V4 has about 17
  times more parameters around it and a much longer path. V4's eight-layer decoder
  starts from strong position+activation queries; latent cross-attention is only a
  residual addition alongside decoder self-attention and FFNs. Its direction/scale
  heads can therefore learn an X/position-conditioned average without using z.
  The small model has four encoder/four decoder blocks and direct joint token mixing.
- V4 residual writes are depth-scaled to roughly 0.16 for weight state, 0.183 for
  latent state and 0.204 for decoder state; the small model uses about 0.354. V4
  also uses LR 5e-5 without warmup versus 1e-4 with 32-step warmup in the small
  experiment. These are real operational differences, but not yet established as
  the primary cause.

### Current causal read

- Established proximal mechanism: V4 rapidly learns decoder latent suppression.
  Its z-shuffle output-relative effect falls from 5.46% at initialization to about
  0.21% by step256 and remains there, while the loss rapidly retreats from the bad
  random output to a roughly 0.962 direction-loss plateau. A fixed-X W swap is
  equally ineffectual. The easiest optimization path is therefore an
  X/position-conditioned common prediction, not reconstruction through z.
- The whole encoder is not literally gradient-dead. At step1024, Perceiver and
  value/output projections still receive gradients around 1e-8 to 1e-7, but the
  late local q/k routing gradients are around 2e-12. The supported statement is
  collapse of W-specific routing/credit, not absence of every encoder gradient.
- Leading but not yet isolated mechanism: repeated normalized latent feedback plus
  repeated activation-value conditioning drives token states toward a common
  context/latent attractor. Once the decoder suppresses z, the resulting weak
  encoder gradient makes this shortcut self-reinforcing. Input normalization fixes
  the starting coordinate scale but cannot fix this topology.

### Remaining discriminators

- H1 tokenizer dilution: W-pair sensitivity is already small immediately after the
  V4 patch tokenizer.
- H2 recurrent encoder common-mode collapse: tokenizer sensitivity is healthy but
  layer-by-layer token/latent sensitivity and participation collapse after feedback
  and conditioning writes.
- H3 decoder shortcut/readout failure: final z remains W-sensitive, but sensitivity
  disappears only across the decoder; matched/shuffled/zero-z interventions and a
  frozen nonlinear z readout distinguish this.
- H4 depth/residual scaling: a depth-4 V4 with the same two-stream topology learns;
  this would implicate depth rather than feedback/conditioning semantics.
- Cheapest next evidence after the 5k checkpoint is one zero-update hooked replay
  measuring fixed-X W-pair deltas after tokenizer, each of ten encoder token/latent
  states, final z, each decoder layer and output, together with matched/shuffled/
  zero-z outputs. This single panel localizes the first loss of W identity without
  training another architecture.

## 2026-08-27 — Exact V4 patch-tokenizer anatomy

- User's leading hypothesis is that the main failure is already in the tokenizer.
  Source audit confirms that V4 does not use the successful small model's direct
  `Linear(16 -> hidden)` content projection and has no guaranteed raw/normalized
  weight skip to the final token.
- For each 128x128 tile, normalization computes one max-abs/q7 scale per output
  column over all 128 input weights, splits the normalized column into eight
  16-value patches, and repeats the same standardized log2 column scale over all
  eight patches. Distribution Encoder supplies an aligned `[16,256]` activation
  descriptor to every 16-value patch; that descriptor is repeated across all 128
  output columns.
- Exact conditioned tokenizer dimensions are `p=16`, `d_patch=64`, two blocks,
  hidden width 256, activation projection width/rank 32. Each scalar weight is
  independently projected `1 -> 64`; each activation descriptor is projected
  `256 -> 32`.
- Each block LayerNorms the 64-vector weight state, adds a base
  `64 -> 256 -> 64` MLP return normalized to RMS 0.5, then forms a multiplicative
  interaction between projected weight state and projected activation. An
  X-conditioned `32 -> 256 -> 64` return is also normalized to RMS 0.5 and added
  through a sigmoid gate initialized to 0.2. Thus X enters token values twice, not
  merely routing/keys, and its write magnitude is normalized independently of
  weight-signal magnitude.
- After two residual blocks, the `[16,64]` state is flattened to 1,024 values,
  LayerNormed across the complete patch, compressed `1024 -> 256 -> 64`, and only
  then receives the separate scale MLP addition. A final `64 -> 1280` projection
  creates the transformer patch token.
- Highest-risk points: (1) the final summary has no direct weight residual and is
  the first unavoidable lossy/nonlinear map; (2) patchwise LayerNorm suppresses
  the 16-value patch's absolute radius even though the stored scale is only one
  per full 128-value column, so relative norms of its eight patches are not
  explicitly carried; (3) repeated common X-value writes can dominate or align
  tokens across output columns; (4) per-scalar conditioned transforms see
  LayerNormed features, leaving exact magnitude only in the residual state that
  is later passed through the final summary LayerNorm.
- These are structural risks, not yet causal proof. The decisive final-checkpoint
  measurement is fixed-X W-pair sensitivity at normalized patch input, after
  `y_proj`, after each conditioned block, before/after summary, after scale add,
  and after `64 -> 1280`. A collapse first at summary would directly validate the
  tokenizer hypothesis.

## 2026-08-27 — Proposed simple replacement tokenizer and small-model test

### Design principle

- User wants a simple tokenizer whose only optional extra role is activation
  conditioning, with a hard structural guarantee that X cannot overwrite the
  normalized weight signal. The proposal is to remove the conditioned residual
  MLP stack, all tokenizer LayerNorms, and the `1024 -> 256 -> 64` summary.
- Use a fixed typed 64-coordinate patch record followed by one bias-free
  `Linear(64 -> model_width)`. The first 16 coordinates always contain the exact
  normalized signed weights; standardized log2 scale and a zero flag occupy
  dedicated coordinates. No additive X value is allowed in those coordinates.

### Two candidate branches

- Conservative A: `[W_norm(16), log_scale(1), zero_flag(1), zeros(46)]`. Activation
  context remains key-only in each encoder attention block, exactly as in the
  successful small normalized model. This is the safest and simplest option.
- Recommended conditioned B: `[W_norm(16), W_norm*tanh(g(X_aligned))(16),
  log_scale(1), zero_flag(1), zeros(30)]`, where `g` is one shared
  `Linear(256 -> 1)` per aligned input position. The original W lane remains exact;
  the bounded interaction occupies a disjoint lane, is zero when W is zero, and
  cannot cancel the base lane. A single `Linear(64 -> width)` embeds the record.
  Keep the existing key-only context in both arms so the sole intervention is the
  token record.

### Proposed experiment after user chooses a branch

- Reuse the known-good 42M depth-4 self-attention encoder/decoder setup, 32x384
  bottleneck, same 48 training operators, B6, 3,000 updates, LR 1e-4 with 32-step
  warmup, and identical Big-VAE structural loss. Run only two simultaneous arms:
  the exact successful normalized control and the chosen tokenizer candidate.
- Share all downstream encoder/decoder/Distribution starts and all batches. Persist
  train curves every 250 steps, depth gradients, latent participation/rank and final
  checkpoints. Add fixed-X W-swap and fixed-W X-shuffle deltas directly at tokenizer
  output, plus zero-W and scale-ladder contracts.
- Candidate success means stable monotone fit, final train structural loss no worse
  than roughly 0.03 / within 20% of the control, healthy latent rank and causal
  dependence on W. If it passes in the small graph, transplant the exact tokenizer
  unchanged into V4; if it fails there, reject it before another 725M run.

## 2026-08-27 — Tokenizer decision deferred until current V4 run completes

- User rejected candidate A as scientifically uninformative: a typed record that
  simply preserves normalized weights is too close to the already known-good direct
  weight input and is therefore expected to train.
- Decision: do not implement or launch another tokenizer arm yet. Let the current
  5,000-step normalized V4 run finish, review its final artifacts, and then test
  conditioned candidate B directly in the known-good small self-attention setup.
- Candidate B remains the intended intervention: an exact protected normalized-W
  lane plus a disjoint bounded interaction lane `W_norm * tanh(g(X_aligned))`, one
  shared `256 -> 1` conditioner, and one bias-free `64 -> width` projection. No
  conditioned MLP stack or tokenizer LayerNorm.
- Current-run snapshot at step 3,200/5,000: train structural loss `0.694953`; the
  exact64 eval at step 3,072 has matched mean direction loss `0.750550`, versus the
  earlier apparent plateau near `0.96`. The run is therefore learning again, but
  latent causality is still extremely weak: z-shuffle output relative delta
  `0.008878` and median direction effect `-4.23e-6`. This is an interim observation,
  not a final verdict; expected remaining wall time is roughly 35 minutes including
  final evaluation/checkpoint writing.

## 2026-08-27 — Normalized V4 run stopped early by user

- User concluded that the training trend was already clear and explicitly requested
  termination. The single process for
  `run_ae_v4_normalized_operator_set.py` was stopped cleanly with SIGTERM.
- Last persisted training point: step `4,330 / 5,000`, structural loss `0.218870`.
  The curve had transitioned from a long near-`0.96` plateau into rapid improvement
  after roughly step 2,300. Snapshot plot:
  `/mnt/shared/weightclip_benchmark/ae_v4_normalized_exact64_v3_5000step/analysis/current_train_loss_curve.png`.
- The run used a final-only checkpoint policy, so early termination produced no
  `.pt` checkpoint and no COMPLETE marker. Logs and operator-set evaluation rows are
  preserved under
  `/mnt/shared/weightclip_benchmark/ae_v4_normalized_exact64_v3_5000step`.
- Next agreed experiment remains the small known-good self-attention setup with the
  conditioned tokenizer B; conservative tokenizer A is rejected as uninformative.

## 2026-08-27 — Small-model tokenizer B experiment completed

### User decision and exact intervention

- User requested a single 3,000-step tokenizer-B arm in the previous known-good
  depth-4 normalized self-attention setup, compared against the archived normalized
  direct-input curve rather than rerunning a redundant control.
- Candidate B uses the exact typed 64-coordinate record:
  `[W_norm(16), W_norm*tanh(g(dist_var))(16), standardized_log2_scale(1),
  zero_flag(1), zeros(30)]`, followed by one bias-free `64 -> 512` projection.
  `g` is one shared bias-free `256 -> 1` map over the production Distribution
  Encoder's per-variable embeddings. Existing activation conditioning through
  encoder attention keys and decoder queries is unchanged.
- Exact known-good downstream starts were reconstructed and verified against archived
  hash `2db1ac6f...`; all 160 common non-tokenizer tensors matched, and the first 16
  columns of the B projection exactly inherited the old direct `16 -> 512` weight.
  The train split, logical sampling stream, LR schedule, objective, and 13 evaluation
  steps were unchanged. Focused tests were 2/2 green and a four-step exact64 CUDA
  smoke passed with all trainable tensors and both tokenizer parameters gradient-live.

### Training result

- Run completed all 3,000 steps in 367 seconds and saved a final checkpoint:
  `/mnt/shared/weightclip_benchmark/conditioned_tokenizer_b_small_3k_v1_20260827T1950Z`.
- Full-train structural loss was monotone across all 13 evals. Candidate versus
  archived normalized direct baseline was: step 250 `0.948416 vs 0.948350`, step
  1000 `0.429517 vs 0.411455`, step 2000 `0.065989 vs 0.064490`, step 2500
  `0.034364 vs 0.034361`, and final `0.023469 vs 0.022274`. Final B is only 5.37%
  worse and passes the precommitted within-20% / below-0.03 stability criterion.
- The small encoder remains healthy: final participation rank `43.24` versus baseline
  `50.17`; latent RMS by depth `0.0879, 0.1554, 0.2093, 0.2514` versus baseline
  `0.0886, 0.1573, 0.2163, 0.2673`. Direction term accounts for almost all of the
  small gap (`0.020692 vs 0.019520`); scale loss is nearly equal.
- The decoder genuinely uses the encoder latent. On the train split, matched loss is
  `0.02344`, operator-shuffled latent gives `0.97760`, and zero latent gives
  `0.98266`; output relative RMS changes are `1.509` and `1.425` respectively.
  The protected record W lane has exact max error zero under matched, W-swapped, and
  X-swapped inputs. Fixed-X W swap changes tokenizer output by relative RMS `1.404`.

### Important mechanistic caveat

- The tokenizer-B experiment establishes that the simple typed tokenizer does not
  break optimization, but it does **not** establish useful activation-specific
  modulation inside the tokenizer. The learned gate is almost a constant negative
  scalar: mean `-0.9648`, RMS `0.9649`, median absolute value `0.9727`, p95 `0.9844`.
  Its projected interaction has `0.909x` the RMS of the base-W contribution.
- Removing the interaction lane after training is destructive (loss `0.22482`), so
  the model uses it; however, shuffling only the per-variable activation embeddings
  while keeping attention-key and decoder-query conditioning matched changes latent
  RMS by only `0.00558`, output RMS by `0.0292`, and loss only
  `0.02344 -> 0.02351`. Therefore the branch has mostly become a second rescaled
  linear W path, not a meaningful W-by-activation interaction.
- Narrow conclusion: GO for the typed tokenizer as a non-destructive weight input;
  NO evidence yet that this exact tanh scalar conditioner is the right activation
  conditioner. Transplanting it unchanged into the large model would preserve a
  simple W path but would overstate what the conditioning experiment demonstrated.

### Artifacts

- Overlay plot:
  `/mnt/shared/weightclip_benchmark/conditioned_tokenizer_b_small_3k_v1_20260827T1950Z/loss_overlay_vs_previous_normalized.png`
- Canonical metrics/summary/checkpoint are in the same run root; zero-update
  tokenizer usage audit:
  `analysis/tokenizer_b_usage.json`.
- Implementation:
  `training/weightclip_benchmark/run_conditioned_tokenizer_b_small.py`; focused
  tests: `tests/weightclip_benchmark/test_conditioned_tokenizer_b_small.py`.

## 2026-08-27 — Proposed 700M scale-up of the simple Direct Normalized model

### User decision

- Scale the already trainable, single-path Direct Normalized self-attention AE to
  approximately the active parameter budget of the original 50k V4 model.
- Preserve the exact serialized bottleneck `32 x 384` and approximately preserve
  the original trainable encoder/decoder parameter allocation.
- Run only one bounded exact64 stability experiment for roughly 2,000–3,000
  updates. Do not launch until the scaled config has first been shown to and
  approved by the user. A successful bounded run may later justify a full 50k run.

### Proposed scale contract (not launched)

- Architecture remains the simple unified Direct Normalized graph: one direct
  bias-free `16 -> d_model` normalized-weight projection, one standardized
  per-output log2-scale MLP, repeated pre-norm RMSNorm self-attention/SwiGLU
  blocks, one `32 x 384` bottleneck, and one repeated decoder stack. The production
  Distribution Encoder conditions encoder attention keys and decoder output
  queries exactly as in the already tested small model. There is no heavy patch
  tokenizer, latent feedback, protected/raw bypass, expert ensemble, or auxiliary
  decoder path.
- Candidate dimensions: `d_model=1536`, `24` attention heads (`head_dim=64`),
  `ffn_dim=5728`, encoder depth `13`, decoder depth `6`, bottleneck slots `32`,
  bottleneck width `384`, dropout `0`.
- Exact candidate count from meta construction: `706,927,488` trainable/total
  parameters. Proposed encoder-side allocation is `479,501,184` (67.829%);
  decoder-side allocation is `227,426,304` (32.171%); ratio `2.108:1`.
- Reference original V4 active allocation, reconstructed from its exact graph, is
  `706,300,801` trainable parameters: encoder `481,391,216` (68.157%), decoder
  `224,909,585` (31.843%), ratio `2.140:1`. The proposed total differs by only
  `+0.089%`; encoder and decoder allocations differ by `-0.393%` and `+1.119%`.
  The old V4 total including frozen legacy modules was about `725.3M`; matching
  trainable capacity is the relevant comparison for learning.
- Proposed bounded training protocol: exact same selected 64 operators and grouped
  48-train/16-diagnostic split as the known-good small run; 3,000 optimizer updates;
  effective six operators/update, implemented as two operators per microbatch times
  gradient accumulation three for memory; AdamW, LR `1e-4`, 32-step warmup, cosine
  schedule, weight decay `.01`, global grad clip `1.0`; eval every 250 updates.
  Objective stays the existing production-style structural loss: direction weight
  `1.0` plus log-scale weight `.1`, with reconstruction/relative/behavioral/KL terms
  disabled. The target is always original raw FP32 W.
- Only function-preserving or already validated engineering aids are proposed:
  BF16 autocast for matrix multiplies, FP32 structural loss/scale statistics,
  fused SDPA, activation checkpointing, zero dropout, residual output initialization
  scaled by depth, and fan-in-scaled nonzero output-head initialization. These are
  not new representational branches.

### Pending user approval

- No implementation or GPU launch has occurred for this scale-up. The user asked
  to see the exact config first and may revise the 48/16 split, 3,000-step horizon,
  or any architecture dimension before launch.

### Clarification requested about the apparent `32 -> 1536` projection

- The user flagged an apparent projection from `32` to roughly `1.5k` as
  potentially pathological. In the proposed graph, `32` is the number of latent
  tokens, not their feature width; there is no learned `32 -> 1536` map.
- The actual bottleneck tensors are `z: [batch, 32, 384]`. The decoder applies the
  same bias-free feature projection independently to every slot,
  `384 -> 1536`, producing `[batch, 32, 1536]`. This expands decoder working width
  but creates no additional bottleneck information. The original V4 used the same
  pattern `384 -> 1280`, so the proposed change is a 20% wider feature expansion,
  not a 48x expansion of 32 coordinates.
- The other potentially confusing map is the input embedding `16 -> 1536` for each
  normalized weight chunk. It is a standard token embedding with only 24,576
  parameters; whether grouping 16 weights per token is appropriate remains a
  separate tokenizer question, but the projection itself is not the source of the
  700M parameter count.

### Input patch-size reconsideration

- User clarified that the concern is the weight input map `16 -> 1536`: whether
  such an extreme expansion is useful and whether a larger weight patch would
  improve optimization.
- Mechanistic distinction: the tall linear map itself is information-preserving
  (rank at most 16 but injective on a 16-dimensional input when full rank) and is
  normally well-conditioned at random initialization. The more concerning property
  is that patch size 16 creates 1,024 weight tokens. Thirty-two latent queries must
  collect their signal through attention over 1,024 items, making credit assignment
  diffuse and the 13-layer 700M model unnecessarily expensive.
- Increasing patch size is not guaranteed harmless: attention can route only whole
  tokens, so overly large patches hide distinctions among input coordinates inside
  an MLP. Patch size 128 would reduce the sequence to one token per output row but
  would also collapse activation conditioning to one full-input summary.
- Current recommendation is patch size 64 as the balanced single-arm choice: two
  tokens per output row, 256 weight tokens plus 32 latent tokens, direct `64 -> 1536`
  projection, one shared full-row scale repeated on the two half-row tokens, and two
  exactly aligned 64-variable Distribution Encoder contexts. Relative to patch 16,
  attention-score matrix size falls to 7.44% and tokenwise FFN work to 27.27%, while
  the input projection remains overcomplete/injective and retains all raw normalized
  values.
- With all other proposed dimensions unchanged (`d=1536`, heads 24, FFN 5728,
  depths 13/6, latent `32 x 384`), generalizing the input/output/query shapes from
  patch 16 to patch 64 gives an estimated exact count of `706,383,744`: encoder
  `480,063,360`, decoder `226,320,384`, ratio `2.121:1`. This is only `+0.0117%`
  from the original V4 active total `706,300,801`, and remains close to its
  encoder/decoder ratio `2.140:1`.
- This patch-size change is a real architectural intervention, not a purely
  function-preserving implementation trick. It should be explicitly approved by
  the user before replacing the conservative patch-16 scale-up config.

## 2026-08-27 — Scaled Direct Normalized p32 run stopped after decisive early result

- User selected patch size 32 and approved one scaled run. The implemented unified
  model used `d_model=1536`, 24 heads, FFN 5728, depths 13/6, and the exact
  `32 x 384` bottleneck. It had `706,284,416` trainable/total parameters, split as
  encoder `479,619,968` and decoder `226,664,448`, only `-0.0023%` from the old V4
  active total. There were 512 weight tokens and sequence length 544.
- A full-model BF16 smoke passed two optimizer updates at 13.38–13.91 GiB peak;
  every trainable tensor and all 13 encoder/6 decoder blocks had finite nonzero
  gradients. Focused tests, compile/Ruff, and an independent pre-launch review all
  passed; the reviewer issued FORMAL GO.
- The real exact64 run was launched at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_3k_v1_20260827T2100Z`.
  It used 48 train/16 diagnostic operators, effective batch six as three B2
  microbatches, and raw-W structural loss `L_direction + 0.1*L_log_scale`.
- Persisted full-train evaluation loss was: step 0 `1.006417`, 250 `0.946449`,
  500 `0.930171`, 750 `0.533313`, and 1000 `0.070363`. The known-good small p16
  Direct Normalized reference was about `0.94835` at step 250 and `0.41146` at
  step 1000. Thus the scaled p32 model retained early stability and entered rapid
  memorization substantially earlier than the small reference.
- User judged the result decisive and explicitly requested termination before 3000.
  The process was interrupted during a later backward pass; GPU memory returned to
  zero. Because the run had a final-only step-3000 checkpoint policy, no checkpoint
  or COMPLETE marker exists. Valid metrics and plots through step 1000 remain in the
  run root, including `metrics.jsonl`, `train_loss_curve.png`, and
  `training_metrics.png`.
- Narrow supported conclusion: the simple single-path Direct Normalized architecture
  scales to roughly 700M and learns the 64-operator bounded problem stably with p32.
  This run does not isolate scale-up from the patch-size change and does not establish
  generalization or readiness of a future 50k production run; it establishes the
  requested optimization-stability gate.

## 2026-08-27 — Direct Normalized p32 promoted to the 500k production stream

### User decision and scope

- After the bounded 700M p32 model reached train structural loss `0.07036` by step
  1000, the user approved a fresh production run with the old 500,000-step data,
  objective, optimizer, and gauge-augmentation contract. The bounded run had no
  checkpoint because it was intentionally stopped before its final-only save, so the
  production run starts from the same seed-42 architecture initialization rather than
  continuing step-1000 weights.
- The user explicitly asked for this concrete simple unified model, not V11 and not a
  return to the original heavy tokenizer or latent-feedback architecture.

### Exact production adaptation

- The encoder/decoder/bottleneck remain the successful p32 graph: 512 normalized
  continuous weight tokens, sequence length 544 including 32 latent slots,
  `d_model=1536`, 24 heads, FFN 5728, 13 encoder blocks, 6 decoder blocks, and the
  serialized `32 x 384` bottleneck. Production Distribution Encoder conditioning
  remains on encoder keys and decoder queries.
- The full bank contains matrices up to `2304 x 256`, versus the bounded arm's
  `1152 x 128`. The original nine row-position embeddings remain unchanged for rows
  0..8; nine production-only row positions and two zero-started column positions were
  added. Partial 27/32/64/128-coordinate tiles receive explicit attention/output
  masks. On the old full `128 x 128`, row-0..8/column-0 path these changes are a no-op,
  and a same-seed regression proves every common state tensor is byte-identical.
- The additions are only 16,896 parameters. Final trainable/total count is
  `706,301,312`, within 511 parameters of the original V4 active budget
  `706,300,801`.

### Frozen production training contract

- Full production operator bank: 14,000 operator groups / 135,100 canonical tiles per
  cycle; five coordinated gauge views; canonical probability 1/6; logical batch 32.
- AdamW uses constant LR `5e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay
  `.01`, no warmup, and global gradient clip `5.0`.
- The loss literally reuses the old Big-VAE functions:
  `behavioral = 50*operator + 1*direction + 10*log-scale`, plus
  `structural = 1*direction + 10*log-scale`. Behavioral and structural coefficients
  are both one; KL/sampling are absent.

### Validation and live launch

- Production smoke root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_p32_production_smoke_20260827T_prod1`.
  It completed 2/2 full-B32 steps, peak CUDA memory `13.397 GiB`, and all parameters /
  all 13 encoder / all 6 decoder blocks had finite nonzero gradients. Step-1 loss was
  total `5.37109`, behavioral `3.83569`, structural `1.53540`; pre-clip global gradient
  norm was `800.17`, so the old clip-5 path is strongly active but finite.
- Focused suite was 5/5 green across the bounded and production contracts; Ruff,
  pycompile, meta parameter construction, and dry-run passed. An independent reviewer
  checked bank orientation, masks, tile maxima, literal loss/optimizer, seeded start,
  resume cursor/RNG, storage, and smoke, then issued FORMAL GO with no P0/P1.
- The real run is live at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_production_500k_v1`;
  launch log:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_production_500k_v1.launch.log`.
  At handoff it had reached step 50/500,000, remained finite, and used `13.40 GiB` peak.
  Startup step 1 reproduced the smoke loss exactly; loss varies with the heterogeneous
  production stream as expected.
- Exact rolling model+Adam+RNG+logical-cursor state is written every 1,000 steps to
  `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_production_500k_v1/resume_latest.pt`.
  A persistent rolling model-only checkpoint is written every 10,000 steps to
  `/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/direct_normalized_scaled_700m_p32_production_500k_v1/model_latest.pt`.
  This split is necessary because persistent filesystems currently have only about
  15 GiB and 5.7 GiB free; old artifacts were not deleted without user authorization.
- Measured eager throughput after bank startup implies roughly 7–10 days for 500k.
  `torch.compile` was deliberately not introduced at promotion time because it would
  be an additional untested intervention; the launched graph is the smoke-validated
  eager path.

### Comet live tracking correction

- The standalone production launcher initially omitted Comet logging. After the user
  requested the link, a read-only sidecar was attached without restarting or changing
  training. It backfilled the existing metric history and now streams new metrics every
  five seconds.
- Live experiment: `https://www.comet.com/mike-5531/big-weight-vae/fcfe9305a9d54dfc995bfcfa060083c1`.
  Local binding metadata is stored in
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_production_500k_v1/comet_experiment.json`;
  sidecar log:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_production_500k_v1.comet.log`.

## 2026-08-27 — Production restart without behavioral/operator loss

- User observed that the activation/operator behavioral objective was unstable and
  explicitly requested that it be disabled and training restarted. The original run
  was stopped around step 5,440; its metrics and step-5,000 rolling resume state were
  preserved under the original v1 paths and were not reused for the changed objective.
- The replacement is a fresh-start, single-intervention run. Architecture, normalized
  p32 input, 706,301,312 parameters, full operator stream, seed 42, batch 32, constant
  LR 5e-5, AdamW, clip 5, and 500k horizon are unchanged. The objective is now only
  `structural direction + 10 * structural log-scale`. Behavioral/operator loss is not
  merely multiplied by zero: its three functions are skipped and absent from autograd.
- New config:
  `/home/coder/project/projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_structural_production_500k.yaml`.
  New run root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_structural_only_500k_v1`.
- A two-step BF16 smoke completed at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_structural_only_smoke_20260827T2230Z`.
  Step 1 had `total=structural=1.535402`, all behavioral fields exactly zero,
  pre-clip gradient norm 171.159, and all 24 tracked parameter groups finite/nonzero.
  An independent targeted reviewer issued FORMAL GO.
- The production restart is live. At the first review point it had reached step 50;
  every behavioral field remained exactly zero and total loss equaled structural loss.
  Comet: `https://www.comet.com/mike-5531/big-weight-vae/d35564e622d446ca82f8a946d1e71b6e`.
  Launch log:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_structural_only_500k_v1.launch.log`.

### Immediate correction: remove only the operator term

- The user clarified that the intended intervention was narrower: remove only
  `50 * behavioral operator`, while retaining behavioral direction and behavioral
  log-scale. The structural-only restart above was therefore stopped and is an invalid
  interpretation of the request; its state is not reused.
- The corrected loss is exactly:
  `behavioral direction + 10 * behavioral log-scale + structural direction + 10 * structural log-scale`.
  `operator_recon_loss` is skipped entirely, while `operator_direction_scale_loss`
  remains active and differentiable.
- Corrected config:
  `/home/coder/project/projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_no_operator_production_500k.yaml`.
  Corrected production root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1`.
- Two-step smoke root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_smoke_20260827T2243Z`.
  Step 1 had operator exactly zero, behavioral direction 1.036727, behavioral scale
  0.136520, behavioral total 2.401923, structural 1.535402, and total 3.937326;
  all 24 gradient groups were finite/nonzero. Independent review issued FORMAL GO.
- The corrected 500k run is live. At step 60, operator remained exactly zero while
  behavioral direction/scale were nonzero; total loss was 3.008276. Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/4d5f4424b59b424c9bf88c963f2ae8cc`.

## 2026-08-28 — No-operator production run plateaus

- User flagged that the live no-operator production run was clearly not learning
  normally. Read-only review at about step 60k confirmed an early plateau rather than
  slow continuing progress. From steps 40k onward, mean total loss was 2.1468,
  behavioral direction 0.9039, and structural direction 0.9524; slopes after 20k were
  effectively zero. Scale losses fell early, but direction remained close to the
  uncorrelated-direction value of one.
- The earlier bounded 700M exact64 result was reclassified correctly: by step 1000 it
  reached train structural direction 0.068 on 48 repeatedly seen operators, while the
  16 held-out operators stayed at 0.980. It demonstrated memorization capacity, not a
  general learned weight codec.
- A zero-update checkpoint-50k intervention was run on the sealed exact64 panel using
  `/home/coder/project/projects/weight-vae/workspace/training/weightclip_benchmark/analyze_direct_normalized_production_latent_use.py`.
  Matched, shuffled-latent, and shuffled-weight-input structural direction losses were
  0.9914, 0.9916, and 0.9916 respectively; zero latent was 0.9958. Zero latent actually
  improved action NRMSE from 1.72 to 1.35. The latent affects output magnitude, but the
  learned variation is not W-specific/useful under these interventions.
- Gradient telemetry shows every logged update exceeded global clip 5. The output head
  accounts for more than 99.98% of squared raw gradient energy at late checkpoints,
  so global clipping attenuates the already-small encoder/decoder credit path. This is
  a supported proximal optimization mechanism, not yet a unique root cause.
- Current supported conclusion: training finds an easy scale/common-template solution
  and fails to learn operator-specific weight direction. Viable deeper causes remain
  (a) broad-stream gradient incoherence versus the bounded memorization regime,
  (b) decoder/X shortcut and credit attenuation, and (c) early 200x stronger relative
  scale weighting versus the bounded run (`20` total scale coefficient versus `0.1`).
  These have not yet been causally separated by interventions.
- Evidence:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_training_plateau_20260828.png`,
  `analysis_training_plateau_20260828.json`, and
  `analysis_checkpoint_050000_latent_use.json`. The live process was not stopped or
  modified during this read-only diagnosis.

### Mechanistic localization at checkpoint 60k: learned common-mode bottleneck

- User explicitly asked to keep the live run running and discuss mechanistically
  interpretable causes rather than stopping it. All diagnostics below were zero-update,
  read-only checkpoint evaluations; training was not signaled or changed.
- The nominal bottleneck is not obviously too small: one 128x128 tile has 16,384 values,
  while the deterministic latent has 32x384 = 12,288 coordinates. The observed failure
  is instead an *effective* bottleneck. Across 64 operators, raw latent pairwise cosine
  is 0.95484. After removing the common mean, participation rank is only 2.82 and the
  largest centered direction contains 55.3% of the remaining energy. Evidence:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_checkpoint_060000_latent_use_v2.json`.
- The decoder is not simply ignoring latent tokens. At step 60k, first-layer decoder
  output queries put 54.3% of their attention mass on latent tokens versus 5.88% under
  uniform attention. Nevertheless, shuffling the weight input makes the final latent
  cosine 0.9945, and the decoder query states stay above 0.99989 cosine at every depth.
  The decoder learned to read the bottleneck, but the encoder supplies an almost-common
  code with little useful operator-specific direction.
- The collapse is learned, not present at initialization. At step 0, encoder attention
  entropy was about 0.971 of maximum at all 13 layers and a weight shuffle changed the
  final latent by 0.957 relative RMS (cosine 0.542). At step 60k, layers 1--12 have
  entropy fractions mostly 0.04--0.09; layer-1 entropy 0.0578 corresponds to only about
  1.4 effective attended tokens out of 544. The same shuffle changes the final latent
  by only 0.105 relative RMS (cosine 0.9945). Thus training converted a broad,
  W-sensitive reader into a nearly one-hot, common reader.
- The most precise failure site is the first encoder block. At initialization its
  latent input/attention write/MLP write RMS values are 0.020 / 0.017 / 0.104. At step
  60k they are 0.022 / 23.16 / 163.78, producing latent-state RMS 172.27 after only one
  block and 260.25 after all 13. The layer-1 attention and MLP writes are themselves
  nearly invariant to shuffled weights (cosines 0.980 and 0.986). Learned parameters
  such as latent slots remain ordinary scale (latent-slot RMS 0.022); the explosion is
  an input-dependent residual write, especially the first SwiGLU MLP, not a giant stored
  latent embedding.
- Architectural interpretation: each pre-norm block reads `RMSNorm(state)`, but writes
  an unconstrained attention residual and an unconstrained SwiGLU residual. The
  depth-scale factor is used only at initialization, not as a persistent multiplier.
  The final latent is RMS-normalized again before projection. This creates a weakly
  identified residual-stream scale: a huge common vector can grow with little direct
  penalty, while the final normalization suppresses small W-specific deviations in its
  direction. Saturated attention first selects a tiny, largely shared subset of token
  values; the first MLP then amplifies that shared summary by roughly 7x and makes it
  dominate every later layer.
- Input tokens mix normalized weight content with scale, group, chunk, and tile
  embeddings in the same value stream; activation context is added to attention keys.
  This gives the saturated router an easy path to stable position/scale/X-dependent
  summaries. The loss evidence is consistent: scale terms became small while direction
  stayed near the no-correlation value. However, attribution specifically to the scale
  term or activation context remains a leading hypothesis until a scale/X ablation is
  run.
- A second supported amplifier is global clipping: every logged update is clipped and
  the output head carries above 99.98% of squared raw gradient energy. The large head
  gradient therefore sets the global clip factor for the much smaller encoder credit
  path. This likely stabilizes the common shortcut, but does not by itself explain why
  the first encoder MLP chooses the common mode.
- Mechanisms now weakened or excluded as primary explanations: hard latent coordinate
  count; complete decoder latent bypass; BF16 underflow; dead gradients; simple gradual
  signal decay through depth. The leading causal chain is instead: saturated first-layer
  routing -> huge common first-MLP write -> pre-norm residual-scale gauge and final
  normalization suppress W-specific deviations -> decoder faithfully reads a
  task-poor/common latent -> head/clip imbalance makes escape difficult.
- Detailed evidence:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_attention_path_initial_v2.json`,
  `analysis_attention_path_060000_v3.json`, and their `.log` files. The diagnostic
  implementation is
  `/home/coder/project/projects/weight-vae/workspace/training/weightclip_benchmark/analyze_direct_normalized_attention_path.py`.
- Cheapest discriminating follow-ups, not yet authorized or run: (1) same checkpoint,
  shuffle/zero activation context separately in encoder keys and decoder queries;
  (2) short same-state forks with a persistent residual-write scale/cap versus current
  blocks; (3) head-only or groupwise clipping versus global clipping; (4) remove/freeze
  scale loss in a matched short fork. These distinguish X/scale shortcut, residual
  common-writer instability, and credit attenuation without redesigning the model.

### Literature match and proposed unified repair

- The exact combined weight-AE failure is task-specific, but its components are known
  Transformer pathologies. Zhai et al., *Stabilizing Transformer Training by Preventing
  Attention Entropy Collapse* (`https://arxiv.org/abs/2303.06296`) identify pathologically
  concentrated softmax attention and connect its entropy to the spectral norm of the
  attention logits. Their sigma-Reparam bounds linear-map spectral gain. Henry et al.,
  *Query-Key Normalization for Transformers*
  (`https://aclanthology.org/2020.findings-emnlp.379/`) directly L2-normalize each query
  and key head and replace uncontrolled dot-product scale with a learned temperature.
- Pre-norm branch/gradient imbalance is also established. NormFormer
  (`https://arxiv.org/abs/2110.09456`) reports larger early-layer than late-layer
  gradients in Pre-LN and adds attention-output/head scaling plus normalization inside
  the FFN. LayerScale in CaiT
  (`https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html`)
  puts a small learnable per-channel multiplier on every attention and FFN residual
  write. DeepNorm (`https://arxiv.org/abs/2203.00555`) is a more extensive residual
  scaling/initialization solution for extreme depth. These papers do not prove our
  root cause, but their targeted failure modes match the measured entropy and branch
  explosions rather than merely sharing generic Transformer terminology.
- A final zero-update decomposition sharpened the first-MLP diagnosis. At step 60k,
  layer-1 SwiGLU gated-hidden RMS is 19.67 versus 0.088 in layer 2; its output residual
  is 163.78. Thus the first MLP has learned a high-gain response to the common,
  saturated-attention direction. Merely normalizing Q/K cannot fix this FFN amplifier.
  Evidence:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_attention_path_060000_v4.json`.
- Recommended minimal architecture is one repeated stabilized pre-norm block, with no
  extra carrier or parallel path:
  1. L2-normalize Q and K per head; use a fixed or bounded learned temperature, so
     attention logits cannot grow without bound.
  2. RMS-normalize the SwiGLU gated hidden activation before the FFN output projection.
     This directly removes the measured 19.67-to-163.78 common writer while retaining
     its direction.
  3. Multiply both attention and FFN residual writes by a small nonzero per-channel
     LayerScale gate (initial proposal 0.1 for depth 13), preferably smoothly bounded
     above so training cannot recreate a 100x branch. Nonzero initialization preserves
     step-1 gradients to branch weights.
  In plain form each sublayer is still just `state <- state + gate * branch(norm(state))`.
  The same rule is used at every depth; this is a unified Transformer, not a collection
  of special paths.
- Proposed causal experiment: compare the already-running baseline against exactly one
  same-seed/same-stream 700M repaired-block run for 10k updates. Do not change tokenizer,
  Distribution Encoder, bottleneck, decoder, loss, optimizer, or data. Persist at
  steps 0/100/500/1k/2k/5k/10k: direction/scale losses, layerwise attention entropy,
  effective attended-token count, attention/MLP write RMS, latent pairwise cosine/rank,
  W-shuffle latent response, and group gradients before/after clipping. Precommitted
  success is not merely finite loss: layer-1 entropy must not collapse toward 0.058,
  no residual write may exceed its input by orders of magnitude, latent W-shuffle
  sensitivity must not decay toward 0.105, and direction loss must improve beyond the
  current plateau. This single comparison tests whether the measured architectural
  failure is causal without adding multiple unrelated arms.

### Immediate cause of encoder attention saturation

- A zero-update logit decomposition at initialization and checkpoint 60k separated
  projected state keys from the Distribution Encoder context-key addition. In encoder
  layer 1, mean query/key L2 norms grew from 6.22/6.25 to 19.74/21.62. Because logits
  multiply Q and K, the implied norm product after the standard divide-by-sqrt(64)
  grew about 11x. Learned Q/K alignment sharpened as well: logit standard deviation
  grew from 0.639 to 22.15 and mean max-minus-min span from 3.55 to 112.37. Mean maximum
  softmax probability therefore rose from 0.0094 to 0.8645, while normalized entropy
  fell from 0.9708 to 0.0578.
- Distribution Encoder key conditioning is not the immediate source of first-layer
  saturation. Removing the context-key term at the same checkpoint leaves entropy
  0.0590, essentially identical to the full 0.0578. The context term alone has entropy
  0.8856. State-key norm is 21.62 versus context-key norm 7.04. Thus the ordinary
  learned QKV projections dominate the pathological logits; X conditioning may still
  affect which token is selected or later layers, but it did not create the scale blowup.
- Recomputing the same checkpoint attention after per-head Q/K L2 normalization and
  a sqrt(head-dim) temperature raises entropy from 0.0578 to 0.7314 and lowers mean
  maximum probability from 0.8645 to 0.0822. This is a direct no-training rescue of the
  attention distribution and strongly supports QK norm/temperature as the correct
  local intervention.
- The argmax token is not a single universal positional sink: its average mode fraction
  across the 18 evaluated tiles is only 0.151 and is not higher than initialization.
  Different examples can choose different tokens. The stronger established failure is
  that high-norm/high-alignment QK makes each query almost discrete, and the downstream
  first MLP maps those selected reads into nearly the same high-gain direction.
- Deeper optimization cause remains partly open. Standard PreNorm controls the input
  state norm but leaves the Q/K projections free to increase their output norms and
  align with a few normalized-state directions; divide-by-sqrt(head-dim) is only a
  fixed initialization-scale heuristic, not a trained logit bound. The current loss
  appears to reward an easy scale/common-template solution, while global clipping and
  weak encoder credit make escape difficult. This explains a plausible positive
  feedback loop, but only the immediate QK-norm/alignment mechanism is directly proven.
- Evidence:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_attention_path_initial_v5.json`
  and `analysis_attention_path_060000_v5.json`.
## 2026-08-28 — Scale-loss gradient versus encoder Q/K

**User question.** If the attention block is repaired but the production loss is left unchanged, can the scale-loss gradient again destroy Q/K and re-create attention saturation?

**Measurement.** Added a read-only component-gradient probe:
`projects/weight-vae/workspace/training/weightclip_benchmark/analyze_direct_normalized_loss_component_gradients.py`.
It computes separate gradients from the actual weighted direction term and the actual weighted scale term (`10 * behavioral_scale + 10 * structural_scale`) on the same forward graph, for every encoder Q/K/V, context-key, attention output, MLP, bottleneck, decoder layer 1, and output head. No optimizer step is taken.

Artifacts:

- Initialization: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_initial_loss_component_gradients.json`
- Step 60000: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/analysis_checkpoint_060000_loss_component_gradients_v2.json`

**Evidence.** At initialization, the weighted scale gradient on encoder Q is 5.25–5.46 times the direction-gradient norm (median 5.32); on K it is 5.25–5.53 times (median 5.33). Direction and scale gradients strongly oppose one another: median cosine about -0.70. At step 60000, scale is still larger than direction on median Q/K (about 1.33–1.36 times), with median cosine about -0.54 for Q and -0.45 for K. For layer 1 at step 60000, Q ratio is 1.40 with cosine -0.57; K ratio is 1.36 with cosine -0.55.

The scale-gradient cosine with the Q/K parameter vectors is approximately zero both at initialization and step 60000. Therefore it is not simply pushing the global Q/K Frobenius norms radially upward. It mainly rotates/reshapes Q/K, which can still create highly aligned or singular routing modes.

**Narrow conclusion.** Keeping scale coefficients at 10 is a serious recurrence risk. Fixed/capped QK normalization blocks the exact norm-amplification route to huge logits, but it cannot prevent the scale objective from consuming Q/K directional capacity and driving common scale-oriented routing. A learned unbounded attention temperature would reopen the exact saturation route.

**Recommended simple production intervention.** Keep direction coefficients at 1 and reduce both behavioral and structural scale coefficients from 10 to 1. Because gradient contribution is linear in this coefficient, the initialization Q/K scale-to-direction ratio is predicted to fall from about 5.3 to about 0.53. This retains scale supervision without letting it dominate the shared encoder. Coefficient 2 would be roughly parity at initialization and is judged unnecessarily risky. For a pure causal architecture-only arm, the old loss may be retained, but that run should be labelled a discriminator rather than the preferred production candidate.

### Follow-up: avoid hand-tuned task coefficients

**User preference.** Seek a mechanism that does not require manually selecting direction-versus-scale loss coefficients.

**Important constraint.** If direction and scale remain two scalar objectives, some relative metric is mathematically unavoidable: an ordinary sum merely hides the choice as equal coefficients. What can be removed is the hand-tuned constant.

**Proposed mechanism: bottleneck gradient-balanced backward.** Treat direction and scale as two raw objectives with no `x10`. On the same forward pass, measure their gradient RMS at the single encoder/decoder bottleneck. Detach those RMS measurements, normalize the two bottleneck gradients to equal voting strength, and combine them along their angular bisector. Use a symmetric reference magnitude such as the harmonic mean of the two original RMS values so the procedure does not arbitrarily amplify the absolute optimizer step. This is one training-time rule; it adds no inference branch, carrier, second decoder, or learned loss-weight parameters.

At the observed initialization cosine near -0.70, the normalized bisector remains a first-order descent direction for both objectives; the 5.3-times larger scale gradient cannot dominate merely through units or coefficient choice. If the gradients become exactly opposite, there is no shared descent direction and the run should report that conflict rather than silently choosing one task.

**Scope and limitation.** Bottleneck balancing prevents scale from dominating the signal entering the encoder but does not itself impose a bound on attention entropy. Pair it with QK normalization and a fixed/capped temperature as a hard block invariant. QK normalization addresses logit-norm saturation; gradient balancing addresses direction-versus-scale credit competition. A learned unbounded temperature is rejected because it gives the scale objective another route to saturation.

**Cost/trade-off.** The method needs separate component-gradient measurements through the decoder before the final backward, so it increases training cost. It should first be tested as a bounded causal arm against the fixed-coefficient baseline, logging bottleneck and Q/K gradient norms, cosines, attention entropy, and temperature.

**Simplest experiment ordering.** Do not stack gradient balancing into the first repair. First keep the existing objective exactly unchanged and impose an architectural stability invariant: per-head L2-normalized Q/K, non-learned bounded cosine logits, and bounded residual writes. This directly tests whether the old loss can still create collapse when it is mathematically unable to inflate attention logits or block-update amplitude. Only if attention remains healthy but representations are still scale-dominated should bottleneck gradient balancing be added as the next discriminator. This preserves the user's simple-to-complex preference and separates saturation from semantic task conflict.

### User decision and implementation: stack the complete repair

**User decision.** Implement the architectural guards and gradient-balanced backward together, and stop the old production run to free resources.

**Old-run stop.** The old production process stopped gracefully at optimizer step 66889 after SIGTERM. It atomically wrote `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_no_operator_500k_v1/resume_latest.pt` (8.476 GB), `/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/model_latest.pt` (2.825 GB), and `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_no_operator_500k_v1/STOPPED.json`. The Comet streamer was then stopped; GPU memory was released.

**Repaired block.** The same 706,301,312-parameter model and byte-identical seed-42 parameter start are retained. Each attention head now L2-normalizes Q/K and uses fixed cosine-logit scale 2.0, so logits cannot grow through Q/K norms. SwiGLU hidden and attention/MLP residual writes use smooth RMS caps with unit small-signal Jacobian; the write cap is derived from depth as `1 / sqrt(2 * max_depth)`. No learned temperature, gates, extra inference branch, or new parameters are introduced.

**Gradient-balancing refinement.** An initial implementation balanced direction/scale at `W_hat`. Full-700M smoke showed this was the wrong boundary: output scale-gradient RMS was about 19 times smaller, producing dynamic scale weight 13.36 and global grad norm 1359. Old parameter probes show the decoder Jacobian amplifies scale relative to direction by roughly 10x before Q/K, so output balancing could worsen the target failure. That variant is rejected and preserved only as diagnostic smoke evidence at `/mnt/shared/weightclip_benchmark/direct_normalized_p32_production_smoke_20260828T085406Z`.

The accepted implementation balances at the serialized bottleneck `z=[B,32,384]`. Raw objectives are `(behavioral direction + structural direction)` and `(behavioral scale + structural scale)` with no x10. Their bottleneck gradients are normalized to equal votes, combined along the bisector, and rescaled to the RMS of their raw equal-weight sum. The resulting detached dynamic weights form one scalar backward through decoder and encoder. There is no hand-selected task coefficient.

**Verification.** Focused CPU contracts: 6/6 pass, Ruff/pycompile green. Full 700M BF16 two-step smoke COMPLETE at `/mnt/shared/weightclip_benchmark/direct_normalized_p32_production_smoke_20260828T085852Z`: bottleneck direction/scale RMS `1.358e-6 / 1.087e-6`, cosine `-0.467`, dynamic weights `0.915 / 1.144`, balanced proxy RMS equals raw-equal-sum RMS, global pre-clip norm 55.85, all 13 encoder and 6 decoder blocks nonzero, no missing gradients, peak 14.66 GiB. A separate initial exact64 parameter probe at `/mnt/shared/weightclip_benchmark/direct_normalized_p32_production_smoke_20260828T085852Z/initial_bounded_balanced_qk_component_gradients_v2.json` shows balanced scale/direction contribution ratios: Q min/median/max `0.840/0.905/0.950`, K `0.854/0.905/0.950`, replacing the old approximately 5.3x scale dominance. This validates the intended immediate mechanism at initialization; longer-run stability/usefulness remains unestablished.

### 2026-08-28 — Repaired 500k production launch

After an independent production review returned FORMAL GO with no P0/P1 blockers, the exact repaired 706,301,312-parameter candidate was launched for 500,000 optimizer steps from a fresh seed-42 start using `/home/coder/project/projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_bounded_balanced_production_500k.yaml`. The run root is `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2`; rolling exact resume state is under `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2`, and rolling persistent model weights are under `/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2`.

The real production path passed startup and early liveness. Step 1 reproduced the accepted smoke exactly: loss 2.171896, pre-clip gradient norm 55.85, bottleneck direction/scale gradient RMS `1.358e-6 / 1.087e-6`, cosine -0.467, dynamic weights `0.915 / 1.144`, and balanced proxy/reference RMS agreement within numerical precision. All 13 encoder and 6 decoder blocks had finite nonzero gradients. By step 10 loss was 2.054898 and the dynamic weights were `1.091 / 0.945`; by step 40 loss was 1.968664 with finite gradients. These few steps establish implementation/liveness, not long-horizon scientific success. Live evidence is in `train_metrics.jsonl`, `gradient_telemetry.jsonl`, and the launch log under the run root.

Comet streaming is live at https://www.comet.com/mike-5531/big-weight-vae/356de4cb988646cfb18cd559a1ac82a7. Monitoring was intentionally released after early validation; the user can request a later causal review at a meaningful checkpoint.

### 2026-08-28 — Current-state diagnosis at rolling checkpoint step 1000

**User request.** Diagnose the live bounded/balanced production state without stopping or mutating training. The live process remained healthy and continued beyond step 1500 during the read-only probes. The exact rolling resume at step 1000 was used for zero-update analysis.

**Failure definition and validity.** This is a scientific learning failure, not a runtime failure. All metrics are finite, no parameter group is missing gradients, GPU memory is stable near 14.67 GiB, and resume step 1000 was saved successfully. However, mean total loss is 1.96496 over logged steps 110–500 and 1.96285 over 510–1000: effectively a plateau. Behavioral and structural direction losses remain about 0.92 and 0.94. Every logged pre-clip global norm is above the configured clip 5. The inspected plot `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/train_loss_curve.png` is readable and shows the same plateau rather than a hidden downward trend.

**Attention saturation is excluded for this checkpoint.** The exact bounded block replay in `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/analysis_attention_path_step_001000.json` gives encoder attention entropy 0.941–0.998 and decoder entropy 0.884–0.989. The old failure had encoder-layer-1 entropy about 0.058. Fixed cosine logits therefore successfully removed the original Q/K-norm saturation mechanism.

**The smooth residual cap itself has saturated.** At initialization, encoder raw/capped attention-write ratios are 1.00–1.01 and raw/capped MLP-write ratios 1.118–1.120. At step 1000 they are 2.29–8.96 and 6.56–11.31 respectively; almost every encoder write is pinned near the cap 0.196. Decoder MLP writes are also compressed by 2.80–3.38. This turns the intended safety bound into a strong radial-gradient attenuator. Correspondingly, median encoder block gradient RMS fell from 3.00e-5 at step 1 to 2.42e-7 at step 1000. This is the strongest directly supported architectural bottleneck.

**Representation collapse is real and was induced by training.** Exact64 latent intervention artifacts are `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/analysis_latent_use_initial.json` and `analysis_latent_use_step_001000.json`. Centered latent participation rank fell from 33.55 at initialization to 3.02 at step 1000; top-1 centered energy rose from 0.110 to 0.541. Weight-shuffle latent relative delta fell from 0.988 to 0.678. The decoder does use the latent, so a completely ignored bottleneck is excluded: matched action NRMSE is 4.43 versus 4.68 with shuffled latent. But this learned latent is harmful overall: zero-latent action NRMSE is better at 3.73, and zero-latent raw NRMSE is 1.191 versus matched 1.224. Thus the path amplifies a low-rank, weakly correct representation rather than reconstructing useful operator variation.

**Credit routing and task conflict.** At step 1000, 99.94% of raw squared gradient energy in the logged group ledger is in the output head, versus 97.65% at step 1; all logged batches are globally clipped. This is a strong proximal head-shortcut/encoder-starvation signature, but raw gradient energy is not the same as the AdamW update, so it is not by itself a final cause. On a fixed exact64 two-operator batch, `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/analysis_loss_component_gradients_step_001000.json` shows direction/scale bottleneck cosine -0.870 and encoder Q/K cosines around -0.999 after their magnitudes are balanced. The balancer removes scale dominance but can leave almost exact parameter-gradient cancellation. Production batches usually report bottleneck cosine near zero, so this conflict is conditional rather than established as the sole global cause.

**Narrow conclusion.** The old attention-logit explosion was fixed. The live replacement still fails because training drives the unified residual branches deep into their smooth caps, reducing amplitude credit; simultaneously the latent collapses from rank about 34 to about 3 and the decoder/head learns a harmful low-rank output. The strongest supported proximal loop is `cap saturation -> weak encoder credit -> common low-rank latent -> head-dominated harmful decode`, while the direction of causality between cap saturation and the downstream head shortcut remains unresolved. Bottleneck balancing alone did not fix this and can conditionally create near-cancelling Q/K gradients. The run was left running because the user requested diagnosis only.

### 2026-08-28 — Stop at step 1898 and revised direction/scale fix

**User decision.** Stop the bounded/balanced production run and redesign the objective path. The user's leading hypothesis is that direction and scale are genuinely different tasks: their gradients can point in different directions and their learning difficulty/progress is not comparable, so equalizing only instantaneous bottleneck gradient RMS is insufficient.

**Graceful stop.** The process stopped at optimizer step 1898 and committed cursor 60736. Evidence: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/STOPPED.json`. Exact resume is `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/resume_latest.pt`; persistent model weights are `/home/coder/project/projects/shared/storage/artifacts/weightclip_benchmark/direct_normalized_scaled_700m_p32_bounded_balanced_500k_v2/model_latest.pt`. The Comet sidecar was stopped and GPU memory was released.

**What the evidence actually says.** Small scalar loss value does not mean an easy or weak task. At step 1000 the scale loss is numerically smaller than direction loss, yet its raw bottleneck gradient is about 4.4 times larger and raw encoder Q/K gradients are about 5 times larger on the fixed exact64 probe. After equal-magnitude balancing, direction and scale Q/K contributions become almost opposite on that batch (cosine near -0.999), so the balancer can convert dominance into cancellation. Production-batch bottleneck cosine is often near zero, therefore pervasive exact opposition is not established; the confirmed defect is that instantaneous gradient magnitude is not a measure of relative task progress or compatible geometry.

**Literature framing.** GradNorm balances tasks by relative training rates, not just raw gradient magnitudes. PCGrad/CAGrad/Nash-MTL treat negative gradient geometry explicitly. FAMO balances loss improvement over time. Impartial MTL seeks equal aggregate-gradient projections. Recon reports that recurring conflicts can survive gradient surgery and motivates making only conflict-heavy layers task-specific. Kendall uncertainty weighting addresses different units/noise, but not near-opposite directions. DeepNorm supplies linear residual scaling and derived initialization; BranchNorm explicitly warns that constraints useful early can undertrain later, matching the observed smooth-cap saturation.

**Recommended coherent fix.** Keep one shared Transformer and one bottleneck, but remove nonlinear residual RMS caps and use a non-saturating DeepNorm-style residual scaling/initialization. Keep normalized Q/K and fixed attention temperature because that intervention demonstrably prevented the old attention collapse. Factor only the final readout into the natural coordinates of a weight block: a direction readout emits a unit-RMS signed shape; a scale readout emits log-radius; their deterministic product is the reconstructed weight. Both readouts consume the same decoder state, so this is one model rather than two pathways. Replace reciprocal instantaneous bottleneck balancing with GradNorm-style relative-progress weighting based on dimensionless ratios `current direction loss / initial direction loss` and `current scale loss / initial scale loss` (fixed canonical alpha 1 for the first test). This addresses different learning speeds. Do not stack PCGrad initially: with near-antiparallel gradients it can leave almost no shared update, and the factorized readout should first remove the avoidable radial-versus-tangential conflict at the output.

**Proposed bounded discriminator.** Compare one repaired candidate against the archived stopped run on the same 64 operators, seed, stream, optimizer, bottleneck and parameter budget for 3000 steps. Log relative progress for both tasks, per-layer task-gradient cosines, residual-write RMS, attention entropy, latent participation rank, and zero/shuffle-latent interventions. Required qualitative success is simultaneous progress of both losses without persistent Q/K near-opposition, no residual saturation, preservation of useful latent rank, and matched latent decoding outperforming zero latent. This is not yet run or established.

**Preferred simplification after literature synthesis.** The factorized-head plus GradNorm route is valid but is not the simplest first repair. Direction and scale are not truly two independent tasks; they are two coordinate views of the same predicted operator action. The preferred first experiment is therefore to remove the multi-objective scalarization entirely. Train with one dimensionless activation-space quadratic loss per operator:

`error = X @ (W_hat - W)`

`loss = mean_valid(error^2) / (mean_valid((X @ W)^2) + eps)`

Equivalently, with empirical activation covariance `A = mean(X^T X)`, this is normalized squared error of `A^(1/2) @ (W_hat - W)`. It is the activation metric used conceptually by GPTQ/OBS-style compression, but no matrix square root is required in code. Direction and log-scale losses remain metrics only and receive no backward. This removes arbitrary task coefficients, relative-rate balancing, and gradient surgery in one step. The earlier rejected operator term was materially different: an unnormalized raw RMSE multiplied by 50; it does not falsify a coefficient-1, per-operator target-energy-normalized quadratic loss.

The architecture repair remains independently required by direct evidence: remove the saturating residual RMS caps and return to non-saturating pre-norm residual updates with depth-scaled output initialization; retain L2-normalized Q/K and fixed cosine temperature because those prevented the old attention-logit collapse. This yields one encoder, one bottleneck, one decoder, one prediction, and one loss. Factorized output heads and GradNorm/CAGrad are demoted to follow-ups only if the single task-space objective still produces a measurable internal conflict or fails to recover useful latents.

### 2026-08-28 — 3000-step single activation-metric experiment

**User decision.** Test the 706M direct-normalized unified model with one
activation-space objective, while continuing to log every legacy behavioral and
structural loss component. The old loss values are diagnostics only and must not
contribute gradients. Remove the previously introduced SwiGLU/residual RMS caps;
retain L2-normalized Q/K with fixed cosine scale 2 because that is a hard attention
stability invariant rather than another objective.

**Exact experiment.** The run used one 13-block encoder, serialized latent
`[32,384]`, one 6-block decoder, p32 continuous normalized weight tokens, and the
production Distribution Encoder conditioning. It has 706,284,416 trainable
parameters. The only backward loss was the per-tile mean
`sum((X @ W_hat - X @ W)^2) / sum((X @ W)^2)`. Logged under `no_grad` were
behavioral operator/direction/scale, structural direction/scale, the old nominal
sum, and prediction-to-target action-amplitude ratios. Training used the exact
48 train operators, all nine tiles per sampled operator, effective B6 through
three B2 microbatches, AdamW (`lr=1e-4`, beta2=.95, eps=1e-8), global clip 1,
320-step warmup, and cosine decay to `1e-5` by step 3000. An independent reviewer
gave FORMAL GO before launch.

Short warmup-32 smokes were rejected: the raw output head caused an immediate
radial overshoot (representative B6 eval loss 1.30 -> 3.99). Warmup 320 produced a
valid 20-step smoke (1.2997 -> 0.9941) and was used for the real run. Rejected and
accepted smoke roots are preserved under
`/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_activation_metric_smoke_v2`
through `..._smoke_v5`.

**Validity and artifacts.** The real run completed all 3000 steps with no nonfinite
rows or missing-gradient groups and wrote a 2,825,236,819-byte final checkpoint.
Root:
`/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_activation_metric_3k_v2_20260828T130359Z`.
Canonical evidence is `metrics.jsonl` (13 full eval rows), `train_metrics.jsonl`
(301 train rows), `gradient_telemetry.jsonl` (13 group-gradient probes),
`all_train_loss_curves.png`, `all_train_loss_curves_live.png`, `summary.json`, and
`model_step_0003000.pt`. Both all-loss plots were visually inspected and are
readable. The legacy-only `training_metrics.png` is not the primary objective plot.

**Observed dynamics.** The full-train activation loss followed
`1.4469@0 -> 1.1042@250 -> 1.0580@500 -> 1.2005@750 -> 1.1602@1000 ->
1.1082@1500 -> .9766@1750 -> .9706@2250 -> .9682@3000`.
The zero-output predictor is exactly 1.0, so the final gain is only 3.18%. The
final median/p95/max per-tile losses are `.9685/.9892/.9919`; the result is real
but weak, not an average hidden by a bad tail. The median predicted action
amplitude is only 12.7% of target. Legacy telemetry is also poor at step 3000:
behavioral direction/scale `.855/.230`, structural direction/scale `.963/.332`,
and raw-weight NRMSE `.9991`.

The latent collapses before the objective becomes useful. Full-train centered
participation rank is `74.6@0 -> 14.2@250 -> 2.56@500 -> 1.33@1250 -> 1.77@3000`.
Thus the nominal 12,288-coordinate bottleneck is used as roughly two common modes.
The late loss below 1 is a small correlated correction to zero, not rich weight
reconstruction.

**Mechanistic gradient evidence.** This run removes direction/scale objective
conflict entirely, removes the saturating residual caps, and mathematically bounds
attention logits, yet collapse persists. Therefore none of those prior mechanisms
is a sufficient primary explanation for this architecture.

The dominant proximal mechanism is the final readout/credit path. The decoder ends
in `RMSNorm(1536) -> Linear(1536,32)`. Its small fan-in-scaled head is necessary to
avoid a huge initial output, but that makes the backward Jacobian to decoder features
small while the direct gradient of the head itself is independent of the head
weight magnitude. Output-head squared gradient-energy share is 97.59% at step 1
and greater than 99.999% at essentially every later probe. All 301 logged train
points have pre-clip norm above 1. After the shared global clip, median encoder-block
gradient RMS is `2.34e-10@250`, `3.51e-11@500`, and `1.66e-9@3000`, versus Adam
epsilon `1e-8`; the output-head post-clip RMS remains approximately `4.51e-3`.
Thus global clipping breaks Adam's usual approximate scale invariance upstream:
encoder updates enter epsilon-dominated attenuation while the head keeps a full
Adam signal. This quantitatively explains why the head can learn the easy near-zero
solution before the encoder learns a useful code.

The final checkpoint strengthens the feedback-loop interpretation: output-head RMS
fell from its seeded expected approximately `5.10e-4` to `3.81e-4` (about -25%),
further attenuating the decoder-to-encoder Jacobian. This is not an isolated dead
tensor: every group remains finite/nonzero, and the model eventually learns a weak
3.18% correction. The narrow supported loop is
`initial harmful output -> head retreats toward zero -> upstream Jacobian weakens ->
global clip pushes encoder gradients below Adam epsilon -> latent collapses ->
near-zero head remains locally favored`.

**What remains unestablished.** Raw gradient-energy dominance alone is not an Adam
update measurement, but the measured post-clip-to-epsilon ratios make it mechanistically
relevant here. The run does not yet distinguish how much collapse comes from the
token/conditioning representation after upstream credit is restored. Intrinsic
bottleneck capacity is not supported as the cause because almost all available rank
is unused.

**Cheapest causal discriminator.** Keep this exact graph, start, data, loss, and LR,
but prevent the output head from globally clipping every upstream group: remove the
shared global clip or clip the head separately while leaving encoder/decoder gradients
unscaled. Log actual Adam effective-update RMS by group and repeat only through the
early collapse window (roughly 500 steps). Prediction: if the proposed mechanism is
causal, encoder post-clip gradients stay above epsilon and latent rank does not fall
from 74 to about 2. If rank still collapses with healthy effective updates, the
remaining leading cause moves to tokenizer/conditioning representation rather than
the readout optimizer geometry.

### 2026-08-28 — Exact objective in the official WeightCLIP release

**User question.** What loss do the WeightCLIP authors actually train with?

**Official released configuration and code.** Both released CNN and ResNet alignment
configs use a three-term joint loss:

`total = batch-normalized masked weight-token MSE + 0.25 * dataset/model SigLIP + 1.0 * dataset-classification CE`.

The SANE `GammaContrastReconLoss` is configured with `gamma=0`, so its internal
two-view NT-Xent term is disabled and reconstruction has weight 1. The reconstruction
target and prediction are standardized using one mean and standard deviation computed
over all valid target entries in the batch, then ordinary masked MSE is applied. Since
the same target mean is subtracted from prediction and target, this is equivalently
raw weight-token MSE divided by one batch target variance. With two canonical views,
the two reconstruction losses are averaged.

The separate WeightCLIP contribution aligns every weight-model latent token with a
DeepSets dataset embedding using normalized cosine similarity, temperature 1, a
learnable SigLIP bias initialized to -4, and pairwise sigmoid/softplus loss. Its
configured weight is 0.25 and `freeze_sane=false`, so this term gives the weight encoder
a direct gradient that does not pass through the reconstruction decoder/output head.
The dataset-identity cross-entropy has weight 1 and primarily trains the dataset encoder.

This is materially different from all local Big-VAE objectives tried so far. It has no
direction/log-scale decomposition and no `X @ W` action loss. It also avoids per-tile
inverse target-energy normalization: one batch variance scales every reconstruction
error. Therefore it does not create the same low-energy-tile gradient amplification as
the latest per-tile relative activation MSE, and its direct latent alignment path can
partly bypass reconstruction-head credit starvation during encoder training.

Primary local sources are the official checkout under
`projects/shared/compatibility/external/weightclip`: `config/loss/gamma_contrast_recon.yaml`,
`config/alignment/default.yaml`, `config/dataset_encoder/{cnn,resnet}.yaml`,
`src/sane/loss/reconstruction.py`, `src/sane/loss/alignment.py`, and
`src/sane/trainer/joint_alignment_trainer.py`.
# 2026-08-28 — Уточнение единицы данных в последнем 3k-прогоне

- Пользователь спросил, обучался ли последний запуск на 64 фиксированных тайлах.
- Уточнение: фиксированы были 64 полные матрицы/оператора размера 1152x128, а не 64 тайла.
- Каждая матрица детерминированно разбивалась на 9 тайлов 128x128: всего 576 тайлов.
- Разбиение делалось по операторам: 48 train-операторов = 432 train-тайла; 16 held-out операторов = 144 eval-тайла.
- На каждом update выбирались 6 train-операторов, каждый раскрывался во все 9 тайлов: effective batch = 54 tile examples (3 microbatch x 2 operators x 9 tiles).
- За 3000 шагов это 18,000 предъявлений операторов / 162,000 предъявлений тайлов, в среднем 375 повторов каждого train-оператора и каждого соответствующего тайла.
- Следовательно, это по-прежнему маленькая фиксированная memorization-задача, но корректная формулировка — 48 фиксированных train-матриц и их 432 фиксированных тайла, а не 64 train-тайла.

## 2026-08-28 — Exact activation-moment loss вместо activation minibatch estimate

### User intent

- Аналитически пронести матожидание по активациям внутрь loss и заранее посчитать все activation-dependent матрицы, чтобы убрать variance оценки loss по minibatch активаций.

### Proposed exact objective

- Для каждого полного оператора сохранить нецентрированный второй момент входов `A = mean(x.T @ x)`; центрировать нельзя, потому что среднее входа также влияет на линейный output.
- Для `E = W_hat - W` exact empirical action loss равен `sum(E * (A @ E)) / sum(W * (A @ W))`, затем ratios усредняются по операторам. Его gradient по `W_hat` равен `2 * A @ E / target_energy`.
- Эквивалентная интерпретация: это normalized Frobenius MSE между `A_sqrt @ W_hat` и `A_sqrt @ W`. Модель по-прежнему предсказывает raw `W`; `A_sqrt` существует только в loss, обратное отображение не требуется.
- Для текущего полного `W[1152,128]`, разбитого на девять row-тайлов, нужен полный `A[1152,1152]` либо все 81 cross-tile блока `A_rs[128,128]`. Только девять диагональных блоков неверны: они теряют cross terms между вкладом разных input tiles в один output.
- На update следует получить predictions всех девяти тайлов выбранного оператора, собрать полный `W_hat`, а loss усреднять по выбранным операторам, не по тайлам.

### Important limitation/current-run correction

- Последний 3k harness уже использовал все 512 фиксированных activation rows для каждого выбранного tile, поэтому внутри него не было случайного activation minibatch. Precomputed `A` не уменьшит уже отсутствующую activation-sampling variance там.
- Реальные преимущества для текущего harness: исправить per-tile surrogate до полного operator-action objective и сделать знаменатель/gradient exact для всего activation bank. Stochastic variance от выбора 6 из 48 train-операторов останется.
- Если production training сэмплирует подмножество activation rows, precomputed full-bank `A` действительно уберёт activation minibatch variance и bias случайного ratio numerator/denominator относительно фиксированного empirical bank.

### Numerical/semantic decisions

- Можно хранить unnormalized Gram `S = sum(x.T @ x)`: множитель числа samples сокращается в normalized ratio.
- `A` может быть singular (здесь rank не больше 512 при input width 1152); это нормально и означает, что empirical activation loss не штрафует направления, невидимые activation bank. Ridge/raw-MSE добавлять в первый тест не предлагается.
- Conditioning через Distribution Encoder остаётся отдельным входом модели; замена loss на precomputed moment не требует менять conditioning path.

## 2026-08-28 — Что именно нестабильно в текущем per-tile activation-relative MSE

### Narrow conclusion

- Loss численно корректен и finite; проблема не в NaN и не в activation minibatch noise. В последнем 3k harness каждый выбранный tile использовал все 512 фиксированных activation rows.
- Основные проблемы — conditioning/scaling objective и доступный zero-output shortcut, а также то, что per-tile objective не является full-operator action loss.

### Exact geometry

- Для tile loss можно записать как `L = 1 + r^2 - 2*r*c`, где `r = ||X W_hat|| / ||X W||`, а `c` — cosine между target и predicted actions.
- Пока encoder не выучил полезное направление и `c` около нуля, условно оптимальная amplitude равна примерно нулю. Поэтому shallow output head может быстро улучшить random-init loss, просто подавив prediction, тогда как обучение направления через глубокий encoder намного труднее.
- Zero prediction имеет loss ровно 1 и не является истинной stationary точкой в output space, но является доступным basin/shortcut для параметризованной глубокой модели при слабом directional credit.

### Stored-run evidence

- Run: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_activation_metric_3k_v2_20260828T130359Z`.
- Full-train loss закончил на `0.968198`, лишь на 3.18% лучше zero-output baseline `1.0`; median predicted/target action RMS упал до `0.12739`.
- Latent participation rank упал примерно `74.60 -> 1.77`; это соответствует retreat/collapse, а не хорошему memorization.
- Logged train minibatch loss за первые 500 шагов: mean `1.4403`, std `0.9061`, max `7.2163`; за последние 500: mean `0.9679`, std `0.00947`. Поздняя «стабильность» возникла одновременно с подавлением output к zero baseline.
- Все 301 залогированный update были globally clipped при cap=1: pre-clip gradient norm min/median/max `6.00 / 285.67 / 7603.80`.

### Direct frozen-bank geometry diagnostic

- На 432 train-тайлах target action energy имеет min/median/max `26.63 / 716.61 / 9629.46`, то есть range `361.6x`.
- Proxy curvature scale `2*||X||_F^2 / ||XW||_F^2` имеет range `22.72x` (median `11.26`, max `106.51`). Поэтому одинаково взвешенные relative tile losses дают существенно разные parameter-gradient/curvature regimes при смене operator minibatch.
- Для полного оператора отношение `||sum_r X_r W_r||^2 / sum_r ||X_r W_r||^2` имеет median `2.03` и range `1.05..3.24`: cross-tile terms велики. Независимый per-tile loss оптимизирует другой objective, не лишь дешёвую оценку full-operator loss.

### Interpretation boundary

- Эти факты показывают badly conditioned optimization и zero-output shortcut, но loss сам по себе не объясняет весь collapse: в direct output space gradient при `W_hat=0` ненулевой. Слабый encoder/decoder Jacobian и head-dominated credit path нужны, чтобы shortcut стал фактическим attractor.

## 2026-08-28 — Возврат от activation-MSE к direction/scale: точная механика конфликта

### User direction

- Отказаться пока от activation-relative MSE и вернуться к точке, где production objective состоял из direction/scale компонентов.
- Понять, почему их gradients стали противонаправленными, и обсудить иной механизм исправления вместо activation loss и instantaneous bottleneck gradient balancing.

### What is and is not inherently conflicting

- Direction и scale согласованы в target optimum. Для одного output-вектора cosine direction gradient является примерно tangential к текущему prediction, а log-norm scale gradient — radial; в непосредственном output space это естественные почти ортогональные координаты, а не противоположные задачи.
- Конфликт появляется после pullback через общий нелинейный decoder/encoder: `g_param = J.T @ g_output`. Анизотропный Jacobian не сохраняет output-space orthogonality и может превратить radial/tangential gradients в отрицательно коррелированные parameter gradients.
- Production mix дополнительно дублировал координатные системы: structural losses измеряют direction/radius raw p16 weight patches, behavioral losses — direction/radius action rows `X @ W`. Radial update в одной системе может менять direction в другой.
- Scale loss вычислялся через `Huber(log(r_hat)-log(r))`. В linear Huber regime его output-gradient magnitude масштабируется примерно как `delta/r_hat`; маленькое prediction norm может дать сильный scale gradient при маленьком scalar loss. Умножение обоих production scale terms на 10 усиливало эффект.

### Existing evidence

- Seed-42 initialization старого no-operator production: weighted scale Q/K gradient был примерно `5.32x` direction и cosine был около `-0.70`.
- Step 60000: scale scalar уже мал, но Q/K scale gradient всё ещё `1.33–1.36x` direction, median cosine отрицательный (`Q -0.54`, `K -0.45`). Scalar loss magnitude поэтому не является мерой task difficulty или gradient pressure.
- Instantaneous bottleneck balancing выровнял magnitudes, но на fixed batch step1000 bottleneck cosine был `-0.870`, а encoder Q/K cosines после pullback около `-0.999`; равные по длине противоположные vectors почти взаимно уничтожились.
- Это не universal opposition на каждом batch: production bottleneck cosine часто был около нуля. Поддержан conditional conflict плюс систематическая scale dominance в старой objective, но не утверждение, что every batch/tasks mathematically incompatible.
- Known-good counterexample: exact same scaled Direct Normalized 700M family с единственным `structural_dir + 0.1*structural_scale` дошла на full train к step1000 с total `1.0064 -> 0.07036`, direction `1.00195 -> 0.06798`, raw scale `0.04469 -> 0.02381`. Следовательно, direction/scale decomposition сама по себе не является причиной failure; material changes were duplicated behavioral geometry and x10 scale pressure.

### Preferred alternative to gradient surgery

- Исправить parameterization final readout, а не пытаться складывать конфликтующие gradients постфактум. Для каждого p16 block один decoder state выдаёт два explicit outputs: `q` для signed shape и scalar `s` для standardized log-radius. Deterministically form `u_hat = q / RMS(q)` and `W_hat_patch = exp(s_hat) * u_hat`.
- Direction loss применяется непосредственно к `u_hat`; scale loss — непосредственно к `s_hat`. Scale gradient больше не проходит через norm of `W_hat`, не содержит `1/r_hat` amplifier и не меняет direction-head weights; direction gradient не меняет scale-head weights. Encoder, bottleneck и decoder остаются едиными; разделяется только естественная последняя coordinate chart.
- Первый тест должен backpropagate only structural shape/scale losses, while behavioral operator/direction/scale remain logged diagnostics. This is grounded by the known-good structural-only run and avoids immediately reintroducing the second incompatible action-space polar geometry.
- Use raw equal coefficients only after standardizing log-radius target from train statistics; do not add instantaneous gradient balancer. If shared-trunk conflict remains after readout factorization, relative-progress GradNorm or CAGrad is a follow-up discriminator, not the first fix. PCGrad on near-antiparallel gradients can collapse the shared update and is not preferred initially.

## 2026-08-28 — Critical correction: scaled Direct Normalized did not preserve the original BigVAE polar decoder

### User correction

- User challenged that reverting to structural-only loss is not a production fix and recalled that the previously successful BigVAE already predicted direction and scale explicitly.

### Source-level finding

- The user is correct. Original `BigWeightVAE` constructs separate `direction_head` and `scale_head`. Decoder computes a signed direction vector, normalizes it, predicts log-scale separately, and deterministically reconstructs each p16 patch as `exp(log_scale) * unit_direction`. It also returns `pred_dirs`, and the production worker passes those explicit directions into `patch_structure_loss`.
- The custom scaled `UnifiedWeightBottleneck` used by the 700M Direct Normalized experiments does **not** preserve this output parameterization. It has a single bias-free `output_head: 1536 -> 32`, directly reshapes those 32 raw values into `W_hat`, and calls `patch_structure_loss` without `pred_dirs`. Direction and log-scale are then re-derived from the same raw prediction inside the loss.
- Its `scale_mlp` is only an input-token embedding of the known normalization scale. It is not a decoder scale prediction head and does not make the output polar-factorized.
- Additionally, the custom model emits one p32 token while the structural objective splits it into two p16 loss patches. Thus two nominal direction/scale units share one decoder query state and one raw 32-value head. Original BigVAE uses one explicit polar output per configured patch.

### Consequence

- The proposed “add explicit direction and scale heads” was not a novel repair; it was reintroducing a mechanism that should have been carried over from the original successful VAE. The prior scale-up matched total width/depth/bottleneck/parameter budget, but not this semantically important decoder contract.
- Therefore gradient-conflict evidence from the custom raw-output wrapper cannot be treated as evidence that the original explicit polar formulation intrinsically fails. The missing polar head likely amplified the observed radial/tangential pullback conflict.

### Structural-only counterargument boundary

- Empirically, structural-only did help the custom exact64 memorization task: total structural loss reached `0.07036` by step1000. It is a valid diagnostic that behavioral+x10-scale additions caused material damage.
- However, user is correct that simply reverting the production wrapper to structural-only is not a satisfactory production fix. It would neither restore parity with the original BigVAE decoder nor prove that the production behavioral objective can coexist with the representation.
- The next design decision should first restore the explicit polar decoder contract. With p32 tokenizer retained, choices are: (a) exact p16 output queries as in original BigVAE, or (b) one p32 query emitting two independently normalized p16 directions and two log-scales. Option (a) is semantically exact but doubles decoder query length; option (b) is cheaper but only approximate parity because the two p16 patches still share one query state.

## 2026-08-28 — Возможное разведение backward paths для direction и scale

### User proposal

- Рассмотреть явное разделение gradient paths direction и scale вместо scalar weighting или bottleneck magnitude balancing.

### Fundamental constraint

- Для почти противоположных gradients на одном shared parameter нет ненулевого update, который гарантированно улучшает обе задачи в первом порядке. Необходимо одно из трёх: раздельные параметры, приоритет одной задачи либо почти нулевой common update. Instantaneous equalization скрыла этот выбор и дала cancellation.

### Minimal coherent design

- Сначала восстановить original p16 polar decoder. Это уже даёт disjoint head parameters: structural direction gradient обновляет `direction_head`, structural scale gradient — `scale_head`; scale no longer backpropagates through `log(norm(raw_W_hat))`.
- Encoder/decoder trunk остаётся shared. Для него proposed asymmetric conflict projection:
  - compute `g_dir = grad(L_behavioral_dir + L_structural_dir)`;
  - compute `g_scale = grad(L_behavioral_scale + L_structural_scale)`;
  - if their per-layer dot product is negative, replace scale gradient with `g_scale_safe = g_scale - dot(g_scale,g_dir)/||g_dir||^2 * g_dir`;
  - send `g_shared = g_dir + g_scale_safe` to the one optimizer.
- Direction owns shared representation/routing because it is the high-dimensional identification task. Scale always updates its explicit scale head; it may update shared layers only through a component that does not oppose direction. At exact opposition its shared contribution becomes zero instead of cancelling direction.
- Apply projection per transformer block (flattened group), not elementwise, and only when cosine is negative. Preserve one forward architecture, one bottleneck, one decoder, and one AdamW; cost is two component backward/gradient evaluations.

### Limitations and ordering

- This rule guarantees only that accepted scale contribution does not oppose direction. If `g_dir` itself worsens scale, no shared-gradient rule can avoid the trade-off without separate capacity; explicit scale-head learning remains available.
- Behavioral scale in action space is not purely radial per weight patch, so an explicit polar head does not mathematically eliminate all behavioral/structural conflict. It does eliminate the artificial raw-head coupling and structural `1/r_hat` amplifier.
- Do not add this projection before restoring the original polar decoder and remeasuring gradients. The missing decoder contract is a concrete implementation mismatch; extra backward surgery should be conditional on conflict that remains after parity restoration.

## 2026-08-28 — Literature-backed architectural answer to direction/scale conflict

### User question

- Найти не очередную loss-weighting/gradient-surgery эвристику, а архитектурный способ развести direction и scale, опираясь на литературу.

### Literature result

- Weight Normalization (Salimans & Kingma, NeurIPS 2016) показывает базовый принцип: длина и направление — разные координаты, и их явная reparameterization улучшает conditioning. В нашем случае original BigWeightVAE уже следовал этому принципу: отдельные direction/scale heads и deterministic recombination. Custom scaled Direct Normalized wrapper этот контракт потерял.
- Recon (Shi et al., ICLR 2023) исследует recurring gradient conflict по shared layers и показывает, что gradient surgery не убирает саму частоту конфликтов. Их архитектурный ответ — измерить conflict score по слоям и сделать high-conflict layers task-specific.
- Learning to Branch (Guo et al., ICML 2020) и Cross-Stitch (Misra et al., CVPR 2016) подтверждают более общий вывод: over-sharing вызывает negative transfer; оптимальная глубина разделения зависит от задач. Cross-Stitch soft-mixes task-specific streams, но для нашего случая это более сложный второй вариант, не первый выбор.

### Recommended architecture, ordered by necessity

1. Сначала восстановить exact original polar output chart. Input tokenizer может остаться p32, но decoder должен выдавать independent p16 output units. Для каждого p16: direction head выдаёт 16 signed logits, они нормализуются; scale head выдаёт один standardized log-radius; raw patch получается как `exp(log_radius) * unit_direction`. Никакого единого raw 32-value output head.
2. После этого заново измерить per-layer direction/scale gradient cosine. Старые конфликты измерены на неправильном raw-output wrapper и не доказывают, что polar-модель требует branch.
3. Если conflict сохраняется, применить Recon-style minimal branch: общий encoder, bottleneck и ранние decoder blocks; последние high-conflict decoder blocks копируются в два одинаковых residual tails — direction и scale. Оба tails используют один и тот же тип Transformer block, затем отдельные polar heads и deterministic recombination. Это не MoE и не набор разнородных механизмов.
4. Branch point выбирается по измеренному layer conflict, а не вручную. Если конфликт только в последнем block/head, добавляется одна task-specific пара blocks. Если он остаётся в shared decoder/encoder, separation сдвигается раньше только для реально конфликтующих layers.

### Important limit

- Полностью shared parameter не может одновременно получить два строго противоположных ненулевых first-order updates. Настоящее gradient separation неизбежно требует либо раздельных параметров, либо приоритета одной задачи, либо нулевого shared update. Архитектурное branching — честный способ заплатить небольшим числом параметров вместо скрытого cancellation.
- Behavioral action-space direction/scale остаются связанными через `X @ W`: radii отдельных patches влияют на итоговое action direction, а directions через сложение влияют на action norm. Поэтому polar heads точно разделяют structural coordinates, но не дают математической независимости behavioral losses. Это ещё одна причина сначала восстановить polar contract и только затем локализовать remaining conflict.

### Narrow proposed comparison

- Arm A: exact p16 polar decoder, полностью shared trunk.
- Arm B: тот же exact polar decoder, но Recon-style branch только в последних conflict-marked decoder blocks.
- Один seed/start/data/order/bottleneck/optimizer; без PCGrad, GradNorm, MMoE и дополнительных loss coefficients. Логировать все прежние losses, per-layer dir/scale cosine и norm, attention entropy, latent rank и обе head trajectories.
- Пока не установлено, что branch вообще нужен: missing polar decoder is a concrete transfer bug; branching is conditional second step if conflict survives its correction.

## 2026-08-28 — Где именно наблюдался failure: full dataset против exact64

### User question

- Уточнить, возникали ли обсуждаемые direction/scale проблемы только при production training на полном dataset.

### Evidence boundary

- На fixed exact64 scaled Direct Normalized 700M уверенно учился в structural-only режиме: full-train total `1.0064 -> 0.07036` к step1000.
- Сильный direction/scale conflict и плохая production dynamics измерялись на full production stream с другой objective: behavioral direction/scale плюс structural direction/scale, сначала с x10 scale terms, затем с bottleneck balancing.
- Поэтому dataset effect не изолирован: одновременно изменились dataset/sampling и loss. Matched fixed64 run с тем же production objective и тем же raw-output wrapper не был проведён.
- Дополнительный fixed64 activation-relative-MSE experiment тоже collapsed к zero-output basin, то есть маленький фиксированный dataset сам по себе не гарантирует обучение при плохой objective. Это другой loss и не является доказательством direction/scale failure на exact64.

### Narrow conclusion

- Для конкретного direction/scale production failure симптом действительно был зарегистрирован на полном dataset, тогда как известный exact64 structural-only control учился.
- Но утверждение «причина только в полном dataset» не установлено из-за confounded loss. Чтобы отделить dataset diversity от architecture/loss, нужен matched exact64/full comparison после восстановления polar decoder, с буквально одинаковой objective и optimizer settings.

## 2026-08-28 — Proposed 700M production-data polar model with one-block task tails

### User proposal and current interpretation

- Взять текущую scaled Direct Normalized 700M family и production dataset, сохранив p32 normalized tokenizer, Distribution Encoder conditioning и прежний bottleneck.
- Восстановить explicit polar output semantics: independent p16 direction units и p16 log-radius targets, затем deterministic recombination `patch = exp(log_radius) * unit_direction`.
- Не ограничиваться двумя linear heads. Shared decoder заканчивается branch point, после которого идут два независимых tails одинакового принципа: один standard Transformer residual block для direction и один такой же block для scale. После каждого tail остаётся только неизбежная минимальная linear projection: 16 logits для direction либо 1 scalar для log-radius.
- Preferred parameter-matched interpretation: из текущих шести decoder blocks первые пять остаются shared, а шестой shared block заменяется двумя independently parameterized copies. Тогда глубина каждого forward path остаётся 6 blocks, а total model увеличивается только на один block. Альтернативная трактовка — сохранить все шесть shared blocks и добавить сверху ещё два tails — дала бы depth 7 и прибавила два blocks; это требует explicit user confirmation и пока не считается выбранным вариантом.
- Common/shared parameters должны сохранять same-seed initialization. Два tails разумно инициализировать одинаковыми копиями исходного шестого decoder block, чтобы branch intervention начинался с одинаковой функции, а специализация возникала из разных losses.
- Direction loss обновляет direction tail/head, scale loss — scale tail/head; оба всё ещё сходятся в shared five-block decoder, bottleneck и encoder. Поэтому experiment проверяет, достаточно ли убрать конфликт в последнем representation/readout layer; он не гарантирует separation во всём trunk.
- Предполагаемый controlled run: production dataset и тот же direction/scale objective без operator behavioral term, без activation loss и без gradient balancing; все существующие loss components и per-layer gradient telemetry остаются logged. Это предположение нужно подтвердить перед implementation/launch.

### Remaining implementation choice

- Нужно явно решить p16 output-query geometry. Exact parity choice: p32 остаётся только input tokenizer granularity, а decoder/tails работают с independent p16 output queries. Cheaper approximate choice: каждый p32 decoder state выдаёт две p16 directions и два scales; эти половины всё ещё делят один pre-tail query state. Предыдущая recommendation подразумевала exact p16 output queries.

## 2026-08-28 — Implemented polar tails and exact loss/backward contract

### User decision

- Оставить прежний no-operator production scalar objective без изменения: `behavioral_direction + 10*behavioral_scale + structural_direction + 10*structural_scale`; `behavioral_operator=0`. Все компоненты должны оставаться logged.

### Implemented architecture

- New schema/config: `weightclip_direct_normalized_scaled_700m_polar_tails_production_v1`, config `projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_polar_tails_production_500k.yaml`.
- Production model retains p32 normalized input tokenizer, Distribution Encoder, 13 encoder blocks, `32x384` bottleneck and decoder blocks 1..5 shared.
- At branch point each of the 512 p32 decoder queries is split into two position-distinct p16 child queries. The former sixth decoder block is the direction tail; an exact initial copy is the scale tail. Each tail processes 1024 p16 queries plus latent slots. Minimal projections produce 16 direction logits or one log-radius.
- Direction is normalized over valid p16 components; scale uses the original BigVAE-style smooth log bound `[-3,6]`; recombination is deterministic. Partial masks have finite zero Jacobians. Exact parameter count is `742,120,833`.

### Necessary backward-routing clarification

- A first smoke with physically separate tails but losses evaluated through one common `W_hat` showed that behavioral scale still updated the direction tail/head strongly: at direction-head scale/direction gradient ratio was `5.67x`. This is expected because direction and norm of `X@W` couple all patch directions/radii even when raw patch coordinates are polar.
- To make the requested gradient-path separation real while preserving every scalar loss value, behavioral direction is evaluated with scale detached, and behavioral scale with direction detached. Structural direction owns direction coordinates; structural scale owns scale coordinates. Forward values and logged losses remain bit-identical to ordinary evaluation; only cross-coordinate backward is stopped. Shared encoder/bottleneck/decoder still receive both tasks.
- Config records this explicitly as `gradient_routing: coordinate_owned_stop_gradient`; this is part of the intervention and must not be described as a pure topology-only comparison.

### Verification evidence

- Focused CPU suite: `tests/weightclip_benchmark/test_direct_normalized_scaled_700m_production.py`, 7 passed. It covers config/count, mask/layout, p16 norm/recombination, old-vs-routed scalar parity, tail-copy initialization, gradient separation and legacy paths.
- Final H100 smoke: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_smoke_v3`, COMPLETE 2/2, peak `14.76 GiB`, `none_parameter_tensors=0`, every encoder/shared-decoder/tail/head group finite and nonzero.
- Step1 scalar loss is unchanged from the common-W backward smoke: total `3.586733`, behavioral `2.173105`, structural `1.413628`, decomposition parity exactly `0.0`; operator exactly zero.
- Coordinate routing makes direction-tail scale gradient exactly zero and scale-tail direction gradient exactly zero. In shared Q tensors, weighted scale/direction RMS ratio is now about `0.105` in encoder and `0.203..0.265` in shared decoder; cosines are near zero to mildly negative rather than scale-dominant. Pre-clip combined norm fell from `641.13` in common-W backward smoke v2 to `97.88` in routed smoke v3.
- Causal boundary: two steps establish wiring and immediate gradient geometry only, not training success. Production launch still requires independent review.

### Primary references

- Weight Normalization: https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html
- Recon: https://openreview.net/pdf?id=ivwZO-HnzG_
- Learning to Branch: https://proceedings.mlr.press/v119/guo20e.html
- Cross-Stitch Networks: https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html

## 2026-08-28 — Polar-tail production launch and first runtime evidence

### Final validation and launch

- Independent review issued FORMAL GO for the exact polar-tail source/config: no P0/P1 blockers; exact model count `742,120,833`, production bank/seed/B32/LR/500k schedule, loss contract and cross-tail gradient isolation were verified.
- Final smoke v4 completed at `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_smoke_v4`: step1 total `3.586760`, behavioral `2.173122`, structural `1.413638`, operator `0`, scalar component parity `0.0`, peak `14.76 GiB`. Opposite-coordinate gradients are exactly zero in both private tails and heads.
- Production run launched from a fresh state at `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1`; resume state is under `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_500k_v1`, persistent rolling model under `projects/shared/storage/artifacts/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1`.
- One initial launcher invocation failed before any run artifact was created because of a mistyped YAML filename; it was immediately relaunched with the reviewed config and does not affect the experiment.

### Early runtime evidence and warning

- The production process passed preflight and reached at least step150 with finite losses and no runtime failure. Step1 matched smoke exactly; peak through step150 was `16.42 GiB`.
- Total loss moved from `3.58676` at step1 to a noisy range around `3.07..3.24` through step150. This is startup evidence only, not a learning conclusion.
- The step100 gradient diagnostic exposed an important unresolved shared-trunk failure: although private tails remain perfectly isolated, direction gradients have already collapsed relative to scale in shared layers. Examples: encoder block13 Q has scale/direction RMS ratio about `1667x`; shared decoder block5 Q about `1283x`. Direction-head gradient RMS fell from about `0.611` at step1 to `0.00160` at step100 while direction losses stayed near `0.94`.
- Therefore the implemented branching fixes cross-coordinate contamination in the private tails but does not yet establish healthy credit assignment in the shared trunk. The run remains active to observe whether this is a transient or persistent collapse; it must not be presented as a solved training problem.

### Comet tracking

- The production trainer itself did not instantiate Comet. The existing live-metrics sidecar was updated so its metadata reflects the current polar-tail model (`742,120,833` parameters and coordinate-owned gradient routing) rather than the previous 706M architecture.
- The sidecar was attached without restarting training and backfilled the existing JSONL rows. Live experiment: https://www.comet.com/mike-5531/big-weight-vae/ffae5c0bdf864ccbb89f92caa5c7c4f8
- Binding is persisted in `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/comet_experiment.json`.

## 2026-08-28 — Polar-tail production status at step4250

### Observed training dynamics

- Run is operationally healthy and still active: step `4250/500000`, about `1.50 steps/s` including startup, peak GPU memory `16.42 GiB`, finite metrics and no runtime failure.
- Scientifically it is not learning well. Total loss falls quickly from `3.58676` at step1 to roughly `3.08` during the first few hundred steps, then stays flat. Over steps `3000..4250`, fitted loss slope is only about `-0.0022` per 1000 steps, negligible relative to minibatch noise.
- Last-100-row means (steps `3260..4250`): total `3.08169`, behavioral direction `0.91774`, behavioral scale `0.06581`, structural direction `0.93335`, structural scale `0.05725`. Both direction terms remain essentially at their initial near-random levels; most early improvement came from scale.
- The curve was visually inspected and is readable: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/train_loss_curve.png`.

### Mechanistic telemetry

- Private-tail isolation remains exact, but shared credit assignment remains badly scale-dominated. At step4200, scale/direction gradient-RMS ratio is about `157.6x` for encoder block13 Q and `63.5x` for shared decoder block5 Q. Across recent probes the ratios fluctuate but remain materially above one.
- Direction-head gradient RMS is about `4.1e-4` at step4200 versus scale-head `7.5e-3`; direction losses do not improve. Thus the one-block split removed direct cross-tail contamination but did not prevent the shared trunk/direction path from entering a weak-gradient plateau.
- Narrow current conclusion: runtime is valid, but the architectural intervention has not solved the training failure. Training remains active because the user asked only for status and did not authorize stopping it.

## 2026-08-28 — Mechanistic analysis of the polar-tail failure at step7000

### User request and evidence bundle

- User asked for a mechanism-level analysis while leaving the production run active.
- A zero-update diagnostic loaded the atomic step-7000/cursor224000 checkpoint and evaluated the same four production samples under the seed-42 initial and trained models. It reconstructed attention, block-output gradients, latent/context interventions, head geometry and per-token/sample cancellation. No training state was modified.
- Canonical bundle: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/mechanistic_step_0007000_v1/`; full report `mechanistic_report.md`, raw `report.json`, spectra `weight_spectrum.json`, frozen stability interventions `stability_counterfactual.json`, temporal telemetry `training_gradient_timeseries.csv`, inspected plots `mechanism_summary.png` and `training_mechanism_timeseries.png`.

### Failure definition and competing mechanisms

- Failure: direction losses stay near `0.92..0.95` and total near `3.08`; scale improves, yet the run is finite and all gradient paths exist.
- Tested mechanisms: double radial gauge/residual spectral runaway; attention saturation; decoder ignoring latent; shared direction/scale conflict; sample/token cancellation; hard latent-capacity limit.

### Supported causal chain

- Direction has two nested forward-invisible radii: direction-tail state is RMS-normalized, then 16 logits are unit-normalized. Unbounded residual blocks can grow either radius without changing decoded direction, while both normalization Jacobians shrink inversely with radius.
- At step7000, encoder block1 full write RMS is `21.84` vs init `.0965`; direction-tail write `154.1` vs `.1437`; median direction-logit norm `11.645` vs `.0864` (135x). Direction-logit gradient RMS drops `.01519 -> 7.44e-6` (2040x); direction-tail output gradient drops about123k times.
- Exact frozen-logit scale test: multiplying logits by `.1` leaves direction loss unchanged but makes logit/head-gradient proxies exactly10x larger; multiplying by10 makes them10x smaller. This establishes the unit-direction radial gauge causally, not correlationally.
- Output projections develop low-rank spectral gain rather than blanket norm explosion: median attention-output top singular grows7.76x and stable rank collapses397->8.5; MLP output grows4.07x and stable rank684->46. Direction-tail attention output is extreme: top singular `.302->4.270`, stable rank398->3.38.
- Representation then collapses: latent effective rank `99.94->1.32`, cross-sample latent cosine `.208->.99968`; predicted direction cross-sample cosine `.9396->.999987` while targets remain near zero cosine and effective rank15.62. Direction becomes a position-dependent common template.
- Latent/context interventions prove readout collapse: rolling latent changes direction only `1.76e-5`; zeroing latent `2.86e-4`; rolling decoder activation context `3.80e-5`. Zero latent still changes log-scale `.743`, so magnitude uses the common latent while direction ignores sample identity.
- The separate input scale MLP becomes a dominant value path: output RMS `.123->.640` while content projection stays `.253->.254`; supplemental step8000 spectrum shows its top singular `1.54->7.30` while content projection changes only `.886->1.014`.

### Discriminating interventions and excluded mechanisms

- Cosine-attention-only raises encoder1 entropy `.519->.980` but leaves write RMS `27.16`, direction logit norm `11.80`, gradient `7.34e-6`, and cross-sample prediction cosine `.99998`. Therefore attention concentration is secondary, not primary.
- Residual-cap-only reduces encoder1 write `21.84->.303`, logit norm `11.65->4.75`, raises direction-logit gradient about7.9x, and lowers cross-sample prediction cosine to `.9849`. It cannot recover information already erased in the frozen latent and worsens immediate loss, so this is a causal mechanism probe, not a hot-fix success claim.
- Shared direction/scale conflict is real but variable: across84 B32 probes through step8300, median scale/direction gradient ratio30.7 (p9096.6, max904); per-batch median cosine ranges `-.992..+.918`, with median53.6% of shared groups negative. It reinforces collapse but is not the whole cause because the private direction head receives exactly zero scale gradient and still collapses by roughly three orders.
- Token/sample cancellation amplifies failure: direction token resultant falls `.0742->.0267`, sample resultant `.893->.544`. It is not the sole initiator because the 135x logit radius and low-rank runaway are independently causal.
- Hard latent capacity is not supported as primary: available latent rank is not used; it collapses from about100 to1.3.

### Narrow conclusion and next architecture test

- Leading mechanism: unbounded low-rank residual gain grows two loss-invisible direction radii; RMSNorm and unit-vector normalization then extinguish direction credit, yielding a common-template decoder. Scale remains learnable, dominates the shared trunk and strengthens a low-rank scale shortcut. Attention concentration and task conflict contribute but are not standalone root causes.
- Recommended no-coefficient next comparison, same start/data/loss for 3000 production steps: (1) one unified repeated pre-norm block with cosine Q/K and smooth fixed RMS cap on every attention/MLP write; (2) the same plus a radius-constrained 16x1536 direction readout (fixed operator norm/orthonormal row frame). Also replace additive `content projection + scale MLP` with one projection of `[32 normalized values, standardized log-scale]` to remove the observed separate scale value path.
- Precommitted early failures: direction-logit median >4x start, latent effective rank<10, or prediction cross-sample cosine>.99 while target remains near zero.

## 2026-08-28 — Concrete mechanism-matched fix proposal

### User question

- User asked for the exact fix after the step-7000 mechanistic analysis.

### Proposed single coherent architecture

- Keep the current data, p32 grouping, `32x384` bottleneck, 13 encoder blocks, five shared decoder blocks, one direction-tail block, one scale-tail block, optimizer and exact scalar loss. Do not add gradient balancing or new loss coefficients.
- Replace every Transformer block by the same bounded pre-norm block:
  - RMSNorm input;
  - cosine-normalized Q/K with fixed attention scale `2`;
  - bias-free attention value/output path;
  - smooth per-token RMS cap on the attention write;
  - RMSNorm, SwiGLU MLP, the same smooth cap on its write;
  - residual identity unchanged.
- Use the already implemented smooth cap `write / sqrt(1 + RMS(write)^2 / tau^2)` with fixed `tau = 1/sqrt(2L)`, where `L=max(encoder_depth, decoder_path_depth)=13`. This gives `tau≈0.196`, is near-linear at the observed healthy initialization writes (`.10..14`), and cannot be defeated by spectral weight growth.
- Replace the additive tokenizer `content_projection + scale_MLP` by one bias-free projection of the 33-vector `[32 normalized signed values, standardized log-scale]`. Positional/tile embeddings remain additive. This removes the separately learnable scale value path that grew to RMS `.640` while content stayed `.254`.
- Keep the polar direction/scale tails, but replace the unconstrained 16x1536 direction matrix by a fixed-gain row-orthogonal readout. The learnable raw frame is orthogonalized as 16 rows and multiplied by the fixed initialization gain. This removes the head’s loss-invisible radial degree of freedom while leaving upstream direction features learnable. Scale head remains unconstrained because its amplitude is supervised.

### Why these changes and not more branching

- Residual cap is the causally supported primary intervention: on frozen weights it reduced encoder block1 write `21.84->.303` and restored direction-logit gradient about7.9x. Cosine attention alone repaired entropy but not the failure, so it is secondary stabilization inside the same repeated block rather than the main fix.
- Fixed-gain direction readout directly removes the exact counterfactual where logit scaling changes gradient inversely while leaving loss unchanged.
- Unified 33D projection removes the measured independent scale shortcut instead of trying to compensate it with loss weights.
- Do not split the encoder or add more experts yet. The private direction head collapses even with exact scale-gradient isolation, so more late branching does not address the proven radial/residual mechanism.
- Do not redesign the decoder into mandatory cross-attention in this first rerun. Current latent ignorance can be downstream of latent collapse; first test whether keeping states and head radius healthy restores sample-specific latent use. If bounded training remains healthy but latent interventions stay null, mandatory latent-value decoding becomes the next isolated architectural change.

### Proposed experiment and fail-fast gates

- One fixed arm versus the already-running failed baseline; same seed42, production stream, B32, LR `5e-5`, loss and 3000-step horizon. No panel of unrelated variants.
- Persist at steps1/100/250/500/1000/2000/3000: all existing losses; per-block input/write RMS; attention entropy/max probability; direction-logit norm; latent effective rank/cross-sample cosine; predicted-vs-target cross-sample direction cosine; latent-roll/zero direction effect; shared direction/scale gradient norms/cosines; approximate top singular values of output projections.
- Hard stop if any block write exceeds `2*tau`, direction-logit median exceeds4x its step1 value, latent effective rank falls below10 after step250, or predicted cross-sample direction cosine exceeds`.99` after step250 while target remains near zero.
- Success at step3000 requires both direction losses to show a sustained downward trend, non-null latent intervention effect, no radial/state runaway, and sample-specific direction predictions. A lower scale loss alone is explicitly not success.

### Operational recommendation

- The current 500k run has already reproduced the failure and can be stopped once the user authorizes it; continuing it no longer provides a plausible recovery trajectory.

## 2026-08-29 — Correction: remove symptom-level caps and eliminate the radial gauges

### User correction

- User rejected the bounded-write / orthogonal-head proposal as overengineered and insufficiently causal. The specific objection is correct: a cap prevents large norms but does not remove the reason those norms are free to grow.

### Revised root cause

- The primary defect is scale non-identifiability, not merely large activations. In the current direction path, the residual-stream radius is discarded by RMSNorm and the direction-logit radius is discarded again by unit normalization. The objective therefore cannot distinguish many states with different norms but the same prediction. Optimization is free to drift along these radial directions; as radius grows, the normalization Jacobians attenuate useful direction gradients.
- Consequently, write caps and spectral constraints are mechanism probes or safeguards, not the clean root fix. The next experiment should instead make both radii identifiable or remove them by standard architectural definitions.

### Revised minimal fix

- Do not change tokenizer, attention mechanism, loss coefficients or introduce gradient balancing in the first causal rerun.
- Replace pre-norm residual blocks with ordinary post-RMSNorm blocks so the evolving state itself has a defined RMS after every sublayer: `h <- RMSNorm(h + Attention(h))`, then `h <- RMSNorm(h + MLP(h))`. This removes the free residual-stream radius rather than clipping it. At depth13 this is a bounded, conventional comparison; gradients must be logged because post-norm can introduce its own depth attenuation.
- Replace the scale-invariant direction objective on unconstrained logits by direct regression to the unit direction target. The direction head emits a raw 16-vector `v`; train it with mean squared error to the unit target `u`. This single loss simultaneously aligns direction and pins `norm(v)` near1. It has no inverse-logit-radius Jacobian. The scale tail continues to predict log-radius with the existing scale losses. Optional unit normalization is applied only when composing the reconstructed weight, not as the only supervision of `v`.
- Remove the proposed residual cap, row-orthogonal direction head and tokenizer rewrite from this first experiment. They are not required to test the revised causal claim.

### Decisive experiment

- One same-start arm for 1000 steps, extend to3000 only if healthy, versus the archived failing baseline. Change only post-RMSNorm blocks and unit-target direction regression; retain the production data, p32 tokens, 32x384 bottleneck, depth/width, optimizer and scale losses.
- The causal prediction is direct: block-state RMS stays defined by construction; raw direction-logit norm remains near1 because it is supervised; direction-logit gradients do not decay inversely with norm; latent rank and sample-specific directions do not collapse. If gradients instead vanish monotonically with depth, post-norm gradient attenuation falsifies this implementation and must be addressed before a longer run.

## 2026-08-29 — Second correction: radial growth is an amplifier; decoder latent bypass precedes it

### User falsification

- User rejected post-norm and direct unit-target regression, correctly noting that modern LLMs train with pre-norm and contrastive systems train with normalized cosine embeddings. Therefore neither pre-norm norm growth nor cosine geometry alone can explain this run.

### New causal ordering from existing artifacts

- The archived random-initialization state already contains a diverse encoder latent: effective rank `99.94`, cross-sample latent cosine `.208`. Nevertheless, predicted directions are already almost common across samples: cross-sample cosine `.9396`.
- At that same initialization, rolling the latent between samples changes predicted direction by only `.000440`, and zeroing it changes direction by only `.00290`; rolling decoder activation context changes direction by `.1969`. This predates the later residual/logit norm explosion.
- At step7000, the initial decoder bypass has hardened: latent rank `1.32`, predicted direction cross-sample cosine `.999987`, latent-roll direction effect `1.76e-5`. Thus the strongest supported ordering is: query/context decoder prior ignores a useful latent at initialization -> weak sample-specific credit reaches bottleneck -> encoder and output collapse toward common template -> scale-invariant radial growth further attenuates the already weak direction gradient.
- The earlier exact logit-rescaling experiment remains valid, but it proves an amplification mechanism, not the initiating root cause. The cap and post-norm proposals are withdrawn as first-line fixes.

### Why LLMs and contrastive encoders differ

- A pre-norm LLM does exhibit residual-scale growth, but every sequence position begins with a sample/token-specific embedding carried by an identity residual path, and next-token cross-entropy uses unnormalized logits whose scale affects confidence. It does not compress input into blank learned slots and then decode through sample-independent output queries.
- Cosine contrastive learning supplies an explicit cross-sample uniformity/negative pressure in addition to positive-pair alignment, and its projection head is directly attached to sample-dependent encoder features. The current independent reconstruction direction loss has no latent-level anti-collapse term, while its decoder can emit a common position/context template without using the latent.

### Revised minimal architecture fix

- Keep pre-norm, the current tokenizer, polar representation and exact losses.
- Remove the decoder's additive query/context value bypass. Construct output query vectors from position and activation context, but use them only as queries. Initialize the decoded state with a bias-free cross-attention read whose keys and values are exclusively the encoder latent: `h0 = CrossAttention(Q=query(position, context), K=z, V=z)`. Do not add the query embedding back to `h0`.
- Run the existing pre-norm residual decoder blocks on this latent-rooted `h0`. For the direction/scale p16 tails, positional half embeddings likewise may affect routing/query coordinates but must not be additive output values. All decoded weight values must descend from `z`; activation context and positions may route them but not manufacture them.
- This is one standard asymmetric attention mechanism, not an auxiliary carrier. The key invariant is `z=0 -> direction value path=0` (scale bias may be logged separately), and sample permutation of `z` must permute decoded directions at initialization.
- First run a same-start step-0/step-1 contract and at most1000 production steps. The decisive prediction is immediate rather than late: at initialization latent-roll/zero direction effects must be material, prediction cross-sample cosine must fall well below the current `.9396`, and nonzero sample-specific gradients must reach the encoder before any norm dynamics develop. If those gates pass but training still collapses, decoder bypass is excluded as the sole cause and objective/sample-gradient cancellation remains viable.

## 2026-08-29 — Precise decoder design error and limit of the cross-attention claim

### User objection and retraction

- User correctly objected that the previous explanation did not show why mandatory cross-attention would learn. The proposal is reclassified from "fix" to a targeted discriminator: it removes one measured bypass but cannot guarantee that the encoder, bottleneck or objective are sufficient.

### Exact semantic error in the current graph

- Current decoder concatenates 32 latent tokens with 512 output-query tokens and runs joint self-attention. Each output query is an additive learned/tile/activation-conditioned vector and remains on the residual value stream through all five shared blocks. It therefore serves simultaneously as an address and as content from which the output head can generate a common template.
- Numerically, at random initialization the decoder queries allocate about `7.1%..10.1%` attention mass to latent tokens in shared blocks and only `4.49%` in the polar tail. The remaining value mass plus the exact residual identity is query/query content. The latent is already diverse (rank `99.94`, cross-sample cosine `.208`), yet the prediction is common (cross-sample cosine `.9396`) and latent roll changes direction only `.000440`. Therefore the current initialization transmits sample-specific latent information only as a small perturbation around a large coordinate/context carrier.
- The core semantic mismatch with an LLM is that an LLM residual token already contains sample-specific token content; our output-query residual contains only an address/prior. With an almost isotropic direction target (target cross-sample cosine `-.00044`), the common-query direction gradients largely cancel, while scale/context have coherent predictable means.

### What cross-attention would and would not establish

- A no-query-residual cross-attention read has algebra `h_i = sum_j attention(q_i, k(z_j)) * v(z_j)`. Position and activation context can select latent values but cannot themselves manufacture output values. With bias-free value/output projections, `z=0` implies `h=0`; swapping `z` necessarily acts on the sole value source unless the learned value/Jacobian path itself is degenerate.
- This only proves removal of the query-value bypass. It does not prove that attention preserves enough information, that the 32x384 bottleneck is adequate, that direction gradients do not cancel, or that later pre-norm blocks will remain healthy.
- The proposal must therefore pass a zero-training discriminator before implementation is considered successful: compare current and no-query-value decoder from the same encoder latent and start; require material latent-roll/zero effects, nondegenerate input-to-output JVP, and sample-specific prediction geometry. If it fails at step0, do not train it. If it passes step0 but collapses during training, the bypass was an amplifier rather than the root cause.

## 2026-08-29 — Latent-rooted decoder zero-training preflight

### User authorization and protocol

- User accepted the address-versus-value diagnosis and authorized about ten minutes of checks without training before deciding whether to implement and train the changed decoder.
- A frozen graph counterfactual reused the same seed-42 initial modules and the exact live step-17000/cursor544000 checkpoint on four real production samples. It performed no optimizer step and did not stop or modify the active run.
- Artifact root: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/latent_rooted_preflight_step_0017000_v1/`; source: `training/weightclip_benchmark/analyze_latent_rooted_decoder_preflight.py`.

### Results

- At seed-42 initialization, current decoder latent-roll direction change is `.001102`; latent-rooted graph is `.903701` (about820x). Current zero-latent direction change is `.006729`; rooted is exactly `1.0`, and rooted raw direction logits at zero latent are exactly zero.
- Mean cross-sample direction cosine changes from `.855130` current to `.120356` rooted. Initial latent is itself diverse on B4: cross-sample cosine `.126403`, effective rank `2.9896/3`.
- Activation context remains a routing signal: context roll changes rooted direction by `.103014`, while it cannot manufacture nonzero direction values at zero latent. Frozen latent probe gradient is finite, all coordinates nonzero, RMS `1.47e-5`.
- The step17000 old-graph checkpoint cannot be rescued by graph surgery: its latent is already collapsed (B4 cosine `.936456`), and rooted output remains common. This is expected and requires training the changed graph from the common initialization.

### Narrow verdict

- GO to native implementation plus step-0/step-1 smoke, not yet to long training. The preflight establishes that removing additive query values eliminates the measured bypass and exposes existing sample-specific latent information without breaking context routing. It does not establish sustained optimization, bottleneck sufficiency or objective coherence; those remain training questions.

## 2026-08-29 — Latent-rooted decoder: native implementation and production launch

### User decision

- After the zero-training preflight, the user authorized implementing the decoder change and launching the full 500,000-step production run immediately rather than doing a bounded 1,000-step experiment first.
- The requested intervention is decoder-only. Data, normalized p32 input, 32x384 bottleneck, 13-layer encoder, parameter initialization, loss, optimizer, batch size and schedule remain matched to the stopped polar-tail run.

### Implemented graph

- The first shared p32 decoder state is now a bias-free cross-attention read: position, tile and activation context define queries; `from_latent(z)` exclusively supplies keys and values. The query vector is not added to the value stream.
- Four ordinary pre-norm self-attention refinements follow, giving the same total of five shared decoder blocks as before.
- Each p16 direction/scale tail starts with the analogous no-query-residual cross-attention: child address is the query, while the shared p32 state exclusively supplies keys and values. The existing private direction and scale heads/loss routing remain unchanged.
- Exact invariant: zero latent produces zero raw direction, zero normalized direction and zero reconstructed weights. Context and positional signals can route latent values but cannot manufacture weight values.
- Parameter count and seeded state remain exactly unchanged at `742,120,833`; only the forward graph differs.

### Verification evidence

- Focused tests: 9/9 pass; native graph agrees with the frozen preflight reference within `1e-6`, latent permutation changes the output, latent gradients are nonzero, and the zero-latent invariant is exact. Compile and Ruff checks pass.
- Two-step production smoke: `/mnt/shared/weightclip_benchmark/direct_normalized_p32_production_smoke_20260829T005622Z/`. Step-1 loss is `3.2942617`, pre-clip gradient norm `69.229`, component-loss parity is exact, all 26 monitored parameter groups have finite nonzero gradients, no parameter tensor is missing a gradient, minimum group RMS is `1.549e-4`, and peak CUDA memory is `15.154 GiB`.
- Independent final reviewer issued FORMAL GO with no P0/P1 blockers.

### Run transition and production status

- The old additive-query polar run was stopped cleanly and preserved at step `18,511`, committed cursor `592,352`: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/STOPPED.json`. Its resume checkpoint (`8.906 GB`) and model checkpoint (`2.969 GB`) remain intact.
- New production root: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1/`.
- Exact config: `conf/weightclip_benchmark/direct_normalized_scaled_700m_latent_rooted_polar_tails_production_500k.yaml`.
- Comet: `https://www.comet.com/mike-5531/big-weight-vae/ee3e55b6331a4a3ab8ca1c96eacada23`.
- Startup is valid. At step 1, loss is `3.2942617`; all monitored groups are finite/live and component parity is exact. By step 20, loss is `3.2297525`, with operator loss exactly zero and all requested behavioral/structural direction and scale components still logged.
- Early step-100 telemetry is valid but already warrants caution: every group remains finite/non-null and component parity is exact, yet the first rooted decoder Q/K gradients have fallen from about `5.7e-4` at step 1 to about `1.7e-10..1.9e-10`; their scale gradient is roughly 3x the direction gradient. `from_latent.weight` changes from direction-dominant at step 1 (scale/direction `.112`) to scale-dominant at step 100 (`5.31`). The stochastic total loss is `3.2396` at step 100 versus `3.2943` at step 1, which is not enough evidence of a learning trend. This is an early warning, not a failure conclusion; the run remains live for the requested production test.

### Interpretation boundary

- This launch establishes that the architecture has removed the measured decoder value bypass and has a healthy initial backward path. It does not yet establish long-horizon learning or prove that the bypass was the only failure mechanism. The decisive evidence will be whether latent sensitivity, representation rank and direction learning remain healthy during training rather than collapsing as in the stopped run.

## 2026-08-29 — Production latent-rooted run: step-1000 mechanistic failure

### User request and failure definition

- The user observed that the new production curve again looked bad and asked what is happening internally. The live process was left running while performing read-only analysis.
- This is a real learning failure, not a noisy single batch: mean total loss rises from `3.253` over steps 1–200 to `3.327` over steps 1201–1440. Both direction losses remain near `.92–.94`; the new run is about `.239` worse than the same-stream old polar run over steps 1201–1440, almost entirely through scale losses.
- Primary training evidence: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1/train_metrics.jsonl`; readable curve: `train_loss_curve.png` in the same root; paired windows: `artifacts/weightclip_benchmark/latent_rooted_hostile_review_step_0001000_v1/paired_training_windows.csv`.

### Validity checks

- The run is alive and finite; logged steps are unique/monotone, committed cursor equals `32*step`, component-loss parity is exact, and there are no missing/nonfinite parameter groups. Gradient clipping is not continuously active after the initial transient.
- The native zero-latent contract still holds exactly: zero latent gives zero decoded weights. Therefore this is not a reintroduced query-value bypass or wrong-forward bug.
- Attention is not saturating to one-hot selections. The checkpoint-1000 probe shows almost perfectly uniform attention instead.

### Found proximal mechanism: absorbing symmetry collapse

- At initialization the learned latent is diverse: effective rank `96.14`, cross-sample cosine `.205`, centered RMS `.767`. At step 1000 it is nearly one common vector: effective rank `1.17`, cosine `.99938`, centered RMS `.063` while total RMS grows to `2.215`.
- Decoded directions likewise collapse: cross-sample cosine becomes `.999999` and effective rank becomes `1.0`, while the target direction rank is `15.30`. Predicted log-scale standard deviation becomes exactly zero versus target `.927`.
- Rooted shared attention becomes uniform: normalized entropy `.99992`, maximum probability `.03254` (approximately `1/32`), score span shrinks from `1.94` to `.071`. Direction-tail attention is already essentially uniform at initialization and remains so: entropy `.9999996`, maximum probability `.005377`, score span `.00537` at step 1000.
- This explains the Q/K gradient death. Q/K learn only from differences among values being mixed. When all latent/shared values are nearly equal, changing attention weights does not change the output, so Q/K receive almost no gradient. The direction tail starts in this weak-routing regime; the first rooted block falls into it by step 100.
- Removing the additive query bypass also removed the only strong token-identity carrier. Output/child addresses now affect values solely through attention weights. Uniform routing averages the same values for every address; subsequent permutation-equivariant blocks cannot recreate position identity from identical states. This is a self-reinforcing, absorbing shortcut.
- Exact artifact: `artifacts/weightclip_benchmark/latent_rooted_hostile_review_step_0001000_v1/report.json`; reproducible probe: `probe_source.py` beside it. Independent native dependency report: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1/latent_rooted_state_step_0001000_v1/report.json`.

### Scale conflict and radial growth are amplifiers, not the initiating event

- At step 1 the first rooted path is direction-dominant (`from_latent` scale/direction gradient ratio `.112`), so scale dominance did not initiate the tail routing weakness. By step 100 the ratio is `5.31`; by step 300 `64.6`; at step 500 gradients are nearly opposite (cosine `-.980`). Scale then captures the already-collapsing common encoder/value dictionary.
- Direction-logit median norm grows from `.0735` to `4.108` (about56x) while unit normalization leaves the loss almost unchanged. The direction-to-latent probe gradient falls from `1.263e-5` to `9.41e-12` (about1.34 million times). This radial gauge further attenuates direction credit but does not explain why tail Q/K were roughly1000x weaker than V on the first backward.
- By step 1000 the scale head accounts for about `99.85%` of the logged raw squared gradient energy. Encoder block 1, decoder conditioner and Distribution Encoder gradients are thousands to tens of thousands of times smaller than in the old graph at the same step.

### Narrow conclusion and remaining decision

- The previous diagnosis was only half right: the old decoder did have a real query-value bypass. The latent-rooted replacement removed it, but the all-cross-attention/no-query-residual formulation has no robust way to bootstrap addressed, position-specific values. It collapses to uniform averaging and an almost constant latent/template.
- Excluded as primary explanations: minibatch/logging noise, one-hot softmax saturation, broken zero-latent contract, and scale dominance as the initial event. Scale conflict and output-radius growth strongly worsen the collapse after it begins.
- The production process is still running because the user asked for diagnosis, not termination. There is no current evidence that continuing this trajectory will recover; the next user decision is whether to stop it and redesign the decoder’s token-identity/value construction rather than tune the loss.

### Clarification of “uniform attention” and proposed repair (not yet authorized)

- “Uniform” is within each sample, not across samples. In the first rooted read, each of 512 output queries assigns almost the same probability `1/32` to each of 32 latent slots. In the p16 tail, each of 1024 child queries assigns almost the same probability to every valid p32 state. Therefore every address receives essentially the same average value.
- This is fatal because addresses are not present on the residual value stream. Once the weighted averages are equal, all later states are equal. The derivative of attention routing depends on differences between each value and their weighted mean; when those differences vanish, Q/K gradients vanish as well. The model cannot bootstrap out of the symmetric state.
- Proposed next decoder removes the softmax bootstrap entirely while retaining the exact no-bypass property. Map the 32 latent slots directly to 512 p32 states through one learned signed slot-mixing matrix of shape `512x32`: each output position receives a different linear combination of latent values from the first forward pass. Zero latent still gives zero state, but positional diversity and a direct output-to-latent Jacobian exist at initialization and do not depend on Q/K learning.
- Then run the same five ordinary pre-norm self-attention refinement blocks. Split each p32 state into two p16 children by multiplicative learned half-gates, not by a second cross-attention bootstrap; run the existing private one-block direction/scale tails afterward.
- Activation/tile conditioning should use the existing conditioner only as bounded multiplicative modulation of the latent-derived state, for example a gate in `(0,2)` initialized to one. It cannot create values when latent is zero. Thus there remains one unified decoded state rather than an additive address carrier or a fixed baseline plus learned residual.
- The key distinction from the failed graph: even if latent slots temporarily become similar, the fixed current mixing rows still give every output a direct signed latent-dependent path, and gradients through that path do not require non-uniform softmax probabilities. A common latent/template shortcut remains possible in principle, but it is no longer an absorbing zero-Q/K-gradient state.
- Recommended protocol after user approval: stop/preserve the current run; implement this single decoder intervention; require step-0 nonuniform p32/p16 state geometry and live direct Jacobians; run the same production stream for 1000 steps with hard gates on latent rank, latent-roll effect, direction diversity and per-depth gradients before authorizing another 500k trajectory.

## 2026-08-29 — Retraction of the learned signed slot mixer

### User correction

- User rejected the proposed learned `512x32` signed slot-mixing matrix as another computationally unstable global mixing mechanism that would itself admit collapse. This objection is correct; the proposal is withdrawn and must not be implemented.

### Why the signed mixer preserves the failure

- If all 32 latent slots collapse to one common vector `c`, output token `i` becomes the row sum of the mixer times `c`. The 512 independently learned rows can therefore memorize a different positional template from a rank-one latent. The encoder is not forced to carry sample-specific information.
- The factorization is not identifiable. A change of latent basis can be compensated by the inverse change in the mixer, and latent amplitude can be traded against mixer amplitude without changing decoded output. Normalization downstream makes this scale freedom even less constrained. This creates a new conditioning/gauge problem rather than fixing the old one.
- Every output error is backpropagated through the same global mixer into every latent slot. Isotropic direction errors can cancel across positions while coherent scale/template gradients accumulate. Thus the mechanism that favored a common template remains available.
- Replacing positive softmax averaging with signed linear averaging removes one implementation of the collapse but not its cause: global all-to-all aggregation with no exclusive responsibility for any latent slot.

### Revised architectural requirement, not yet an authorized implementation

- The first decoder expansion must contain no learned global all-to-all mixing or routing. Each bottleneck slot needs hard structural ownership of a disjoint output subset and a direct gradient from that subset.
- Encoder and decoder ownership should be mirrored. A minimal unified candidate is a symmetric hourglass: shared pairwise merges `512 -> 256 -> 128 -> 64 -> 32`, the fixed `32x384` bottleneck, and shared pairwise splits `32 -> 64 -> 128 -> 256 -> 512 -> 1024`. Grouping defines responsibility, not a claim of image-like locality.
- Split/merge weights are shared across positions; fixed child-role/address embeddings distinguish the two children. There is no independent 512-row positional dictionary. If all parents become equal, the decoder can only repeat a small shared child pattern, while different owned targets send different gradients to different parent slots. Hence the observed arbitrary rank-one positional-template shortcut is no longer an easy stationary solution.
- Ordinary pre-norm self-attention may be used for global interaction after ownership-specific states exist and remain on the residual stream. It must not be the mechanism that creates token identity from blank queries. The same repeated hourglass block family should be used throughout; no protected baseline, carrier, learned global mixer, or second decoder path is proposed.
- This is a mechanistic candidate, not established evidence for weight autoencoding. Required discriminator before training: compare equality-state Jacobians of the failed rooted decoder and the hard-ownership hourglass; then run at most 1,000 matched steps and require latent slot/sample diversity, latent-roll sensitivity, non-rank-one predictions, and live early encoder gradients. No production run is authorized by this discussion entry.

## 2026-08-29 — Independent multi-step risk review; naive hard ownership is also NO-GO

### User correction and process failure

- User identified a repeated research-process failure: proposed fixes were optimized against the latest observed failure mode but were not subjected to a second-order review of the new shortcuts, gauges, gradient deadlocks and capacity restrictions introduced by the fix itself.
- This criticism is accepted. The rooted decoder removed the old additive query bypass but created a routing bootstrap collapse. The subsequently proposed signed mixer removed the softmax bootstrap but restored a global rank-one template and latent/mixer gauge. The initial hard-ownership proposal likewise addressed global mixing before its own future failure modes had been evaluated.
- No new model implementation or training is authorized from this discussion. Three independent reviewers were asked to attack the design from Jacobian/optimization, information-flow/symmetry and hostile-shortcut perspectives.

### Independent verdicts

- All three reviewers reject the naive hard-owned learned merge/split design as currently specified. It is better localized than a global mixer but not a solved architecture.
- A shared split tree can map an equal nonzero parent into one learned 32-role motif repeated across 32 groups. Additive group/role/context values make this a full template carrier again. A high-rank multiplicative context modulation can also decode from context plus a constant latent while still passing the weak `z=0 -> output=0` test.
- Strict ownership makes each 384-dimensional slot responsible for 512 continuous weight coordinates. This is a local 25% compression whose adequacy has not been established; aggregate bottleneck size alone cannot justify it because strict owners cannot borrow capacity from one another.
- Learned merge/split levels can discard sibling-difference modes while retaining coherent common/scale modes. Their singular-value defects and latent-basis/scale gauges compound over multiple levels.
- Later global attention can numerically erase initial ownership. Separate tails also do not remove shared direction/scale conflict.
- The existing unit-normalized direction output retains an exact invisible-radius gauge: raw direction norms can grow without changing the forward prediction while useful tangent gradients shrink. None of the proposed ownership changes fixes this amplifier.
- Query-residual variants remain vulnerable to the old positional template bypass. Autoregressive variants admit teacher-forced posterior collapse or a constant-latent/BOS template generator. Neither is recommended as the next experiment.

### Stronger required contract

- Zero-latent output is necessary but insufficient. Every candidate must pass a constant-nonzero equality-state intervention: force all slots and samples to the same latent while using heterogeneous real targets; owner gradients must remain nonzero, heterogeneous and aligned with their own target residuals.
- Every decoded value needs sole ancestry from sample-derived weight content. Learned blank latent seeds, additive group/position/role/context values, decoder biases and unique per-group dictionaries are disallowed on the value path.
- Ownership must be visible in the full-depth Jacobian, not just the graph topology. Before global refinement the owner Jacobian must be block diagonal; after every block a material diagonal component must remain.
- Any grouping requires a local-versus-global capacity audit and production permutation/gauge audit. Fixed grouping is bookkeeping only if it does not silently assume unavailable locality.
- Common and sibling-difference modes must both survive every dimension-changing transition. Learned averaging is not an acceptable merge for noisy weights.
- Direction-radius and latent-basis sweeps must demonstrate that a loss-invisible internal rescaling cannot arbitrarily extinguish the useful VJP.
- Direction and scale task gradients must be measured separately at the token field, ownership boundary, bottleneck, decoder expansion and output tails. A healthy total gradient is not sufficient.

### Conditional candidate, not a selected architecture

- The only candidate surviving as worth preflight is deterministic field folding, not a learned tree: create one sample-derived field per p32 token, concatenate/reshape fields without averaging from `512x24` to `32x384`, and exactly unfold them in the decoder. Pure fold/unfold has unit conditioning and cannot hide collapse in learned slot aggregation.
- This candidate still has unresolved risks: the explicit local `32 weights -> 24 features` compression may discard direction information; a global bottleneck Transformer may overwrite the folded field with a common dictionary; context modulation may become a carrier; and the direction-radius gauge remains. Therefore it is not approved for training before capacity, carrier, equality-state and full-depth Jacobian tests.

### Evidence

- Jacobian/optimization review: `/home/coder/project/docs/notes/decoder_architecture_design_risk_review_20260829.md`.
- Joint encoder/decoder information-flow review: `/home/coder/project/artifacts/weightclip_benchmark/hard_ownership_architecture_review_20260829/report.md`.
- Hostile shortcut/risk review: `/home/coder/project/projects/weight-vae/workspace/artifacts/weightclip_benchmark/group_owned_decoder_hostile_review_v1/report.md`.

## 2026-08-29 — Facts-only reset and one first-principles architecture bet

### User direction

- User rejected further local patches to the accumulated token/query/polar architecture. The requested reset is to reconstruct the observed training phenomena from raw evidence, distinguish intrinsic data/task difficulty from complexity introduced by our own formulation, consult genuinely analogous solved problems, and make one reasoned architectural bet rather than launch a panel of defensive micro-experiments.

### Retractions and evidence-backed corrections

- The earlier claim that output queries or a particular decoder routing graph are the root cause was overconfident. Perceiver IO is a direct counterexample showing that a fixed latent bottleneck plus semantic output queries can work; the question is whether that machinery is necessary and well-conditioned in this codec, not whether queries are intrinsically invalid.
- Tokenizer B, Distribution Encoder conditioning, p32 tokenization, and 700M scale are not individually sufficient causes of collapse. Each appears in a bounded fixed-operator run that learns well.
- The cleanest isolated learn-versus-collapse transition in the archive is the backward objective on the same small normalized model, start, data and optimizer. Action-relative reconstruction reaches only NRMSE `.916` at step512 with latent effective rank `1.95`, whereas the normalized structural objective reaches loss `.266` at step500 and `.013` at step4000 with latent rank `17.72` and `20.78`.
- A simple direct-normalized 700M p32 model learns the exact64 set (`1.006 -> .070` by step1000), so parameter count and shallow normalized weight input do not themselves prevent optimization.
- The exact64-to-production transition changed dataset diversity, gauge views, batch size, LR schedule, loss composition and scale weighting simultaneously. Production has never received a clean long run of the simplest successful normalized reconstruction formulation. Therefore the production root cause is not experimentally identified.
- Removing only operator loss, adding gradient balancing, polar tails, bounded attention, or replacing the decoder did not rescue production. These failures invalidate those patches; they do not establish a universal decoder theorem.
- Low latent rank and loss of sample/permutation sensitivity are the most consistent failure correlates. Gradient concentration, clipping, and attention saturation are not necessary distinguishing causes because some successful runs also exhibit head-dominated raw gradients.

### Intrinsic task versus self-created task

- Intrinsic task: map one fixed 128x128 weight tile (16,384 continuous values with heterogeneous row scales) into exactly 12,288 latent numbers and reconstruct it. This is a real 25% undercomplete codec problem. If the source is locally full-dimensional, some information must be lost; no routing trick removes that rate limit.
- Self-created task: we additionally imposed semantic latent slots, long attention routing, blank output queries, activation context as a decoder path, p32/p16 conversions, separate unit-direction and log-scale heads, duplicated behavioral/structural objectives, loss-invisible direction radius, and later gradient surgery. None of these is required by the bottleneck contract.

### Selected leap of faith: a dense nonlinear-PCA autoencoder

- Treat the bottleneck as one vector of length 12,288; reshape it to `32x384` only at the external interface. Do not assign semantic slot ownership.
- Use one invertible input representation: for each of 128 rows, store 128 signed weights divided by that row's q7/max-absolute scale, plus one standardized log-scale scalar. The resulting input and reconstruction target both have length 16,512. Padding/zero masks remain explicit.
- Encoder: one bias-free dense map `16512 -> 12288`.
- Bottleneck refinement: one ordinary residual GELU block `z = h + M2(GELU(M1(h)))`, with `M1` and `M2` both `12288 -> 12288`.
- Decoder: one bias-free dense map `12288 -> 16512`, followed by the deterministic inverse normalization to raw weights.
- Parameter count is `707,788,800`, matching the requested scale without depth, attention, queries, tails, Distribution Encoder, context carrier, normalization layers, or auxiliary paths.
- Train with one masked mean-squared reconstruction loss in the standardized normalized representation. Direction and scale are not separate tasks: all signed normalized coordinates and all standardized log-scales are coordinates of the same target. Raw-weight and activation-action errors remain logged diagnostics, not competing backward objectives in the first run.

### Why this is a standard reset rather than a new patch

- Its linear special case is the classical undercomplete linear autoencoder/PCA result. The single residual nonlinear block adds capacity without changing the one-path codec structure.
- Strong image/audio first-stage codecs likewise first learn deterministic sample-derived latent codes with direct reconstruction losses; they do not require the downstream generative/task objective to teach the codec from scratch. MAE also succeeds with normalized reconstruction targets. Hyper-representation work on neural weights specifically reports that unnormalized global weight MSE ignores small/narrow layers and uses layer-wise normalized reconstruction to fix this imbalance.
- Activations are relevant for eventual task-aware distortion, but feeding them as a second decoder information source makes the representation conditional rather than canonical. The first codec should assign one code to one weight matrix. If functional fine-tuning is later needed, retain the reconstruction loss and add a full-rank action metric; do not replace the codec objective with an activation-relative seminorm that has a large null/easy region.

### Risks considered before implementation

- The 12,288 bottleneck may be genuinely too small for the production distribution. Failure with a healthy noncollapsed optimizer would then be a rate/capacity result, not a reason to add routing machinery.
- Dense matrices assume a fixed padded tile layout and are expensive but fit the current fixed 128x128 contract and deliberately make no unsupported locality assumption.
- The usual invertible change-of-basis gauge between encoder and decoder remains, but unlike the polar direction radius it is not loss-invisible to only one task and does not create an alternate decoder carrier. A shallow path limits conditioning products.
- Gauge/permutation augmentation may make compression harder by presenting functionally equivalent matrices as different raw targets. Start with the canonical production representation; add equivariance only if the product requirement actually demands it.
- This proposal intentionally omits activation conditioning. That is a feature of the first canonical-codec run, not a claim that activations are useless. They are reserved for evaluation and possible second-stage metric fine-tuning after reconstruction is established.

### Minimal next action, not yet launched

- Implement this one architecture, run only a short wiring smoke, then one real approximately-2k-step production-stream training. No architecture panel is proposed. The decisive read is simply whether normalized reconstruction decreases while latent sample rank and held-out-batch reconstruction diversity remain alive.
- The currently running latent-rooted production process was not stopped or modified during this retrospective.

### Evidence and literature

- Facts-only retrospective: `/home/coder/project/docs/notes/weight_ae_facts_only_retrospective_20260829.md`.
- First-principles architecture report: `/home/coder/project/projects/weight-vae/workspace/artifacts/weightclip_benchmark/first_principles_dense_ae_bet_v1/report.md`.
- Literature review: `/home/coder/project/artifacts/weightclip_benchmark/literature_fixed_bottleneck_review_20260829/report.md`.
- Classical linear autoencoder/PCA result: Baldi and Hornik (1989).
- Weight hyper-representation normalization evidence: Schuerholt et al., NeurIPS 2022.
- Relevant reconstructive analogues: VQ-VAE, MAE, latent-diffusion first-stage autoencoders, SoundStream/EnCodec, and Perceiver IO as a query-based counterexample.

## 2026-08-29 — Second-order correction to the dense-AE target

- Before presenting the dense reset, the primary agent considered simplifying the representation further to one fixed global signed transform `asinh(W / s_ref)` with transformed-space MSE. Independent reviewers agreed that this is a defensible companding baseline but rejected treating it as a neutral main-task representation.
- The reason is mechanistic: transformed MSE locally weights raw error approximately as `raw_error^2 / (s_ref^2 + W^2)`. It deliberately discounts absolute errors on large weights, while inverse `sinh` amplifies transformed-space tail errors. One global `s_ref` is also poorly matched to a production mixture of row/operator scales, and it redundantly stores row scale across all row entries.
- Existing project evidence favors explicit local scale conditioning: under the same small-model structural setup, per-row normalized continuous input reached loss `.013` and latent rank `20.8`, whereas raw signed input ended at `.557` and rank `2.4`. Therefore global companding is not selected as the primary leap of faith.
- A second issue was found in the initially written `q7 values + 128 log-scale coordinates` reconstruction target: a flat coordinate MSE could underallocate credit to only 128 scale coordinates among 16,512 total coordinates, recreating an avoidable two-lane allocation problem.

### Final selected dense reset

- **Encoder input only:** the empirically successful invertible per-row representation: 16,384 signed q7/maxabs-normalized values plus 128 standardized row log-scales.
- **Latent/core:** unchanged simple dense proposal: `16512 -> 12288`, one residual GELU block with two `12288 -> 12288` maps, and reshape to `32x384` only at the interface.
- **Decoder output:** directly predict all 16,384 raw weight values divided by one fixed corpus reference RMS. The decoder does not output separate direction and scale heads and receives no scale side channel; all scale information must pass through the 12,288-dimensional code.
- **Single backward objective:** per-example full-rank relative weight MSE, verbally: squared error over every valid raw-weight coordinate divided by the fixed target weight energy for that example, then averaged over examples. Its zero-output baseline is one; at zero output its gradient points directly toward the complete target. There is no activation nullspace, unit-direction normalization, radial gauge, or competing scale/direction gradient.
- **Diagnostics only:** raw Frobenius error, old structural direction/scale metrics, and activation-action error remain logged but detached. They do not alter training in the first run.
- Exact bias-free parameter count is `706,215,936`: encoder `16512x12288`, two core maps `12288x12288`, decoder `12288x16384`.

This correction preserves the representation with direct positive evidence, but makes the learned mapping and loss one-path and one-target. It is the selected proposal; neither `asinh` nor the earlier dual reconstruction target should be presented as the main run.

Targeted concept review of the rejected compander: `/home/coder/project/artifacts/weightclip_benchmark/asinh_dense_ae_concept_check_20260829/report.md`.

## 2026-08-29 — Hard-contract correction: arbitrary matrix size and activation distribution

### User correction

- The proposed dense `16512 -> 12288 -> 16384` reset violates a hard product requirement: the weight matrix has arbitrary `d_in x d_out`, while latent size is a fixed architectural hyperparameter. Reconstruction quality may degrade as the number of source values grows, but the model graph must not require one fixed input/output dimension.
- A second hard requirement is that the representation meaningfully account for the distribution of input activations.
- The fixed-layout dense proposal is therefore retracted as the production architecture. It remains at most a fixed-tile diagnostic and must not be presented as the solution.

### Correct standard problem class

- The actual contract is `variable-size structured input -> fixed-rate latent -> variable-size structured output`. This is exactly the problem class addressed by Perceiver IO and set/conditional neural-process codecs, not by a flat fixed-dimensional autoencoder.
- A fixed `32x384` latent supplies 12,288 real-valued code coordinates for every matrix. As `d_in*d_out` increases, fewer code coordinates are available per source value; graceful quality degradation is expected. No architecture can promise constant fidelity for unbounded full-dimensional matrices at fixed rate.

### Selected unified architecture family

- Tokenize a `d_in x d_out` matrix into a variable sequence of p32 weight chunks. Each token carries 32 continuous signed locally normalized values, its local log-scale, a validity mask, deterministic row/column/chunk address, and matrix-shape metadata. Local chunk scaling avoids max-statistic drift with arbitrary row length.
- Run the production Distribution Encoder over corresponding valid input-activation patches. Its context conditions the encoder attention for aligned weight chunks. Activation context affects routing/bit allocation, while weight-derived features remain the content being compressed; there is no activation-to-output decoder path.
- One Perceiver input cross-attention reads the arbitrary number of conditioned weight tokens into a fixed latent array. A stack of identical-structure pre-LN latent Transformer blocks supplies model capacity. The final encoder projection produces exactly `32x384`.
- For any requested output shape, create one deterministic coordinate query per p32 output chunk. A deliberately shallow Perceiver decoder performs one cross-attention read from the fixed latent and a shared linear p32 head. Query count is arbitrary; decoder parameters do not depend on matrix dimensions.
- There are no polar heads, separate direction/scale tails, output self-attention tower, second reconstruction path, or activation carrier around the bottleneck. Parameter count can be scaled toward 700M by the number/width of repeated latent Transformer blocks, not by changing latent rate or adding mechanisms.

### One activation-aware reconstruction objective

- Raw relative weight MSE alone cannot force meaningful activation use: conditioned on the exact target W, its optimum is independent of activation covariance. Activation tokens could be ignored or used only as dataset identity.
- Use one full-rank activation-aware quadratic distortion instead. Let `A` be the activation covariance over input coordinates, normalized to have mean eigenvalue one, and let `G = I + A_normalized`. The loss is the reconstruction-error energy measured by G, divided by the target-weight energy measured by the same G.
- Verbal form: ordinary raw weight error plus activation-action-weighted error, normalized together as one metric. The identity term makes every raw-weight direction costly and removes the nullspace that made action-only loss admit an easy basin. The covariance term makes errors on frequently or strongly activated input directions more expensive. There are no separate direction and scale objectives or tuned loss coefficients.
- Distribution Encoder context tells the encoder how to allocate the fixed code for the current activation distribution; the same distribution defines which reconstruction errors matter. Decoder still sees only the fixed latent and output addresses.

### Known risks retained honestly

- Vanilla Perceiver routing can still learn a mean/template solution; a shallow decoder limits this shortcut but cannot mathematically prohibit it. The reason to make this bet is that Perceiver IO is a proven variable-input/fixed-latent/variable-output architecture, while the project evidence already shows fixed latent and query decoding can learn on bounded data.
- The production archive supports normalized weight tokens, useful Distribution Encoder conditioning, fixed `32x384` latents, query decoding, and variable-shape plumbing separately. It does not establish this exact composition on the full production distribution.
- A full activation covariance can be expensive for very large `d_in`; implementation may use the sufficient statistic already available for the production activation sample or a full-rank shrinkage approximation. Any approximation must preserve the identity floor and the single-metric semantics.
- Address encoding must support arbitrary shapes without a learned fixed-size positional table. Deterministic coordinates plus shape metadata satisfy this interface but do not impose image-like smoothness as a scientific assumption.

### Evidence

- Corrected architecture/literature report: `/home/coder/project/artifacts/weightclip_benchmark/arbitrary_matrix_perceiver_io_architecture_20260829/report.md`.
- Facts-only component assessment remains: `/home/coder/project/docs/notes/weight_ae_facts_only_retrospective_20260829.md`.

## 2026-08-29 — Retraction of the activation-SPD loss bet and compute-scalable decoder correction

### User correction

- User rejected two parts of the preceding proposal: a single shallow output write is not an adequate decoder scaling story for a future approximately-10B foundation model, and the proposed activation-covariance/full-rank loss is too close to objectives that already failed to converge.

### Exact loss-history audit

- The literal metric `G = I + normalized_A` has never been implemented or run. None of the archived action-loss runs contains an identity/raw-MSE term.
- However, two close activation-covariance objectives already failed decisively:
  - Small 34.85M full-operator action-relative run: `/mnt/shared/weightclip_benchmark/gptq_token_bottleneck_v1_20260827T151252Z`. The normalized arm reached action NRMSE `.91624` against zero baseline `1`, held-out NRMSE `1.02013`, and latent participation rank about `1.92` by step512.
  - 706M per-tile action-relative run: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_activation_metric_3k_v2_20260828T130359Z`. It ended at loss `.96820` against zero baseline `1`, prediction/target action RMS `.12739`, raw NRMSE `.99909`, and latent rank `1.77` after 3000 steps. It already used all 512 activation rows, so precomputing covariance would not remove a missing activation-row minibatch variance.
- Writing action loss as `trace(error^T A error)` or precomputing `A` is algebraically the same objective. Full-operator aggregation would repair missing cross-tile terms in the 706M run, but the small run already used the full operator and still collapsed.
- Adding a nontrivial identity term is mathematically different and removes activation-null directions, but it does not remove shrink-to-zero geometry. Under any positive-definite quadratic metric, while prediction direction is initially uncorrelated with target direction, reducing prediction amplitude moves loss toward the zero baseline. A small identity ridge is nearly the failed objective; a large ridge is a substantive raw/action mixture requiring the very weighting decision the proposal claimed to avoid.
- Therefore the `I+A` backward objective is withdrawn. Exact novelty is not evidence of likely success. Activation-action quantities remain diagnostics until there is a better-grounded objective.

### Scalable encoder/decoder topology

- A scalable Perceiver codec need not have a one-layer decoder. Distinguish the deep **decoder trunk** from the final variable-size **output renderer**.
- Encoder side: variable weight tokens with production Distribution Encoder context are read into 32 internal tokens; a deep pre-RMSNorm latent Transformer stack processes them; a projection produces the fixed code `z[32,384]`.
- Decoder side: expand `z` from width384 to foundation width `D`, then run a separate deep stack of untied pre-RMSNorm Transformer blocks on those same 32 tokens. Only after that deep decoder memory is formed does one shared output cross-attention render an arbitrary number of p32 coordinate queries.
- Repeated decoder block:
  - residual self-attention over 32 memory tokens;
  - residual SwiGLU MLP;
  - ordinary depth-scaled residual-projection initialization, with no nonlinear caps.
- At `D=4096`, one dense MHA+SwiGLU block is approximately `.20B` parameters. Roughly 25 encoder-memory blocks plus 25 decoder-memory blocks yield about 10B parameters while keeping the fixed bottleneck `32x384`. The final output write remains linear in the number of requested p32 chunks and can be streamed exactly.
- This places comparable capacity on both sides of the bottleneck. A 10B block stack is not run independently for every output coordinate; doing so would multiply foundation-model compute by the matrix token count, while output self-attention would additionally become quadratic in matrix size.
- The final renderer can still be a capacity bottleneck and a deep decoder can still learn a conditional mean/template. The proposal does not claim otherwise. The key correction is that decoder expressivity scales through a deep nonlinear transformation of the complete fixed code before rendering, while every output query shares that transformed memory.

### Current loss/conditioning stance

- Do not combine this architecture change with another speculative loss. The only strongly positive objective evidence remains locally normalized structural reconstruction (`direction + 0.1 log-scale`) on fixed64; production-scale generality remains unestablished.
- Distribution Encoder conditioning remains encoder-side. A successful small run showed it was materially used: context shuffling changed normalized-run loss from about `.022` to `.798` and changed latent RMS materially. This establishes that activation context can influence a reconstruction model without activation loss.
- This satisfies “activation information is available and used by the encoder” in demonstrated bounded settings. It does not yet guarantee principled operator-aware rate allocation on production. No claim stronger than that is currently supported.

### Evidence

- Exact historical audit: `/home/coder/project/docs/notes/weight_ae_facts_only_retrospective_20260829.md` and the two run roots above.
- Scalable decoder review: `/home/coder/project/artifacts/weightclip_benchmark/scalable_deep_perceiver_io_codec_20260829/report.md`.

## 2026-08-29 — Proposal to regularize latent collapse in the old polar two-tail model

### User hypothesis

- User proposed retaining the previous two-head polar architecture and adding a contrastive/in-batch latent loss so different matrices or tiles cannot map to the same latent. The desired regularizer should make collapse unfavorable while leaving the decoder free to choose the actual latent coordinate system.

### What the archive establishes

- Cross-sample collapse is real. In the old additive-query polar run, sample-mean latent cosine changed `.208 -> .999679`, pooled sample/slot effective rank changed `99.94 -> 1.325`, and predicted directions became essentially identical. Evidence: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_500k_v1/mechanistic_step_0007000_v1/report.json`.
- In the latent-rooted run, pooled rank changed `96.14 -> 1.17`, sample-mean cosine `.205 -> .99938`, and sample-specific latent-roll output effect collapsed. Evidence: `/home/coder/project/artifacts/weightclip_benchmark/latent_rooted_hostile_review_step_0001000_v1/report.json`.
- The stored `effective_rank` mixes batch and slot axes. A clean within-each-sample 32-slot rank was not persisted; near-uniform decoder attention proves averaging but does not by itself prove that all 32 encoder slots were literally identical.
- Critically, old polar direction decoding was already almost latent-identity-insensitive at initialization, before latent collapse: latent rank was about `99.94`, sample cosine `.208`, yet latent roll changed predicted direction only `.000440`. Training-time latent diversity loss therefore targets one measured failure but not the pre-existing weak direction-decoder use. Scale/amplitude did use the latent materially; the decoder did not ignore all latent information uniformly.

### Why naive pooled InfoNCE is insufficient

- It can be solved by assigning one operator/activation fingerprint `c_i` to every sample while all 32 slots inside sample i remain equal. Samples become perfectly contrastive, yet attention still mixes identical values and its Q/K routing gradient remains zero.
- A projection head can isolate an ID subspace that the reconstruction decoder ignores. Activation Distribution Encoder context, layer identity, shape and tile address are easy nuisance fingerprints.
- Treating each slot as a contrastive instance can instead be solved by fixed slot IDs and imposes an arbitrary slot correspondence. Random in-batch negatives can also be false negatives for tiles/views of the same underlying operator.
- Most importantly, a latent-only loss sends no direct gradient to the decoder/tails. When the reconstruction Jacobian into z is already weak, the auxiliary can dominate the encoder and train instance discrimination rather than useful reconstruction.

### Minimal anti-collapse regularizer worth considering

- Do not begin with InfoNCE, a trainable projector, or explicit negatives. Use a VICReg-style variance floor directly on `z[B,32,D]`, but remove both trivial sample-only and slot-only components first.
- Construct a two-way-centered interaction residual in words: subtract each sample's mean over its 32 slots; subtract each slot index's mean over the batch; add back the global mean. Any common vector, pure sample ID, pure slot ID, or additive `sample ID + slot ID` then becomes exactly zero.
- Normalize this residual by the global RMS of the original z so increasing latent norm cannot satisfy the constraint.
- Use hinge floors on two quantities: for every slot, require nonzero variation across matrices; for every matrix, require nonzero variation across its 32 slots. Above the floor the penalty is zero. It does not require orthogonal slots, a fixed target covariance, pairwise repulsion, or a particular latent basis, so the decoder retains rotational/coordinate freedom.
- This is adapted from the anti-collapse variance component of VICReg, not a claim that the full VICReg or Barlow Twins objective transfers directly. Full `D x D` covariance is statistically rank-limited and expensive when latent dimension greatly exceeds batch size.

### Essential interpretation boundary

- The regularizer can still be satisfied by a deterministic content hash or low-rank sample-slot interaction that is useless to reconstruction. It must not be called a fix merely because latent rank/cosine improves.
- The decoder-coupled evidence is the latent swap reconstruction matrix: decode latent j with the queries/context of target i and compare matched diagonal errors with nuisance-matched off-diagonal errors. A useful code requires the diagonal to be materially better. Direction and scale should be reported separately because the old scale path used z while direction did not.
- A stronger trainable alternative would contrast matched versus swapped reconstruction errors rather than latent vectors. That directly trains decoder use and leaves latent geometry free, but costs extra decoder passes, still permits training-set ID memorization, and requires a temperature/margin plus the absolute reconstruction objective. It is not selected automatically.

### Narrow recommendation

- The proposed anti-collapse idea is mechanistically relevant and worth a bounded intervention, but **not** as naive pooled InfoNCE and **not** with the claim that it alone repairs the old architecture.
- If implemented, use the two-way-centered variance hinge as the smallest direct anti-collapse term, keep the existing reconstruction objective, and judge success only by reconstruction improvement plus matched-versus-shuffled latent use. The old direction decoder's near-zero latent sensitivity at initialization remains an independent risk.

### Literature and review

- VICReg: https://arxiv.org/abs/2105.04906
- Barlow Twins: https://arxiv.org/abs/2103.03230
- Targeted design review: `/home/coder/project/artifacts/weightclip_benchmark/latent_anticollapse_regularizer_review_20260829/report.md`.

## 2026-08-29 — Production test of two-way latent anti-collapse; geometry preserved, direction use not rescued

### User decision and selected intervention

- User authorized a production test of latent anti-collapse mechanics in the old additive-query polar two-tail model.
- The primary run was fresh seed42/cursor0, not resumed from a collapsed checkpoint. Model, production bank, permutation stream, B32, optimizer, LR, clipping, normalization and the old behavioral-direction/scale plus structural-direction/scale objective were unchanged.
- The auxiliary acted on the exact deterministic decoder-input latent `z[B,32,384]`. In plain terms, it removed both sample-only and slot-only means, divided by a detached global latent RMS, and imposed margin `0.10` on cross-sample and within-sample interaction dispersion. Epsilon was `1e-6`; coefficient was `1.0`; calculations were FP32 over the full physical B32.
- Launcher/config/sidecar/tests were frozen and independently reviewed before launch. Production-faithful smoke completed 2/2 with exact 742,120,833 parameters, all groups live and zero auxiliary at healthy initialization. This established wiring and non-interference, not efficacy.

### Production run and artifacts

- Run root: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_500k_v1`.
- Comet: `https://www.comet.com/mike-5531/big-weight-vae/8411da20d6be4fb4a8a8490994df219e`.
- The run was stopped cleanly at step1,485 after the step1,000 causal probe answered the question. `STOPPED.json`, exact optimizer/RNG resume state and a model-only checkpoint were saved.
- Step1,000 probe: `latent_geometry_step_0001000_v1/report.json`; analyzer source: `/home/coder/project/projects/weight-vae/workspace/training/weightclip_benchmark/analyze_polar_latent_anticollapse_checkpoint.py`.
- Integrated review: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_500k_v1/analysis_review.md`.

### Experimental evidence

- The barrier did operate as designed, but only transiently. It was nonzero at logged steps30,40 and600. The worst logged cross/within minima were `.0923/.0690`; after each activation they rebounded above margin. At step1,000 the train-batch minima were `.328/.367`, and the auxiliary was zero.
- Step600 telemetry proves an actual active backward path: anti gradient RMS at z was `1.235e-7`, `2.03x` direction and `.56x` scale, with near-zero cosine to both. However, no same-step old-baseline latent-geometry probe exists, so an improvement in effective rank relative to baseline is not established.
- The unchanged legacy objective did not materially improve or degrade. Across 149 paired same-step rows through1,480, new mean was `3.08864`, old mean `3.09262`, paired mean delta `-0.00399` (-.13%) with paired RMS `.02731`. Both remained on the same roughly `3.0-3.2` plateau.
- The step1,000 latent was not globally constant: flattened cross-sample effective rank `5.65/31`, mean sample cosine `.432`, and sample×slot interaction RMS/total `.542`. But within-sample slots were still low-dimensional: median rank `1.77`, minimum `1.06`.
- The exact-layout swap probe was decisive. Swapping z only between equal `(tile_row,tile_col,d_in_mask,d_out_mask)` layouts changed weighted direction loss `1.816240 -> 1.816623`, only `+.021%`; it changed weighted scale loss `1.171985 -> 1.395728`, `+19.1%`.
- This is one aggregated B32 checkpoint probe, not a population theorem. It is nevertheless sufficient to diagnose the current checkpoint because the effect-size separation between direction and scale is about three orders of magnitude and is corroborated by direction rank/cosine.
- Predicted directions remained a common low-rank template: cross-sample cosine `.99590`, rank `2.06/16`, versus target cosine `.00219` and rank `15.86/16`. Mean prediction-target direction cosine was only `.0409`.
- At the exact decoder-input latent, scale/direction gradient RMS ratio was `95.7x` at step1,000 and `29.4x` at step1,400. Thus the direction credit path remained starved even while scale used z.
- The auxiliary caused one transient clipped spike when first active (preclip norm12.17 at step40), but no persistent instability.

### Supported conclusion

- The latent-only regularizer successfully preserved the dispersion statistic it targeted, so pure global latent collapse is not the whole cause.
- It did not fix the production failure because the direction decoder can ignore sample-specific latent information while scale uses it. A low-rank or decoder-irrelevant interaction can satisfy the hinge.
- The production test therefore supports the previously stated hostile counterexample rather than the hoped-for rescue: latent diversity is not equivalent to decoder-usable reconstruction information.
- Continuing to500k would have spent compute on an intervention that was zero and blind to the observed direction failure, so the run was stopped at1,485 with evidence preserved.

### Next framing

- Do not tune the latent margin or replace it with naive InfoNCE; those operate on the same wrong axis.
- Any next intervention must directly couple sample identity to decoded direction quality, or replace the direction decoder/objective so sample-specific information is required. The exact design is unresolved and should be derived from this branch-specific failure rather than layered on as another latent-only penalty.

## 2026-08-29 — Loss implication: target the collapsed direction predictions, not latent variance

### User inference

- User asked whether the anti-collapse pressure should be placed on the directional decoder-head predictions rather than only on the latent.

### Clarification

- Broadly yes, because the measured failure is now at the direction output: latent variation exists and scale uses it, while final direction predictions remain almost identical.
- However, a generic “make predictions diverse” term is still insufficient. It can reward arbitrary noise, and a pred-pred cosine/Gram penalty has a stationary collapsed point when all normalized predictions are equal.
- The current absolute direction loss already acts on predictions, but it only pulls each prediction toward its own target. It does not explicitly require prediction i to be more compatible with target i than with other same-layout target directions; the common-template solution can therefore remain an easy batch-averaged basin.

### Narrow next loss bet

- Remove the latent-only anti-collapse term rather than stack mechanisms.
- Keep the existing absolute direction objective.
- Add one same-layout prediction-to-target contrastive term on the final unit direction fields:
  - group samples with exactly matching layout/masks;
  - for each prediction i, score its mean componentwise cosine against every detached target-direction field j in that group;
  - use its own target as the positive and the others as negatives;
  - apply row-wise cross-entropy with one fixed temperature (initial proposal `.1`).
- In plain words: each decoded direction field must identify its own real weight-direction target among other structurally compatible targets. Identical direction predictions cannot solve these different labels, while arbitrary diversity that is unrelated to targets is not rewarded.

### Why latent-swap stays diagnostic rather than backward loss

- A matched-vs-swapped ranking loss directly tests z use, but it has a cheap gaming direction: make deliberately mismatched pairs worse, or encode an ID in z and compare it with activation-context identity, without improving matched reconstruction.
- It also requires a second decoder pass. Keep exact-layout swap as the causal evaluation: after output contrastive training, it tells whether improvement actually flows through z or only through activation context/query side information.

### Honest limit

- Output InfoNCE does not guarantee that the decoder will use z. It may learn the direction field through activation context if that path is expressive enough. If absolute/contrastive direction quality improves but the same-layout z-swap gap remains negligible, the remaining failure is architectural/routing rather than another loss-weight problem.
- This is a single coherent loss intervention aimed at the observed output collapse; it is not yet implemented or experimentally validated.

## 2026-08-29 — Production restart with retained latent barrier plus direction-target InfoNCE

### User decision

- User overrode the preceding suggestion to replace the latent auxiliary: retain the existing two-way latent anti-collapse loss and add the new loss on final direction predictions.
- Relaunch the old additive-query polar two-tail production model from the exact fresh seed42/cursor0 start. Once the launch is correct, do not stop it for an unfavorable diagnostic or early curve without an explicit user command.

### Exact intervention

- The old task objective is unchanged: behavioral direction + `10 *` behavioral log-scale + structural direction + `10 *` structural log-scale; operator loss remains disabled.
- The existing latent auxiliary is unchanged: FP32 two-way-centered interaction RMS hinge on the exact decoder-input `z[32,32,384]`, margin `.10`, coefficient `1`, detached denominator, full physical B32.
- One additional target-aware output loss is applied to the final unit direction predictions. Samples are grouped only when tile row, tile column, full input mask and full output mask all match. For each prediction, its own detached true direction field is the positive and the other same-layout true direction fields are negatives. Similarity is mean corresponding-patch cosine; row-wise cross-entropy uses temperature `.1`; group losses receive equal weight; coefficient is `1`.
- This loss directly penalizes the observed common direction template. It does not by itself prove use of `z`: activation context remains a possible path, so an exact-layout latent-swap probe remains the later discriminator. The swap intervention is diagnostic only and is not in backward.

### Verification and launch evidence

- Frozen hashes: launcher `005abeb54b78e...`, Comet sidecar `ff9add2c731b...`, config `cf3fe763af13...`, focused tests `e37562c93541...`.
- Focused suite passed `16/16`; Ruff, py_compile and config dry-run passed. An independent reviewer issued FORMAL GO for these exact files.
- Production-faithful B32 smoke: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_smoke_20260829T134057Z`. It completed 2/2 with 742,120,833 parameters, all parameter groups live and peak `14.76 GiB`.
- Smoke step1 preserved the old task value exactly (`3.586760044`). Latent auxiliary was zero at healthy initialization; direction InfoNCE was `.979880273`; total was `4.566640377` with zero decomposition error. Ten same-layout groups covered 27/32 samples. InfoNCE latent-gradient RMS was `1.84x` the old direction gradient and nearly orthogonal to scale; its scale-head gradient was exactly zero.
- Fresh production root: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_500k_v1`.
- Comet: `https://www.comet.com/mike-5531/big-weight-vae/c3d58de96a23405aa025473e228c033f`.
- Early live rows through step50 are finite and correctly decomposed. Same-layout eligibility varies from `.5625` to `.84375`; both auxiliaries are active when required. Trainer and Comet sidecar are live. By explicit user instruction, the correct run must remain running until the user directly requests a stop.

### Live component scale snapshot (step840 metrics / step800 gradient telemetry)

- The exact scalar objective is `behavioral_direction + 10*behavioral_scale + structural_direction + 10*structural_scale + latent_anticollapse + direction_InfoNCE`; operator loss is zero.
- At step840: behavioral direction `.92079`, weighted behavioral scale `.61819`, structural direction `.94367`, weighted structural scale `.29104`; legacy total `2.77368`, latent auxiliary `0`, InfoNCE `.86763`, total `3.64131`.
- The latest component-gradient checkpoint is step800. At the exact decoder-input latent, RMS gradients were: old direction `4.51e-8`, weighted scale `6.65e-7`, direction InfoNCE `6.64e-7`, latent anti-collapse `0`. Scale and InfoNCE were each about `14.7x` the old direction gradient. InfoNCE aligned strongly with old direction (`cosine .899`) and was only weakly opposed to scale (`cosine -.100`); old direction versus scale remained opposed (`cosine -.377`).
- Step800 global pre-clip norm was `4.357`; step840 was `3.065`, both below clip threshold `5`. Total-gradient group RMS at step800 included direction head `8.79e-3`, scale head `6.73e-2`, Distribution Encoder `6.74e-4`, bottleneck projections `9.51e-5`, encoder block1 `7.21e-5`, encoder block13 `1.64e-6`, shared decoder block1 `2.56e-5`, and block5 `8.86e-6`.
- Narrow interpretation: the new loss currently supplies the missing direction-side signal at the latent at approximately the same magnitude as scale and mostly in the desired direction. The old absolute-direction pull alone is still much weaker, and depth-wise gradient attenuation remains observable. This is a current snapshot, not yet a convergence result; the run remains live and untouched.

## 2026-08-29 — In-place LR transition from step1,235

### User decision

- User explicitly requested a graceful stop, an LR increase from `5e-5` to `3e-4`, and continuation of the same production run rather than a fresh start.
- The architecture, data stream, losses, optimizer family and all non-LR hyperparameters remain unchanged. The run must remain live unless the user explicitly asks to stop it.

### Preserved state and transition semantics

- The original process stopped cleanly at optimizer step `1,235`, committed logical cursor `39,520`.
- The rolling checkpoint preserves the exact model, Adam first/second moments and step counters, Python/NumPy/CPU/CUDA RNG states, and data cursor: `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_500k_v1/resume_latest.pt`.
- A plain YAML LR edit would not have worked because `optimizer.load_state_dict` restores the checkpoint LR. The launcher now validates the declared transition and changes only the loaded Adam parameter-group LR after optimizer-state restoration. It does not reset moments or replay from cursor0.
- Resume config: `/home/coder/project/projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_polar_tails_latent_anticollapse_direction_infonce_production_500k_lr3e4_resume.yaml`.
- The former stop marker was retained as `STOPPED.step_000001235.json`; the transition is recorded in `resolved_resume_config_step_000001235.json` and `resume_events.jsonl` under the existing run root.

### Verification and live evidence

- Focused production tests passed `18/18`; Ruff and py_compile passed. The corrected frozen implementation received an independent FORMAL GO. Relevant frozen hashes are launcher `765ae4bd...`, resume config `974d6f28...`, tests `3d941661...`, and unchanged Comet sidecar `ff9add2c...`.
- Startup reported `checkpoint_step=1235`, prior LR `5e-5`, effective LR `0.0003`, optimizer state entries `253`, start cursor `39,520`, and the unchanged 742,120,833-parameter model.
- The first new persisted row is step `1,240`, cursor `39,680`, with actual optimizer LR `3e-4`, finite total loss `3.873511`, legacy task loss `3.017489`, direction InfoNCE `.856021`, latent auxiliary `0`, and finite pre-clip gradient norm `6.7919`.
- Rows through step `1,290` continue monotonically at LR `3e-4`. Step1,290 had a finite pre-clip norm spike of `127.77`; global clipping remains `5`. This is an observed consequence/risk of the requested instantaneous 6x LR increase, not a resume discontinuity or crash.
- The trainer and the original Comet sidecar remain live on the same run and experiment: `https://www.comet.com/mike-5531/big-weight-vae/c3d58de96a23405aa025473e228c033f`.

## 2026-08-29 — Mechanistic audit after the in-place LR increase

### User question and run validity

- User asked whether the live state had degraded after continuing from step1,235 with LR `3e-4` and requested a mechanistic interpretation. The trainer was inspected read-only and remains live; no stop or state mutation was performed.
- Resume continuity is valid: the first new row is step1,240/cursor39,680; rows are ordered and unique with cursor=`step*32`; Adam/RNG/cursor were restored; all metrics remain finite and objective decomposition parity is exact.
- Evidence: `/home/coder/project/artifacts/weightclip_lr3e4_continuation_audit_step2450.md`, `/home/coder/project/artifacts/weightclip_benchmark/direction_infonce_step2000_mechanistic_audit_v1.md`, and the live run JSONLs under `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_500k_v1`.

### Scalar behavior: acute shock, partial recovery, no demonstrated gain

- Against a matched 300-step pre-transition window, the early post-transition legacy loss worsened about `4.0%`, InfoNCE `9.6%`, and weighted scale `9.0%`. Median pre-clip norm rose `3.45 -> 7.18`; points exceeding clip5 rose `8/30 -> 20/30`, with one maximum `127.77`.
- By steps2,060–2,350 most scalar shock had recovered: total remained about `1.0%` worse, legacy `0.36%` worse, InfoNCE `3.5%` worse and scale `2.3%` worse; absolute direction alone was about `0.8%` better. Thus there is no catastrophic divergence, but no evidence that LR `3e-4` improved the objective.
- InfoNCE group composition stayed matched across windows, excluding variable group size as the reason for its regression. Its top-1 remains only modestly above the correct variable-layout chance rate: excess was about `+14.4pp` immediately before the LR jump, dropped to `+8.9pp` just after it, and recovered only to about `+11.5pp` through step2,820.

### Found internal degradation: the direction credit path no longer reaches z

- The scalar recovery hides a sharp internal breakpoint between telemetry steps1,300 and1,400. Direction-gradient RMS at the exact decoder-input latent fell `2.39e-7 -> 5.90e-10` and stayed mostly below `1.2e-9` through step2,400.
- Geometric means before versus late after transition: absolute-direction gradient at z fell about `503x`, direction-InfoNCE gradient about `166x`, and scale gradient about `14x`. The scale/direction ratio grew from roughly `4.9` to `116.6`.
- Head gradients remain nonzero while upstream direction gradients vanish. This is not global numerical death: every logged parameter group remains finite/live, and the scale/context paths still train. It localizes the failure to the direction head-to-latent credit path.

### Mechanism supported by the step2,000 checkpoint

- The direction head has an exact radial gauge: it emits 16 raw logits and only their unit-normalized direction enters every direction/InfoNCE loss. Scaling those raw logits changes neither loss but divides the upstream normalization Jacobian approximately by the radius.
- From checkpoint1,235 to2,000, the direction-head weight norm grew `2.65x`, while representative trunk/projection norms changed only about `0.5–6.6%`. At step2,000 the raw direction-logit norm median is `11.267` (p05 `9.73`, p95 `13.95`), approximately the same pathological radius as the failed old polar model at step7,000 (`11.645`) and about `130x` the initialization median (`.0864`). This directly explains why direction loss can stay near `.94` while almost no gradient reaches z.
- Exact-layout causal swaps show where the surviving sample-specific signal goes. Swapping z changes direction loss by `-0.024%` and unit directions by only `.00893`: direction practically ignores z. Swapping activation Distribution-Encoder context changes direction loss by `+3.345%` and unit directions by `.4894`, about `55x` the z-swap effect. The output contrastive objective escaped through the direct activation/query route rather than teaching the weight latent to carry direction.
- Direction outputs are now more varied than in the anti-only failure (cross-sample cosine `.280` versus `.996`), so InfoNCE did break the common-template symptom. But this is not accurate content: matched target cosine is only `.014`, predicted effective rank `1.73/16` versus target `15.93/16`, and own-minus-hardest-negative similarity is negative on the B7 probe.
- The latent hinge also misses the remaining geometry failure. At step2,000 its interaction ratio and floor statistics are healthy and the auxiliary is zero, yet within-sample 32-slot effective rank has median `1.011`. A nearly rank-one sample-by-slot interaction can therefore satisfy the two-way RMS barrier.
- Scale still uses z somewhat, but weakly: exact-layout z swap worsens scale `2.72%`, versus `19.1%` in the earlier anti-only step1,000 probe; activation-context swap worsens it `11.32%`.

### Narrow conclusion and caveat

- There is real internal degradation despite non-divergent scalar loss: the run has moved into a large-radius, direction-gradient-starved solution. InfoNCE mostly buys output diversity through activation context, while the fixed weight latent remains nearly irrelevant to direction; the latent anti-collapse hinge is satisfied by low-rank slot geometry and therefore stays inactive.
- The immediate shock follows the isolated 6x LR change, but strict attribution of the persistent mechanism specifically to LR is not established because there is no parallel continuation of checkpoint1,235 at `5e-5`, and related radial/credit collapse existed in earlier polar runs. The supported statement is that LR `3e-4` did not cure the mechanism and coincided with a sharp re-entry into it.
- The production run remains live by the user's explicit instruction not to stop a correctly running job without a direct command.

## 2026-08-29 — Removing the direction-head radial gauge without another loss coefficient

### User question and terminology

- User asked how to eliminate the observed norm inflation “before RMS.” In the exact source, the measured inflation is in the raw 16D direction logits before their final per-patch L2 normalization, not in the final RMSNorm itself. The preceding `direction_output_norm` is an affine RMSNorm and participates in the same factorization.
- Current chain in plain text: `state -> affine RMSNorm gain gamma -> linear head W -> raw logits a -> unit direction a/||a||`.

### Important causal correction

- A pure global rescaling `W -> cW` is erased by the final direction normalization. Its factor also cancels from the derivative back to the head input because the numerator contains `W` while the normalization Jacobian contributes `1/||Wh||`. Therefore head Frobenius growth alone does **not** explain the 100–500x loss of direction gradient at z.
- What is established jointly is more specific: checkpoint1,235→3,000 composite head Frobenius grew `3.82x`, spectral norm `5.64x`, and stable rank fell `3.71 -> 1.70`; the model also routes direction mainly through activation context. Spectral concentration and bypass can kill useful tangent directions even though a pure scalar factor would cancel.

### Minimal exact gauge fix

- Fold the affine RMSNorm gain into one effective matrix once: `B = W * gamma[None,:]`.
- Replace the learned direction-output RMSNorm with non-affine RMSNorm (`gamma` fixed exactly to one). This removes the exact per-channel `W/gamma` compensation lane without losing the composite linear function.
- Constrain the single effective matrix `B` to a fixed Frobenius sphere, with no learned magnitude `g`. After an optimizer update, retract `B` to the chosen radius, exclude it from decoupled weight decay, and project its first-moment update into the sphere tangent. This is a hard gauge choice rather than a soft norm penalty.
- Folding gamma and positive global renormalization preserve every decoded direction exactly because the final unit normalization cancels the common scalar. Thus the minimal Frobenius-sphere version preserves the existing direction-function class and loss while preventing arbitrary secular head-radius growth.

### Why the tempting alternatives were rejected

- Another RMSNorm or standard weight normalization with a learned gain only relocates/reintroduces the same radius.
- A soft norm penalty introduces a coefficient, leaves the symmetry in the model, and allows scale to migrate into RMSNorm gain or the tail.
- Direct raw-unit MSE is not a clean replacement. For a wrong predicted direction its optimal raw radius is approximately `max(cosine,0)`, not one; with current cosine near zero it initially rewards shrinking logits toward zero, where the normalized behavioral/InfoNCE paths become ill-conditioned.
- Hyperspherical-angle charts introduce poles, periodicity and anisotropic optimizer geometry. A custom stop-gradient/backward rule ceases to be the gradient of the stated forward objective.

### Remaining risk and honest scope

- Fixed Frobenius removes the exact global radial gauge but does not prevent singular-value concentration at fixed norm, nor the activation-context shortcut, nor possible growth of the pre-RMS residual state. A Stiefel/row-orthogonal head would prevent spectral collapse but restricts the head spectrum/function class and is a materially stronger intervention; it is not the first minimal fix.
- Required telemetry for any implementation: pre-RMS state RMS; raw-logit norm percentiles; effective-head Frobenius, spectral norm and stable rank; gradients at logits, before/after final normalization, direction tail and z; exact-layout z/context swaps. If head radius is fixed but stable rank or z-sensitivity still collapses, the radial gauge is exonerated and the remaining target is tail/routing, not another norm tweak.
- Applying this to a live Adam checkpoint is not a semantics-preserving optimizer continuation: the forward can be converted exactly, but Adam moments live in the old factored coordinates. A clean causal run should start fresh, or explicitly reset/transport only the converted terminal-head state and declare that intervention.

## 2026-08-29 — Gauge-fixed direction head implemented at the original LR

### User decision

- Add the minimal direction-head gauge fix and return the new fresh production arm to the original constant LR `5e-5`.
- The existing LR-`3e-4` production continuation remains live and must not be stopped without a separate explicit user command.

### Exact implementation

- The old affine direction-output RMSNorm gain is folded into the direction-head matrix at seeded construction; the direction RMSNorm is then non-affine. The scale branch is unchanged.
- The stored target radius is the exact seed-42 composite-head Frobenius norm, `0.0806206539`, rather than a tuned constant.
- The actual trainable `direction_head.weight` is retracted to that sphere after every AdamW step and before any checkpoint/save. Its Adam first moment is projected tangent after retraction; the second moment and step counter are preserved.
- The direction head is the sole member of a named AdamW group with weight decay zero. All other trainable parameters remain in the ordinary weight-decay `.01` group. Both groups use the original LR `5e-5`, betas and epsilon.
- Existing final per-p16 unit normalization, direction/scale routing, old behavioral+structural task loss, latent anti-collapse hinge and same-layout direction InfoNCE are unchanged. Parameter count decreases by exactly 1,536 to `742,119,297` because the affine direction RMS gain is removed.
- Flat telemetry now records target/actual Frobenius radius and error, radial first-moment fraction, spectral norm, stable rank, row norms and valid raw-logit norm percentiles in both training and gradient artifacts.

### Verification evidence

- Focused gauge tests passed `8/8`; the complete production test file passed `26/26`; Ruff, py_compile, config dry-run and an in-memory two-group checkpoint/optimizer resume roundtrip passed.
- Production-faithful B32 smoke: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_gauge_fixed_smoke_20260829T1305Z`.
- Smoke completed 2/2, cursor64, all 27 logged parameter groups finite/live, peak `14.76 GiB`. Step1 preserved the exact old task start (`legacy=3.586760044`); InfoNCE `.979880154`, latent auxiliary zero, total `4.566640377`.
- After the first real Adam step, head norm was `.0806206465`, relative radius error `9.24e-8`, and radial first-moment fraction `9.60e-9`. Valid raw-logit norm median remained `.08616` (p01 `.07169`), so the constraint did not create a near-zero-logit singularity at startup.
- Frozen quartet hashes: launcher `56b4563173ec...`, config `74ae7a0c825e...`, sidecar `23c97e38dcc8...`, tests `9894a48e9693...`. Final independent production GO is pending at the time of this entry.

### Scientific scope

- This intervention removes the exact affine-factorization/global-radius gauge while preserving the initial direction function. It is not claimed to solve fixed-radius spectral/rank collapse, pre-RMS state growth or activation-context bypass.
- The decisive later evidence is therefore trajectory telemetry plus the same-layout `z`/context swap probe. If radius stays fixed but head stable rank and latent direction sensitivity still collapse, the radial-gauge hypothesis is falsified and the remaining mechanism lies upstream/routing-side.

## 2026-08-29 — Gauge-fixed 500k production launch and first 100-step evidence

### Launch status

- The independent reviewer issued FORMAL GO for the exact frozen quartet and smoke. No implementation or launch blocker remained.
- Fresh trainer PID `820670` and Comet sidecar PID `821337` are live. Comet: `https://www.comet.com/mike-5531/big-weight-vae/28e6b04c5eb44941b232754e775185a3`.
- Production root: `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_latent_anticollapse_direction_infonce_gauge_fixed_500k_v1`.
- The prior LR-`3e-4` trainer and its sidecar remain live and untouched, per the user's explicit no-stop instruction.
- First rows validate the operational contract: step1/cursor32, LR exactly `5e-5`, finite loss and gradients, exact decomposition, and all gauge invariants. Step1 total/legacy/InfoNCE are `4.566640377 / 3.586760044 / .979880154`, matching the fair old start.

### Early mechanistic result: the global-radius fix is already insufficient

- The hard constraint works numerically: through step100, Frobenius relative error stays below about `1e-7` and the first-moment radial fraction below about `5e-9`.
- Despite fixed total norm, direction-head stable rank fell `13.478 -> 4.049` by step100 while spectral norm rose `.02196 -> .04006`. Raw-logit median grew `.0862 -> 1.328` (p01 `.0717 -> 1.205`). This demonstrates immediate scale migration into spectral concentration and/or the pre-head state rather than violation of the Frobenius constraint.
- Same-step comparison to the otherwise matched prior InfoNCE arm at step100 is adverse for the intended credit path. Direction-gradient RMS at the decoder-input latent is `1.65e-7` versus old `7.92e-7` (`4.8x` lower); InfoNCE latent gradient `1.45e-7` versus `5.28e-7` (`3.6x` lower); encoder-block1 group RMS `3.03e-5` versus `3.39e-4` (`11.2x` lower); Distribution Encoder `1.87e-4` versus `9.57e-4` (`5.1x` lower).
- Scale gradient at the latent remains `4.08e-7`, so it is already `2.47x` the absolute-direction gradient. InfoNCE and scale are strongly opposed at this checkpoint (`cosine -0.527`).
- Scalar loss is not divergent and happens to be lower on the step100 batch (`3.8037` vs old `3.9271`), but this does not override the mechanistic evidence: the requested global gauge was fixed exactly while the relevant direction credit became weaker through another available geometry.

### Narrow interpretation and action

- This is strong early evidence against the hypothesis that unconstrained global head radius was the root cause. It was a real symmetry, but removing it does not prevent the model from concentrating the fixed norm and increasing raw logits through upstream state geometry.
- It is not yet a long-horizon convergence conclusion. The new run remains live, as instructed; no automatic stop was issued. Later telemetry and the scheduled same-layout `z`/context swap remain necessary to localize fixed-radius spectral concentration versus upstream state inflation/bypass.

## 2026-08-29 — Why fixed-radius logits still inflate, where direction credit weakens, and the latent-loss norm loophole

### User observation about the latent auxiliary

- User correctly observed that the current latent anti-collapse loss does not keep the latent norm small because it subtracts means before measuring diversity.
- Exact decomposition: a latent can contain a global component, a sample-only component shared by all slots, a slot-only component shared by all samples, and a true sample-by-slot interaction. Two-way centering removes the first three and constrains only the interaction.
- The denominator is the RMS of the full latent but is detached. Therefore growth of an omitted global/sample/slot component can activate the hinge, but the auxiliary has no gradient that shrinks that component. It responds only by increasing the interaction component until the ratio clears the `.10` floor. The loss is an anti-collapse floor, not a norm prior or upper bound.
- This creates a plausible ratchet: task gradients can grow common latent components; the hinge then grows interaction variance to keep up, while neither term restores the total radius. In the gauge run latent RMS already grows from `.782` at step1 to about `2.06` at step200. Causality is not yet isolated, but the norm loophole is exact from the objective.

### Established head geometry through step200

- Read-only reports: `/home/coder/project/artifacts/weightclip_benchmark/gauge_fixed_early_mechanistic_probe_step200_v1.md`, `/home/coder/project/artifacts/weightclip_gauge_fixed_vs_old_infonce_causal_trajectory_step200_20260829.md`, and independent review `/home/coder/project/docs/notes/gauge_fixed_direction_causal_review_20260829.md` with exact CSV alongside.
- The head Frobenius constraint and tangent first moment remain exact, excluding a broken implementation. Nevertheless spectral norm rises `.02196 -> .04006 -> .05151` and stable rank falls `13.48 -> 4.05 -> 2.45` at steps1/100/200. Top-mode squared-energy share rises `7.4% -> 24.7% -> 40.8%`; row norms remain live, so this is correlated singular concentration rather than dead rows.
- Non-affine RMSNorm bounds post-RMS hidden norm by `sqrt(1536)`. Median raw-logit use of the rigorous `spectral_norm * sqrt(1536)` bound rises `.099 -> .846` by step100. Thus raw-logit inflation is not caused by post-RMS hidden magnitude; normalized hidden directions co-adapt with the head's high-gain right-singular subspace.

### Leading proximal mechanism and gradient localization

- For a raw logit vector followed by unit normalization, backward first removes the component parallel to the predicted direction and divides the remaining tangent gradient by raw-logit norm. It then multiplies by the transpose of the head matrix.
- If normalized hidden states and logits align with the dominant singular pair, the unit-normalization projector deletes the dominant left-singular component. The transpose can return credit only through the smaller remaining singular modes. This explains why the head weights can still receive material gradients while the hidden/tail path receives little useful angular signal. Uniform head scaling alone cannot do this because its scale cancels in forward normalization/backward; anisotropy plus alignment is required.
- Narrow established boundary: the first extra attenuation is already inside the direction-exclusive readout/tail subsystem, before the shared encoder. At matched step100, gauge/old absolute-direction head gradient is `1.12x`, but tail-Q is `.19x`; InfoNCE head is `.79x`, but tail-Q `.049x`. Shared decoder-5 QKV is `.18x/.125x` and latent activation `.209x/.276x` for absolute direction/InfoNCE. Scale conflict later amplifies the imbalance; it is not the sole origin because the private direction tail is already affected.
- Exact operation-level attribution between final unit normalization, affine-free RMSNorm backward, and tail attention is not yet established because the run does not log activation gradients at raw logits/direction hidden. The leading spectral-alignment mechanism is strongly supported; pre-RMS state inflation, per-example batch cancellation and tail-attention degeneration remain viable additions until the step1000 checkpoint probe.

### Connection between the two findings

- The growing latent norm cannot directly explain post-RMS raw-logit inflation, because RMSNorm erases forward magnitude. It can still worsen backward conditioning: RMSNorm's backward Jacobian contains an inverse pre-RMS radius. If large latent/common components propagate into the direction-tail state, they provide a second attenuation factor before the head geometry.
- The step1000 read-only probe must therefore decompose latent norm into global/sample/slot/interaction components and measure pre-RMS state RMS plus gradients at unit predictions, raw logits, post-RMS hidden, pre-RMS tail state, shared state and `z`. This is the first probe capable of separating the exact latent-radius loophole from the already established spectral/angular bottleneck.

## 2026-08-29 — Exact initialization boundary for behavioral-direction credit

### Exact evidence

- A deterministic B32 replay of the gauge-fixed run measured activation VJPs of `behavioral_direction` alone. Step-1 losses matched live exactly; pre-clip total gradient norm differed by `7.63e-6`, within the `1e-5` gate, and behavioral direction matched exactly at `.9676770568`.
- Backward activation L2 norms were: predicted directions `.224277`, raw logits `2.668913`, post-RMS hidden `.0536544`, pre-RMS direction state `.114914`, direction-tail output `.114914`, final shared state `.155017`, and latent `z` `.00196436`.
- The exact initialization contractions are therefore raw logits to post-RMS hidden (`49.7x`) and, more strongly, final shared state to `z` (`78.9x`). Unit normalization amplifies rather than kills the init tangent gradient (`11.9x`); RMSNorm/private-tail transport is live. Total predicted-direction-to-`z` attenuation is `114x`.
- Direction-tail and final-shared attention are high-entropy at init (normalized entropy `.9971/.9957`, max probability `.00456/.00832`) with nonzero token-query mass to latent keys (`.0524/.0881`). This excludes softmax saturation at init, but not later latent bypass.

### Step-10 evidence limit

- The step-10 replay failed the full scalar parity gate despite a close behavioral scalar (`.93822` replay versus `.93128` live); maximum full-gate discrepancy was `.9666`. The extra step-1 VJP through activation checkpointing perturbed the later optimizer trajectory even after RNG restoration. Per the agreed precommit, no more retries were launched and no numerical step-10 VJP is accepted.
- Exact live forward geometry still establishes early angular attenuation: median raw-logit radius grew `.08616 -> .96544`, head spectral norm `.02215 -> .03096`, and stable rank fell `13.25 -> 6.78`. The scale `spectral_norm / raw_radius` fell about `8x`, so the normalized-readout angular Jacobian is already substantially weaker by step10.
- Invalid replays consistently suggested the same two weak intervals, especially shared state to `z`, but this remains a leading hypothesis only. Exact step-10 localization requires either a saved checkpoint or a clean replay with no probe before step10 and full scalar parity within `1e-5`.
- Full report: `/home/coder/project/artifacts/weightclip_benchmark/gauge_fixed_behavioral_direction_replay_step10_v1/final_boundary_report.md`.

## 2026-08-29 — Correction and exact decomposition of the apparent 79x shared-to-z attenuation

- An exact init-only B32 behavioral-direction VJP split passed live scalar parity exactly. The previously cited `norm(grad final_shared) / norm(grad z) = 78.9` is not a 79x contraction through the shared decoder: it compares the full 544x1536 shared tensor with the 32x384 latent code.
- Backward through the five shared blocks increases total gradient norm from `.155017` at block5 output to `.219533` at shared input (`1.416x`). Latent-slice norm also increases backward from `.000696` to `.004990` (`7.17x`). No shared block is an init gradient sink.
- At shared input, query512 norm is `.219476` versus latent32 `.004990`, a `43.98x` ratio. Token-count dimensionality explains `4x`; query per-coordinate RMS is still `10.99x` larger. Query tokens contain `99.948%` of shared-input gradient energy, latent tokens only `.052%`.
- Source attribution: the 512 query tokens are independently built from output/tile queries plus Distribution Encoder context, then preserved by ordinary residual identities. They can carry behavioral credit directly to the query conditioner without `z`. Only diffuse self-attention transfers query credit into the 32 latent positions. Init attention is already high entropy and assigns latent keys only about `7--10%` of token-query mass, so saturation is not the cause.
- `from_latent^T` reduces total norm `.004990 -> .001964` (`2.5405x`), but this exactly matches seeded random-projection algebra: observed reverse ratio `.39362` versus `sqrt(384) * weight_RMS(.0200539) = .39298`. This is ordinary width/scale behavior, not anomalous conditioning.
- Narrow causal conclusion: the init weakness is a topological query residual bypass plus diffuse latent coupling, not depth-wise gradient vanishing. Report and raw evidence: `/home/coder/project/artifacts/weightclip_benchmark/gauge_fixed_behavioral_direction_replay_step10_v1/init_shared_to_z_causal_report.md` and `init_shared_to_z_probe.json`.

## 2026-08-29 — Categorical GPTQ decoder design and hostile review

- Recommended categorical representation is one transformer token per ordered p16 code vector, with 16 independent 15-way logits per token. Literal scalar tokens would increase sequence length `1024 -> 16384` and dense-attention work about `256x`; one categorical ID for a whole p16 would require `15^16` classes and is VQ if replaced by a finite learned codebook.
- Freeze the current symmetric GPTQ transform for a causal comparison: codes `-7..7`, one scale per 128-weight output group, no zero point. Production shapes are code targets `[B,128,8,16]`, logits `[B,128,8,16,15]`, and scales `[B,128,1]`.
- Minimal full AE: 1024 p16 code tokens plus 128 non-repeated group-scale metadata tokens enter the same encoder and fixed `32x384` bottleneck. Keep production DE; repeat each p32 context to its two p16 children, with zero aligned context for scale tokens.
- Remove both polar tails, unit direction normalization, polar scale bounding and gauge machinery. Use one six-block decoder whose first read is `Q=conditioned coordinate queries, K/V=from_latent(z)` with no query-value residual; remaining blocks operate on the resulting latent-rooted state. Code and scale observation heads sit on the same final state.
- Minimal objective is GPTQ-representation likelihood only: mean scalar code CE plus `0.5 * MSE` on standardized log2 group scale. Old behavioral/structural/InfoNCE/anti-collapse objectives stay out of the first causal arm; their metrics are evaluation-only.
- The change genuinely removes the per-vector unit-normalization `1/r` tangent attenuation and polar direction/scale heads. It does not solve latent use if current residual queries remain: exact init evidence says that path owns `99.948%` of shared-input behavioral gradient energy. Ground-truth scale at decoder input would be another forbidden side channel.
- Main second-order risks: majority/near-zero code priors, softmax margin saturation, plain CE ignoring ordinal code distance and GPTQ Hessian/error-feedback coupling, residual activation-context modulation even in a rooted read, and about `5x` denser encoder attention from 1152 versus 512 data tokens.
- Full design/review: `/home/coder/project/artifacts/weightclip_benchmark/categorical_gptq_decoder_design_review_20260829.md`.

## 2026-08-29 — Integrated categorical-GPTQ reset recommendation

### User proposal and semantic correction

- The user proposed replacing the polar continuous reconstruction head and its
  regression losses with ordinary categorical prediction of an offline GPTQ
  discretization, while charging nearby-bin mistakes less than distant-bin
  mistakes.
- GPTQ itself is an offline activation-aware assignment/compensation algorithm,
  not a categorical training loss.  The frozen project codec produces symmetric
  scalar codes `-7..7`, one scale per 128-weight output group and fixed zero
  point zero.  A p16 patch therefore contains 16 ordered 15-way symbols; one
  flat class for the entire p16 would require `15^16` classes and would instead
  be a VQ/RVQ design.
- The earlier `gptq_token` experiment discretized only the encoder input and
  still regressed floating-point weights at the output.  Its negative result
  does not test the newly proposed categorical decoder.  Its fixed GPTQ
  dequantizer did establish a good representation floor (held-out action NRMSE
  `.07997`).

### Recommended single model, not an experiment panel

- Keep the normalized continuous p32 encoder, production Distribution Encoder,
  13 encoder blocks and fixed `z[32,384]`.  Activation-distribution conditioning
  is encoder-side only, so all sample-specific activation information must pass
  through `z`.
- Use persistent coordinate queries as address/symmetry-breaking state, but do
  not inject Distribution-Encoder context directly into decoder residual values.
  Decoder blocks use residual cross-attention from those queries to `z`; this
  avoids both established bad extremes: the old sample-specific query bypass
  and the failed no-query-residual latent-rooted uniform-averaging graph.
- Preserve p32 compute states and split them into ordered p16 children.  Replace
  the two private polar tails by one p16 categorical tail.  The scalar-code head
  is `1536 -> 16*15`, reshaped to
  `[B,d_out,ceil(d_in/16),16,15]`, and is evaluated in parallel without
  autoregressive teacher forcing.
- Predict decode-critical scale metadata rather than passing ground-truth scale
  to the decoder.  Use one ordered categorical log2-scale symbol per
  `(output,input-group-128)`, with a fixed train-only global binning and explicit
  overflow bins.  The current symmetric codec needs no zero-point head.  A
  deterministic argmax decoder reconstructs `W_hat = scale_center * signed_code`.
- Remove from backward both polar heads/tails, final unit normalization, bounded
  continuous log-scale, Frobenius gauge machinery, behavioral/structural
  direction/scale losses, direction InfoNCE and latent anti-collapse.  Retain
  dequantized raw/action, old structural metrics and latent/context swaps as
  detached diagnostics.  Removing the duplicate private tail naturally returns
  the model to roughly the 707M class before any decoder-attention redistribution.

### Distance-aware categorical objective

- Use ordinary logits and softmax probabilities, but train them with a weighted
  cumulative/ordinal log-loss.  For target class `y`, threshold `r` predicts
  whether the code is at or below `r`; its BCE weight is
  `abs((r+1-y)^2 - (r-y)^2) = abs(2*(r-y)+1)`.  A point prediction displaced by
  `d` bins crosses thresholds whose total discrete cost is `d^2`, so a neighbor
  is cheaper than a distant code without `CE + lambda*MSE`, a smoothing
  temperature, or a blurred non-one-hot optimum.
- Compute cumulative log probabilities in FP32 with prefix/suffix `logsumexp`,
  not BF16 `softmax+cumsum`.  Normalize threshold weights once with a frozen
  dataset-wide constant, never per example.  Apply the same ordered loss family
  to the log-scale classes; report code and scale components separately.
- The first loss uses normalized lattice distance `(k-y)^2`.  Exact GPTQ local
  Hessian costs are retained as metrics, not backward weights: GPTQ/Hessian
  information already changes the teacher code, while adding heavy-tailed
  `scale^2/Hinv_diag` weights risks recreating the previous gradient-dominance
  failure.

### What this intervention does and does not establish

- It directly removes the observed polar pathologies: the unit-vector tangent
  projector, inverse raw-logit-radius Jacobian, direction-head radial gauge and
  the multiplicative direction/scale reconstruction parameterization.
- It does not make bottleneck use automatic.  Coordinate queries can still learn
  position-dependent majority-code priors, especially because central GPTQ codes
  are frequent.  Success therefore requires improvement over the frozen
  per-position marginal baseline plus matched-vs-shuffled/zero-`z` degradation;
  raw accuracy alone is invalid evidence.
- Required diagnostics are code NLL relative to the marginal prior, balanced
  per-code recall, off-by-one and mean squared bin distance, scale-bin error,
  dequantized raw/action NRMSE, code entropy, and separate `z`/activation-context
  swaps.  Autoregressive teacher forcing is rejected because previous true codes
  would become a direct output side channel and introduce exposure error.
- A literal one-ID-per-p16 design remains a separate VQ/RVQ decision.  It must
  match the GPTQ bitrate with multiple code stages and use codebook-vector
  distances, not numeric ID distance; it is not bundled into this reset.

## 2026-08-29 — Exact initialization boundary for behavioral-direction credit; step-10 state was not retained

### User question and run action

- User asked for the concrete operation at which the pure behavioral-direction gradient is lost at initialization and after roughly ten updates, rather than the later step100/200 parameter-gradient localization.
- User explicitly authorized stopping the current gauge-fixed production run. It was stopped gracefully at step627/cursor20,064. `STOPPED.json` and the full 8.3-GiB resume checkpoint were written under the existing run roots; the gauge Comet sidecar was also stopped. The older LR-`3e-4` run was not touched.

### Exact behavioral-only activation VJP at initialization

- A deterministic B32 replay matched the live step1 scalar trajectory to max absolute error `7.63e-6`. It isolated `behavioral_direction` alone, with scale detached and without structural direction.
- Valid-coordinate gradient RMS along the backward path was: final unit direction `3.876e-4`; masked raw logits `4.612e-3`; post-RMS hidden `9.463e-6`; pre-RMS tail state `2.027e-5`; direction-tail output `1.979e-5`; final shared-decoder state `3.690e-5`; bottleneck `z` `3.133e-6`.
- Therefore the largest exact initialization contraction is the transpose of the tiny initialized 16x1536 direction head: raw-logit gradient to post-RMS hidden drops about `487x`. The final unit normalization is not killing the initial gradient; because raw-logit norm is only about `.086`, it amplifies the tangent gradient from unit prediction to raw logits by about `11.9x`. RMSNorm then amplifies by about `2.14x`, not attenuates, on this batch. A second aggregate contraction of about `11.8x` exists from final shared-decoder state back to `z`, but this spans the complete shared decoder/read path rather than one measured operation.
- Tail and shared-decoder attention are high-entropy at initialization (normalized entropy `.9971` and `.9957`), so softmax saturation is excluded for this exact batch. Near-uniform routing may still be weakly informative, but it is not saturated one-hot attention.

### What is and is not known at step10

- The original production run retained only scalar/geometry telemetry at step10; objective-separated gradients were logged at steps1,100,... and the first checkpoint was scheduled at step1000. There is no exact step10 state to inspect.
- Exact live facts by step10: behavioral direction is still unsolved (`.9677 -> .9313`), raw-logit median rises `.0862 -> .9654` (`11.2x`), and head stable rank falls `13.25 -> 6.78`. Thus the final unit-normalization tangent multiplier loses approximately the same `11.2x` gain, while the head becomes more anisotropic, within ten updates.
- Several short replays reproduced step1 exactly but failed the full step10 scalar/gradient parity gate because the original step1 production telemetry performs additional checkpointed autograd passes. Their step10 VJPs are retained only as an invalid diagnostic, not evidence. They consistently suggested a new large shared-decoder-to-`z` contraction, but this must be called a leading hypothesis, not an established live-run boundary.
- Canonical replay source/report: `/home/coder/project/artifacts/weightclip_benchmark/gauge_fixed_behavioral_direction_replay_step10_v1/replay_probe.py` and `report.json`; `report.json.validation.passed=false` explicitly prevents accidental use as exact step10 evidence. Archive-only localization report: `/home/coder/project/artifacts/weightclip_gauge_fixed_early_direction_gradient_localization_20260829.md`.

### Narrow conclusion

- Exact at initialization: the primary behavioral-direction credit bottleneck is `raw direction logits -> direction_head^T -> post-RMS hidden`, not softmax saturation and not the final unit normalization.
- Exact by step10: terminal conditioning has already degraded sharply through raw-radius growth and spectral-rank loss, but the precise additional operation where credit to `z` disappears is not recoverable from the retained live artifacts. Naming the shared decoder, RMSNorm, or attention as the exact step10 edge would overclaim.

### Metric clarification for the initialization boundary

- Two preceding summaries used different, both useful notions of “gradient death.” Across tensors of different width, total L2 norm measures total returned gradient energy, while RMS measures the update-sized signal per activation coordinate.
- By total L2 norm, the largest measured interval is final shared state to `z` (`78.9x`), followed by raw logits through `direction_head^T` to post-RMS hidden (`49.7x`).
- By per-coordinate RMS, the head transpose is much harsher (`487x`) because it spreads 16 output-coordinate gradients over 1,536 hidden coordinates; shared state to `z` is `11.8x`. This is not contradictory. For optimization of the upstream wide trunk, the per-coordinate head contraction matters; for total credit reaching the fixed bottleneck, the shared-decoder-to-`z` energy contraction matters.
- The user-facing conclusion must therefore name both bottlenecks and state the metric rather than selecting one without qualification.

## 2026-08-29 — Corrected cause of the apparent shared-decoder-to-z gradient loss

### Exact latent/query split

- A new exact-init B32 behavioral-only VJP split the 544-token shared decoder state into the 32 tokens actually produced from `z` and the 512 z-independent output queries. Artifact: `/home/coder/project/artifacts/weightclip_benchmark/gauge_fixed_behavioral_direction_replay_step10_v1/init_shared_to_z_probe.json`; source alongside as `init_shared_to_z_probe.py`. Behavioral scalar matches live step1 exactly (`.9676770568`).
- At the input of shared decoder block1, total gradient norm is `.219533`, but `.219476` is on the 512 query tokens and only `.004990` on the 32 latent tokens. The query slice has `43.98x` more total norm and `10.99x` more per-coordinate RMS; more than `99.94%` of squared gradient energy is on the z-independent query slice.
- The `from_latent` projection itself is not a severe bottleneck: its output gradient norm `.004990` becomes z-gradient norm `.001964` (`2.54x` total contraction), while per-coordinate RMS changes only `3.979e-6 -> 3.133e-6` (`1.27x`).
- The five shared residual blocks do not exhibit vanishing depth. Full-state gradient norm is `.219533` at their input, `.198752` after block1 and `.155017` after block5; backwards, block input receives more rather than less total gradient. The prior `79x` final-shared-total/z comparison was mostly invalid attribution because it counted gradients on 512 query states that are not descendants of z.

### Supported mechanism

- Exact graph: `z -> from_latent -> 32 latent tokens`; separately, 512 learned address/tile queries are conditioned directly by Distribution-Encoder activation context; both sets are concatenated and processed by five ordinary residual self-attention blocks. Only query/child tokens are read by the output head.
- The 512 query tokens therefore have an exact autonomous carrier and identity-gradient highway around z. At exact init, final shared attention gives query tokens only `.0881` mass to latent keys and about `.912` to non-latent keys; the direction tail gives only `.0524` to latent keys. Attention entropy is near one, so this is not saturation: it is token-cardinality dilution plus a strong query residual bypass.
- Causal intervention corroborates the graph. At seeded init, rolling activation context changes predicted direction by `.1969`, while rolling z changes it by only `.000440` (`448x` smaller); output relative-RMS changes are `.6265` versus `.0296` (`21.2x`). Thus direction output is almost z-independent before optimization starts.
- Correct conclusion: gradient is not progressively killed by five decoder blocks or mainly by `from_latent`. Most behavioral credit never enters the z-owned slice: it follows the shorter learned-query/activation-context residual route. The apparent shared-state-to-z death is a credit-routing/ownership failure built into the decoder topology.

## 2026-08-29 — User clarification: categorical GPTQ decoder is autoregressive

- The user clarified that the intended categorical Transformer is explicitly
  causal/autoregressive with teacher forcing, not the parallel factorized writer
  assumed in the preceding recommendation.
- The coherent interpretation is autoregression over ordered p16 compute tokens,
  each step predicting the structured tuple of 16 scalar GPTQ codes in parallel;
  scalar-by-scalar autoregression would lengthen a 128x128 tile to 16,384 steps
  and is a materially different compute contract.  Group scale symbols must be
  serialized before the codes that use them, or predicted as an additional
  field of the group-opening token.
- A causal decoder receives shifted ground-truth payload embeddings, causal
  position/shape embeddings, and cross-attends to the fixed latent `z` in every
  block.  Distribution-Encoder information remains encoder-side only; otherwise
  activation context is a second sample-specific conditioning path around `z`.
- The distance-aware ordinal/CDF categorical loss remains applicable to every
  scalar field.  Autoregression changes conditioning and inference order, not the
  15-way code alphabet or the need to predict scale metadata.
- The principal scientific risk is now explicit: a powerful teacher-forced
  decoder may model local GPTQ-code transitions while ignoring `z`.  This is not
  a reason to reject the requested AR model, but train NLL alone cannot certify a
  useful fixed code.  Matched-vs-shuffled/zero-`z` NLL and free-running
  reconstruction are mandatory evaluation, and teacher-forced and free-running
  metrics must not be conflated.

## 2026-08-29 — User retracts AR; direct p16 categorical output is selected

- The AR clarification was sent accidentally.  The selected formulation returns
  to parallel prediction: one p16 decoder state predicts the complete structured
  GPTQ patch in one forward pass.
- For the frozen project codec, the minimal code head is affine-free RMSNorm
  followed by one bias-free linear map `1536 -> 16*15`, reshaped to 16
  independent ordered 15-way logits.  There is no final vector normalization,
  cosine classifier, learned temperature or giant `15^16` patch vocabulary.
- Scale remains mathematically necessary for literal GPTQ: code `k` represents
  `scale*(k-zero)`, so codes alone identify only the normalized lattice shape.
  Omitting scale either makes the representation non-decodable, introduces an
  external side channel, or changes the tokenizer to a global absolute/VQ
  codebook.  None is the selected GPTQ AE contract.
- Scale need not remain a continuous regression lane.  Predict one ordered
  categorical log2-scale symbol per `(output,input-group-128)` from a masked
  pooling of its eight p16 states.  The current symmetric codec has fixed zero
  and therefore needs no zero-point head.  Codes and scale use the same
  distance-aware cumulative categorical loss family and are dequantized only for
  detached reconstruction/action metrics.

## 2026-08-29 — Exact GPTQ scale target and proposed categorical scale head

- Offline target scale reuses the frozen symmetric project codec.  For each
  output channel and consecutive valid input group of at most 128 weights,
  `s = max(abs(W_group))/7`, clamped below by `1e-8`; padding is excluded.  This
  scale is fixed while GPTQ sequentially chooses codes `round(v/s)` in `-7..7`
  and compensates later working weights.  In this codec activations/Hessian
  affect code assignment, not the max-absolute scale itself.
- The scale target is categoricalized in log space.  Freeze a train-only global
  interval for `log2(s)`; use an ordered vocabulary such as 256 total classes,
  with boundary classes serving as underflow/overflow and uniform interior bin
  centers.  Decode class `k` as `s_hat = 2**center[k]`.
- No extra scale token or Transformer tail is needed in the parallel writer.
  For each 128-weight group, take the eight final p16 decoder states and compute
  a valid-component-count-weighted mean.  Apply affine-free RMSNorm and one
  bias-free linear classifier `1536 -> K_scale`.  Initialize its weight with the
  same ordinary categorical scale as the code head, not zero and not the tiny
  former regression-head initialization.
- The code and scale heads read the same p16 categorical-tail representation,
  but code logits do not consume predicted scale.  Scale participates only in
  deterministic dequantization after classification, avoiding a multiplicative
  training path while keeping the payload self-contained.

## 2026-08-29 — Parallel categorical GPTQ production implementation and launch contract

### User decision

- The user approved a single 500k production launch of the non-autoregressive
  p16 categorical GPTQ formulation.  All losses from the immediately preceding
  production formulation must remain available for debugging/analysis, but none
  may contribute to backward.
- After a technically correct launch, training must not be stopped because of a
  plateau, bad early dynamics or mechanistic warning.  Intervention is allowed
  only for a fatal technical failure such as NaN/Inf, crash/OOM, wrong
  data/objective/schema, dead graph, checkpoint/cursor corruption or a proven
  implementation bug.  Otherwise diagnostics are read-only.

### Frozen implementation

- Launcher:
  `/home/coder/project/projects/weight-vae/workspace/training/weightclip_benchmark/run_parallel_categorical_gptq_700m_production.py`.
  Config:
  `/home/coder/project/projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_parallel_categorical_gptq_production_500k.yaml`.
- Encoder is the existing normalized-p32 13-block production encoder with the
  Production Distribution Encoder and fixed `z[32,384]`.  Decoder sample content
  comes only from `z`: five p32 residual cross-attention blocks, p32-to-p16
  address split, then one residual cross-attention categorical tail.  Learned
  coordinate-query residuals remain as address/symmetry breaking; the direct
  decoder Distribution-Encoder conditioner is deleted.
- Heads are affine-free RMSNorm plus bias-free `1536 -> 16*15` code logits, and
  valid-count pooling of eight p16 states followed by affine-free RMSNorm plus
  `1536 -> 256` log2-scale logits.  Exact trainable count is `696,787,328`.
- Teacher targets are generated online after production gauge permutation and
  re-tiling.  The mask-aware implementation is bit-exact with the prior codec on
  full tiles: symmetric codes `-7..7`, first 256 activation rows, damping `.01`,
  IEEE FP32, per-output maxabs/7 scale.  Partial input/output dimensions and
  padded activation/value entries are masked before Hessian construction and
  sequential compensation.
- Scale uses 256 uniformly ordered log2 centers over `[-12,0]`; boundary classes
  absorb under/overflow.  Backward is exactly `L_code_ordinal +
  L_scale_ordinal`, where each cumulative threshold is weighted by the change in
  squared lattice distance and normalized by maximum squared lattice distance.
  No behavioral, structural, InfoNCE, latent hinge, Hessian-weighted or gauge
  term enters backward.
- Hard argmax dequantization is used under `torch.no_grad()` to log the old
  behavioral direction/scale, structural direction/scale, operational metric,
  direction InfoNCE and latent anti-collapse statistics under
  `diagnostic_old_*`.  The currently disabled operational coefficient remains
  zero in `diagnostic_old_behavioral_operator`; its actual unweighted value is
  separately logged as `diagnostic_old_behavioral_operator_actual`.
- Fresh optimizer contract is single-group AdamW, seed42, LR `5e-5`, constant,
  betas `(.9,.999)`, eps `1e-8`, WD `.01`, B32, clip5, 500k.  No old regression
  checkpoint, LR3e-4 transition or gauge optimizer state is reused.

### Focused evidence before production

- Six focused tests pass.  They cover exact parameter/schema contract,
  p16/scale shapes and z-dependence, full-tile GPTQ bit parity, mixed
  `d_in=27/32/64/128` sliced-reference parity and padding invariance, ordinal
  distance ordering/extreme-logit finite gradients, and masked argmax decode.
- Production-faithful B32 smoke completed at
  `/mnt/shared/weightclip_benchmark/parallel_categorical_gptq_700m_smoke_20260829T180206Z`.
  Step1 total/code/scale losses are `.275373/.164047/.111326`, preclip gradient
  norm `2.472`, peak memory `13.20 GiB`; every parameter tensor and all 13
  encoder, five shared decoder, tail, both head, Distribution Encoder and
  bottleneck groups have finite positive gradients.  Latent code/scale gradient
  RMS are `1.70e-6/2.75e-6`, cosine `-.00838`.
- GPTQ representation floor is valid.  Continuous versus binned-scale raw NRMSE
  is `.165138/.165355` (`1.0013x`); calibration operator metric
  `.00111057/.00114158` (`1.0279x`); heldout operator metric
  `.00137395/.00140249` (`1.0208x`).  Scale-bin p99 relative error is `1.6126%`;
  smoke log2 scales span `[-9.823,-3.389]`, so neither boundary is hit.
- Online GPTQ host timing logged `.362s` on step1 and is a material throughput
  cost, but not a correctness failure.  The two-step smoke is implementation
  evidence only; its one-point plot is visually non-informative and is not used
  as learning-quality evidence.
- The prior LR3e-4 polar run was gracefully stopped at its current committed
  step before the smoke, with `STOPPED.json`, rolling resume and persistent model
  saved.  This avoids concurrent GPU contention.  Its artifacts are preserved.
- A final independent source/config/smoke review is pending; production launch
  requires its explicit GO.

## 2026-08-29 — Parallel categorical GPTQ 500k production launched

- Independent final review returned `FORMAL GO`, P0=0/P1=0.  It independently
  checked the frozen launcher/config/sidecar/tests and the production-faithful
  smoke: categorical-only backward, detached old metrics, post-permutation
  mask-aware GPTQ, decoder latent ownership, parameter/optimizer contract,
  all-live gradients, fresh roots and storage/runtime feasibility.
- Fresh production root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1`.
  Rolling state:
  `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/resume_latest.pt`.
  Persistent model:
  `/var/tmp/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/model_latest.pt`.
- Trainer PID at launch is `886431`; Comet sidecar PID is `886982`.  Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/33ef8bc85e3b4d09a08ad80814f42d6d`.
- Production step1 exactly reproduced smoke: total/code/scale
  `.275373/.164047/.111326`, hard-decode raw NRMSE `30.54376`, detached old
  legacy metric `6.92006`, preclip gradient norm `2.472`, peak `13.20 GiB`.
  By step10 total was `.102666` and hard NRMSE `1.03473`; by step20 total was
  `.101003` and hard NRMSE `1.03423`.  These early values establish a valid live
  process and rapid removal of the random scale-head explosion, not final
  learning quality.
- Warm online GPTQ time dropped from the first-call `.357s` to `.022-.023s` at
  steps10/20, so the initial throughput concern is not sustained.
- Per the user's explicit instruction, the correct live run is now left
  untouched.  Bad loss dynamics or a mechanistic warning are not stop
  conditions; only fatal technical failures authorize intervention.

## 2026-08-29 — Step-1000 categorical run diagnosis: broadcast-latent / code-prior collapse

- The user asked what is happening internally and suspected that the latents
  simply collapsed.  Read-only diagnostics were run without signalling or
  changing the production trainer.  The run remained finite, cursor-consistent,
  and all parameter groups retained nonzero gradients; this is a scientific
  failure mode, not a fatal runtime failure, so the run remains live as ordered.
- Failure definition: the categorical objective fell from `.27537` at step1 to
  about `.083-.085`, yet hard decoded raw NRMSE reached only the zero-prediction
  level (`~1.0016` at step1000), while detached direction metrics stayed near
  chance.  The GPTQ teacher floor is much better (`~.15-.17` raw NRMSE), so the
  tokenizer and scale discretization do not explain the plateau.
- An exact held-out unconditional-prior calculation established that the code
  head learned the input-independent GPTQ marginal, not conditional codes.
  Prior versus live steps910-1000: ordinal loss `.083400` vs `.082586`, entropy
  `.736152` vs `.734060`, accuracy `.169671` vs `.170523`, off-by-one `.470856`
  vs `.474102`, MAE `1.94768` vs `1.93853`, MSE `6.30319` vs `6.26174`.  Both
  hard argmax to code zero.  Evidence and reproduction:
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_gptq_hostile_review_step1000_20260829.md`,
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_gptq_prior_review_step1000_20260829.json`,
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_gptq_prior_counts_20260829.json`, and
  `/home/coder/project/projects/weight-vae/workspace/scripts/analyze_parallel_categorical_gptq_prior.py`.
- The scale branch does learn conditional information: live scale MAE is about
  `5.85` bins versus `14.20` for a global prior.  But the zero code prediction
  makes the reconstructed weights essentially zero, so this does not improve
  hard reconstruction.
- The direct checkpoint-1000 probe refines "all latents are identical."  Of the
  centered latent variance, `98.69%` is a sample-main vector shared across all
  32 slots, `0.72%` is slot-main, and only `0.59%` is sample-by-slot interaction.
  Thus `z` still identifies the sample globally, but has lost almost all
  address-specific slot structure: `z ~= global + sample_vector[b]`.
- Every decoder cross-attention read is essentially uniform over the 32 slots:
  normalized entropy `.99964-.99970`, mean max probability `.0342-.03436`
  versus uniform `1/32=.03125`.  It broadcasts the same mean `V(z)` to every
  coordinate.  The first write is large, but write/query RMS falls across the
  five p32 blocks from `.62` to `.166`; its sample-centered fraction falls from
  `.382` to `.234`.
- A same-layout cyclic `z` swap supplies the functional discriminator.  It
  changes only `2.12%` of code argmaxes and worsens code loss by only `.37%`
  (`.085282 -> .085599`), while it changes `89.23%` of scale argmaxes and makes
  scale loss `6.38x` worse (`.002849 -> .018174`).  Zeroing `z` does worsen code
  loss to `.10975`, so the decoder uses the shared statistical latent state,
  but it practically ignores the correct sample identity for code prediction.
  Artifact:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/mechanistic_probe_step1000.json`;
  analyzer:
  `/home/coder/project/projects/weight-vae/workspace/training/weightclip_benchmark/analyze_parallel_categorical_checkpoint.py`.
- Supported causal mechanism: residual coordinate queries and heads can realize
  the positional/unconditional code prior without sample content.  Uniform
  cross-attention rewards only the slot mean, so the encoder places the useful
  global scale signal in a broadcast sample vector; slot differences cease to
  matter.  As values become homogeneous, routing gradients weaken further,
  forming a self-reinforcing broadcast-latent basin.  Scale can use the global
  vector, but per-coordinate GPTQ codes cannot.  This is not the previous
  direction-versus-scale gradient conflict: code/scale latent-gradient cosines
  are near zero or positive, while both code credit into `z` and slot-specific
  structure collapse.
- At step1000 `dL_code/dz` was `2.50e-9` versus code-head gradient RMS
  `1.12e-5`; all groups were nevertheless finite/live.  The nonfatal telemetry
  bug `code_valid_count=scale_valid_count=1.0` is caused by constructing counts
  with a boolean tensor's `new_tensor`; it does not affect masks, loss, or
  backward and was not patched mid-run.

## 2026-08-29 — Fix discussion: why the old latent hinge is insufficient

- The user asked whether the previous two-way latent anti-collapse loss is the
  required fix.  Conclusion: it can be a secondary guardrail, but it is not a
  mechanism-matched primary fix.
- Direct counterevidence is now especially strong.  In the previous anti-only
  polar run, the hinge reached zero and interaction/RMS reached `.542`, yet a
  same-layout latent swap changed direction loss by only `+.021%`.  In the
  current categorical run, interaction/RMS recovered on its own to `.1046` by
  about step2080, already above the former `.1` margin, while hard NRMSE remained
  `1.00026`.  Therefore satisfying that constraint neither guarantees decoder
  use nor conditional reconstruction.
- Algebraically the old hinge grows the two-way-centered interaction `r[b,s]`,
  for which the slot mean is exactly zero.  The current nearly uniform
  cross-attention reads approximately the slot mean, so it is almost blind to
  the component the auxiliary would create.  The encoder could satisfy the
  hinge with a low-rank sample-by-slot hash in an ignored subspace.  Because the
  live categorical gradient at `z` is only around `2.5e-9`, the auxiliary could
  dominate encoder training without improving codes.
- Two remedies were distinguished.  The smallest loss-level rescue is a
  same-layout prediction-to-target retrieval objective on the final code
  logits: `S_ij = -ordinal_cost(pred_i, target_j)`, followed by row-wise CE with
  the diagonal as positive.  With identical layouts and no decoder-side DE
  input, sample discrimination must pass through `z`; the unconditional prior
  is not a solution.  It is still an auxiliary with temperature/scalarization
  and can learn a small target fingerprint, so it is a diagnostic/rescue rather
  than a root topology fix.  Review:
  `/home/coder/project/docs/report/parallel_categorical_gptq_anticollapse_fix_review_20260829.md`.
- The recommended root fix is deterministic, balanced slot ownership plus a
  content-rooted decoder.  Each input p32 patch and corresponding output p16
  patches have a fixed owner among 32 slots; the owner slot directly initializes
  the output content state.  Coordinates may route or multiplicatively gate
  content, but no additive coordinate/query value path reaches the categorical
  head.  A deep Transformer over the 32 owned latent tokens supplies global
  mixing and is where model capacity scales.  Consequently softmax does not
  first have to discover routing, and equal slots receive different
  patch-specific gradients, so the present uniform-attention equality basin is
  not stationary for the same reason.
- A softer alternative is fixed distinct slot anchors in Q/K plus a terminal
  carrier-free cross-attention writeout.  It preserves arbitrary output length
  and avoids the former no-query-residual uniform bootstrap, but learned soft
  routing can still re-uniformize; deterministic ownership is the stronger
  causal intervention.
- If retained after the topology change, the old hinge should be described only
  as a zero-above-margin safety fuse.  Success must be functional: categorical
  loss below the sealed unconditional prior, meaningful matched-versus-shuffled
  `z` code gap, sustained code-gradient at `z`, and hard NRMSE below one—not
  latent dispersion alone.

## 2026-08-29 — User-observed directional improvement revises the collapse verdict

- The user noticed that detached directional and contrastive diagnostics were
  improving while the optimized categorical curve looked flat.  This was
  confirmed and is not explained by minibatch composition: exact-layout group
  eligibility/count/size remain stationary over the compared rolling windows.
- From steps `501-1000` to `3001+`, direction InfoNCE falls `.881 -> .784`,
  top-1 rises `.446 -> .624` against layout-dependent chance near `.39`, and
  diagonal-minus-hardest-negative rises `.0015 -> .0238`.  Behavioral direction
  improves `.975 -> ~.947`, structural direction `.964 -> ~.946`, and latent
  interaction/RMS grows `.045 -> ~.237`.  Scale cannot explain structural
  direction or InfoNCE because the decoded p16 directions are explicitly
  normalized before those metrics.
- The optimized code loss is not literally flat: it falls
  `.082641 -> ~.08233` (`~.36-.38%`), while total falls about `.57%`.  The code
  trend is smaller than the per-batch standard deviation near `.001`, so it is
  visually obscured.  The sensitivity mismatch explains the larger diagnostic
  movement: ordinal loss is a soft cumulative score averaged over all code
  components, whereas the diagnostics apply hard argmax and then normalize
  sparse p16 vectors.  Small probability reorderings crossing a few argmax
  boundaries can strongly change angular/ranking metrics while barely changing
  the average ordinal scalar, exact-code accuracy, MAE, or raw NRMSE.
- A read-only step-3000 checkpoint probe confirms a current conditional signal.
  Same-layout `z` swap changes code loss `.082462 -> .083275` (`+.99%`) even
  though only `1.43%` of hard code argmaxes change.  Attention is no longer
  uniform (block-2 entropy `.963`, max probability `.0806` versus uniform
  `.03125`), and the checkpoint's centered latent variance decomposes into
  `81.65%` sample-main, `11.02%` slot-main, and `7.34%` interaction.  This is a
  weak partial routing escape, not full reconstruction.
- Important comparison caveat: step-1000 and step-3000 probes used their
  respective next-cursor batches (`32000` and `96000`), not one fixed paired
  batch.  Their numerical differences alone are not a paired checkpoint
  trajectory; the temporal conclusion is supported by hundreds of rolling live
  batches, while step3000 independently establishes the current functional
  state.
- The earlier step-1000 diagnosis remains valid for that checkpoint but the
  stronger claim of a persistent/absorbing full code collapse is retracted.
  The model appears to be slowly symmetry-breaking out of the prior basin.
  It remains far from useful reconstruction: hard NRMSE is still around
  `1.001`, and aggregate code accuracy/MAE are nearly flat.  Therefore no
  architecture or anti-collapse intervention should be made to this live run
  now; continue observing as the user requested.
- Evidence:
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_gptq_directional_escape_step3000_20260829.md`,
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_gptq_prior_review_step3000_20260829.json`, and
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/mechanistic_probe_step3000.json`.

## 2026-08-29 — Scale diagnosis: global-radius shortcut, not row-scale transmission

- The user observed that scale appears to learn poorly.  The pure categorical
  scale metrics do improve substantially: global-prior MAE `14.20` bins and
  ordinal loss `.01712` become roughly `5.7` bins and `.00245`.  A step-3000
  same-layout `z` swap worsens scale MAE `6.38 -> 20.60`, scale loss
  `.00313 -> .07265`, and changes `68%` of scale argmaxes.  Thus the scale head
  is live and strongly conditional on the bottleneck.
- The old detached `behavioral_scale` and `structural_scale` metrics are not
  clean scale-head measures: they evaluate norms of the product
  `hard_code * decoded_scale`.  With hard codes often equal to signed zero,
  predicted weight norms vanish even if the categorical scale token is right.
- Residual error is nevertheless material.  The 256 centers over `[-12,0]`
  have spacing `.04706 log2`; late MAE near `5.7` bins corresponds to about
  `.268 log2`, approximately `20%` multiplicative error.  Binning itself is not
  the floor: p50/p99 scale quantization error is only about `.8%/1.6%`, and
  observed targets do not approach the configured boundaries.
- The step-5000 read-only probe establishes the missing decomposition.  The
  model nearly perfectly predicts the per-sample global scale (`corr=.988`,
  sample-mean MAE `.0919 log2`) but not the 128 output-row residuals
  (`within-sample corr=.064`, RMSE `.323 log2`).  A trivial oracle that broadcasts
  one true median scale over all output rows is better than the model:
  `4.89` versus `5.42` bins MAE.  Model physical errors are p50 `12.9%`, mean
  `17.0%`, p90 `35.6%`.  About `70.2%` of target variance is sample-global, so
  this is the easiest shortcut and explains the large `z`-swap sensitivity.
- This is particularly diagnostic because encoder input already contains the
  exact continuous teacher scale: both `_prepare_normalized_inputs` and GPTQ
  compute `row_maxabs/7`; standardized log2 scale is repeated in the four p32
  tokens of every output row.  The failure is therefore loss/routing of known
  row-specific information through the fixed bottleneck, not inference of an
  unavailable target.
- Objective geometry enables the shortcut.  The ordinal point cost is
  `d^2/(K-1)^2`; for scale, a five-bin error costs only `25/255^2=3.84e-4`,
  whereas a two-bin code error costs `4/14^2=.0204`.  Scale contributes only
  about `2.7-3.3%` of the total scalar, making the last several scale bins cheap.
  This is not a global gradient-death explanation: late `dL_scale/dz` is often
  comparable to or larger than `dL_code/dz`, and their cosine is near zero.
  The supported mechanism is a weak full-range-normalized precision incentive
  combined with the broadcast latent/routing shortcut, which preserves the
  dominant sample mean and discards zero-mean row deviations.
- No static target/bin/mask/pooling bug was found.  One remaining measurement
  caveat is that the ordinal objective and argmax decode are not decision-rule
  matched; expected-bin/CDF-median decoding may improve the reported hard scale
  error and should be checked read-only before changing training.
- Current best next-run fix, if the argmax caveat is small: keep GPTQ codes
  categorical but predict one continuous standardized `log2(scale)` scalar per
  output row with MSE/Huber (ideally separating and variance-normalizing the
  sample-global mean and row-centered residual).  This removes the arbitrary
  `255^2` geometry.  If row residuals still fail, add a deterministic row-owned
  carrier from encoder scale tokens to the corresponding decoder row rather
  than an anti-collapse loss; scale already uses `z`.
- Evidence:
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_scale_audit_step5000_20260829.md` and
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/scale_audit_checkpoint_latest_20260829.json`.

## 2026-08-30 — Current categorical state: no collapse; code-center shrinkage and unstable conditional carrier

- The user requested a mechanistic audit of the live categorical GPTQ model.
  All measurements were read-only.  The trainer and Comet sidecar remain live;
  the latest checked row is step `101290`, finite, with exact cursor continuity.
- Validity checks exclude an operational failure: no NaNs/infs, no dead gradient
  groups, no clipping (`max pre-clip norm 2.472 < 5`), and calibration/heldout
  operator metrics agree.  The known `*_valid_count=1` bool-cast telemetry bug is
  diagnostic-only and does not affect masks, targets, or backward.
- The training history has four regimes: early prior/broadcast collapse, routing
  escape around `8k-17k`, useful reconstruction growth through roughly `32-40k`,
  then a slow noisy plateau.  Broad online medians are approximately NRMSE
  `.913/.908/.911/.912` over `20-40k/40-60k/60-80k/80-98k`.
- The old latent-collapse diagnosis is now false.  At exact step `98k`, centered
  latent variance decomposes into `72.9%` slot-main, `22.1%` sample-by-slot
  interaction, and `5.0%` sample-main; centered and interaction effective ranks
  are `19.7` and `31.0`.  Exact-layout `z` swap raises code loss `1.56x` and
  scale loss `39.2x`, changing `31.7%` and `91.1%` of their argmaxes.  The falling
  normalized interaction ratio is caused by total latent RMS growth; absolute
  interaction RMS rises rather than collapses.
- Scale is no longer the principal bottleneck at this checkpoint.  Expected
  log2-scale has correlation `.952`, R2 `.907`, row-centered correlation `.725`,
  and interaction correlation `.741`.  Crossed decoding gives GPTQ teacher
  NRMSE `.1677`, oracle-code plus predicted-scale `.2287`, but predicted-code
  plus oracle-scale `.8845`.  Thus current end-to-end error is dominated by the
  categorical code payload.
- The code distribution is strongly center-seeking: target zero-code frequency
  is `17.0%`, while the step-98k model predicts zero `79.4%`.  Expected-code
  decoding improves oracle-scale NRMSE only `.8845 -> .8778`, so argmax is not
  hiding a good soft distribution.
- Sharp attention is not, by itself, the cause.  The tail has entropy about
  `.05` and max probability about `.95`, but winner traffic uses about `9.3`
  effective slots rather than one, all slots receive soft mass, and causal
  inference-temperature interventions (`0.5x`, `2x`, `4x`, plus all-block
  `2x`) are neutral or worse.  This is presently useful sparse ownership, not a
  simple attention-saturation failure.
- A same-batch checkpoint discriminator reveals instability hidden by online
  batch composition.  On one fixed B64, steps `90k/98k/100k` give hard NRMSE
  `1.0227/.8856/1.0133`, scale row-centered correlation
  `.0169/.7255/.0396`, and code `z`-swap loss ratios
  `1.018x/1.558x/1.018x`.  The strong `98k` carrier is therefore a real
  transient and is mostly forgotten by `100k` on that batch.
- This is not pure global optimizer churn.  On a broader paired panel of eight
  B32 batches, `90k -> 100k` improves code ordinal loss
  `.08545 -> .08385` in `7/8` batches and hard NRMSE
  `1.02525 -> 1.01451` in `8/8`.  However predicted zero frequency becomes even
  worse, `87.23% -> 90.57%`, while target NLL also improves
  `2.498 -> 2.458`.  Hence the strongest supported causal mechanism is that the
  ordinal objective rewards a cheap central/broad distribution much more
  reliably than it rewards conditional code amplitude; the weak conditional
  carrier can form and then be forgotten around that basin.  Constant LR is a
  plausible operational amplifier, but is not established as the sole cause.
- Excluded as primary current causes: total latent collapse, sample-global-only
  scale, simple attention-temperature saturation, argmax decision mismatch,
  code/scale gradient conflict, gradient clipping/numerical stall, and
  calibration overfit.  Remaining viable ambiguity is objective geometry versus
  longer-term ownership/capacity limits.  The clean next discriminator is a
  paired continuation from one checkpoint with constant versus decayed LR on a
  sealed multi-batch panel, followed independently by a point-error-aware code
  objective; no change was made to the live run.
- Evidence:
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_current_state_trajectory_20260830/report.md`,
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_current_state_mechanistic_probe_step98000_20260830.md`,
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_current_state_hostile_review_step98000_20260830.md`,
  `/home/coder/project/projects/weight-vae/workspace/docs/report/parallel_categorical_paired_90k_100k_mechanistic_probe_20260830.md`,
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/current_state_mechanistic_probe_latest_20260830.json`,
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/paired_90k_100k_mechanistic_probe_20260830.json`, and
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_gptq_categorical_parallel_500k_v1/paired_90k_100k_core_panel_8x32_20260830.json`.

## 2026-08-30 — Why the categorical objective weakly rewards code amplitude

- The user asked for the exact source of the weak amplitude incentive.  The
  current cumulative ordinal objective assigns a deterministic class error of
  distance `d` the lattice cost `d^2/(K-1)^2`.  For the 15-way signed GPTQ code
  vocabulary, a one-bin miss costs `1/196=.0051` and a two-bin miss costs
  `4/196=.0204`; all valid scalar positions are then averaged.
- More importantly, the code loss is expressed only in normalized code-index
  units.  Physical dequantization error is proportional to
  `scale^2 * (pred_code-target_code)^2`, but the backward code objective omits
  the row/group `scale^2` and GPTQ/Hessian importance.  Correcting a code in a
  high-amplitude or activation-important row is therefore rewarded no more
  than correcting the same bin displacement in a tiny row.  This was an
  intentional scale-invariant reset, not an implementation bug, but it removes
  the physical amplitude incentive.
- Under incomplete conditional information, expected squared lattice risk is
  minimized near the conditional mean.  The signed target distribution is
  approximately centered, so positive and negative possibilities cancel and
  class zero is the safe prediction.  Small probability improvements can lower
  the threshold log loss while zero remains the argmax.
- Averaging across all code coordinates further dilutes sparse correct
  non-zero predictions.  Patch-normalized directional diagnostics amplify
  exactly those sparse sign/pattern improvements, explaining why they can fall
  much faster than the optimized scalar objective.
- Exact implementation evidence is
  `/home/coder/project/projects/weight-vae/workspace/training/weightclip_benchmark/run_parallel_categorical_gptq_700m_production.py:501-551` and
  `:1101-1112`.

## 2026-08-30 — Active WeightCLIP operator dataset uploaded to private Hugging Face storage

- The user explicitly requested uploading the dataset, not merely documenting
  its local location, and asked that another agent on another machine be able
  to find and use it. The live categorical production trainer was not stopped
  or modified.
- The uploaded scope is exactly the active dataset referenced by the current
  production pair manifest, not the full multi-generation local zoo. It
  contains `weight_tile_bank-9020c363ee2005e8`,
  `context_bank-234e31b2ec93af4f`, the immutable pair manifest, coverage, and
  JSONL/Parquet permutation metadata: 3,622 payload files and
  41,724,270,793 bytes (38.859 GiB).
- The private Hub repository is
  `https://huggingface.co/datasets/EnriFermi/weightclip-resnet18slim-operator-bank`.
  The completed payload revision is
  `3bc8cabb959f7dfd5682e9e36c6d69a51a00f444`; the final revision including
  dataset metadata is `79d9b8c97362a3892d1f8475f71face6f9859b72`.
- Upload used the resumable Hugging Face large-folder path with an exact
  allowlist and two low-priority workers. One early uploader process was
  resumably restarted so Xet cache lived on `/var/tmp` instead of the nearly
  full project filesystem; only the uploader was stopped, never training.
- Post-upload validation established that the repository remains private, all
  3,622 payload paths have their expected remote sizes, and the downloaded
  small manifests/coverage/permutation files match their frozen SHA-256 values.
  A remote-manifest relocation smoke also produced an absolute-path resolved
  manifest successfully without re-downloading the 38.9 GiB payload.
- The original pair manifest is retained byte-for-byte for provenance. Because
  its runtime paths are machine-specific, the new downloader creates a separate
  resolved pair manifest and prints its new SHA-256. Only the top-level bank,
  coverage, and permutation paths are rewritten; nested absolute provenance
  paths remain historical metadata.
- Discovery and machine-readable pin:
  `/home/coder/project/projects/weight-vae/workspace/docs/weightclip_operator_dataset_hf.md`
  and
  `/home/coder/project/projects/weight-vae/workspace/conf/weightclip_benchmark/operator_dataset_remote_hf.json`.
  Download helper:
  `/home/coder/project/projects/weight-vae/workspace/scripts/download_weightclip_operator_dataset_from_hf.py`.
  Access from another machine requires a Hugging Face token with read access to
  the private repository. The current loader performs one full payload SHA scan
  when opening the downloaded banks, so first open will read the full dataset.

## 2026-08-30 — Download on the current machine is blocked only by HF authentication

- The user requested downloading the operator dataset that was uploaded to
  Hugging Face. The intended source is the pinned private repository
  `EnriFermi/weightclip-resnet18slim-operator-bank` at revision
  `79d9b8c97362a3892d1f8475f71face6f9859b72`.
- The checked target is
  `/home/coder/project/datasets/weightclip_resnet18slim_operator_dataset`;
  its filesystem has about 409 GiB free, sufficient for the 38.859 GiB payload.
- The current machine has neither `HF_TOKEN` nor a cached Hugging Face token.
  An anonymous Hub metadata request returned HTTP 401, confirming that the
  private repository cannot be downloaded without read authorization.
- No payload files were downloaded. Next action: provide `HF_TOKEN` with read
  access (preferably through the environment, not chat), then run the committed
  download helper and inspect its inventory/hash validation plus resolved pair
  manifest.

## 2026-08-31 — Private HF operator dataset downloaded and validated locally

- The user supplied a fine-grained Hugging Face token. It was saved in the
  standard per-user Hugging Face credential store, not Git credentials. Both
  credential files were tightened from the CLI-created mode `0644` to `0600`.
  Because the secret was sent through chat, token rotation after this task is
  recommended.
- The pinned private revision
  `79d9b8c97362a3892d1f8475f71face6f9859b72` was downloaded to
  `/home/coder/project/datasets/weightclip_resnet18slim_operator_dataset`.
  The directory mode is `0700` and its materialized size is about 39 GiB.
- The initial eight-worker transfer hit the documented anonymous/fine-grained
  account quota of 1000 API requests per five minutes at roughly 30%. The
  download was resumable; after the quota window reset it completed with one
  worker without re-downloading completed files. The committed helper needed
  `python -I` because its sibling `scripts/inspect` package otherwise shadows
  the Python standard-library `inspect` module.
- Helper validation passed: weight bank `793` files / `11,181,671,316` bytes;
  context bank `2,825` files / `30,517,934,053` bytes. The helper also verified
  the frozen SHA-256 values of the small immutable manifests, coverage, and
  permutation metadata. No full 38.9 GiB rehash was repeated.
- Resolved runtime manifest:
  `/home/coder/project/datasets/weightclip_resnet18slim_operator_dataset/operator_dataset-resolved-c478efe716506765.json`,
  SHA-256
  `c394c33203a9d2161a09167498a3d54470bab522c7c1a02e944fed14f5b77c09`.
  Independent post-run checks confirmed that both bank paths, coverage, and
  both permutation paths exist and that there are no `.incomplete` files.

## 2026-08-31 — K=1 raw-direction MSE arm implemented and smoke-validated

### User decision and frozen comparison

- The user selected a single-component (`K=1`) direction regression arm on one
  A100. The other A100 is reserved for a separate user experiment.
- The architecture and production setup are the original 742,120,833-parameter
  p32 polar regression model with separate p16 direction and scale tails, before
  latent anti-collapse, direction InfoNCE, gauge fixing, and categorical GPTQ.
- Every prediction-direction cosine objective and InfoNCE term is absent from
  this arm's backward. Final per-p16 L2 normalization remains only to compose
  reconstructed weights from the direction and scale coordinates.

### Exact intervention

- The direction tail's raw 16-vector is regressed to `r0 * target_unit_direction`
  with `r0=0.08615882694721222`, matching the seed-42 initialization scale.
  The scalar is `0.5 * squared_L2 / r0^2`, with the original structural
  `sqrt(target_patch_radius)` within-output weighting. At equal predicted and
  target radius this has the same dimensionless tangent scale as the old
  `1-cosine`, while additionally supervising the previously invisible radius.
- There is one deterministic vector per p16 (`K=1`); there is no mixture head,
  winner assignment, contrastive term, latent anti-collapse term, or head gauge
  constraint.
- The existing behavioral and structural log-scale objectives remain at weight
  10 each. A dedicated behavioral scale-only implementation preserves the old
  scalar exactly without even constructing its cosine sibling. Direction is
  detached from both scale paths.
- Production config:
  `projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_polar_tails_raw_direction_mse_k1_production_500k.yaml`.
  Focused contract tests:
  `projects/weight-vae/workspace/tests/direct_normalized_raw_direction_mse_test.py`.

### Smoke evidence and launch status

- Four focused tests pass, including exact-target zero loss, an orthogonal
  equal-radius unit-scale check, exact parity of the new scale-only helper with
  the old scale scalar, and direction/scale autograd separation. Py-compile,
  config dry-run, and `git diff --check` also pass.
- A production-faithful two-step B32 smoke ran on physical GPU0 only and
  completed with exact parameter count, finite loss/gradients, and all
  encoder/shared-decoder/tail/head groups live. Peak allocation was 14.68 GiB.
  Artifact root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_raw_direction_mse_k1_smoke_v1`.
- At step 1, total loss was `2.610457`: raw direction MSE `.924007`, weighted
  scale objective `1.686450`, and exact component-sum parity error `0`. Raw
  direction norm median was `.0861438` against target `.0861588`; cosine and
  InfoNCE loss coefficients are zero/absent. Pre-clip norm was `368.85`, so the
  inherited clip-5 regime is active from initialization and must be monitored
  in the real trajectory rather than interpreted from the smoke alone.
- Smoke Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/35d4c18187b946fb8ddfb1abbe6ee95f`.
  It is explicitly tagged `smoke` with realized horizon 2 and scientific
  horizon 500,000.
- The 500k production trainer has not been started. Repository rules require an
  independent reviewer to inspect the final setup and issue explicit GO before
  a large run; reviewer agents are unavailable in this side conversation. The
  remaining action is independent prelaunch review in the main thread, then
  launch trainer plus Comet sidecar on GPU0 without changing this frozen arm.

## 2026-08-31 — User corrected the launch hold; K=1 production is live on GPU0

- The user explicitly objected to holding the already requested launch. This
  superseded the prior side-thread hold: the frozen 500k config was launched on
  physical GPU0, while GPU1 remains completely idle for the user's other arm.
- Live production root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_raw_direction_mse_k1_500k_v1`.
  Production Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/427906517d894ce0920bed65c6feeb63`.
- At the first stable read through step 140, median interval speed after startup
  was `1.077 step/s` (`.929 s/step`, B32), implying a no-checkpoint-overhead ETA
  of about `5.37 days`. GPU0 was at 100% utilization with about 21.4 GiB
  allocated; GPU1 was at 0 MiB.
- The early trajectory already exposes the predicted raw-MSE radial effect.
  Median raw radius fell from `.08614` at step 1 to roughly `.015-.027` over
  steps 40-140 despite target `.08616`; raw direction MSE fell `.924 -> ~.50`.
  This is consistent with the direct-vector MSE optimum `radius ~= r0*cosine`
  while directions are poorly aligned, rather than evidence that the configured
  target was not applied. It is a scientifically important warning, not yet a
  final outcome; the trainer was not stopped or modified.
- Exact step-100 component parity remained zero and all named groups remained
  numerically live, but encoder gradient RMS had already fallen to order
  `1e-7` while direction-head RMS was `.56`; all logged pre-clip norms remained
  far above clip 5. Continued monitoring is required for direction/radius and
  latent credit collapse.

## 2026-08-31 — User selected a strict two-loss restart

- The user removed behavioral scale from the K=1 experiment and required
  exactly two optimized terms: `raw_direction_MSE + 10*structural_log_scale`.
- The preceding three-loss production was gracefully stopped at optimizer step
  559. Its `STOPPED.json`, resume checkpoint, persistent model, metrics, and
  Comet run were preserved under
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_raw_direction_mse_k1_500k_v1`.
- A distinct frozen schema
  `weightclip_direct_normalized_scaled_700m_polar_tails_raw_direction_mse_structural_scale_production_v1`
  prevents the old and new objectives from sharing provenance. In the new
  config, all behavioral coefficients, structural direction, reconstruction,
  relational, cosine, InfoNCE, and anti-collapse terms are zero/absent.
- Behavioral scale is not merely multiplied by zero: the two-loss branch skips
  `_operator_scale_only_loss`, verified by a focused test that patches that
  helper to raise if called. Four focused tests, py-compile, `git diff --check`,
  and config dry-run pass.
- Fresh production root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_raw_direction_mse_k1_structural_scale_500k_v1`.
  Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/54f559fefc854942be7e6f1b69fc52ca`.
- The live integration check passes. At step 1,
  `.9240074 + 10*.0481200 = 1.4052074` within `2.24e-8`; behavioral and both
  direction-cosine fields are exactly zero and component parity is exactly
  zero. At step 20 total is `1.085335`, raw direction MSE `.579880`, raw
  structural scale `.0505455`, and behavioral scale remains exactly zero.
  All encoder/shared-decoder/tail/head groups are gradient-live.

## 2026-08-31 — Original polar regression with a strictly frozen decoder launched on GPU1

### User decision and baseline

- The user selected the original pre-categorical p32 polar regression model as
  the baseline: 742,120,833 parameters, five shared decoder blocks, separate
  p16 direction and scale tails, seed 42, B32, constant LR `5e-5`, and the
  unchanged coordinate-owned objective
  `behavioral_direction + 10*behavioral_scale + structural_direction +
  10*structural_scale`. No latent anti-collapse, InfoNCE, gauge fix, raw-vector
  MSE, GPTQ categorical head, VQ, or autoregression is present.
- The requested intervention is encoder-only training from a fresh seeded
  initialization. To make "fully frozen decoder" strict rather than nominal,
  only parameters whose effect reaches reconstruction through `z` are
  trainable: input content/scale/group/chunk embeddings, latent slots, all 13
  encoder blocks, latent norm, and `to_latent`. The Distribution Encoder and
  tile embeddings are frozen because they also feed decoder conditioning
  directly. `from_latent`, decoder queries/conditioner, all shared decoder
  blocks, both tails, both output norms, and both heads are frozen.

### Implementation, data and verification

- Config:
  `projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_polar_tails_encoder_only_production_500k.yaml`.
  Focused contract test:
  `projects/weight-vae/workspace/tests/direct_normalized_encoder_only_scope_test.py`.
- Exact partition is 101 trainable tensors / 474,221,568 parameters and 152
  frozen tensors / 267,899,265 parameters. The optimizer is built only from the
  trainable identity set. Step-1 fail-closed checks reject any missing
  trainable gradient or any gradient on a frozen parameter. Full loss,
  direction/scale components, per-encoder-block gradients, rate, LR, clip norm,
  cursor, and VRAM remain logged.
- The local private-HF snapshot is complete and used through resolved manifest
  `/home/coder/project/datasets/weightclip_resnet18slim_operator_dataset/operator_dataset-resolved-c478efe716506765.json`,
  SHA-256 `c394c33203a9d2161a09167498a3d54470bab522c7c1a02e944fed14f5b77c09`.
  Both banks and coverage/permutation paths exist, no `.incomplete` file exists,
  and the production loader opened all 135,100 canonical tiles / 14,000 groups.
- Focused tests passed 5/5 together with the already present K=1 tests;
  py-compile, config dry-run and `git diff --check` passed. A B32 two-step smoke
  completed at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_encoder_only_smoke_v1`:
  step-1 loss `3.586198`, exact component parity `2.38e-7`, all 13 encoder blocks
  live, no frozen gradients, and peak allocation `11.57 GiB`. Its one-point
  plot is readable but is only a smoke artifact, not trend evidence.
- Independent prelaunch review issued explicit GO with P0=0/P1=0 after an exact
  production partition/optimizer reconstruction and path/GPU collision check.
  Frozen source/config/test hashes were respectively `12844946ec97...`,
  `85b90a1f204e...`, and `d2bf10187028...`.

### Live production state and early evidence

- Trainer PID `71669` is detached under its own session and live on physical
  CUDA1. Root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_encoder_only_500k_v1`.
  Comet sidecar PID `72733`:
  `https://www.comet.com/mike-5531/big-weight-vae/1dade012915b43f7ac45d34894b55454`.
- At the checked step 130, loss was finite (`3.062354`), cursor was 4,160, LR
  remained `5e-5`, and peak allocation was `12.34 GiB`. The latest five
  post-start intervals had median speed `1.089 step/s`, giving a provisional
  no-checkpoint-overhead ETA of about `5.31 days`. These are different stream
  batches, so the early scalar decrease is not a matched-batch convergence
  claim.
- Step-100 telemetry still has exactly 152 no-gradient tensors, matching the
  declared frozen inventory, while all 13 encoder blocks and `to_latent` remain
  live. Direction/scale gradients are already opposed in representative
  encoder locations: cosine `-.536` and scale/direction RMS `1.46x` in encoder
  block-1 Q, cosine `-.205` and ratio `2.29x` in block-13 Q, and cosine `-.367`
  with ratio `1.63x` at `to_latent`. This is early mechanism telemetry only; it
  does not yet establish the long-horizon result of encoder-only training.
- During this final read, the separate GPU0 K=1 trainer was no longer live and
  its own `STOPPED.json` recorded step 559/cursor 17,888. This encoder-only task
  did not signal or modify that run; GPU1 selection and paths remained
  isolated throughout.

## 2026-08-31 — GPU0 strict two-loss run live at 10.7k with persistent radial collapse

- A live status review confirmed that the strict two-loss production process
  (`raw_direction_MSE + 10*structural_log_scale`) is still running on physical
  GPU0, with trainer PID `77327` and Comet sidecar PID `77570`. There is no
  `FAILED.json` or `STOPPED.json`; metrics were only about 13 seconds old and
  advanced from step 10,730 to 10,750 during the review.
- Physical GPU0 holds about 21.3 GiB. Short `nvidia-smi dmon` sampling showed
  bursty compute (`0/97/0/0/98%` SM across five one-second samples), while the
  recorded run-average rate is `0.4896 step/s`. At that rate, 500k completion
  is approximately `2026-09-11 22:08 UTC` (about 11.6 days remaining).
- Operational validity is clean: all metrics are finite, steps strictly
  increase, `committed_logical_index == step*32` for every row, and the latest
  step-10,700 gradient telemetry has zero missing parameter tensors. Both the
  persistent model and resumable optimizer checkpoint were written at the
  10k boundary.
- The scientific trajectory is concerning. Window medians for raw direction
  MSE are `.5029` at steps 1-200, `.4876` near 1k, `.4828` near 5k, `.4834`
  near 10k, and `.4838` over the latest 500 steps: useful direction progress
  has effectively plateaued since about 5k on these online stream windows.
  Meanwhile mean predicted radius falls `.0171 -> .00831 -> .00509 -> .00458
  -> .00429`, versus configured target `.08616`. The recent radius is only
  `5.0%` of target and radius MAE is `95.0%` of target. This extends the earlier
  evidence for the raw-vector-MSE radial shortcut; it is not a matched-batch
  checkpoint comparison.
- Gradients are live but imbalanced. At step 10,700, structural-scale gradient
  RMS exceeds direction by `11.9x` at encoder block 1 Q, `5.68x` at encoder
  block 13 Q, and `6.46x` at `to_latent`; the latter two component cosines are
  mildly negative. Every one of the latest 50 logged pre-clip norms exceeds
  the clip-5 threshold (median `9.32`). No intervention was made because the
  user asked for status, not a stop or objective change.
- Evidence root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_raw_direction_mse_k1_structural_scale_500k_v1`,
  especially `train_metrics.jsonl`, `gradient_telemetry.jsonl`,
  `resolved_config.json`, and the 10k checkpoints.

## 2026-08-31 — User stopped the GPU0 strict two-loss run and freed the device

- The user explicitly requested shutting down the GPU0 run and releasing the
  GPU. The target was re-resolved as trainer PID `77327` with
  `CUDA_VISIBLE_DEVICES=0` and the strict two-loss config; the separate GPU1
  encoder-only trainer was left untouched.
- `SIGTERM` produced a graceful trainer exit after 25 seconds. Final
  `STOPPED.json` records step `10,975` and committed cursor `351,200`.
  A fresh 8.906 GB resumable optimizer checkpoint was written at
  `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_raw_direction_mse_k1_structural_scale_500k_v1/resume_latest.pt`,
  and a fresh 2.969 GB persistent model was written under the run root.
- The Comet sidecar PID `77570` was then terminated gracefully and exited after
  eight seconds. A repeated `nvidia-smi` check showed physical GPU0 at 0%
  utilization, 3 MiB / 81,920 MiB, and no compute process. GPU1's independent
  encoder-only process remained live and was not signaled.

## 2026-08-31 — Warm-encoder / gradual-decoder-thaw branch launched on GPU0

### User decision and branch semantics

- The user requested branching from the latest saved strict encoder-only
  checkpoint available at request time, keeping encoder training active while
  gradually unfreezing the entire decoder through its learning rate. Physical
  CUDA0 was to be used after the strict two-loss run released it; the existing
  CUDA1 encoder-only production was to remain live.
- The moving encoder-only `resume_latest.pt` was pinned by hardlink before its
  next atomic replacement. The immutable branch source is
  `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_encoder_warm_decoder_ramp10k_500k_v1/source_encoder_only_step_00013000.pt`:
  source step `13,000`, committed logical cursor `416,000`, original polar-tail
  schema, and strict encoder-only optimizer with 101 state entries.
- Branch step is intentionally reset to zero while the data cursor continues
  from 416,000. Encoder LR remains constant at `5e-5`. Decoder LR is linear in
  branch step: `5e-5 * min(step / 10,000, 1)`, hence `5e-9` at the first update,
  `2.5e-5` at step 5,000, and `5e-5` from step 10,000 onward. Encoder Adam
  moments and checkpoint RNG state are restored; all 152 decoder/direct-
  conditioning tensors begin with empty Adam state.

### Implementation and validation

- Dedicated config:
  `projects/weight-vae/workspace/conf/weightclip_benchmark/direct_normalized_scaled_700m_polar_tails_encoder_warm_decoder_ramp10k_production_500k.yaml`.
  The objective, p32/p16 architecture, normalization, seed 42, B32, data bank,
  AdamW hyperparameters, and all original two-tail regression coefficients are
  unchanged from the encoder-only source; only the requested training scope and
  LR schedule differ.
- The production optimizer has two named groups over all 742,120,833 trainable
  parameters: 101 encoder tensors retain Adam state and constant LR; 152
  decoder/direct-conditioning tensors start without moments and receive the
  ramped LR. Resume checkpoints store the branch-relative step, initial cursor,
  source provenance, model, both optimizer groups, and RNG. Cursor validity is
  `committed == 416000 + branch_step * 32`.
- Focused tests passed 4/4, including schedule endpoints 0/1/5k/10k/20k and
  exact one-group-to-two-group Adam-state transfer. Py-compile and
  `git diff --check` passed. Final small-file hashes were trainer
  `4417f6c67d50...`, sidecar `e4f54d978cee...`, config `4fcdc7e5122c...`, and
  test `97ec2c7f15c5...`.
- B32 two-step smoke completed at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_encoder_warm_decoder_ramp10k_smoke_v1`.
  `warmstart_event.json` records encoder Adam 101, decoder Adam 0, and decoder
  tensor count 152. Step-1 loss was `2.975929`, cursor advanced to 416,032,
  decoder LR was `5e-9`, and every encoder/shared-decoder/tail/head group had a
  nonzero gradient. The smoke completed at cursor 416,064 with peak allocated
  VRAM 14.77 GiB. Its one-point plot is not trend evidence.
- One independent final reviewer issued explicit GO with P0=0/P1=0 after
  inspecting the final code/config, checkpoint and data contracts, smoke
  artifacts, CUDA allocation, telemetry, and branch-resume semantics.

### Live production state and early evidence

- Detached trainer PID `234801` is live on physical CUDA0. Production root:
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_encoder_warm_decoder_ramp10k_500k_v1`.
  Comet sidecar PID `235038`:
  `https://www.comet.com/mike-5531/big-weight-vae/fb7204dd1f09410cac409581c98e5c40`.
  The independent CUDA1 encoder-only trainer was not stopped or modified.
- The production process independently confirmed all 742,120,833 parameters
  trainable, encoder Adam state 101, decoder Adam state 0, and source cursor
  416,000. At branch step 1, loss was `2.975929`, behavioral `1.805029`,
  structural `1.170900`, pre-clip norm `15.637`, encoder LR `5e-5`, decoder LR
  `5e-9`, and all named gradient groups were live. At step 10, loss was
  `3.181875`, cursor `416,320`, and decoder LR `5e-8` (exact multiplier .001).
  CUDA0 was at 100% with about 17.97 GiB reported used and peak PyTorch
  allocation 14.77 GiB.
- These first ten online batches establish correct execution and telemetry, not
  whether gradual thaw improves convergence. The meaningful read is the
  trajectory through the 10k ramp and after decoder/encoder LR parity; all
  ordinary loss, component, gradient, rate, cursor, LR, and VRAM metrics are
  streamed to the artifact root and Comet for that review.
- A follow-up at branch step 100/110 confirmed that thaw updates remain live:
  step-100 decoder LR was `5e-7` (1% of encoder LR), and the Distribution
  Encoder, decoder conditioner, all five shared decoder blocks, both tails,
  both heads, and all encoder blocks had nonzero gradient RMS. No named group
  was dead; the ledger remained 253 trainable tensors / zero frozen. Across the
  online rows through step 110, loss ranged `2.9759..3.3883`; this short noisy
  range does not establish improvement or deterioration. Median post-start
  interval throughput was `0.487 step/s`, implying roughly 5.6 hours to the
  10k LR-parity boundary and roughly 11.9 days for 500k before checkpoint
  overhead. Decoder Adam allocation raised peak PyTorch memory to 16.42 GiB.

## 2026-08-31 — User stopped the gradual-decoder-thaw branch for poor quality

- The user judged the decoder-thaw run's quality to be poor and explicitly
  requested shutdown. This is a user conclusion/decision; no claim was made
  here that the online loss alone establishes a causal explanation.
- The exact CUDA0 trainer PID `234801` received `SIGTERM` and exited gracefully
  after its current optimizer step. `STOPPED.json` records branch step `6,326`,
  committed logical cursor `618,432`, source encoder-only step `13,000`, and
  initial cursor `416,000`.
- Final resumable state was preserved at
  `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_encoder_warm_decoder_ramp10k_500k_v1/resume_latest.pt`
  (8.3 GiB), and the model checkpoint at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_encoder_warm_decoder_ramp10k_500k_v1/model_latest.pt`
  (2.8 GiB). The branch had not reached LR parity: at step 6,326 its scheduled
  decoder LR was approximately `3.163e-5` versus encoder LR `5e-5`.
- Comet sidecar PID `235038` was then terminated gracefully. No matching thaw
  trainer, loader worker, or sidecar remained. Repeated `nvidia-smi` showed
  physical CUDA0 free at 3 MiB / 0% utilization. The separate CUDA1 strict
  encoder-only trainer and its Comet sidecar remained live and were not
  signaled.

## 2026-08-31 — Proposed 10M mini polar AE for much faster experiments

### User goal

- The user proposed replacing expensive 742M exploratory runs with an
  approximately 10M-parameter model, reducing hidden and latent widths,
  changing the input patch to 16, and using 32x32 weight tiles. The explicit
  goal is experiments that run many times faster.
- This entry is an architecture proposal, not yet an implemented or measured
  speed/result claim. "VAE" is provisionally interpreted as the current
  deterministic polar weight bottleneck; adding a stochastic posterior/KL
  would be a separate scientific intervention.

### Recommended topology and exact parameter budget

- Preserve the scientifically relevant two-tail decoder topology: 5 shared
  decoder blocks, one direction tail, and one scale tail. Reduce encoder depth
  from 13 to 10 rather than disproportionately starving the decoder.
- Proposed dimensions: tile size 32; input and output patch size 16;
  `hidden_dim=192`, `heads=6` (head width 32), `mlp_dim=704`,
  `latent_dim=48`, and `latent_slots=16`. The latent code therefore has
  `16*48=768` scalars for 1,024 tile weights, exactly the large model's latent
  density `32*384/16384 = 0.75`.
- Proposed mini Distribution Encoder: `k_s=16`, `Kq=32`, `d_var=64`,
  `d_dist=64`, 2 variable-attention layers / 4 heads, 2 DCN cross layers, and a
  32-wide two-layer deep tower. Keep covariance enabled at patch size 16.
- With a 32-row group embedding, 2 chunks per row, 64 weight tokens, preserved
  global tile coordinates (72 row positions / 8 column positions), and one p16
  child per input token, the exact analytical parameter count derived from the
  current module definitions is `9,938,689`. Breakdown: encoder blocks
  5,656,320; five shared decoder blocks 2,766,720; tails 553,344 each;
  Distribution Encoder 135,936; decoder conditioner 172,608; all remaining
  embeddings/projections/norms/heads 100,417.

### Expected speed mechanism and required implementation

- Current encoder sequence length is `512+32=544`, and polar tails process
  `1024+32=1056` states. The mini design uses `64+16=80` everywhere: 6.8x
  shorter in the encoder and 13.2x shorter in each tail. Parameters fall 74.7x.
  A dense-projection-plus-attention FLOP proxy computed from the exact block
  shapes is about 563x lower per example. This is only a theoretical compute
  ratio; small-matrix utilization, the Distribution Encoder, Python, and data
  loading will bound realized speed.
- The current implementation cannot accept this config unchanged: tile width,
  masks, group IDs, reshape logic, distribution patch indices, losses, and
  polar output shapes contain fixed 128 constants. They must be generalized to
  a `tile_size` field; the p32-to-two-p16 split becomes a p16-to-one-p16 path.
- Reuse the sealed 128x128 bank through a deterministic 4x4 subtile view rather
  than downloading another dataset. Each record needs stable parent/subrow/
  subcol lineage, matching 32-wide activation slices and masks, and new
  normalization statistics computed on the 32x32 training view. A local
  subtile operator objective is a proxy for one contribution to the full
  128-wide operator, not numerically identical to the old task; absolute loss
  levels must not be compared across tile sizes.
- Disable activation checkpointing for the 10M model. Start the throughput
  benchmark at B512 (same number of raw weight scalars per step as old B32) and
  fall back to B256 only if measured memory/kernel behavior requires it. Report
  examples/s and weight-scalars/s, not just steps/s. The first acceptance gate
  should be an executable contract test plus a short measured A100 throughput
  comparison; no end-to-end speedup number is established before that test.

### Distribution Encoder clarification

- The recommended `9,938,689`-parameter count includes a mini Distribution
  Encoder rather than silently removing activation-distribution conditioning.
  Its proposed dimensions are `k_s=16`, `Kq=32`, `d_var=64`, `d_dist=64`, two
  variable-attention layers with four heads, two DCN cross layers, a 32-wide
  two-layer deep tower, and covariance at patch size 16. Its own exact parameter
  count is 135,936, versus 5,384,576 in the current production model.
- The total also includes 122,880 parameters for 64-wide distribution-key
  projections in ten encoder blocks and 172,608 parameters in the decoder query
  conditioner. Thus conditioning and its decoder bypass remain represented in
  the topology and parameter budget.
- A paired `use_distribution_conditioning=false` ablation would contain
  9,507,265 parameters after removing those three pieces. It is useful as a
  speed/causal control, but it changes available information and must not
  replace the conditioned mini baseline without an explicit user decision.

## 2026-08-31 — Implemented and measured the conditioned 9.94M mini regression baseline

### User decision and historical objective

- The user asked to implement and launch the mini model in the last ordinary
  regression setup immediately before categorical/VQ/autoregressive work. The
  selected objective is the original coordinate-owned two-tail polar loss:
  `behavioral_direction + 10*behavioral_scale + structural_direction +
  10*structural_scale`. Behavioral direction and scale are measured through
  the local operator action `XW`; standalone raw operator MSE has coefficient
  zero. Categorical targets, VQ, autoregression, InfoNCE, KL, and later raw
  direction-MSE variants are absent.
- The user explicitly asked about the Distribution Encoder. The implemented
  baseline retains it and its decoder query conditioner; this is a
  deterministic polar autoencoder, not a newly introduced stochastic
  posterior/KL VAE.

### Implementation and data contract

- The standalone trainer is
  `projects/weight-vae/workspace/training/weightclip_benchmark/run_mini_polar_regression_production.py`;
  the production config is
  `projects/weight-vae/workspace/conf/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_production.yaml`.
  The exact model has 9,938,689 trainable parameters: tile32/p16,
  hidden 192, MLP 704, ten encoder blocks, five shared decoder blocks, one
  direction and one scale tail, latent 16x48, and the proposed 64-wide mini
  Distribution Encoder.
- The sealed 128x128 operator-bank stream is expanded deterministically into
  nonempty 32x32 subtiles. Completely padded subtiles are skipped instead of
  becoming zero-loss training examples. A composite next-consumed cursor
  `(parent_logical_index, subpatch_index, emitted_logical_index)` is saved with
  model, AdamW, and RNG state for exact process-stop resume.
- Distribution conditioning is deduplicated only for identical
  `(parent,input-subtile)` activation contexts inside a batch, then gathered
  back to each output subtile. A focused equality test established that this
  gives the same Distribution Encoder outputs as evaluating duplicates
  independently. The behavioral implementation similarly shares the exact
  target `XW` calculation; a focused test establishes bitwise equality of the
  direction and scale scalar terms with the historical loss helper.
- A sampled tile32 normalization artifact was computed from 32,768 valid
  subtiles / 1,048,576 valid output rows and stored at
  `/mnt/shared/weightclip_benchmark/calibration/mini_p16_tile32_norm_seed42_32768.json`.
  The measured `log2(maxabs/7)` mean is `-6.81575982` and standard deviation is
  `0.905690685`. This replaces the incompatible p128/p32 scale statistics.

### Executable and throughput evidence

- `tests/mini_polar_regression_production_test.py` passes three focused tests:
  exact parameter count/full forward-backward with every parameter live,
  nonempty-subtile/resume behavior, Distribution Encoder dedup parity, and
  exact historical behavioral-loss parity. Python compilation and
  `git diff --check` also pass.
- The final measured B128 smoke is
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_smoke_20260831T141129Z`.
  It completed four updates with finite losses `3.0762, 3.1600, 2.9842,
  2.6386`; step 1 had zero missing parameter gradients and every named group,
  including all ten encoder blocks, Distribution Encoder, conditioner, shared
  decoder, both tails, and both heads, had positive gradient RMS. Peak
  allocation was 1.96 GiB. Post-step-1 throughput was `3.641, 2.854, 2.466`
  step/s (median 2.854) and median 365.3 mini-tiles/s.
- The B512 volume-throughput control is
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_smoke_20260831T140357Z`;
  its post-step-1 median was 0.654 step/s and peak allocation 7.44 GiB. Because
  the user's goal is much faster experiment iteration, the production config
  selects B128. Relative to the observed approximately 0.487 step/s large-model
  run, this is about 5.9x faster in optimizer steps and about 23x higher in
  examples/s. These are short throughput measurements, not convergence or
  quality conclusions.
- The smoke loss curve at
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_smoke_20260831T141129Z/train_losses.png`
  was inspected and is readable. Four noisy batches are insufficient to infer
  a trend; the production trajectory must be reviewed after meaningful
  checkpoints. The mini local 32-wide `X_sub @ W_sub` task is a proxy for one
  input-block contribution, so its absolute losses are not directly comparable
  to the original 128-wide task.

### Production prelaunch state

- Physical CUDA0 was free at 3 MiB / 0% while the separate original strict
  encoder-only run remained isolated on CUDA1. Final B128 production output and
  `/dev/shm` resume paths were absent. The Comet sidecar was extended to record
  the mini architecture, parameter count, Distribution Encoder settings, and
  ordinary loss/gradient/rate/VRAM metrics.
- An independent final review of the exact source/config/test/sidecar hashes
  was requested as the mandatory production gate. Production launch is pending
  that explicit GO; the user has already authorized immediate launch after the
  gate, without another approval prompt.

## 2026-08-31 — Mini conditioned polar regression production launch

- The independent final reviewer returned explicit `GO` with `P0=0, P1=0` for
  the final B128 setup. The review confirmed the exact historical objective,
  9,938,689-parameter conditioned model, Distribution Encoder path, nonempty
  subtile stream and composite resume cursor, fused-loss parity, normalization
  artifact, checkpoint/telemetry behavior, sidecar compatibility, physical
  CUDA0 availability, and unique production paths.
- Production was launched on physical CUDA0 from the Weight-VAE workspace with
  `PYTHONPATH=.`. Trainer PID is `296265`; Comet sidecar PID is `296447`.
  Output root is
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1`,
  stdout log is the sibling
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1.launcher.log`,
  and the live Comet run is
  `https://www.comet.com/mike-5531/big-weight-vae/301106138caa4a638eca142a35f922e6`.
  The separate original encoder-only job remains isolated on CUDA1.
- Production step 1 exactly reproduced the B128 smoke start: loss `3.0761845`,
  behavioral `1.9503107`, structural `1.1258738`, raw operator-MSE contribution
  zero, pre-clip gradient norm `27.0537`, cursor 128, and peak allocation about
  1.88 GiB. Both step-1 and step-100 gradient ledgers have zero missing tensors,
  zero dead named groups, and all 23 model groups live.
- At the later live read around step 170 / cursor 21,760, every stored scalar
  was finite. Loss was `2.38239`, with the early logged range `2.27086..3.07618`;
  cumulative throughput was `2.596 step/s`, implying about 53.5 hours to the
  500k horizon if sustained. These are early operational measurements, not a
  convergence or quality conclusion. The first resumable checkpoint is due at
  step 1,000 and the first persistent model checkpoint at step 10,000.

## 2026-08-31 — Diagnosed the mini run's pathological-looking directional-loss trace

- The user flagged the directional-loss graph as abnormal. Live inspection at
  roughly step 17.5k confirmed a long early behavioral-direction plateau near
  `0.87`, followed by macro improvements near 7k and 10.5k, plus a strong
  short-period sawtooth. The run was still finite and live: behavioral direction
  had improved from `1.0029` at step 1 to about `0.52..0.56`, structural
  direction from `0.6225` to roughly `0.33..0.44`; every stored gradient ledger
  through 17.5k had live direction tail/head gradients. Thus explosion, NaN,
  and a disconnected direction head were excluded.
- The high-frequency "schizophrenic" trace has a measured data-order mechanism.
  B128 counts subtiles rather than independent parent operators. It consumes
  only about `13.26` parent p128 records per optimizer step; the other examples
  are correlated output/input siblings. The balanced bank plan cycles through
  200 strata round-robin, so one stratum sweep is approximately
  `200 / 13.26 = 15.1` optimizer steps. Metrics sampled every ten steps alias
  this into an approximately 30-step oscillation.
- After local detrending over the post-12k trace, lag-30 autocorrelation was
  `0.81` for total loss, `0.73` for structural direction, `0.50` for behavioral
  direction, and `0.82` for behavioral scale. The separate large p128/B32 run
  shows a weaker sampler harmonic at lag 50, consistent with its 32 parent
  records per step and much greater batch lineage diversity. This discriminates
  sampler/batch correlation from an intrinsically oscillatory directional-tail
  optimizer failure.
- The slow macro plateau and the two later downward phases are not fully
  explained by the 30-step alias. The second transition occurs after the first
  135,100-parent bank cycle boundary and new cycle/gauge assignments, but not
  exactly at the boundary; a causal claim that the gauge-cycle switch triggered
  it is therefore not established. A clean fix/test would interleave subtiles
  from a deterministic reservoir of many parents, preserve a resumable shuffle
  cursor, and compare the same-start directional trajectory. No such mutation
  or production stop was performed in this diagnostic-only turn.

## 2026-08-31 — Causal diagnosis of the mini directional plateau and two macro drops

### Failure definition and validity checks

- The user clarified that the target is the repeatable macro pattern, not the
  approximately 30-step sampler sawtooth: behavioral direction sits near
  `0.87` through about step 7k, improves gradually, falls sharply around
  10.6k--11k, plateaus near `0.62` through about 15k, then enters a second
  sustained fall to about `0.44` by 19.4k.
- This is genuine model learning rather than changing online batch difficulty.
  The same fixed B128 probe at stream cursor zero gives behavioral-direction
  loss `1.00287` at seed initialization, `0.75296` at the preserved step-10k
  model, and `0.46889` at the preserved step-19k state. Structural direction on
  that same probe changes `0.62248 -> 0.48771 -> 0.33961`. The complete fixed
  probe and intervention report is
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1/direction_phase_diagnostic_v1/fixed_probe_report.json`;
  the preserved checkpoints are in the same directory.
- External schedule explanations are excluded. The optimizer LR is constant at
  `5e-5`; clipping frequency is zero during the 500--7k plateau and below 0.3%
  in the later phases. There is no LR restart or thaw schedule. The first bank
  cycle ends near step 10.19k, but loss does not discontinuously improve there;
  the second macro fall begins around 15k--17k, far from a bank-cycle boundary.
  Thus a new gauge assignment may stimulate the first transition but cannot be
  the common mechanism of both transitions.

### Competing mechanisms and discriminators

- H1, loss-invisible radial growth recreates the old polar weak-gradient trap.
  Prediction: a fall should coincide with raw-logit/state radius shrinking and
  restored normalization Jacobian. H2, a decoder/query common-template saddle
  initially ignores the weight latent, followed by stagewise recruitment of
  sample-specific latent modes. Prediction: latent permutation sensitivity,
  prediction diversity, target alignment, and effective rank should increase
  across the macro transitions. H3, the decoder escapes only through the direct
  Distribution Encoder context bypass. Prediction: permuting DE context should
  dominate permuting `z`. H4, changing data/gauge composition only makes the
  online batches easier. Prediction: gains should disappear on a fixed probe.
  H5, the scale task or clipping/scheduler unlocks direction. Prediction: only
  behavioral direction should improve, or the transitions should align with an
  optimizer event.

### Discriminating results and supported mechanism

- Initialization is almost a common-template decoder: predicted-direction
  cross-sample cosine is `0.9781` while target cross-sample cosine is `0.00094`;
  rolling `z` across samples changes behavioral direction by only `-0.00113`.
  At step 10k the prediction cosine is `0.0133` and rolling `z` worsens direction
  by `+0.26159`; at step 19k they are `0.00402` and `+0.52452`. Therefore the
  mini model really transitions from ignoring sample identity in `z` to relying
  causally on it.
- DE context is also causally used, but it is not the sole escape route. Its
  same-probe roll penalty grows `-0.00429 -> +0.09041 -> +0.22243`, whereas the
  latent-roll penalty is larger at both learned checkpoints. From step 10k to
  19k, prediction/target patch cosine rises `0.2172 -> 0.4550` and latent entropy
  effective rank rises `20.97 -> 38.08`. The second phase therefore recruits
  additional useful latent modes and stronger joint latent/context decoding,
  rather than merely learning an activation-only bypass.
- The temporal gradient ordering repeats. Before the first sharp loss fall,
  median DE gradient RMS rises from `0.000552` at steps 9401--9800 to `0.00103`
  at 10201--10600 while behavioral direction is nearly flat
  (`0.7529 -> 0.7377`). At 10601--11000, bottleneck gradient rises to `0.00323`,
  direction-tail to `0.000190`, direction-head to `0.00759`, and behavioral
  direction falls to `0.6498`. Before the later fall, DE gradients again burst
  during 15.4k--17.4k; during 17.8k--19.4k direction-tail/head and downstream
  decoder gradients increase roughly two to threefold while the loss falls
  from about `0.51` to `0.44`. This supports a repeated two-stage process:
  representation/context alignment accumulates first, then useful credit
  reaches the bottleneck and directional decoder and the observable loss moves.
- H1 contributes to the length of the plateau but does not trigger the falls.
  The direction-head Frobenius norm grows `0.0787 -> 0.2923 -> 0.3595` and raw
  logit norm grows `0.0833 -> 0.7904 -> 0.8706`; the second fall happens without
  a radius collapse. Unlike the old 700M failure, direction-tail state RMS is
  bounded (`0.1664 -> 0.1564` from 10k to 19k) and the latent does not collapse
  to rank one. The smaller model can therefore cross the coupled
  encoder/decoder saddle that the old runaway-radii model could not cross.
- H4 and H5 are excluded as primary mechanisms: the fixed-probe gains remain;
  structural direction improves alongside behavioral direction even though
  scale is detached from the direction route; and there is no scheduler or
  clipping event. Checkpoint saves also occur repeatedly without matching every
  transition.

### Narrow conclusion and remaining discriminator

- The best-supported mechanism is delayed, stagewise symmetry breaking of a
  coupled autoencoder path. Learned output queries and DE context initially let
  the decoder emit a nearly sample-independent direction template, so the
  encoder and decoder face a chicken-and-egg saddle: `z` is not yet useful to a
  decoder that does not listen to it, and the decoder receives little useful
  sample-specific signal from `z`. Representation/DE gradients accumulate;
  once an aligned mode crosses the sensitivity threshold, bottleneck/tail
  credit increases and the loss drops. A second set of modes is recruited later,
  producing the second fall. This is a grokking-like phase transition in route
  usage, not evidence of a discrete scheduler change.
- The exact trigger attribution between DE, encoder and decoder remains a
  leading mechanism rather than a completed intervention proof because no
  pre-7k or pre-15k fixed-probe checkpoints exist. A decisive future run should
  save fixed-probe geometry every 250--500 steps and branch the same optimizer
  state immediately before a transition into normal, frozen-DE, frozen-latent
  encoder, and context-dropout continuations. No production mutation or stop
  was made during this diagnosis.

## 2026-08-31 — Literature map for encoder lag, decoder latent ignoring, and staged loss drops

- User framing: the repeated pattern is representation collapse caused by a
  weak encoder at initialization, after which the decoder learns to ignore it.
  Terminology needs one qualification. In a stochastic VAE, `posterior
  collapse` strictly means that the approximate posterior approaches the prior.
  This mini model has no KL or stochastic posterior, so `latent-pathway
  collapse`, `latent ignoring`, or `decoder-side shortcut collapse` is the more
  exact name. The training mechanism is nevertheless directly covered by the
  posterior-collapse literature.
- He et al., *Lagging Inference Networks and Posterior Collapse in Variational
  Autoencoders* (ICLR 2019, https://arxiv.org/abs/1901.05534) is the closest
  match to the proposed causal ordering. It reports that early in training the
  inference/encoder network lags the moving true posterior, which encourages
  the model to ignore the latent; their intervention is aggressive inference
  network updates before model updates.
- Fu et al., *Cyclical Annealing Schedule* (NAACL 2019,
  https://aclanthology.org/N19-1021/) gives the closest architectural picture.
  A low-quality early `z` makes the decoder discard latent Path A and use the
  easier autoregressive/bypass Path B. In this project, the analogous Path B is
  the learned output-query plus direct Distribution Encoder context route. The
  paper's KL schedule is not directly applicable because the mini model has no
  KL; its two-path diagnosis is applicable.
- Chen et al., *Variational Lossy Autoencoder* (ICLR 2017,
  https://openreview.net/forum?id=BysvGP5ee) calls the broader effect the
  information-preference property: a sufficiently expressive decoder models
  predictable structure without spending latent information. Dang et al.,
  *Beyond Vanilla VAEs* (ICLR 2024,
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/26ded5c8ee8ec1bc4caced4e1c9b1584-Abstract-Conference.html)
  extends collapse theory to conditional VAEs and shows that the relationship
  between conditioning input and output affects collapse. This is especially
  relevant to the activation-derived DE decoder side channel.
- Hu et al., *Complexity Matters: Rethinking the Latent Space for Generative
  Modeling* (NeurIPS 2023,
  https://papers.nips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html)
  is the closest deterministic-autoencoder result. It argues for a relatively
  weak decoder during representation learning and proposes Decoupled
  AutoEncoder training: first train the encoder with an auxiliary weaker
  decoder, then freeze that encoder and train the full decoder. This supports a
  controlled encoder-first/decoder-later experiment more directly than VAE KL
  remedies do.
- Saxe et al., *Exact Solutions to the Nonlinear Dynamics of Learning in Deep
  Linear Neural Networks* (ICLR 2014, https://arxiv.org/abs/1312.6120) gives a
  theoretical analogue for the visible curve shape: deep factorized paths can
  have long plateaus followed by rapid drops and learn correlation modes in
  stages. This supports the interpretation of the two drops as sequential mode
  recruitment, but it is an analogue rather than a proof for the nonlinear
  conditioned transformer.
- Other useful boundaries: Lucas et al., *Don't Blame the ELBO!* (NeurIPS 2019,
  https://papers.nips.cc/paper/9138-dont-blame-the-elbo-a-linear-vae-perspective-on-posterior-collapse)
  establishes that collapsed local solutions are not solely a KL or powerful
  decoder artifact. Dieng et al., *Avoiding Latent Variable Collapse with
  Generative Skip Models* (AISTATS 2019,
  https://proceedings.mlr.press/v89/dieng19a.html) shows that architecturally
  enforcing stronger latent-to-likelihood links increases mutual information
  and reduces collapse.
- Project-specific synthesis, not a literature-established fact: the fixed
  probe shows that the mini decoder ignores `z` at initialization even though
  the initial latent has nontrivial diversity. Therefore `weak/low-quality
  encoder route -> decoder chooses bypass -> encoder later loses credit` is
  better supported than saying the encoder representation itself is already
  collapsed at step zero. The literature-backed next discriminator would use
  the same start and compare normal joint training against (1) several encoder
  updates per decoder update, (2) a weak/minimal decoder warm-up, and (3) a
  temporarily gated or dropped DE decoder bypass, followed by a controlled
  decoder ramp. No experiment change was authorized or launched in this
  literature-review turn.

## 2026-08-31 — Mini baseline review and acceleration priority

- User believes the 10M mini polar-regression baseline may already be trained
  sufficiently based on its Comet curves and asked for the experiment link
  again. The next intended direction is to make training substantially faster.
- Live operational evidence at review time: the production process was still
  active and had reached step `61,690 / 500,000` (12.3%), not the configured
  terminal step. Its run root is
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1`,
  and Comet is
  `https://www.comet.com/mike-5531/big-weight-vae/301106138caa4a638eca142a35f922e6`.
  Therefore process completion is false; whether the learning curves have
  converged sufficiently is not yet established and remains a user/artifact
  review question.
- No stop or training change was requested in this turn. Keep the current run
  intact pending the user's inspection, then profile and optimize the measured
  training bottleneck under the same regression contract.

## 2026-08-31 — Mini baseline stopped; latent-preference objective framing

- User requested a checkpoint-preserving stop of the current 10M mini baseline
  and proposed two related additions: (1) a small two-layer projection head with
  a contrastive representation loss, with the positive/negative identity still
  undecided (dataset versus layer or another grouping); and (2) a decoder loss
  that prefers the existing losses under the correct latent over the same losses
  under a shuffled latent. User suggested treating the latter as a wrapper over
  all current behavioral/structural losses and compared the idea to an RL-style
  preference baseline.
- Operational evidence: SIGTERM was sent only to the mini trainer. The run
  stopped cleanly at step `72,611`; the trainer workers and Comet sidecar then
  exited. `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1/STOPPED.json`
  records `status=stopped`, the exact composite cursor, and both checkpoint
  paths. Fresh full resume state is at
  `/dev/shm/weightclip_mini_polar_regression_10m_p16_tile32_b128_500k_v1/resume_latest.pt`;
  the persistent model is at
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1/model_latest.pt`.
  Physical CUDA 0 was verified free at 3 MiB and 0% with no compute process;
  the unrelated encoder-only run remains on CUDA 1.
- Post-run review: the inspected plot
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1/train_losses.png`
  is readable and shows the large staged decreases followed by a much shallower
  noisy regime. The final `60k-72.6k` medians were total `0.2597`, behavioral
  `0.1237`, structural `0.1385`, behavioral direction `0.07343`, and diagnostic
  operator relative MSE `0.1667`. A fit to 1k-block medians over `40k-72k`
  still decreased by about 8.2% of the median total loss per 10k steps, so exact
  convergence is not established, although marginal gains are much smaller.
- A fresh fixed-probe audit was stored at
  `/mnt/shared/weightclip_benchmark/mini_polar_regression_10m_p16_tile32_b128_500k_v1/direction_phase_diagnostic_v1/final_step_00072611_report.json`.
  At step 72,611 the fixed-probe base loss was `0.4661`; rolling latents across
  samples raised it by `+2.4625`, whereas rolling direct distribution context
  raised it by `+0.6382`. Latent entropy effective rank was `98.05` (versus
  `22.51` at initialization), stable rank `11.55`, and top energy fraction
  `0.0866`. Therefore the final checkpoint does not exhibit a decoder that
  ignores `z` under this random-roll intervention. The observed initial
  near-independence was transient; a new objective would aim to establish the
  dependency earlier or under harder matched negatives, not repair proven
  final-step collapse.
- Recommended formulation, not yet implemented: retain the absolute original
  positive loss and add a bounded pairwise preference term. For per-example
  `L_pos` and same-loss `L_neg`, define a relative gap
  `g=(L_neg-L_pos)/stopgrad(L_pos+eps)` and optimize a smooth hinge/logistic
  `softplus((margin-g)/temperature)`. This is DPO/RankNet-like energy
  preference with reward `-L`, not reinforcement learning. Do not replace the
  original loss: pure difference optimization can succeed by making the
  shuffled branch arbitrarily bad. A finite margin makes the negative pressure
  stop after sufficient separation.
- Negative latents should be detached and matched on nuisance identifiers such
  as dataset, layer/operator role/depth, tile row/column, and mask signature,
  while differing in checkpoint/lineage. Random shuffles or shuffles across
  tile coordinates permit trivial dataset/layer/coordinate shortcuts. Current
  batches do not carry all available sample metadata, so the collator would
  need to retain `dataset`, `lineage_id`, `checkpoint_sha256`, `layer_key`,
  `gauge_id`, and coordinates. The current model already exposes separate
  `encode` and `decode_polar`, so a second latent-only decoder pass is otherwise
  straightforward; the routed loss must be refactored to expose per-example
  values rather than only a batch aggregate.
- The representation head and decoder preference are complementary. A head-only
  loss can learn layer/dataset identity while the decoder still ignores `z`;
  preference-only training can exploit a coordinate mismatch. For the head,
  positives defined merely as “same layer” are not recommended if the goal is
  instance-specific operator information, because they pull distinct operators
  together. The cleanest candidate is two controlled views of the same p32
  sample, with hard negatives from the same layer and coordinates but another
  checkpoint. The existing full-operator gauge views cannot automatically be
  paired by equal p32 coordinates because permutations cross tile boundaries.
  This positive identity remains a material user decision before implementation.
- Literature boundaries: VICReg (https://arxiv.org/abs/2105.04906) provides an
  explicit variance/covariance anti-collapse alternative when reliable
  negatives are unavailable; SupCon (https://arxiv.org/abs/2004.11362) supports
  multiple label-defined positives but does not resolve what semantic identity
  should be used here; DPO (https://arxiv.org/abs/2305.18290) motivates the
  smooth pairwise-classification shape, but its policy/reference derivation does
  not directly transfer to this deterministic regression decoder.

## 2026-08-31 — Final mini latent-preference experiment design

- User requested the final setup design. The frozen design is stored at
  `projects/weight-vae/workspace/docs/mini_polar_latent_preference_final_setup_20260831.md`;
  it is a design artifact only, and no new training was launched in this turn.
- Main scientific decision: the causal panel starts from the exact fresh seed-42
  initialization, not the warm step-72,611 checkpoint, because the fixed-probe
  audit already shows strong final latent dependence. The stopped checkpoint is
  retained as the quality/reference endpoint. Starting from it would not test
  whether the proposed losses prevent the early weak-latent phase.
- Positive identity is fixed to the same individual p32 operator sample under
  two projector-dropout views. “Same layer” and “same dataset” are rejected as
  positive labels because they allow coarse prototypes and can remove
  instance-specific information.
- Hard negatives are exact nuisance-matched pairs: same dataset, training epoch,
  layer key, operation/depth/role, parent and p32 coordinates, masks and graph
  gauge, but different lineage/checkpoint. The primary negative is a different
  lineage at the same epoch; adjacent epochs in one lineage are evaluation-only
  because near-identical weights can become false negatives.
- The 3k-step causal panel is canonical-only and includes a paired-sampler
  control, representation-only, preference-only and combined arm. Historical
  baseline metrics are contextual rather than the formal control because the
  paired canonical view schedule changes ordering/views. A winning arm must
  later pass a shared-gauge confirmation in which exactly the same verified
  gauge is applied to both pair members; equal view indices from different
  checkpoints are not sufficient.
- The representation head is training-only `RMSNorm -> flatten(768) ->
  256-GELU-dropout(0.1) -> 128 -> L2`, with symmetric binary InfoNCE at
  temperature 0.1. The decoder preference is a relative smooth hinge over the
  exact original per-example routed loss, margin 0.5 and temperature 0.1. The
  original positive loss always remains the anchor. Foreign latents and direct
  conditioning/query state are detached in the negative branch so the
  preference gradient targets decoder use of `z` rather than a conditioning
  mismatch shortcut.
- Static auxiliary coefficients are calibrated once on a fixed paired probe to
  initial gradient ratios 0.25 (preference/base) and 0.10
  (representation/base), stored in an artifact, and linearly ramped during the
  first 1k steps. No adaptive online rescaling is allowed. A weighted auxiliary
  ratio above 0.75 stops the run diagnostically.
- Panel arms run for 3k steps from identical starts; the winner extends to 30k,
  covering the historical transition regions, and only then may extend to
  72,611. Required gates cover positive-loss preservation, hard matched gap,
  component non-cheating, latent rank, paired-swap sensitivity, exact pairing,
  gradients and throughput. An independent reviewer GO is required before a
  long confirmation launch.
- Math-preserving speed work is part of every arm: background four-batch CPU
  prefetch, pinned transfer, overlap, and removal of unconditional per-step CUDA
  synchronization. Fused AdamW and `torch.compile` are intentionally excluded
  from the causal loss comparison. The optimized paired control targets at
  least 1.5x historical median throughput; each loss arm must retain at least
  70% of that control throughput.

### 2026-08-31 — Exact representation-loss definition

- User asked for the exact `L_repr` after confirming the original routed polar
  regression `L_base`.
- For every nuisance-matched pair `(i,j)`, form two independent dropout views
  `h_i^a,h_i^b` and `h_j^a,h_j^b` with the training-only normalized projector.
  The positive for an anchor is its other view of the exact same tile; the only
  training negative is the matched tile from the other lineage.
- Define `ell(u,v,n) = -log(exp(cos(u,v)/0.1) /
  (exp(cos(u,v)/0.1) + exp(cos(u,n)/0.1)))`. The exact symmetric pair loss is
  the mean of `ell(h_i^a,h_i^b,h_j^b)`, `ell(h_i^b,h_i^a,h_j^a)`,
  `ell(h_j^a,h_j^b,h_i^b)`, and `ell(h_j^b,h_j^a,h_i^a)`, then averaged over
  the 64 matched pairs in B128. Gradients flow through the encoder and
  projector; there is no stop-gradient in `L_repr`.
- This is deliberately binary matched-negative InfoNCE, not all-batch InfoNCE:
  easy negatives from unrelated layers/datasets must not dominate the signal.
  The projector is discarded at decoding/evaluation time. Raw `L_repr` is near
  `log(2)` when positive and negative similarities are indistinguishable and
  approaches zero when the exact-sample positive wins cleanly.

## 2026-09-01 — Combined mini latent-preference production launch

- User overrode the staged 3k/30k causal panel and approved an immediate fresh
  seed-42 combined run for the full 500,000-step horizon, with no additional
  approval gate. This is not a warm-checkpoint continuation: the experiment is
  intended to test whether the two auxiliary losses prevent the early weak-z
  phase that a warm converged checkpoint cannot reproduce.
- Implemented production trainer/config/test/design artifacts at
  `projects/weight-vae/workspace/training/weightclip_benchmark/run_mini_polar_latent_preference_production.py`,
  `projects/weight-vae/workspace/conf/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_production_500k.yaml`,
  `projects/weight-vae/workspace/tests/mini_polar_latent_preference_production_test.py`,
  and `projects/weight-vae/workspace/docs/mini_polar_latent_preference_final_setup_20260831.md`.
  The historical pre-categorical two-tail polar regression remains the base
  objective; the 9,938,689-parameter core is augmented by a 229,808-parameter
  training-only projector. B128 contains 64 exact-layout, same-checkpoint-index,
  different-lineage hard pairs from the unchanged pinned operator bank.
- The first independent review returned NO-GO with two P1 blockers: the Comet
  sidecar could not parse the new gradient schema, and the planned causal
  telemetry was incomplete. Both were fixed. The final trainer records full
  positive/foreign component losses and gap distributions, pair distances,
  latent geometry, four fixed-probe interventions, and objective-split module
  gradients; the sidecar streams these data and five readable plots. The final
  independent re-review returned GO with P0=0 and P1=0 after a current-tree
  nine-test pass and inspection of the final B128 smoke at
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_smoke_20260831T234940Z`.
- Smoke calibration fixed `lambda_pref=7.293307614` and
  `lambda_repr=0.018456987` without clipping. The fresh production calibration
  reproduced these values (`7.29331231` and `0.01846324`). Pair validity,
  lineage mismatch, checkpoint mismatch, and same checkpoint-index fractions
  were all exactly 1.0; no NaN/Inf was observed.
- Production is live on physical CUDA 0 with trainer PID 408988 and sidecar PID
  410002. Run root:
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_b128_500k_v1`;
  launch log is the adjacent `.launch.log`; exact resume checkpoint is
  `/dev/shm/weightclip_mini_polar_latent_preference_10m_p16_tile32_b128_500k_v1/resume_latest.pt`.
  Comet experiment:
  `https://www.comet.com/mike-5531/big-weight-vae/5860bf85c2c84743833b1c2847df2d3a`.
  CUDA 1 and its unrelated encoder-only run were not touched.
- Early evidence only, not a convergence claim: at step 120, all rows were
  finite, peak allocated VRAM was 2.62 GiB, representation pair accuracy was
  1.0, and median preference gap was 0.088 while p10 was approximately -0.001.
  The step-100 auxiliary/base gradient ratio was 0.331 at ramp 0.10, below the
  0.75 diagnostic stop. Median throughput from steps 20-120 was only 0.452
  steps/s (about 12.8 days projected), dominated by roughly 2-3 seconds of hard
  pair data wait versus roughly 0.08-0.35 seconds of GPU work. Therefore the
  scientific run is valid and active, but the requested speedup is not yet
  established; hard-pair I/O is the leading operational bottleneck to optimize
  without changing the loss contract.

### 2026-09-01 — Production diagnostic stop at step 200

- The preceding live-status entry was superseded minutes later by the approved
  fail-closed guard. At attempted step 200, weighted auxiliary/base gradient
  ratio reached `0.99662264`, exceeding the fixed `0.75` threshold. Step 200 was
  not committed. The trainer saved an exact resume checkpoint and persistent
  model at committed step 199, wrote `DIAGNOSTIC_STOP.json` and `STOPPED.json`,
  and released CUDA 0. The Comet sidecar uploaded and flushed all available
  rows, then was selectively terminated; CUDA 1 remained untouched.
- Validity checks passed: all stored metric/gradient values are finite, pair
  contract fractions remain exactly 1.0, and this was not a data, resume, GPU,
  or logging failure. The five final plots were inspected and are readable.
- Directly supported proximal mechanism: first-batch static calibration did not
  remain predictive as training changed the two objectives. Base gradient L2
  fell from `51.92` at step 1 to `7.62` at step 100, while weighted preference
  gradient L2 rose from `0.0130` to `2.525`; at step 100 this already produced
  auxiliary/base `0.331` with only 10% ramp. At step 200, 20% ramp crossed the
  guard. Objective-split telemetry at step 100 shows preference pressure in the
  encoder, decoder trunk and especially direction tail, rather than a projector
  logging artifact. The deeper cause of the continued preference-gradient
  amplification is not yet distinguished from the simultaneous decay of the
  easy base-regression gradient.
- Early scientific signal was mixed: representation pair accuracy reached 1.0,
  but preference p10 stayed negative and the margin-satisfied fraction remained
  near zero through the stored metrics. Thus bypassing the guard is not
  justified. Changing `lambda_pref`, the 1k ramp, or using online gradient
  budgeting would define a new experiment and requires an explicit protocol
  choice; the stopped step-199 state is preserved for either resume diagnostics
  or comparison, but a fresh start is the fair test of a revised prevention
  schedule.

### 2026-09-01 — Why the combined run did not continue

- User asked what specifically failed. The model and representation objective
  did execute; the failed assumption was that one random-initialization batch
  could calibrate a fixed preference coefficient for the later non-stationary
  optimization. Calibration chose `lambda_pref=7.2933` because the initial
  unweighted preference gradient L2 was `1.7798` versus base `51.9217`, targeting
  a full-strength ratio of 0.25.
- The linear coefficient ramp controlled only the scalar coefficient, not the
  realized gradient ratio. From step 1 to step 100, base gradient L2 fell
  `51.92 -> 7.62` (6.8x) while the ramp-corrected raw preference gradient rose
  about 1.94x. Together these changes made the raw preference/base ratio about
  13.2x larger than at calibration. Therefore even 10% ramp produced an actual
  ratio of `0.331`, and 20% ramp reached `0.997` at attempted step 200.
- The preference hinge remained active because the requested relative margin
  was 0.5 while the observed median gap was only roughly 0.04-0.09 and p10 was
  usually negative. In contrast, the representation gradient at step 100 was
  only `0.00116` versus preference `2.5246`; representation was not the source
  of the stop. Narrow conclusion: this coefficient-calibration/schedule is
  invalid for the changing gradient geometry. It does not establish that the
  decoder-preference idea itself is invalid.

### 2026-09-01 — Comet visibility correction

- User reported that the Comet experiment appeared to contain no logs. Remote
  API verification showed the scalar upload was intact: 209 metric series,
  including 20 points each for `train/loss`, `train/base_loss`,
  `train/preference_loss`, `train/representation_loss`, and the preference-gap
  metrics. The initial sidecar had intentionally disabled stdout capture, so
  the Comet text Logs view did not contain the trainer launch log even though
  scalar metrics existed.
- Reopened the same experiment key, uploaded the trainer stdout as both a text
  sample and asset, uploaded all metric/gradient/probe/config/stop JSON(L)
  evidence, and attached all five final diagnostic PNGs at step 199. Remote API
  then listed 14 assets. No duplicate experiment was created.

## 2026-09-01 — Revised preference v2 production launch

- After confirming that v1 truly stopped at step 199, the user approved an
  immediate corrected launch. The evidence-directed v2 intervention is narrow:
  fresh seed 42, preference calibration target `0.25 -> 0.025`, and linear ramp
  `1k -> 10k`; the historical base objective, model, data order, hard pairs,
  representation loss, preference formula, optimizer and 0.75 hard guard are
  unchanged. V2 has unique persistent and `/dev/shm` paths, so v1 artifacts are
  preserved.
- Focused tests passed 8/8 locally; the independent final reviewer reran the
  expanded current-tree suite (10 passed), verified exact final hashes and
  returned GO with P0=0 and P1=0. Final B128 smoke at
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_smoke_20260901T082444Z`
  completed 2/2 with `lambda_pref=0.72933088`, unchanged
  `lambda_repr=0.01845857`, no clamps, exact pair validators 1.0, finite
  objective-split telemetry and 2.53 GiB peak allocation.
- Fresh 500k v2 production is live on physical CUDA 0 with trainer PID 507258
  and Comet sidecar PID 508419. Run root:
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_500k_v2`;
  Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/26e6abb3e78746f19eb69a82cc0a0994`.
  Remote API verification found 22 loss rows through step 210 and the gradient
  audit through step 200; a live trainer stdout asset was also attached.
- V2 crossed the exact v1 failure point and continued: auxiliary/base gradient
  ratio was `0.001955` at step 100 and `0.006406` at step 200, versus v1
  `0.996623` at attempted step 200 and the unchanged 0.75 stop threshold. All
  observed rows were finite. This establishes that the revised schedule fixes
  the immediate step-200 takeover; it is early telemetry, not evidence that the
  full 10k ramp or 500k convergence is safe.
- The known performance limitation remains: roughly 0.4 steps/s because
  dynamic hard-pair materialization is I/O-bound. The scientific run remains
  active; throughput optimization is a separate follow-up and must preserve the
  exact paired sample stream.

### 2026-09-01 — Representation objective is too easy

- User observed that the representation loss is too simple. Live v2 telemetry
  supports this: by steps 840-910, binary representation accuracy was 1.0,
  raw loss was roughly `0.0006-0.004`, projector positive cosine was about
  `0.944` and matched-negative cosine about `0.02-0.08`. At the step-900
  gradient audit, weighted representation gradient L2 was only `5.91e-5`
  versus base `1.997` and preference `0.1204`.
- The current task has three related shortcuts: the positive consists of two
  dropout passes through the projector over the exact same already-computed
  latent; each anchor has only one binary negative; and the two-layer projector
  can amplify a small lineage-specific difference without requiring the raw
  latent geometry to carry a broadly useful representation. Consistent with
  the head-shortcut hypothesis, raw paired-latent cosine remained roughly
  `0.30-0.65` and generally above random cross-sample cosine `0.17-0.42`, while
  the projector already separated the pair almost perfectly.
- This establishes saturation and vanishing useful representation pressure,
  not which shortcut is individually dominant. A discriminating replacement
  should bundle: projector-versus-direct-z ablation, multiple exact-matched
  negatives per anchor, and encoder-input views rather than projector-only
  dropout. A low-capacity linear projection over normalized slotwise z plus
  K-way hard InfoNCE (for example K=8 exact nuisance-matched lineages) and a
  small direct-z variance/covariance floor is the current leading design. The
  active production was not changed or stopped merely from this discussion.

### 2026-09-01 — Selective CUDA 1 encoder-only pause

- User requested stopping only the CUDA 1 encoder-only production with exact
  continuation possible later. CUDA 0 mini preference v2 was left untouched.
- Sent SIGTERM only to trainer PID 71669. Its cooperative handler finished the
  current optimizer step and atomically saved at step 82,132 / committed cursor
  2,628,224, then wrote `STOPPED.json` and exited. The matching Comet sidecar
  PID 72733 was terminated only after trainer completion and flushed normally.
- Exact process-resume checkpoint (model, AdamW, RNG, config and cursor) is
  `/dev/shm/weightclip_direct_normalized_scaled_700m_p32_polar_tails_encoder_only_500k_v1/resume_latest.pt`
  (6,762,464,202 bytes). Because `/dev/shm` is host-volatile, an additional
  atomic persistent copy was stored at
  `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_polar_tails_encoder_only_500k_v1/resume_stopped_step_82132.pt`
  with the same recorded size. The persistent model-only checkpoint at
  `model_latest.pt` was also refreshed at step 82,132.
- Final hardware verification: physical CUDA 1 had no compute process, 3 MiB
  used and 0% utilization; CUDA 0 retained only the mini v2 trainer. Restart
  should restore the persistent exact resume back to the config-declared
  `/dev/shm` resume path if the host was rebooted, then launch the unchanged
  encoder-only production config.

### 2026-09-01 — Mini optimizer-timescale arm (`AdamW beta1=0.99`)

- User requested a CUDA 1 copy of the live CUDA 0 mini latent-preference v2,
  strongly increasing Adam first-moment smoothing without increasing batch size
  or gradient accumulation. The chosen intervention is `beta1 0.90 -> 0.99`;
  `beta2=0.999`, fresh seed 42, data order, B128, LR `5e-5`, model, historical
  two-tail base regression, hard pairs, auxiliary losses/calibration/ramp and
  500k horizon are unchanged. The only other config differences are physical
  `cuda:1` and unique output/resume/model paths. Thus this is a fair optimizer
  timescale arm, not a warm start from the already-trained control.
- Pre-flight mechanism audit of the CUDA 0 metrics found that the visible raw
  base-loss sawtooth is not established as optimizer noise. Over the last 300
  logged rows available at audit time, base loss had autocorrelation `0.736` at
  lag 3 (logging is every 10 steps, hence a 30-step period), while lag 1 was
  only `0.080`; the same phase separation was strongest in structural loss.
  Base loss correlated `0.795` with structural and `0.696` with behavioral.
  Meanwhile the fixed probe changed only `2.124 -> 2.154 -> 2.129` at steps
  1000/2000/3000. Source evidence is the control
  `train_metrics.jsonl` and `fixed_probe_metrics.jsonl` under
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_500k_v2`.
- Competing mechanisms and discriminators are therefore: (1) periodic
  data/stratum composition predicts the same 30-step phase pattern in both
  arms but a much smoother fixed probe; (2) Adam first-moment update noise
  predicts lower matched-step residual/first-difference variation and a smoother
  fixed probe for `beta1=0.99`; (3) structural-component domination predicts
  that any improvement is localized to structural loss rather than all loss
  components. Phase-conditioned raw loss, equal-step fixed probes and
  component-wise variation will distinguish them; no causal conclusion is
  claimed from the launch alone.
- Final focused tests passed (`9 passed` in the parent run; independent reviewer
  reran the directly relevant file with `8 passed`). B128 smoke at
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_smoke_20260901T093343Z`
  completed 2/2 on CUDA 1: 10,168,497 total parameters, all parameter groups
  live, finite telemetry, exact pair validators 1.0, committed cursor 128 and
  2.53 GiB peak. The independent reviewer verified hashes/diff/AdamW wiring and
  returned final GO with P0=0, P1=0.
- Production is live on physical CUDA 1 with trainer PID 525305 and Comet
  sidecar PID 526222. Config:
  `projects/weight-vae/workspace/conf/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_pref025_ramp10k_beta1_099_production_500k.yaml`;
  run root:
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_beta1_099_500k_v3`;
  Comet:
  `https://www.comet.com/mike-5531/big-weight-vae/6d170f78df8b4e038c9b748bc4d1d8d4`.
  Remote API verification found 705 metric records through step 40 and both a
  trainer stdout text sample and asset, so the Comet page contains real scalars
  and logs rather than only an experiment shell.
- Early equal-step evidence through step 100 is mixed and does not yet support
  smoothing. For base loss, first-difference std was `0.277` at beta1 0.90
  versus `0.322` at beta1 0.99, and detrended std was `0.199` versus `0.203`;
  structural first-difference std improved slightly `0.170 -> 0.160`. Step-100
  base loss was `2.6470 -> 2.5816`, but eleven early logged points are far too
  few and still strongly non-stationary. The beta1 arm remains running for the
  decisive matched 1k+ comparison.
- Operationally, running both hard-pair loaders concurrently reduced each run
  to roughly `0.27 step/s` in the current window (about 3.4-3.7 seconds data
  wait per step), while GPU compute is much shorter. This is host data-loader
  contention, not GPU saturation or a model failure. Changing loader topology
  now would confound the optimizer arm; throughput optimization remains a
  separate experiment.

### 2026-09-01 — Current mini batch-diversity contract

- User asked how the active mini dataloader works and how batch diversity is
  controlled. The current pipeline has deterministic cycle-level hierarchical
  balancing, but no strict diversity constraint on the final p32 minibatch.
  The parent p128 stream round-robins 200 shuffled
  `(dataset, operation, depth_index, role)` strata. Entries inside each stratum
  are shuffled and cyclically repeated so large tiled operators do not dominate
  merely by tile count. An I/O locality reorder groups operators only within
  each stratum while preserving the exact global stratum sequence; lineage and
  checkpoint temporal order are explicitly not preserved by that contract.
- Each p128 parent is then expanded sequentially into its nonempty p32
  subtiles, with no second shuffle. A training step takes the next 64 contiguous
  p32 anchors and resolves one deterministic hard partner for every anchor,
  producing physical B128 as 64 adjacent pairs. Every pair is fail-closed to
  different lineage and different checkpoint payload while matching dataset,
  checkpoint index, layer/operator, tile/subtile coordinates, masks and
  canonical gauge. Which member is the positive target alternates by pair and
  step. Loader workers and prefetch change materialization latency only, not
  the committed logical order.
- Consequently the hard guarantee is pair validity and cycle-level stratum
  exposure, not a minimum number of lineages/checkpoints/contexts per final
  B128. Live beta1=0.90 telemetry over steps 2370-5360 shows only 8-24 unique
  lineages per physical B128 (median 14, mean 15.15) and 32-62 unique
  Distribution Encoder contexts (median 46, mean 45.34; mean reuse 2.88x).
  The stream consumed about 66 parent p128 tiles per ten optimizer steps, or
  roughly 6.6 parents per 64-anchor batch, explaining the within-parent
  clustering. Unique-lineage count had near-zero correlation with base loss
  (`-0.043`) and context count only weak correlation (`-0.162`), so these two
  counts alone do not explain the 30-step sawtooth.
- Direct inspection of the saved production fixed batch confirms the user's
  suspected same-matrix multiplicity. Its 64 anchor pairs form only seven
  consecutive parent-matrix blocks with p32 patch counts
  `[1, 16, 16, 4, 4, 16, 7]`; every block has a corresponding partner matrix,
  so physical B128 contains only 14 checkpoint payloads/lineages. Six of those
  payloads contribute 16 samples each. Thus one matrix can contribute up to 16
  p32 patches to a single batch (and its matched partner another 16), making up
  32/128 samples from one exact matrix-pair block.
- Historical p128/B32 did not have this same-matrix concentration at meaningful
  scale because it batched parent tiles directly, before p128-to-p32 expansion.
  Its exhaustive cycle-0 locality audit over 4,222 B32 batches reported mean
  31.35 unique checkpoint payloads (p5 30) and mean 30.71 unique lineages (p5
  29), with only 0.218% adjacent same-lineage entries. Thus a p128 batch was
  normally 30-32 distinct matrices/checkpoints rather than seven matrix pairs.
  The late p128 loss did show a separate periodic autocorrelation peak at 50/100
  optimizer steps, while its means by `step mod 30` were nearly equal. This
  means p128 was not free of composition periodicity, but it did not have the
  mini run's p32 block-amplification mechanism or exact 30-step phase pattern.
- The missing discriminator is per-batch p32 stratum/operator/parent occupancy
  and entropy. A future truly batch-balanced arm should buffer p32 anchors and
  constrain max anchors per parent/operator while round-robining strata, retain
  the exact hard-pair resolver, and log stratum entropy/max-repeat. That would
  change sample order and is therefore a separate experiment, not a live edit
  to the beta1 comparison.

#### Identity terminology clarification

- `lineage_id` is one complete source-model training trajectory, concretely
  `<dataset>:seed=<N>`. A lineage contains many epoch checkpoints; each
  checkpoint contains many layer matrices; each matrix contains many p128 tiles
  and each mini parent tile yields p32 subtiles. The exact matrix identity is
  `(checkpoint_sha256, layer_key)`, not `lineage_id` alone.
- Therefore tiles from one exact numerical matrix cannot belong to different
  lineages. Tiles with the same architectural `layer_key` can belong to
  different lineages, but they are different learned matrices. Conversely,
  many distinct matrices/checkpoints/layers can share one lineage. Unique
  lineage count is consequently only a coarse lower bound on matrix diversity,
  not an exact matrix count.

### 2026-09-01 — Exact 500-step checkpoint retention for the v2 kink analysis

- User expects a sharp direction-loss transition around step 10,000 in
  `mini-polar-latent-preference-pref025-ramp10k-production-500k-v2` and asked
  to stop it safely, preserve full checkpoints every 500 global steps, and
  continue from the exact state. The user explicitly rejected any
  window-triggered or rolling retention policy: checkpointing is unconditional
  at every multiple of 500 with no deletion window. The projected storage of
  about 120.9 GB decimal (112.6 GiB) through step 500,000 was explicitly
  approved.
- The CUDA 0 trainer stopped cooperatively after its current committed update at
  step 6006, not the estimated 6500. Its exact cursor was
  `parent_logical_index=39832`, `subpatch_index=4`,
  `emitted_logical_index=384384`. The stop payload contains the model,
  projector, full AdamW state, Python/NumPy/CPU/CUDA RNG state, normalization,
  calibration, exact config and cursor. The original stop record was preserved
  as
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_500k_v2/STOPPED_step_000006006.json`.
- The trainer now accepts an opt-in retained-checkpoint cadence and directory.
  After each committed optimizer step at a requested global multiple, it first
  writes the full exact `resume_latest.pt` payload and then atomically hardlinks
  that immutable inode as `resume_step_<step>.pt`. Later latest-checkpoint
  replacements cannot mutate prior retained inodes. No scientific config,
  optimizer setting, loss, data order, or ordinary checkpoint cadence changed.
  A resume-start snapshot is also retained so the intervention boundary itself
  is recoverable.
- Focused and integrated tests passed (`12 passed` in the independent final
  review). A B128 two-step CUDA smoke at
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_retained_checkpoint_smoke_20260901T103300Z`
  loaded both retained steps independently and verified distinct immutable
  inodes, full 184-entry optimizer state, all RNG keys, exact committed cursors,
  finite metrics, and `retention_window=null`. The independent prelaunch review
  returned GO with P0=0 and P1=0.
- Production resumed on physical CUDA 0 with trainer PID 544553 from exact step
  6006. The first new log is step 6010, with emitted anchor 384640, so the
  sequence is continuous (`5970, 5980, 5990, 6000, 6010`) rather than restarted
  or duplicated. The immutable start snapshot is
  `/dev/shm/weightclip_mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_500k_v2/retained_checkpoints/resume_step_000006006.pt`;
  the first periodic snapshot is due at step 6500 and subsequent ones at 7000,
  7500, and so on through 500000. The machine had about 289.2 GB free in
  `/dev/shm` before launch, leaving roughly 168 GB after the projected retained
  set.
- The existing Comet sidecar stayed attached to the same experiment. Remote API
  verification observed resumed training metrics through step 6020, and the
  retention policy/index were uploaded as experiment assets:
  `https://www.comet.com/mike-5531/big-weight-vae/26e6abb3e78746f19eb69a82cc0a0994`.
  CUDA 1 beta1=0.99 production and its sidecar were left untouched.

### 2026-09-01 — CUDA 1 beta1=0.99 run paused with exact resume state

- User requested stopping the CUDA 1 training while leaving CUDA 0 running.
  Only the beta1=0.99 trainer and its Comet sidecar were targeted. The trainer
  received cooperative SIGTERM and stopped after committed step 4583 with
  cursor `parent_logical_index=30396`, `subpatch_index=0`,
  `emitted_logical_index=293312`.
- The full resumable payload is
  `/dev/shm/weightclip_mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_beta1_099_500k_v3/resume_latest.pt`
  (122,273,962 bytes). It was loaded after shutdown and verified to contain the
  exact step/cursor, beta values `[0.99, 0.999]`, 184 AdamW parameter states,
  model/projector state, config, normalization/calibration and all four RNG
  domains. The persistent model snapshot and `STOPPED.json` are under
  `/mnt/shared/weightclip_benchmark/mini_polar_latent_preference_10m_p16_tile32_b128_pref025_ramp10k_beta1_099_500k_v3`.
- Trainer PID 525305 and sidecar PID 526222 both exited. Physical CUDA 1 was
  then verified at 3 MiB and 0% utilization with no compute process. The CUDA 0
  v2 run remained live and had advanced to logged step 7060 during the final
  check.

### 2026-09-01 — Repository state publication for exact mini reproduction

- User requested committing and pushing the entire current repository state so
  the same mini implementation can be launched on another machine. The target
  is the existing remote branch `weightclip-experiment-state-20260830`, whose
  remote tip was verified to equal the detached local base
  `750dd475ec2a81c72ae88047d5e7affce2d65078`, allowing a normal fast-forward
  publication rather than a forced update.
- The publication scope is all 20 current worktree changes: mini regression and
  latent-preference trainers/configs/tests/analysis, the relevant 700M
  encoder-only/raw-direction/decoder-thaw work, Comet streaming changes,
  experiment design documentation, and this append-only discussion record.
  No runtime checkpoint, generated metrics, Comet/HF credential, or large data
  artifact is inside Git.
- A sensitive-value scan found no credential material in the change set; the
  only matches were environment-variable names and the existing statement that
  the private HF credential lives outside Git. The combined directly relevant
  test panel passed `20 passed` with two known Transformer warnings, and
  `git diff --check` was clean before staging.
- Exact execution on another machine still requires the pinned private HF
  operator-bank snapshot and local runtime paths referenced by the production
  config. The downloader and dataset contract are already tracked under
  `projects/weight-vae/workspace/scripts/download_weightclip_operator_dataset_from_hf.py`
  and `projects/weight-vae/workspace/docs/weightclip_operator_dataset_hf.md`;
  checkpoints and the normalization cache remain external runtime artifacts,
  not repository source.
