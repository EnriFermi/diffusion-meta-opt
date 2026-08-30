# Decoder architecture design-risk review

Date: 2026-08-29 UTC

Scope: zero-training, read-only review after the latent-rooted cross-attention
collapse. No implementation or training was performed. The rejected learned
global `512 x 32` signed mixer is not reconsidered as a candidate.

## Established starting point

The failed decoder did satisfy the algebraic contract `z = 0 -> decoded weight =
0`, but this was not enough. Its tails were almost uniform at initialization,
and by step 1000 the 32 latent values were nearly identical. Attention routing
then had an approximately zero Jacobian because changing weights over identical
values cannot change their weighted sum.

Primary evidence:

- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1/gradient_telemetry.jsonl`
- `/mnt/shared/weightclip_benchmark/direct_normalized_scaled_700m_p32_latent_rooted_polar_tails_500k_v1/latent_rooted_state_step_0001000_v1/report.json`
- `/home/coder/project/artifacts/weightclip_benchmark/latent_rooted_hostile_review_step_0001000_v1/report.json`

The required property is therefore stronger than zero-latent correctness:
equality of latent slots or equality across samples must not be an absorbing
state of the decoder/encoder Jacobian.

## Common risks that no candidate automatically removes

1. **Population common-template stationary point.** For any deterministic AE,
   `E(W) = c` and `D(c) = conditional mean template` is stationary when the
   expected target residual is orthogonal to the decoder Jacobian. Architecture
   can make this point less expressive and non-absorbing, but cannot prove it
   absent.
2. **Polar direction-radius gauge.** If the emitted 16-vector is used only after
   unit normalization, multiplying it by a positive scalar preserves the
   prediction and shrinks its tangent gradient inversely with the scalar. This
   remains an amplifier in all four candidates.
3. **Encoder/decoder factorization gauge.** A learned latent basis change can be
   undone by the inverse decoder basis change. Multiple learned merge/split
   transitions create multiple such gauges.
4. **Shared-parameter cancellation.** Hard ownership separates activation and
   latent gradients, but a decoder shared across groups still sums its parameter
   gradients over heterogeneous targets. Ownership is not a substitute for
   measuring per-group gradient resultants.
5. **Activation-context bypass.** A zero-latent test must be repeated for every
   tile/context. Additive context or role embeddings anywhere on the value path
   allow a context-conditioned template even if weight content is ignored.

## Candidate 1: flat hard-ownership grouped encoder/decoder

Concrete interpretation for the review: partition the 512 p32 tokens into 32
fixed groups of 16. Group `g` produces latent `z_g`; only `z_g` directly expands
to its 16 owned p32 outputs and their p16 children. Encoder and decoder functions
are shared across groups, but there is no learned all-to-all slot mixer.

### Exact or approximate collapse solutions

- **Shared role template:** `E_g(W_g) = c` for every group and a shared decoder
  maps `c` to the same 16-role template in every group. This is an exact
  population stationary solution if the role-wise mean residual is zero. It is
  much less expressive than a 512-row dictionary, but it is not impossible.
- **Context-coded constant:** if group/tile/context embeddings enter additively,
  `E_g` can ignore weights and emit a group-specific constant. The local decoder
  can then recreate an arbitrary positional template. This restores the exact
  shortcut ownership was meant to remove.
- **Dead local Jacobian:** a saturated gate, zero output projection, or one-value
  attention inside a group can make `dD_g/dz_g = 0`. Encoder gradients then die
  even though ownership is formally correct.
- **Single-value attention trap:** if one latent token is used as the sole K/V
  for 16 queries, softmax is exactly 1 and independent of every query. All 16
  outputs are identical unless a separate latent-derived expansion exists. This
  is the rooted failure in a smaller group, not a repair.

### Jacobian two to three optimizer steps ahead

- At step 1, the ideal Jacobian is block diagonal: error in group `g` reaches
  `z_g` without averaging over the other 31 groups. Isotropic direction errors
  may still cancel in the *shared decoder weights*, but not in the 32 distinct
  latent activations.
- If all `z_g` become equal after early scale/common-mode updates, different
  owned targets still generate different `dL/dz_g` provided the local expansion
  Jacobian is live. Equality is therefore normally not an invariant manifold.
- Adam can still favor a coherent shared scale/template update over noisy local
  direction updates. The design buys a nonzero escape gradient; it does not
  balance the two objectives.

### Gauge, capacity, and conditioning risks

- A learned local encoder matrix and decoder matrix retain one latent `GL(384)`
  basis gauge and an amplitude trade. Avoid adding a learned global slot basis.
- Each group compresses 16 x 32 = 512 input scalars to 384 latent scalars before
  output. This matches the intended 25% compression locally but prevents using
  spare capacity in one group to help another.
- Purely local groups cannot model global weight/operator correlations. Adding
  all-to-all mixing before ownership can reintroduce the original common-mode
  route; adding it after a direct local residual is safer but changes the causal
  test.
- Grouping must follow exact matrix axes and masks. Arbitrary flattened groups
  can mix padded coordinates or couple rows with different activation roles.

### Unique zero-training falsifiers

Reject before training if any fails:

1. `z_g` perturbation changes any unowned output above numerical tolerance, or
   fails to change multiple owned outputs in linearly independent directions.
2. The local Jacobian `d owned_output_g / d z_g` has collapsed effective rank,
   a huge top-to-median singular-value ratio, or materially different spectra
   across masks/group positions.
3. Setting all 32 latents equal makes the 32 latent gradients equal on a batch
   whose owned targets differ. That would show equality is still absorbing.
4. Zero latent with arbitrary position, activation context, tile and masks gives
   a nonzero value output.
5. A local latent roll changes outputs outside the corresponding rolled groups,
   or the matched local roll has negligible effect.

## Candidate 2: symmetric hourglass merge/split

Concrete interpretation: mirrored ownership through `512 -> 256 -> 128 -> 64
-> 32` merges, a `32 x 384` bottleneck, then `32 -> 64 -> 128 -> 256 -> 512 ->
1024` splits. Transitions are shared across positions. Global interaction is
allowed only after ownership-specific value states exist.

### Exact or approximate collapse solutions

- **Common-mode-only encoder:** each merge discards the sibling-difference mode
  and retains only their mean. The decoder uses only a common split branch. If
  target differences are zero in expectation, the discarded difference branch
  can be a stationary zero-gradient solution.
- **Identical-child split:** for `child0 = U0 parent` and `child1 = U1 parent`,
  `U0 = U1` produces equal siblings. Repeating it yields a small periodic role
  template rather than sample-specific leaves. If the differential gradients
  cancel across shared positions, this manifold is stationary.
- **Additive child-role bypass:** `child = U parent + role_embedding` lets a
  constant or zero parent generate token identity. Composed over four levels it
  becomes a 16-role positional dictionary repeated across the 32 root groups.
- **Context-coded root:** different roots can be manufactured from additive
  group/context embeddings even when all weight inputs are ignored, restoring
  a much richer template.

### Jacobian two to three optimizer steps ahead

- If merges are literal reshapes/concatenations and splits are literal channel
  partitions, ancestry is exact and the structural transitions have singular
  values 1. Different leaf errors reach different parent channel slices on the
  first backward.
- If each transition is learned, four small conditioning defects multiply in
  the encoder and five in the decoder. A per-level median singular value of
  `0.5` becomes about `0.002` over nine transitions; a value of `2` becomes
  about `512`. Pre-norm blocks do not repair this across dimension-changing
  transitions.
- Shared child transforms aggregate common-mode scale gradients coherently,
  while antisymmetric direction gradients can cancel. Adam's second moment can
  suppress the noisy difference branch after only a few updates, after which
  the common-mode-only solution becomes hard to escape.
- Unlike rooted attention, equal parents are not automatically absorbing if
  owned leaf residuals produce distinct parent-channel gradients. This benefit
  exists only when the split retains distinct channel paths at initialization.

### Gauge, capacity, and conditioning risks

- Learned merge/split pairs introduce an independent invertible-basis and scale
  gauge at every resolution. This is worse conditioned than one flat local
  bottleneck unless transitions are fixed isometries or tightly tied inverses.
- Averaging is not an acceptable merge for noisy weights with no assumed local
  smoothness. A valid merge must preserve both common and difference channels
  until the explicit `512 -> 384` per-root bottleneck.
- Literal concatenation requires width to double as token count halves. Holding
  width fixed silently imposes extra compression at every merge and makes the
  nominal `32 x 384` bottleneck description misleading.
- Pairing order must be stable under masks and semantically aligned with matrix
  row/column layout. Gauge/permutation views can make an arbitrary 1-D tree a
  poor inductive bias even though it is mathematically valid.

### Unique zero-training falsifiers

Reject before training if any fails:

1. For every merge level, compare common perturbation `[delta, delta]` with
   contrast perturbation `[delta, -delta]`. Contrast gain must not be near zero
   relative to common gain before the explicit bottleneck.
2. For every split, one parent perturbation must create two linearly independent
   child perturbations. Identical or nearly collinear siblings are a hard fail.
3. Randomized end-to-end JVP/VJP estimates through merge plus split must show a
   bounded nonzero spectrum on the retained subspace; report each level rather
   than only the product.
4. With all 32 root latents equal but heterogeneous leaf targets, gradients to
   roots and to sibling channel slices must remain heterogeneous.
5. Zero root values with arbitrary child roles/context must give zero leaves.
6. The actual scalar count at each level must prove that no unreported
   compression occurs before the 384-dimensional root bottleneck.

## Candidate 3: autoregressive decoder

### Exact or approximate collapse solutions

- **Teacher-forcing posterior collapse:** set all latent cross-attention/value
  weights to zero and learn the next weight token from the ground-truth prefix,
  position and context. This is an exact stationary solution for the latent
  path. Low training loss would not establish an autoencoder.
- **Free-running template generator:** without teacher forcing, a constant
  latent plus BOS and causal state acts as a clock and can generate an entire
  position-specific mean sequence. This is more expressive than the rejected
  512-row mixer, not less.
- **Prefix domination:** even if early tokens use `z`, later tokens can ignore it
  once the recurrent prefix carries enough information. Gradients to `z` decay
  with sequence position.

### Jacobian two to three optimizer steps ahead

- Under teacher forcing, the shortest and highest-signal path is target prefix
  to output. Adam reinforces it before the noisier latent path learns; latent
  ignorance is the expected optimization outcome.
- Under free running, later-token gradients are products through hundreds of
  recurrent steps. Values below or above unit gain cause vanishing or exploding
  credit, while early prediction errors change every later input.
- Scheduled sampling does not remove either stationary solution and adds an
  estimator/interpolation choice.

### Gauge, capacity, and conditioning risks

- The order over matrix patches is arbitrary and not invariant to the existing
  permutation/gauge views. A row-major causal prior can model serialization
  artifacts rather than operator structure.
- Sequence length up to 1024 makes free-running production substantially slower
  and creates train/inference exposure mismatch.
- Residual/KV norms and the polar direction-radius gauge remain. Latent scale can
  also trade against cross-attention projection scale.

### Unique zero-training falsifiers

Reject before training if any fails:

1. Ground-truth-prefix perturbation has a larger output effect than latent roll
   under the intended training graph.
2. `norm(d output_t / d z)` decays materially with token index, or late-token
   Jacobians are dominated by prefix Jacobians.
3. Teacher-forced and free-running outputs/Jacobians disagree strongly before
   learning.
4. A constant latent plus BOS produces high-rank position-specific outputs.
5. One-token output perturbations amplify down the free-running chain.

## Candidate 4: standard query-residual decoder with anti-bypass constraints

Standard cross-attention state is `query + attention(query, z)`. The query
residual fixes the uniform-address bootstrap problem, but it reopens the exact
position/content bypass measured in the old production graph.

### Exact or approximate collapse solutions

- **Direct query template:** set cross-attention value/output weights to zero;
  let residual queries and the head emit a position/context template. Encoder
  gradient is exactly zero. This is an exact stationary solution.
- **Latent-gated template:** for output `gate(z) * F(query)`, the encoder emits a
  constant `c`, `gate(c)` becomes a constant, and `F(query)` reconstructs the
  positional mean. Zero-latent correctness does not prevent this.
- **Zero-subtraction separability:** in `D(query,z) - D(query,0)`, a separable
  `D = F(query) + G(z)` cancels `F` and leaves a position-independent `G`; mixed
  query-latent weights can stay zero when their expected residual gradient
  cancels. Large cancelling `F` terms also create BF16 precision risk.
- **Dead multiplicative gate:** a zero/saturated anti-bypass gate can satisfy
  zero-latent correctness while giving approximately zero gradient to both the
  query interaction and the encoder.

### Jacobian two to three optimizer steps ahead

- At step 1, the query residual gives the output head a much shorter gradient
  path than the latent cross-attention path. Coherent scale/template gradients
  update this path first; sample-specific direction gradients largely cancel.
- A multiplicative constraint makes the mixed Jacobian bilinear. If either the
  latent factor or query factor shrinks during the first updates, the other
  receives weaker credit, producing another self-reinforcing dead path.
- A query-only residual remains position-rich even when all latents collapse.
  Therefore latent equality is not an absorbing *output* state; it is an easy
  high-capacity template state with no incentive to escape.

### Gauge, capacity, and conditioning risks

- Query norm, gate norm and head norm can trade amplitude. Zero-subtraction adds
  a large-term cancellation gauge. The polar direction radius remains free.
- Anti-bypass penalties, latent dropout, shuffle losses or auxiliary dependence
  losses add coefficients and do not create an architectural invariant.
- A clean two-stream design in which queries retain residuals only in an address
  stream, while the head reads only a separately rooted value stream, is not a
  standard query-residual decoder. It should be reviewed as a different class.

### Unique zero-training falsifiers

Reject before training if any fails:

1. Zeroing cross-attention produces a material output, proving the direct query
   stationary solution is reachable.
2. Holding one nonzero latent constant across samples while varying query,
   position/context produces a high-rank positional dictionary. This detects
   the latent-gated template that the zero-latent test misses.
3. The mixed derivative `d2 output / d query d latent` is low rank, tiny, or much
   smaller than the direct `d output / d query` path.
4. In BF16, zero-subtracted output differs materially from an FP32 reference, or
   the two cancelled branch norms greatly exceed their difference.
5. Latent roll effect is small relative to query/context roll at the exact
   initialized parameter scale.

## Ranking

Ranking is for the next causal architecture test, not a claim of final quality.

1. **Flat hard ownership/grouped encoder-decoder — conditional first choice.**
   It has the shortest, block-diagonal credit path and the fewest transition
   gauges. Equal latents are not automatically absorbing because different
   owned targets can produce different latent gradients. Its principal risk is
   insufficient global capacity, making it a strong causal baseline rather than
   an established production architecture.
2. **Symmetric hourglass — conditional second choice, potentially higher final
   capacity.** It preserves ownership with shared multi-scale computation, but
   only if merge/split transitions are information-preserving reshapes/channel
   partitions. Learned averaging and learned untied transitions create exactly
   the difference-mode collapse and multi-level conditioning problem that the
   design is supposed to avoid.
3. **Query-residual plus anti-bypass — no-go without a stronger invariant.** The
   usual zero-latent gate is insufficient; a constant nonzero latent unlocks the
   entire positional dictionary. Most proposed constraints either restore the
   old bypass or create a dead bilinear gate.
4. **Autoregressive decoder — no-go for the next test.** Teacher forcing leaks
   targets and free running turns a constant latent into a powerful sequence
   template while introducing a long recurrent Jacobian and large cost.

## Mandatory pre-launch conditions

No candidate should be trained, even for 1000 steps, until all applicable
conditions below pass on the exact production masks/context and both random
synthetic and real B4/B32 samples:

1. **Zero-value invariant:** zero latent/content gives exactly zero decoded
   values for every position, tile and activation context; scale-head bias must
   not create reconstructed weights.
2. **Equality-state escape:** force all latent slots and all samples to the same
   nonzero vector, apply heterogeneous owned targets, and show heterogeneous,
   nontrivial gradients back to the owned latent slots. This is the decisive
   discriminator missing from the previous preflight.
3. **No hidden dictionary:** a constant nonzero latent must not unlock an
   arbitrary group/position-specific template through address/context alone.
   Report output rank and cross-position diversity under this intervention.
4. **Ownership audit:** perturbation and gradient support must match the declared
   ownership graph exactly before any optional global refinement.
5. **Jacobian spectrum:** store per-stage JVP/VJP gain, effective rank and
   top-to-median spectrum. Do not accept only a nonzero scalar gradient. Both
   common and contrast modes must survive.
6. **Direction-radius audit:** report raw direction norms and the change in loss
   gradient under frozen 0.1x/1x/10x logit rescaling. The architecture test must
   have a hard early-stop gate for renewed radial gradient attenuation even if
   that gauge is not changed in the first arm.
7. **Matched dependence interventions:** latent roll, local-group roll,
   zero-latent, context roll and position/role roll must be measured separately.
   The correct local latent intervention must dominate unowned/context-only
   changes.
8. **Step-1 task pullback:** measure direction and scale gradients separately at
   leaf, every split/merge level, bottleneck and early encoder. A healthy total
   gradient can hide cancellation or a scale-only path.
9. **No unreported compression:** list token count, channel count and total scalar
   dimension at every stage, including masks. Compression is allowed only where
   explicitly intended.
10. **Fail-fast bounded run only after the above:** first authorize at most 1000
    matched steps, not another 500k launch. Persist equality-state escape,
    latent rank, per-group rank, direction diversity and Jacobian gains at steps
    1/10/100/250/500/1000. A scalar loss decrease or a live zero-latent gradient
    is not sufficient success.

## Narrow recommendation

The highest-information next test is a flat hard-owned group autoencoder with a
direct rank-rich local expansion, no attention over a single latent value, no
additive group/context value embeddings, and no global refinement in the first
arm. It tests whether exclusive local credit alone prevents the equality-state
collapse with the shortest possible Jacobian. If that passes dependence and
optimization gates but lacks reconstruction capacity, the symmetric hourglass
is the coherent capacity extension. Starting with the hourglass makes a failure
ambiguous between ownership, difference-mode loss and multi-stage conditioning.
