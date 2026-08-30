# Weight-AE global downstream-salience hypothesis

Date: 2026-08-17 UTC

Status: proposed central paper hypothesis. It passed iterative conceptual and
hostile review, but no experiment has established it.

## Plain-language claim

> **From local weights and their input activations, a Weight-AE predicts which
> weight changes matter to the final behavior of the complete model, and the
> same rule transfers across vision, text, and audio.**

## Precise claim

> A behavioral Weight-AE trained only on vision weights and local operator
> behavior `XW`, without downstream task loss, gradients, Fisher, or
> Gauss--Newton supervision, learns a transferable context-conditioned rule for
> retaining weight information. Among perturbation directions of the same
> weight tile that are matched in norm, spectrum, and immediate local operator
> effect, its decoder preserves more strongly the directions with greater
> canonically normalized end-to-end predictive-KL/Gauss--Newton salience.
> Changing only a real pre-operator modality context causally reverses this
> retention ordering when downstream salience reverses, and a law fitted only
> on vision predicts the effect without target refitting in shared multimodal
> weights and independent NLP/audio checkpoints.

The claim is deliberately narrower than saying that the latent uniquely
represents Fisher/Gauss--Newton. The identified claim is causal allocation of
decoded weight retention aligned with unseen downstream salience.

## Intuition

Take one layer `W` and two perturbations `Delta_A` and `Delta_B`. Match them so
that they have the same size and change the immediate layer output by the same
amount:

```text
||X Delta_A|| approximately equals ||X Delta_B||.
```

Suppose the rest of the network strongly amplifies `Delta_A` but suppresses
`Delta_B`. The hypothesis predicts that the Weight-AE preserves `Delta_A` more
accurately, although this downstream sensitivity was never provided during AE
training.

For the causal cross-modality test, use the same physical shared-multimodal
weight `W`. Choose fixed matched directions whose downstream importance
reverses across modality contexts:

```text
image: Delta_A matters more than Delta_B
text:  Delta_B matters more than Delta_A.
```

Encode symmetric interventions once under a fixed anchor context, hash the
codes, and hold them bitwise fixed. Decode them under image and text contexts.
The reconstructed weights themselves must preserve `Delta_A` more in the image
condition and `Delta_B` more in the text condition. Use a separately designed
audio-versus-text pair.

## Mathematical object

For context `c`, let the local activation metric be

```text
G_c = Sigma_X_c tensor I.
```

Let `F_c` be the per-output predictive-KL Gauss--Newton metric of the full
model. For direction `Delta_k`, define

```text
q_c,k = (Delta_k^T F_c Delta_k) / (Delta_k^T G_c Delta_k)
q_tilde_c,k = q_c,k / (trace(G_c^dagger F_c) / rank(G_c)).
```

The active-subspace/ridge convention must be frozen before target access. The
normalization is required so arbitrary rescaling of a task loss cannot create
or destroy the cross-domain law.

For anchor context `C_0`, encode symmetric perturbations:

```text
z_plus_k  = Q_R(E(W + alpha Delta_k, C_0))
z_minus_k = Q_R(E(W - alpha Delta_k, C_0)).
```

The quantized codes are held bitwise fixed. Decode each pair under every codec
context `C_c` and form the central response:

```text
Delta_hat_c,k =
  [D(z_plus_k,C_c) - D(z_minus_k,C_c)] / (2 alpha)

r_c,k = <Delta_hat_c,k,Delta_k>_F / ||Delta_k||_F^2.
```

`r_c,k` is the primary context-free, weight-side retention measurement.

## Decisive same-weight reversal

Select matched directions `A,B` for the same physical multimodal tile such
that normalized downstream salience reverses, for example:

```text
q_tilde_image,A > q_tilde_image,B
q_tilde_text,A  < q_tilde_text,B.
```

The hypothesis predicts the same reversal in `r_c,k` when only decoder context
changes. The primary difference-in-differences is

```text
I_weight =
  (r_image,A - r_image,B)
  - (r_text,A - r_text,B).
```

## Crossed-ruler control

Run the complete factorial `C_codec x C_eval`: every decoded response produced
under every decoder context is evaluated with every fixed local/end-to-end
evaluation context, plus the context-free weight projection above.

The primary reversal must follow `C_codec` and already be present in the
decoded weights. If it appears only because `C_eval` changes, the result is a
changed-ruler tautology and the hypothesis fails.

## Breadth test

After the same-physical-weight causal anchor succeeds on at least two
independent shared-multimodal checkpoints, fit a predeclared two-parameter
monotone relevance--retention law on held-out vision only. Without target
refitting, test it on independent NLP and audio checkpoints.

Normalized downstream salience must add preregistered out-of-sample predictive
value beyond role, normalized depth, model/checkpoint identity, weight
statistics, local `G`, and decoder-Jacobian alignment. Fisher/Gauss--Newton
shuffling is a required null.

## Falsifier

After all assay-validity gates pass, the hypothesis is false if any of the
following occurs:

- decoder context does not change weight-side retention ordering;
- the ordering does not follow preregistered normalized downstream salience;
- the vision-fitted relationship requires refitting on NLP or audio;
- Fisher/Gauss--Newton shuffling explains the effect equally well;
- matched local-MSE, weight-only, context-shuffled, untrained, or equal-rate
  conditional linear codecs reproduce the effect;
- normalized downstream salience adds no held-out information beyond local
  activation geometry and registered covariates.

## Required controls and eligibility gates

1. `C` must contain only fixed pre-operator information from the unperturbed
   model: no labels, downstream gradients/Fisher, post-operator outputs, or
   information recomputed from perturbed weights.
2. Historical training must be confirmed to use local `XW`-level behavioral
   direction/scale rather than end-to-end task supervision.
3. A real native-context decoder replay must beat zero, C-only, and untrained
   controls. The old encoder-only hash check with `decoder_calls=0` is not a
   positive control.
4. Symmetric `+alpha/-alpha` responses must have stable nonzero dynamic range.
5. Directions must be paired within one tile/context and matched in Frobenius
   norm, rank/singular spectrum, local `G` energy, alpha, role/depth/model, and
   weight-distribution likelihood.
6. Direction construction and evaluation must use independent `F`, `G`, and
   activation splits. Require a preregistered salience separation and publish
   all attempted-pair counts and acceptance rates.
7. Quantization/bin-boundary effects must be controlled with an alpha sweep,
   dithering, and a decomposition of latent-symbol changes versus decoder
   response conditional on a symbol change.
8. Shared causal-anchor weights must be physically hash-identical across
   modalities and have no modality-specific adapter on the tested path.
9. A same-rate oracle single code must be able to cover the union of
   modality-relevant subspaces. Otherwise a failure is information-theoretic
   invalidity at that rate, not negative evidence about the AE.

Required matched baselines include behavioral, local-MSE, weight-only,
context-shuffled and untrained AEs; the strongest equal-rate source-trained
conditional linear codec; local activation-aware analytical codecs;
C-only/zero/wrong-code and Fisher-shuffle controls; and a target-Fisher codec
only as a privileged ceiling.

Only real native contexts are used. The failed fixed/global-context run is not
evidence for or against this hypothesis. Latent distances, PCA, clustering,
retrieval, linear probes, and latent-space model optimization are not primary
evidence.

## Verdict semantics

- **Invalid:** any context/objective, decoder, response, stimulus,
  quantization, oracle-rate, or shared-weight eligibility gate fails.
- **Negative:** gates pass but weight-side reversal is absent, controls
  reproduce it, or the vision law fails to transfer.
- **Positive:** one frozen vision law predicts weight-side retention; decoder
  context reverses causal ordering on at least two shared-multimodal
  checkpoints; and the effect transfers without target fitting to independent
  NLP/audio checkpoints with a material preregistered advantage over the
  strongest matched baseline.

## Honest novelty wording

Do not claim the first learned weight codec, behavioral weight AE,
activation-aware or sensitivity-aware compression method,
decoder-side-information codec, or multimodal shared transformer.

The narrow proposed novelty is:

> **Implicit emergence and cross-domain invariance of global downstream
> relevance inside a weight representation trained only with local operator
> behavior, established by same-weight causal context reversal rather than
> latent clustering or probes.**

## Review record

After iterative author and independent hostile-review cycles, the repaired
exact hypothesis received:

- causal identifiability: 91--92/100;
- novelty/literature: 94/100;
- feasibility/methodology: 93/100, conditionally 94/100 after verifying the
  activation-condition contract.

These scores assess hypothesis quality, not the probability that the current
checkpoint will produce a positive result.

