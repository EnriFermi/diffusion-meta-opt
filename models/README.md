# WeightQuantileVAE

This folder contains the `WeightQuantileVAE` model implementation:

- `models/weight_quantile_vae.py` - model code
- `models/__init__.py` - public exports

## What The Model Does

`WeightQuantileVAE` compresses a layer weight matrix `W` into a fixed number of latent tokens and reconstructs `W_hat`.
Conditioning comes from input batch statistics (`X`) via per-feature quantiles.

Core tensors:

- `X`: `[n, d_in]` - input activations batch
- `W`: `[d_in, d_out]` - layer weights
- `W_hat`: `[d_in, d_out]` - reconstructed weights

Forward:

```python
W_hat, kl_loss, aux = model(X, W)
```

## Architecture Overview

1. Quantile extraction:
- Compute `Q = quantile(X, tau, dim=0)` with shape `[k, d_in]`.
- `tau_i = (i + 0.5) / k`.

2. Input distribution encoder (`InputDistributionEncoder`):
- Convert to feature-major: `Q_feature_major = Q.T` -> `[d_in, k]`.
- Shared MLP per feature: `k -> k_mlp`.
- One self-attention layer across the `d_in` dimension.
- Patch-average over `patch_size=P` rows.
- Output `dist_patches`: `[n_patches, k_mlp]`, where `n_patches = ceil(d_in / P)`.
- This is computed once and reused for all `d_out`.

3. Patch token build:
- Patchify `W` along `d_in`: `W_patches` -> `[d_out, n_patches, P]`.
- Concatenate per patch: `[k_mlp] + [P]` -> `[k_mlp + P]`.
- Shared MLP head -> patch tokens `[d_out, n_patches, d_tok]`.

4. Row encoder (`RowMixerBlock` stack):
- Add one CLS token per output row (`d_out`) with deterministic output-index positional signal.
- Row sequence per output: `[1 + n_patches, d_tok]`.
- `encoder.self_attn_mode`:
  - `full`: full self-attention inside the row
  - `cls_only`: only CLS attends to row tokens (patch-patch attention disabled)

5. Perceiver resampler on CLS only:
- Take `CLS_all` from all output rows: `[d_out, d_tok]`.
- Learned latent base `[m_lat, d_lat]` cross-attends to CLS context.
- Then latent self-attention + latent FFN (`resampler.n_layers` times).
- Produce `Z: [m_lat, d_lat]`.

6. VAE head:
- `mu, logvar = Linear(Z)`.
- Reparameterization: `Z_sample = mu + eps * exp(0.5 * logvar)`.
- KL term returned as scalar `kl_loss`.

7. Decoder:
- Build query tokens for each `(patch_index, output_index)` from deterministic Fourier features.
- Cross-attention: queries -> latent tokens (no query self-attention).
- Decode each query to patch vector of length `P`.
- Unpatch to `W_hat: [d_in, d_out]`.

## ModelConfig Parameters

Main fields (`ModelConfig`):

- `k`:
  - Number of quantiles per input feature.
  - Controls granularity of distribution summary from `X`.

- `k_mlp`:
  - Output width of the input-distribution feature encoder.
  - Used instead of raw quantile patch vectors in patch-token construction.

- `patch_size` (`P`):
  - Patch size along `d_in` for both distribution stream and weight stream.
  - `n_patches = ceil(d_in / P)`.

- `d_tok`:
  - Token width for row encoder tokens (patch tokens and CLS tokens).

- `m_lat`:
  - Number of latent tokens in the Perceiver/VAE bottleneck.
  - Fixed compression budget.

- `d_lat`:
  - Latent token width.

- `n_heads`:
  - Attention heads for row mixer, resampler, and decoder cross-attention.
  - Constraints: `d_tok % n_heads == 0` and `d_lat % n_heads == 0`.

- `pos_fourier_dim`:
  - Fourier/sinusoidal feature width before learned positional projections.
  - Used for output index and `(patch_index, output_index)` positions.

- `row_mlp_mult`:
  - Hidden multiplier for FFN blocks inside row mixer.

- `resampler_mlp_mult`:
  - Hidden multiplier for FFN blocks inside Perceiver resampler layers.

- `decoder_mlp_mult`:
  - Hidden multiplier for decoder FFN and patch decoding MLP.

- `dropout`:
  - Dropout probability used across MLP/attention blocks.

Nested config: `encoder` (`EncoderConfig`)

- `encoder.n_row_layers`:
  - Number of row mixer blocks.

- `encoder.self_attn_mode`:
  - `"full"` or `"cls_only"` (see row encoder behavior above).

Nested config: `resampler` (`ResamplerConfig`)

- `resampler.n_layers`:
  - Number of Perceiver resampler layers.

## Notes

- Quantile computation and distribution encoding are reused across all `d_out` for compute efficiency.
- The decoder is not constrained to low-rank forms; it predicts full patch values.
- `aux` from forward includes shape diagnostics for debugging.
