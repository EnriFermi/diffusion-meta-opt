# BigVAE Latent Flattening

Post-hoc BigVAE latent-space flattening experiment based on the IRVAE flattening module.

The trained BigVAE is frozen. A RealNVP-style flow `i` is trained so the effective decoder

```text
z_flat -> i^{-1}(z_flat) -> BigVAE decoder
```

has a more isotropic pullback metric. The flow is identity-initialized, so reconstruction is unchanged at step 0.

Run:

```sh
BIG_VAE_CHECKPOINT=/path/to/latest.pt \
OFFLINE_ROOT=/path/to/offline_big_vae_dataset \
post_train_research/big_vae_latent_flattening/run_big_vae_latent_flattening.sh
```

Useful overrides:

```sh
MAX_STEPS=2000 BATCH_SIZE=2 MAX_T_PATCHES=4 MAX_D_OUT=16 PROBES=1 \
FLOW_LAYERS=8 FLOW_HIDDEN_DIM=512 \
BIG_VAE_CHECKPOINT=/path/to/latest.pt \
OFFLINE_ROOT=/path/to/offline_big_vae_dataset \
post_train_research/big_vae_latent_flattening/run_big_vae_latent_flattening.sh
```

Outputs are written under `post_train_research/big_vae_latent_flattening/artifacts/runs/*`.

