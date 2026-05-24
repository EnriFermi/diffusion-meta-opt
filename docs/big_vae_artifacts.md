# BigVAE Artifact Layout

Active BigVAE training and post-train jobs use one artifact root:

```text
${BIG_VAE_ARTIFACT_ROOT:-./artifacts/big_vae}/
  runs/<run_id>/
    logs/
    reports/
    crashes/
    config_resolved.yaml
    artifact_layout.json
  checkpoints/
    train/default/stage_<N>/
    train/v2/stage_<N>/
    latent_diffusion_prior/stage_<N>/
  datasets/
    offline/big_vae/stage_<N>/offline_dataset/
    presliced/big_vae/<config>/stage_<N>/presliced_offline_dataset/
    heldout/big_vae/offline_dataset/
    latent_diffusion/stage_<N>/
  eval/
    suite/
    heldout/
    latent_flattening/
    vit_latent_scaling/
    tinyvit_latent_h1/
    loss_landscape/
  tmp/
```

`BIG_VAE_ARTIFACT_ROOT` is the only root-level override. Use narrower env vars
only when a specific tool must read or write somewhere else:

```text
BIG_VAE_CHECKPOINT
BIG_VAE_OFFLINE_DATASET_DIR
BIG_VAE_LATENT_DIFFUSION_DATASET_DIR
BIG_VAE_EVAL_SUITE_ROOT
BIG_VAE_HELDOUT_ROOT
BIG_VAE_LATENT_FLATTENING_ROOT
BIG_VAE_VIT_SCALING_ROOT
BIG_VAE_TINYVIT_H1_ROOT
```

Legacy MiniVAE/procedural artifacts are not part of this layout. Historical
files under `artifacts/training` can stay on disk, but new active BigVAE runs
should not write there by default.
