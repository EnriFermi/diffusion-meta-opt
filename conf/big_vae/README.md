# BigVAE Configs

The active BigVAE train entrypoints live in `conf/big_vae/train/`.
Latent diffusion configs live in `conf/big_vae/latent_diffusion/`.

Use them directly:

```bash
scripts/launchers/big_vae/train.sh
BIG_VAE_CONFIG=big_vae/train/v2 scripts/launchers/big_vae/train.sh
scripts/launchers/big_vae/latent_diffusion_dataset_build.sh
scripts/launchers/big_vae/latent_diffusion_train.sh
```

To add a new BigVAE train config, create one file:

```text
conf/big_vae/train/my_run.yaml
```

Start from `v2.yaml`, keep the defaults local to that file, and put all run
overrides in the same file.

The active model parameter files are:

```text
conf/big_vae_experiment/model_parameters_specific_to_big_vae_stage.yaml
conf/big_vae_experiment/model_parameters_specific_to_big_vae_stage_v2.yaml
```

Those files own the full model shape: `patch_size`, `big_vae.*`,
`big_vae.distribution_encoder.*`, and `big_vae.patch_tokenizer.*`.
