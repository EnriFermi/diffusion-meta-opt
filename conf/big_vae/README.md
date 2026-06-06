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

To add a new BigVAE run, create one file:

```text
conf/big_vae/train/my_run.yaml
```

Start from `v2.yaml`, keep the defaults local to that file, and put all run
overrides in the same file.
