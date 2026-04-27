# ViT Latent Scaling Experiments

This experiment compares two optimization parameterizations for ViT classifiers:

- `raw`: optimize ViT parameters directly with AdamW.
- `latent`: load a raw AdamW checkpoint, encode its named tensors with a frozen BigVAE encoder, decode latent slots back into ViT tensors on every forward pass, freeze BigVAE, and optimize only latent slots with AdamW.

The latent runs use the LR grid:

```text
1e-4 1e-3 1e-2 1e-1 1e0
```

The same grid is also run for raw baselines. `RAW_INIT_LR=1e-3` selects the raw checkpoint used to initialize latent slots. Metrics include both `step` and `total_steps_with_raw_init`, so the latent-only continuation and the end-to-end raw+latent budget are both visible.

## Runner

Use the POSIX shell runner:

```bash
BIG_VAE_CHECKPOINT=artifacts/training/checkpoints/weight_quantile_vae/stage_1/latest.pt \
post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh cifar10_small
```

The single positional mode chooses what to run:

```text
all
mnist, cifar10, imagenet
mnist_tiny, mnist_small, mnist_medium
cifar10_tiny, cifar10_small, cifar10_medium
imagenet_tiny, imagenet_small, imagenet_base
<config>_raw
<config>_latent
<config>_pipeline
```

Examples:

```bash
post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh mnist_tiny_raw
BIG_VAE_CHECKPOINT=... post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh mnist_tiny_latent
BIG_VAE_CHECKPOINT=... post_train_research/vit_latent_scaling/run_vit_latent_scaling.sh cifar10
```

`*_latent` automatically runs the raw init checkpoint first if it does not exist.

## Defaults

Data roots:

```text
MNIST_DATA_DIR=$PROJECT_ROOT/data/mnist
CIFAR10_DATA_DIR=$PROJECT_ROOT/data/cifar10
IMAGENET_DATA_DIR=$PROJECT_ROOT/data/imagenet
```

ImageNet expects either an ImageFolder layout:

```text
$IMAGENET_DATA_DIR/train/<class>/*
$IMAGENET_DATA_DIR/val/<class>/*
```

or a torchvision `ImageNet` root with the required metadata.

Model grid:

```text
mnist_tiny:     image=28 patch=4  hidden=64  depth=4  heads=4
mnist_small:    image=28 patch=4  hidden=128 depth=6  heads=4
mnist_medium:   image=28 patch=4  hidden=192 depth=8  heads=6
cifar10_tiny:   image=32 patch=4  hidden=192 depth=6  heads=3
cifar10_small:  image=32 patch=4  hidden=256 depth=8  heads=4
cifar10_medium: image=32 patch=4  hidden=384 depth=12 heads=6
imagenet_tiny:  image=224 patch=16 hidden=192 depth=12 heads=3
imagenet_small: image=224 patch=16 hidden=384 depth=12 heads=6
imagenet_base:  image=224 patch=16 hidden=768 depth=12 heads=12
```

Outputs are written under:

```text
post_train_research/vit_latent_scaling/artifacts/<dataset>/<size>/<setup>_lr_<lr_tag>/
```

Important files:

```text
config.json
metrics.csv
summary.json
checkpoints/raw_final.pt
checkpoints/latent_final.pt
```

Set `FORCE=true` to rerun an output directory that already has `summary.json`.

## Hydra Runner

There is now an additional Hydra entrypoint for single targeted runs:

```bash
python post_train_research/vit_latent_scaling/run_vit_latent_scaling_hydra.py \
  --config-name config_vit_latent_scaling_ae_encoded \
  vit_latent_scaling/preset=cifar10_small \
  vit_latent_scaling.raw_checkpoint=/path/to/raw_final.pt \
  vit_latent_scaling.big_vae_checkpoint=/path/to/big_vae.pt
```

The three convenience configs are:

```text
config_vit_latent_scaling_ae_encoded
config_vit_latent_scaling_latent_random
config_vit_latent_scaling_ae_diffusion_prior
```

They correspond to:

```text
ae_encoded          -> setup=latent, big_vae_latent_init=encoded
latent_random       -> setup=latent, big_vae_latent_init=random
ae_diffusion_prior  -> setup=latent, big_vae_latent_init=diffusion_prior
```

There are also self-contained shell wrappers next to this README:

```text
run_vit_latent_scaling_raw.sh
run_vit_latent_scaling_ae_encoded.sh
run_vit_latent_scaling_latent_random.sh
run_vit_latent_scaling_ae_diffusion_prior.sh
```

Each script has editable variables at the top for:

```text
PRESET
OUTPUT_ROOT
RAW_CHECKPOINT
BIG_VAE_CHECKPOINT
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT
```

For the fair from-scratch comparison, use:

```text
run_vit_latent_scaling_raw.sh
run_vit_latent_scaling_latent_random.sh
run_vit_latent_scaling_ae_diffusion_prior.sh
```

`RAW_CHECKPOINT` is only needed by the optional `ae_encoded` path, because that mode initializes latents by encoding an already existing raw ViT solution.

By default the Hydra runner and these wrappers write outputs under:

```text
post_train_research/vit_latent_scaling/artifacts/
```

You can also use the general config and switch groups explicitly:

```bash
python post_train_research/vit_latent_scaling/run_vit_latent_scaling_hydra.py \
  vit_latent_scaling/preset=cifar10_small \
  vit_latent_scaling/setup=ae_diffusion_prior \
  vit_latent_scaling.raw_checkpoint=/path/to/raw_final.pt \
  vit_latent_scaling.big_vae_checkpoint=/path/to/big_vae.pt \
  vit_latent_scaling.big_vae_diffusion_prior_checkpoint=/path/to/prior.pt
```
