#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME="${BIG_VAE_LATENT_DIFFUSION_CONFIG:-config_big_vae_latent_diffusion_prior}"

exec "$SCRIPT_DIR/../_run_python_module.sh" \
  experiments.train_big_vae_latent_diffusion_prior \
  --config-name "$CONFIG_NAME" \
  "$@"
