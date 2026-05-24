#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME="${BIG_VAE_LATENT_DIFFUSION_DATASET_CONFIG:-config_big_vae_latent_diffusion_dataset_build}"

exec "$SCRIPT_DIR/../_run_python_module.sh" \
  experiments.build_big_vae_latent_diffusion_dataset \
  --config-name "$CONFIG_NAME" \
  "$@"
