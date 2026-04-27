#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="onerec"
PRESET="cifar10_small"
OUTPUT_ROOT="./artifacts/training/post_train_research/vit_latent_scaling"
RAW_CHECKPOINT=""
BIG_VAE_CHECKPOINT=""
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT=""

: "${RAW_CHECKPOINT:?Edit RAW_CHECKPOINT in this script before running}"
: "${BIG_VAE_CHECKPOINT:?Edit BIG_VAE_CHECKPOINT in this script before running}"
: "${BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT:?Edit BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT in this script before running}"

run_python() {
  if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python "$@"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$@"
    return
  fi

  python "$@"
}

run_python \
  "$SCRIPT_DIR/run_vit_latent_scaling_hydra.py" \
  --config-name config_vit_latent_scaling_ae_diffusion_prior \
  "vit_latent_scaling/preset=$PRESET" \
  "vit_latent_scaling.output_root=$OUTPUT_ROOT" \
  "vit_latent_scaling.raw_checkpoint=$RAW_CHECKPOINT" \
  "vit_latent_scaling.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "vit_latent_scaling.big_vae_diffusion_prior_checkpoint=$BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT"
