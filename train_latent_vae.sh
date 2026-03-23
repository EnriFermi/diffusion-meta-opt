#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"
MICROMAMBA_ENV_NAME="${MICROMAMBA_ENV_NAME:-distribution-encoder}"

# Comet credentials — replace "..." with real values or override via environment.
COMET_API_KEY_DEFAULT="RrClhd4FveFQKO4qLo4jBjrKu"
COMET_WORKSPACE_DEFAULT="dont4rootme"
COMET_PROJECT_NAME_DEFAULT="latent-set-vae"
COMET_EXPERIMENT_NAME_DEFAULT="vae-deepsets-baseline"
export COMET_API_KEY="${COMET_API_KEY:-$COMET_API_KEY_DEFAULT}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-$COMET_WORKSPACE_DEFAULT}"

# If running inside a conda environment, prefer its runtime libraries.
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Examples:
#   ./train_latent_vae.sh
#   ./train_latent_vae.sh latent_vae.device=cuda:1
#   ./train_latent_vae.sh latent_vae.max_steps=10000 latent_vae.batch_size=128
#   COMET_API_KEY=xxx COMET_WORKSPACE=yyy ./train_latent_vae.sh
#
# Entrypoint: python -m distribution_encoder.train_latent_vae
# Config: conf/distribution_encoder/config.yaml
# All arguments are passed as Hydra overrides.

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m distribution_encoder.train_latent_vae "$@"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  exec python -m distribution_encoder.train_latent_vae "$@"
fi

if command -v micromamba >/dev/null 2>&1; then
  exec micromamba run -n "$MICROMAMBA_ENV_NAME" python -m distribution_encoder.train_latent_vae "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m distribution_encoder.train_latent_vae "$@"
fi

exec python -m distribution_encoder.train_latent_vae "$@"
