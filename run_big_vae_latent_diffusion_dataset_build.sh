#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m experiments.build_big_vae_latent_diffusion_dataset "$@"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m experiments.build_big_vae_latent_diffusion_dataset "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python -m experiments.build_big_vae_latent_diffusion_dataset "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m experiments.build_big_vae_latent_diffusion_dataset "$@"
fi

exec python -m experiments.build_big_vae_latent_diffusion_dataset "$@"
