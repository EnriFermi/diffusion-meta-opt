#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Примеры:
#   ./run_big_vae_latent_diffusion_dataset_repair.sh \
#     ./artifacts/training/datasets/big_vae_latent_diffusion/stage_1
#
#   BIG_VAE_LATENT_DIFFUSION_DATASET_DIR=./artifacts/training/datasets/big_vae_latent_diffusion/stage_1 \
#   BIG_VAE_LATENT_DIFFUSION_REPAIR_PREFIX_FRACTIONS=0.01,0.05,0.1,0.25,0.5,1.0 \
#   ./run_big_vae_latent_diffusion_dataset_repair.sh
#
# Entrypoint:
#   python -m experiments.repair_big_vae_latent_diffusion_dataset
#
# Если аргументы не переданы, скрипт попробует взять root из
# BIG_VAE_LATENT_DIFFUSION_DATASET_DIR и подставит базовые опции repair-а.

if [ "$#" -eq 0 ]; then
  DATASET_ROOT="${BIG_VAE_LATENT_DIFFUSION_DATASET_DIR:-}"
  if [ -z "$DATASET_ROOT" ]; then
    echo "usage: $0 <latent_diffusion_dataset_root> [repair args...]" >&2
    echo "or set BIG_VAE_LATENT_DIFFUSION_DATASET_DIR" >&2
    exit 2
  fi
  OUTPUT_DIR="${BIG_VAE_LATENT_DIFFUSION_REPAIR_OUTPUT_DIR:-$DATASET_ROOT/analysis}"
  PREFIX_FRACTIONS="${BIG_VAE_LATENT_DIFFUSION_REPAIR_PREFIX_FRACTIONS:-0.01,0.05,0.1,0.25,0.5,1.0}"
  TOPK="${BIG_VAE_LATENT_DIFFUSION_REPAIR_TOPK:-20}"
  set -- \
    "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --prefix-fractions "$PREFIX_FRACTIONS" \
    --topk "$TOPK"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m experiments.repair_big_vae_latent_diffusion_dataset "$@"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m experiments.repair_big_vae_latent_diffusion_dataset "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python -m experiments.repair_big_vae_latent_diffusion_dataset "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m experiments.repair_big_vae_latent_diffusion_dataset "$@"
fi

exec python -m experiments.repair_big_vae_latent_diffusion_dataset "$@"
