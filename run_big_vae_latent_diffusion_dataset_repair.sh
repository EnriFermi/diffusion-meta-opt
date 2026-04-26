#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Настройки repair-а редактируются прямо в этом файле.
# Скрипт не принимает ни positional args, ни env overrides.
DATASET_ROOT="./artifacts/training/datasets/big_vae_latent_diffusion/stage_1"
OUTPUT_DIR="$DATASET_ROOT/analysis"
PREFIX_FRACTIONS="0.01,0.05,0.1,0.25,0.5,1.0"
TOPK="20"
#
# Entrypoint:
#   python -m experiments.repair_big_vae_latent_diffusion_dataset

if [ "$#" -ne 0 ]; then
  echo "this script does not accept positional arguments" >&2
  exit 2
fi

if [ -z "$DATASET_ROOT" ]; then
  echo "DATASET_ROOT is empty in run_big_vae_latent_diffusion_dataset_repair.sh" >&2
  exit 2
fi

set -- \
  "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --prefix-fractions "$PREFIX_FRACTIONS" \
  --topk "$TOPK"

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
