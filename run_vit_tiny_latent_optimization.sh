#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
export PYTHONUNBUFFERED=1
exec conda run --no-capture-output -n "$CONDA_ENV_NAME" python -m experiments.compare_vit_tiny_latent_optimization "$@"
