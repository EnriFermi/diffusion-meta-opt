#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
exec conda run -n "$CONDA_ENV_NAME" python -m experiments.compare_vit_tiny_latent_optimization "$@"
