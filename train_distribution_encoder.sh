#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Comet credentials — replace "..." with real values or override via environment.
COMET_API_KEY_DEFAULT="..."
COMET_WORKSPACE_DEFAULT="..."
export COMET_API_KEY="${COMET_API_KEY:-$COMET_API_KEY_DEFAULT}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-$COMET_WORKSPACE_DEFAULT}"

# If running inside a conda environment, prefer its runtime libraries.
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Examples:
#   ./train_distribution_encoder.sh
#   ./train_distribution_encoder.sh train.device=cuda:1
#   ./train_distribution_encoder.sh train.max_steps=5000 train.batch_size=128
#   COMET_API_KEY=xxx COMET_WORKSPACE=yyy ./train_distribution_encoder.sh
#
# Entrypoint: python -m distribution_encoder.train
# Config: conf/distribution_encoder/config.yaml
# All arguments are passed as Hydra overrides.

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m distribution_encoder.train "$@"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  exec python -m distribution_encoder.train "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m distribution_encoder.train "$@"
fi

exec python -m distribution_encoder.train "$@"
