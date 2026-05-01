#!/usr/bin/env bash

set -a
. "./mom.env"
set +a
export COMET_API_KEY="${COMET_API_KEY:-$COMET_API_KEY_DEFAULT}"
export COMET_WORKSPACE="${COMET_WORKSPACE:-$COMET_WORKSPACE_DEFAULT}"


if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m experimen s.train_big_vae "$@"
fi 

if [ -n "${CONDA_PREFIX:-}" ]; then
  exec python -m experiments.train_big_vae "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m experiments.train_big_vae "$@"
fi

exec python -m experiments.train_big_vae "$@"