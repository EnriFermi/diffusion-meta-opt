#!/usr/bin/env bash

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m legacy.procedural_simple_vae_pipeline.train "$@"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  exec python -m legacy.procedural_simple_vae_pipeline.train "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m legacy.procedural_simple_vae_pipeline.train "$@"
fi

exec python -m legacy.procedural_simple_vae_pipeline.train "$@"
