#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT_PATH="experiments/diagnose_big_vae_identity.py"

if [ -n "${DIAGNOSTICS_CONDA_ENV:-}" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found" >&2
    exit 1
  fi
  exec conda run -n "$DIAGNOSTICS_CONDA_ENV" python "$SCRIPT_PATH" "$@"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  exec python "$SCRIPT_PATH" "$@"
fi

echo "Set DIAGNOSTICS_CONDA_ENV=<env_name> or activate a conda env first." >&2
exit 1
