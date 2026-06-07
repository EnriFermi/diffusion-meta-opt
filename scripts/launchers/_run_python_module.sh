#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <python.module> [args...]" >&2
  exit 2
fi

MODULE_NAME="$1"
shift

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -f "./mom.env" ]; then
  set -a
  . "./mom.env"
  set +a
fi

if [ -n "${COMET_API_KEY_DEFAULT:-}" ]; then
  export COMET_API_KEY="${COMET_API_KEY:-$COMET_API_KEY_DEFAULT}"
fi

if [ -n "${COMET_WORKSPACE_DEFAULT:-}" ]; then
  export COMET_WORKSPACE="${COMET_WORKSPACE:-$COMET_WORKSPACE_DEFAULT}"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m "$MODULE_NAME" "$@"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"

if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV_NAME" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m "$MODULE_NAME" "$@"
fi

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "$CONDA_ENV_NAME" bash -c '
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    module_name="$1"
    shift
    exec python -m "$module_name" "$@"
  ' _ "$MODULE_NAME" "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m "$MODULE_NAME" "$@"
fi

exec python -m "$MODULE_NAME" "$@"
