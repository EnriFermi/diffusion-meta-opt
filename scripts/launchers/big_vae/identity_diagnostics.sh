#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CONDA_ENV_NAME="${DIAGNOSTICS_CONDA_ENV:-${CONDA_ENV_NAME:-onerec}}"

exec "$SCRIPT_DIR/../_run_python_module.sh" \
  big_vae.entrypoints.identity_diagnostics \
  "$@"
