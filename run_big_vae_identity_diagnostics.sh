#!/usr/bin/env bash

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Примеры:
#   ./run_big_vae_identity_diagnostics.sh
#   ./run_big_vae_identity_diagnostics.sh diagnostics.frozen_target.source_kind=synthetic train.device=cpu
#   ./run_big_vae_identity_diagnostics.sh train.resume_checkpoint=/abs/path/latest.pt
#
# По умолчанию конфиг уже фиксирует single-device / no-noise debug режим:
# distributed=false, amp=false, compile=false, dropout=0, weight_decay=0, grad clipping off.

ENTRYPOINT=(python experiments/diagnose_big_vae_identity.py)
REQUIRED_IMPORTS='import torch, omegaconf'

if python -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
  HAVE_RUNTIME=1
else
  HAVE_RUNTIME=0
fi

if [ -n "${PIPENV_ACTIVE:-}" ] && [ "$HAVE_RUNTIME" -eq 1 ]; then
  exec "${ENTRYPOINT[@]}" "$@"
fi

if [ -n "${CONDA_PREFIX:-}" ] && [ "$HAVE_RUNTIME" -eq 1 ]; then
  exec "${ENTRYPOINT[@]}" "$@"
fi

if command -v conda >/dev/null 2>&1; then
  exec conda run -n "${DIAGNOSTICS_CONDA_ENV:-onerec}" "${ENTRYPOINT[@]}" "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run "${ENTRYPOINT[@]}" "$@"
fi

exec "${ENTRYPOINT[@]}" "$@"
