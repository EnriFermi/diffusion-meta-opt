#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/mom.env" ]; then
  set -a
  . "$PROJECT_ROOT/mom.env"
  set +a
fi

export HELDOUT_ROOT="${HELDOUT_ROOT:-$PROJECT_ROOT/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset}"
export HELDOUT_LOG_DIR="${HELDOUT_LOG_DIR:-$PROJECT_ROOT/post_train_research/big_vae_heldout_eval/artifacts/logs}"
export HELDOUT_RECORDS_PER_PAIR="${HELDOUT_RECORDS_PER_PAIR:-1024}"
export HELDOUT_TARGET_SIZE_GB="${HELDOUT_TARGET_SIZE_GB:-20}"
export HELDOUT_OVERWRITE="${HELDOUT_OVERWRITE:-false}"
export HELDOUT_MAX_SEEN_SAMPLES="${HELDOUT_MAX_SEEN_SAMPLES:-0}"
export HELDOUT_X_CHUNK_SIZE_RECORDS="${HELDOUT_X_CHUNK_SIZE_RECORDS:-128}"
export HELDOUT_LOG_EVERY_SEEN_SAMPLES="${HELDOUT_LOG_EVERY_SEEN_SAMPLES:-1000}"
export HELDOUT_SAMPLE_WAIT_TIMEOUT_S="${HELDOUT_SAMPLE_WAIT_TIMEOUT_S:-1800}"
export HELDOUT_SAMPLE_WAIT_STATUS_EVERY_S="${HELDOUT_SAMPLE_WAIT_STATUS_EVERY_S:-60}"
export HELDOUT_SAMPLE_WAIT_POLL_S="${HELDOUT_SAMPLE_WAIT_POLL_S:-1}"
export HELDOUT_FAIL_ON_SAMPLE_WAIT_TIMEOUT="${HELDOUT_FAIL_ON_SAMPLE_WAIT_TIMEOUT:-true}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python "$SCRIPT_DIR/build_heldout_offline_dataset.py" "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python "$SCRIPT_DIR/build_heldout_offline_dataset.py" "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python "$SCRIPT_DIR/build_heldout_offline_dataset.py" "$@"
fi

exec python "$SCRIPT_DIR/build_heldout_offline_dataset.py" "$@"
