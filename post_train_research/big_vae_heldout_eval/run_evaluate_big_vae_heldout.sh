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

: "${BIG_VAE_CHECKPOINT:?Set BIG_VAE_CHECKPOINT=/path/to/stage_N/latest.pt or step_XXXXXXX.pt}"

export HELDOUT_ROOT="${HELDOUT_ROOT:-$PROJECT_ROOT/post_train_research/big_vae_heldout_eval/artifacts/offline_dataset}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
export EVAL_MAX_RECORDS="${EVAL_MAX_RECORDS:-0}"
export EVAL_MAX_SLICES_PER_SOURCE="${EVAL_MAX_SLICES_PER_SOURCE:-0}"
export EVAL_SAVE_RECORD_METRICS="${EVAL_SAVE_RECORD_METRICS:-true}"
export EVAL_LOG_EVERY_RECORDS="${EVAL_LOG_EVERY_RECORDS:-100}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python "$SCRIPT_DIR/evaluate_big_vae_heldout.py" "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python "$SCRIPT_DIR/evaluate_big_vae_heldout.py" "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python "$SCRIPT_DIR/evaluate_big_vae_heldout.py" "$@"
fi

exec python "$SCRIPT_DIR/evaluate_big_vae_heldout.py" "$@"
