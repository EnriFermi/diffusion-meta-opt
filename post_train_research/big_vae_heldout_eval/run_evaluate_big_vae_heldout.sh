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


ARTIFACT_ROOT="${BIG_VAE_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts/big_vae}"
export BIG_VAE_CHECKPOINT="${BIG_VAE_CHECKPOINT:-$ARTIFACT_ROOT/checkpoints/train/default/stage_1/latest.pt}"
export HELDOUT_ROOT="${HELDOUT_ROOT:-$ARTIFACT_ROOT/datasets/heldout/big_vae/offline_dataset}"
export HELDOUT_LOG_DIR="${HELDOUT_LOG_DIR:-$ARTIFACT_ROOT/eval/heldout/logs}"
export HELDOUT_REPORTS_DIR="${HELDOUT_REPORTS_DIR:-$ARTIFACT_ROOT/eval/heldout/reports}"
export HELDOUT_CRASHES_DIR="${HELDOUT_CRASHES_DIR:-$ARTIFACT_ROOT/eval/heldout/crashes}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
export EVAL_MAX_RECORDS="${EVAL_MAX_RECORDS:-0}"
export EVAL_MAX_SLICES_PER_SOURCE="${EVAL_MAX_SLICES_PER_SOURCE:-0}"
export EVAL_SAVE_RECORD_METRICS="${EVAL_SAVE_RECORD_METRICS:-true}"
export EVAL_LOG_EVERY_RECORDS="${EVAL_LOG_EVERY_RECORDS:-100}"
export EVAL_REQUIRE_FULL_COVERAGE="${EVAL_REQUIRE_FULL_COVERAGE:-true}"
export EVAL_LATENT_DUMP_ENABLED="${EVAL_LATENT_DUMP_ENABLED:-true}"
export EVAL_LATENT_DUMP_MAX_ENTRIES="${EVAL_LATENT_DUMP_MAX_ENTRIES:-256}"
export EVAL_LATENT_DUMP_MAX_SLICES_PER_SOURCE="${EVAL_LATENT_DUMP_MAX_SLICES_PER_SOURCE:-1}"
export EVAL_LATENT_DUMP_BALANCE_ENABLED="${EVAL_LATENT_DUMP_BALANCE_ENABLED:-true}"
export EVAL_LATENT_DUMP_BALANCE_KEYS="${EVAL_LATENT_DUMP_BALANCE_KEYS:-dataset,model,layer_type,depth_label}"
export EVAL_LATENT_DUMP_MAX_PER_GROUP="${EVAL_LATENT_DUMP_MAX_PER_GROUP:-1}"
export EVAL_LATENT_PLOT_ENABLED="${EVAL_LATENT_PLOT_ENABLED:-true}"
export EVAL_LATENT_PLOT_TSNE="${EVAL_LATENT_PLOT_TSNE:-true}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m post_train_research.big_vae_heldout_eval.evaluate_parts.runner "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python -m post_train_research.big_vae_heldout_eval.evaluate_parts.runner "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m post_train_research.big_vae_heldout_eval.evaluate_parts.runner "$@"
fi

exec python -m post_train_research.big_vae_heldout_eval.evaluate_parts.runner "$@"
