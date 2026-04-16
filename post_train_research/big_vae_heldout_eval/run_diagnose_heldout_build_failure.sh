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
export HELDOUT_DIAG_JSON_OUT="${HELDOUT_DIAG_JSON_OUT:-$PROJECT_ROOT/post_train_research/big_vae_heldout_eval/artifacts/heldout_diag.json}"

exec python "$SCRIPT_DIR/diagnose_heldout_build_failure.py" \
  --heldout-root "$HELDOUT_ROOT" \
  --log-dir "$HELDOUT_LOG_DIR" \
  --json-out "$HELDOUT_DIAG_JSON_OUT" \
  "$@"
