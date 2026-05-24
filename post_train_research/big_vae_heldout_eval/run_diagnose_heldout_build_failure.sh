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
export HELDOUT_ROOT="${HELDOUT_ROOT:-$ARTIFACT_ROOT/datasets/heldout/big_vae/offline_dataset}"
export HELDOUT_LOG_DIR="${HELDOUT_LOG_DIR:-$ARTIFACT_ROOT/eval/heldout/logs}"
export HELDOUT_REPORTS_DIR="${HELDOUT_REPORTS_DIR:-$ARTIFACT_ROOT/eval/heldout/reports}"
export HELDOUT_CRASHES_DIR="${HELDOUT_CRASHES_DIR:-$ARTIFACT_ROOT/eval/heldout/crashes}"
export HELDOUT_DIAG_JSON_OUT="${HELDOUT_DIAG_JSON_OUT:-$ARTIFACT_ROOT/eval/heldout/heldout_diag.json}"

exec python "$SCRIPT_DIR/diagnose_heldout_build_failure.py" \
  --heldout-root "$HELDOUT_ROOT" \
  --log-dir "$HELDOUT_LOG_DIR" \
  --reports-dir "$HELDOUT_REPORTS_DIR" \
  --crashes-dir "$HELDOUT_CRASHES_DIR" \
  --json-out "$HELDOUT_DIAG_JSON_OUT" \
  "$@"
