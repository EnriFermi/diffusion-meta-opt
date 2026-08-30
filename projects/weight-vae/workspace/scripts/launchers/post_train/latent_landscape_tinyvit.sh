#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)

exec "$REPO_ROOT/post_train_research/loss_landscape_analysis/run_diagnose_latent_landscape_tinyvit.sh" "$@"
