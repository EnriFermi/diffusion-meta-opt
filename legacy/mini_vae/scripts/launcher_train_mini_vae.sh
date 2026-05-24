#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)

exec "$REPO_ROOT/scripts/launchers/_run_python_module.sh" \
  legacy.mini_vae.experiments.train_mini_vae \
  --config-name mini_vae_train \
  "$@"
