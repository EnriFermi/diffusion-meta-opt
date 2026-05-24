#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-diff-meta-opt312}"
exec "$SCRIPT_DIR/scripts/launchers/post_train/vit_tiny_latent_optimization.sh" "$@"
