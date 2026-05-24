#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME="${BIG_VAE_CONFIG:-big_vae/train/default}"

exec "$SCRIPT_DIR/../_run_python_module.sh" \
  experiments.build_big_vae_offline_dataset \
  --config-name "$CONFIG_NAME" \
  "$@"
