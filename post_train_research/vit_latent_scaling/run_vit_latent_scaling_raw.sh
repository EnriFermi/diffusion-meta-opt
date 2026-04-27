#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="onerec"
PRESET="cifar10_small"
OUTPUT_ROOT="./post_train_research/vit_latent_scaling/artifacts"

run_python() {
  if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python "$@"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$@"
    return
  fi

  python "$@"
}

run_python \
  "$SCRIPT_DIR/run_vit_latent_scaling_hydra.py" \
  --config-name config_vit_latent_scaling_raw \
  "vit_latent_scaling/preset=$PRESET" \
  "vit_latent_scaling.output_root=$OUTPUT_ROOT"
