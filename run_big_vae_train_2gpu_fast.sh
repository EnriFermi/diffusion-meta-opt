#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda environment
CONDA_ENV="onerec"
PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/${CONDA_ENV}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "       Make sure conda env '${CONDA_ENV}' exists."
    exit 1
fi

# Install missing deps if needed
"$PYTHON" -c "import hydra" 2>/dev/null || {
    echo "Installing hydra-core..."
    "${PYTHON%/python}/pip" install hydra-core
}

echo "=== Big VAE Training (2GPU Fast) ==="
echo "Python:  $PYTHON"
echo "Workdir: $SCRIPT_DIR"
echo ""

DEFAULT_OVERRIDES=(
  "collector=collector_profile_interleaved"
  "streaming=streaming_profile_local_disk_gpu_parallel"
  "train.distributed=true"
  "train.num_gpus=2"
  "train.amp=bf16"
  "train.compile=false"
  "train.compile_dynamic=false"
  "train.log_every=50"
)

exec "$PYTHON" experiments/train_big_vae.py "${DEFAULT_OVERRIDES[@]}" "$@"
