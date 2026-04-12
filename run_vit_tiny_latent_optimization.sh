#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Examples:
#   ./run_vit_tiny_latent_optimization.sh \
#     --setup both \
#     --big-vae-checkpoint artifacts/training/checkpoints/weight_quantile_vae/stage_1/latest.pt \
#     --epochs 20 --batch-size 128
#
#   CUDA_VISIBLE_DEVICES=1 ./run_vit_tiny_latent_optimization.sh \
#     --device cuda:0 --setup bigvae_latent \
#     --big-vae-checkpoint artifacts/training/checkpoints/weight_quantile_vae/stage_1/latest.pt
#
#   ./run_vit_tiny_latent_optimization.sh --setup lowrank_latent \
#     --max-steps 20 --train-subset 512 --test-subset 256 --no-save-checkpoints

if [ -n "${CONDA_PREFIX:-}" ] && python -c "import torch, torchvision" >/dev/null 2>&1; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m experiments.compare_vit_tiny_latent_optimization "$@"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m experiments.compare_vit_tiny_latent_optimization "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python -m experiments.compare_vit_tiny_latent_optimization "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m experiments.compare_vit_tiny_latent_optimization "$@"
fi

exec python -m experiments.compare_vit_tiny_latent_optimization "$@"
