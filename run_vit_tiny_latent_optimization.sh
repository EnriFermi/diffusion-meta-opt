#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-diff-meta-opt312}"
exec conda run -n "$CONDA_ENV_NAME" python -m experiments.compare_vit_tiny_latent_optimization "$@" \
  --setup both \
  --big-vae-checkpoint artifacts/training/checkpoints/weight_quantile_vae_gpu0/stage_1/latest.pt \
  --epochs 20 \
  --batch-size 128 \
  --device cuda:0
