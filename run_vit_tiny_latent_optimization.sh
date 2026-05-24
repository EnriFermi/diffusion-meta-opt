#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-diff-meta-opt312}"
exec conda run --no-capture-output -n "$CONDA_ENV_NAME" python -m experiments.compare_vit_tiny_latent_optimization "$@" \
  --setup bigvae_latent \
  --big-vae-checkpoint artifacts/training/checkpoints/weight_quantile_vae_gpu0/stage_1/latest.pt \
  --epochs 20 \
  --batch-size 128 \
  --device cuda:0 \
  --latent-lr 1e-1 \
  --big-vae-latent-noise-std 0.02 \
  --big-vae-tile-t-patches 4 \
  --big-vae-tile-d-out 64 
