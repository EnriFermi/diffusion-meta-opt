#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_ROOT="${BIG_VAE_ARTIFACT_ROOT:-./artifacts/big_vae}"
BIG_VAE_CHECKPOINT="${BIG_VAE_CHECKPOINT:-$ARTIFACT_ROOT/checkpoints/train/default/stage_1/latest.pt}"

exec "$SCRIPT_DIR/../_run_python_module.sh" \
  big_vae.entrypoints.vit_tiny_latent_optimization \
  "$@" \
  --setup bigvae_latent \
  --big-vae-checkpoint "$BIG_VAE_CHECKPOINT" \
  --epochs 20 \
  --batch-size 128 \
  --device cuda:0 \
  --latent-lr 1e-1 \
  --big-vae-latent-noise-std 0.02 \
  --big-vae-tile-t-patches 4 \
  --big-vae-tile-d-out 64
