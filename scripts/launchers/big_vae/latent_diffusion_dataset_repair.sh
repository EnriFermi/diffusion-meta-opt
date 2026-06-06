#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "this script does not accept positional arguments" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_ROOT="${BIG_VAE_ARTIFACT_ROOT:-./artifacts/big_vae}"
DATASET_ROOT="${BIG_VAE_LATENT_DIFFUSION_DATASET_ROOT:-$ARTIFACT_ROOT/datasets/latent_diffusion/stage_1}"
OUTPUT_DIR="${BIG_VAE_LATENT_DIFFUSION_REPAIR_OUTPUT_DIR:-$DATASET_ROOT/analysis}"
PREFIX_FRACTIONS="${BIG_VAE_LATENT_DIFFUSION_REPAIR_PREFIX_FRACTIONS:-0.01,0.05,0.1,0.25,0.5,1.0}"
TOPK="${BIG_VAE_LATENT_DIFFUSION_REPAIR_TOPK:-20}"

if [ -z "$DATASET_ROOT" ]; then
  echo "BIG_VAE_LATENT_DIFFUSION_DATASET_ROOT is empty" >&2
  exit 2
fi

exec "$SCRIPT_DIR/../_run_python_module.sh" \
  big_vae.entrypoints.latent_diffusion_dataset_repair \
  "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --prefix-fractions "$PREFIX_FRACTIONS" \
  --topk "$TOPK"
