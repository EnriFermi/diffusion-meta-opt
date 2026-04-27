#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="onerec"

STAGE1_CONFIG_NAME="vit_latent_scaling/config_ae_diffusion_prior"
STAGE2_CONFIG_NAME="vit_latent_scaling/config_latent_random"

STAGE1_PRESET="cifar10_small"
STAGE2_PRESET="mnist_small"

EXPERIMENT_ROOT="./post_train_research/vit_latent_scaling/artifacts/latent_transfer"
STAGE1_OUTPUT_DIR="$EXPERIMENT_ROOT/stage1_source"
STAGE2_OUTPUT_DIR="$EXPERIMENT_ROOT/stage2_target"

BIG_VAE_CHECKPOINT=""
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT=""
BIG_VAE_INIT_CALIBRATION_BATCHES="1"

STAGE1_OPTIMIZER_NAME="AdamW"
STAGE2_OPTIMIZER_NAME="AdamW"

STAGE1_LATENT_LR_SCHEDULER="cosine_decay_to_floor"
STAGE1_LATENT_LR_FLOOR_RATIO="0.1"
STAGE1_LATENT_LR_DECAY_STEPS="5250"

STAGE2_LATENT_LR_SCHEDULER="cosine_decay_to_floor"
STAGE2_LATENT_LR_FLOOR_RATIO="0.1"
STAGE2_LATENT_LR_DECAY_STEPS="5250"

: "${BIG_VAE_CHECKPOINT:?Edit BIG_VAE_CHECKPOINT in this script before running}"

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
  --config-name "$STAGE1_CONFIG_NAME" \
  "vit_latent_scaling/preset=$STAGE1_PRESET" \
  "vit_latent_scaling.output_dir=$STAGE1_OUTPUT_DIR" \
  "vit_latent_scaling.optimizer_name=$STAGE1_OPTIMIZER_NAME" \
  "vit_latent_scaling.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "vit_latent_scaling.big_vae_diffusion_prior_checkpoint=$BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT" \
  "vit_latent_scaling.big_vae_init_calibration_batches=$BIG_VAE_INIT_CALIBRATION_BATCHES" \
  "vit_latent_scaling.latent_lr_scheduler=$STAGE1_LATENT_LR_SCHEDULER" \
  "vit_latent_scaling.latent_lr_floor_ratio=$STAGE1_LATENT_LR_FLOOR_RATIO" \
  "vit_latent_scaling.latent_lr_decay_steps=$STAGE1_LATENT_LR_DECAY_STEPS"

STAGE1_LATENT_CHECKPOINT="$STAGE1_OUTPUT_DIR/checkpoints/latent_final.pt"

run_python \
  "$SCRIPT_DIR/run_vit_latent_scaling_hydra.py" \
  --config-name "$STAGE2_CONFIG_NAME" \
  "vit_latent_scaling/preset=$STAGE2_PRESET" \
  "vit_latent_scaling.output_dir=$STAGE2_OUTPUT_DIR" \
  "vit_latent_scaling.optimizer_name=$STAGE2_OPTIMIZER_NAME" \
  "vit_latent_scaling.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "vit_latent_scaling.latent_checkpoint=$STAGE1_LATENT_CHECKPOINT" \
  "vit_latent_scaling.latent_lr_scheduler=$STAGE2_LATENT_LR_SCHEDULER" \
  "vit_latent_scaling.latent_lr_floor_ratio=$STAGE2_LATENT_LR_FLOOR_RATIO" \
  "vit_latent_scaling.latent_lr_decay_steps=$STAGE2_LATENT_LR_DECAY_STEPS"
