#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="diff-meta-opt312"
PRESET="cifar_mili"
OUTPUT_ROOT="./post_train_research/vit_latent_scaling/artifacts"
BIG_VAE_CHECKPOINT="./artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt"
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT="./artifacts/training/checkpoints/big_vae_latent_diffusion_prior_AE/stage_1/latest.pt"
BIG_VAE_LATENT_PARAMETERIZATION="sphere"
BIG_VAE_INIT_CALIBRATION_BATCHES="4"
OPTIMIZER_NAME="Adam"
ADAM_BETA1="0.9"
ADAM_BETA2="0.95"
ADAM_EPS="1e-10"
LATENT_LR_SCHEDULER="cosine_decay_to_floor"
LATENT_LR_FLOOR_RATIO="1.0"
LATENT_LR_DECAY_STEPS="700"
LATENT_DEBUG_LOG_DELTAS="true"
LATENT_DEBUG_JACOBIAN_EPS="1e-3"
LATENT_DEBUG_JACOBIAN_PROBES="16"

: "${BIG_VAE_CHECKPOINT:?Edit BIG_VAE_CHECKPOINT in this script before running}"
: "${BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT:?Edit BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT in this script before running}"

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
  --config-name vit_latent_scaling/config_ae_diffusion_prior \
  "vit_latent_scaling/preset=$PRESET" \
  "vit_latent_scaling.output_root=$OUTPUT_ROOT" \
  "vit_latent_scaling.optimizer_name=$OPTIMIZER_NAME" \
  "vit_latent_scaling.adam_beta1=$ADAM_BETA1" \
  "vit_latent_scaling.adam_beta2=$ADAM_BETA2" \
  "vit_latent_scaling.adam_eps=$ADAM_EPS" \
  "vit_latent_scaling.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "vit_latent_scaling.big_vae_latent_parameterization=$BIG_VAE_LATENT_PARAMETERIZATION" \
  "vit_latent_scaling.big_vae_diffusion_prior_checkpoint=$BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT" \
  "vit_latent_scaling.big_vae_init_calibration_batches=$BIG_VAE_INIT_CALIBRATION_BATCHES" \
  "vit_latent_scaling.latent_lr_scheduler=$LATENT_LR_SCHEDULER" \
  "vit_latent_scaling.latent_lr_floor_ratio=$LATENT_LR_FLOOR_RATIO" \
  "vit_latent_scaling.latent_lr_decay_steps=$LATENT_LR_DECAY_STEPS" \
  "vit_latent_scaling.latent_debug_log_deltas=$LATENT_DEBUG_LOG_DELTAS" \
  "vit_latent_scaling.latent_debug_jacobian_eps=$LATENT_DEBUG_JACOBIAN_EPS" \
  "vit_latent_scaling.latent_debug_jacobian_probes=$LATENT_DEBUG_JACOBIAN_PROBES"
