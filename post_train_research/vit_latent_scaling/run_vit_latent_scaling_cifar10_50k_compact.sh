#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="diff-meta-opt312"
CONFIG_NAME="vit_latent_scaling/config_cifar10_50k"
SETUP_GROUP="latent"
INIT_GROUP="diffusion_prior"
OUTPUT_ROOT="./post_train_research/vit_latent_scaling/artifacts"

BIG_VAE_CHECKPOINT="./artifacts/training/checkpoints/weight_quantile_vae_gpu0_square/stage_1/latest.pt"
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT="./artifacts/training/checkpoints/big_vae_latent_diffusion_prior_AE/stage_1/latest.pt"
RAW_CHECKPOINT=""
LATENT_CHECKPOINT=""

BIG_VAE_LATENT_PARAMETERIZATION="euclidean"
BIG_VAE_INIT_CALIBRATION_BATCHES="4"
BIG_VAE_DECODE="weights"
BIG_VAE_TILE_T_PATCHES="4"
BIG_VAE_TILE_D_OUT="64"

OPTIMIZER_NAME="AdamW"
ADAM_BETA1="0.9"
ADAM_BETA2="0.95"
ADAM_EPS="1e-9"
LATENT_LR_SCHEDULER="cosine_decay_to_floor"
LATENT_LR_FLOOR_RATIO="1.0"
LATENT_LR_DECAY_STEPS="1"

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

case "$SETUP_GROUP" in
  raw)
    INIT_GROUP="noop"
    ;;
  latent)
    : "${BIG_VAE_CHECKPOINT:?Edit BIG_VAE_CHECKPOINT in this script before running latent setup}"
    case "$INIT_GROUP" in
      base|random)
        ;;
      encoded)
        : "${RAW_CHECKPOINT:?Edit RAW_CHECKPOINT for INIT_GROUP=encoded}"
        ;;
      diffusion_prior)
        : "${BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT:?Edit BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT for INIT_GROUP=diffusion_prior}"
        ;;
      from_checkpoint)
        : "${LATENT_CHECKPOINT:?Edit LATENT_CHECKPOINT for INIT_GROUP=from_checkpoint}"
        ;;
      *)
        echo "Unsupported INIT_GROUP=$INIT_GROUP. Use one of: base random encoded diffusion_prior from_checkpoint" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported SETUP_GROUP=$SETUP_GROUP. Use raw or latent." >&2
    exit 1
    ;;
esac

set -- \
  "$SCRIPT_DIR/run_vit_latent_scaling_hydra.py" \
  --config-name "$CONFIG_NAME" \
  "vit_latent_scaling/setup=$SETUP_GROUP" \
  "vit_latent_scaling/init=$INIT_GROUP" \
  "vit_latent_scaling.output_root=$OUTPUT_ROOT" \
  "vit_latent_scaling.optimizer_name=$OPTIMIZER_NAME" \
  "vit_latent_scaling.adam_beta1=$ADAM_BETA1" \
  "vit_latent_scaling.adam_beta2=$ADAM_BETA2" \
  "vit_latent_scaling.adam_eps=$ADAM_EPS" \
  "vit_latent_scaling.latent_lr_scheduler=$LATENT_LR_SCHEDULER" \
  "vit_latent_scaling.latent_lr_floor_ratio=$LATENT_LR_FLOOR_RATIO" \
  "vit_latent_scaling.latent_lr_decay_steps=$LATENT_LR_DECAY_STEPS" \
  "vit_latent_scaling.raw_checkpoint=$RAW_CHECKPOINT" \
  "vit_latent_scaling.latent_checkpoint=$LATENT_CHECKPOINT" \
  "vit_latent_scaling.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "vit_latent_scaling.big_vae_latent_parameterization=$BIG_VAE_LATENT_PARAMETERIZATION" \
  "vit_latent_scaling.big_vae_diffusion_prior_checkpoint=$BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT" \
  "vit_latent_scaling.big_vae_init_calibration_batches=$BIG_VAE_INIT_CALIBRATION_BATCHES" \
  "vit_latent_scaling.big_vae_decode=$BIG_VAE_DECODE" \
  "vit_latent_scaling.big_vae_tile_T_patches=$BIG_VAE_TILE_T_PATCHES" \
  "vit_latent_scaling.big_vae_tile_d_out=$BIG_VAE_TILE_D_OUT"

run_python "$@"
