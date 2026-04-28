#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="onerec"
DATA_DIR="./data/cifar10"
OUTPUT_DIR="./post_train_research/loss_landscape_analysis/artifacts/latent_landscape_tinyvit"
DEVICE="auto"

SEEDS="0 1 2 3 4"
TRAIN_SUBSET="10000"
TEST_SUBSET="2000"
BATCH_SIZE="128"
EVAL_BATCH_SIZE="256"
NUM_WORKERS="4"
DIAG_BATCH_SIZE="1024"
DIAGNOSTIC_SPLIT="train"

STEPS="3000"
EPOCHS="1000"
LR="100"
WEIGHT_DECAY="0"
GRAD_CLIP_NORM="0"
LOG_EVERY="25"
EVAL_EVERY="100"
CHECKPOINT_STEPS="1 50 100 200 500"

BIG_VAE_CHECKPOINT=""
BIG_VAE_LATENT_INIT="diffusion_prior"
BIG_VAE_LATENT_PARAMETERIZATION="sphere"
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT=""
BIG_VAE_DIFFUSION_PRIOR_STEPS="50"
BIG_VAE_DIFFUSION_PRIOR_SAMPLER="ddim"
BIG_VAE_DIFFUSION_PRIOR_ETA="0.0"
BIG_VAE_DECODE="all"
BIG_VAE_TILE_T_PATCHES="16"
BIG_VAE_TILE_D_OUT="8"
BIG_VAE_ENCODER_CONTEXT_ROWS="64"
BIG_VAE_ENCODER_CONTEXT_STD="1.0"
BIG_VAE_ENCODER_BATCH_SIZE="16"
BIG_VAE_INIT_CALIBRATION_BATCHES="1"

IMAGE_SIZE="32"
PATCH_SIZE="4"
HIDDEN_DIM="128"
DEPTH="3"
NUM_HEADS="4"
MLP_RATIO="2.0"
DROPOUT="0.0"
ATTENTION_DROPOUT="0.0"
NUM_CLASSES="10"
IN_CHANNELS="3"

RADIAL_SCALES="1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 0.3 1 3 10 30 100 300 1000"
THETA_GRID="-1.5707963267948966 -1.0471975511965976 -0.5235987755982988 -0.2617993877991494 0 0.2617993877991494 0.5235987755982988 1.0471975511965976 1.5707963267948966"
ANGULAR_RANDOM_DIRS="4"
JACOBIAN_EPS_REL="1e-3"
JACOBIAN_TANGENT_DIRS="8"

COMPUTE_ACCESSIBILITY="false"
COMPUTE_HESSIAN="false"
HESSIAN_SUBSPACE_DIM="128"
HESSIAN_TOPK="5"
HESSIAN_TRACE_PROBES="16"
COMPUTE_2D_SLICES="false"
SLICE_GRID_SIZE="31"
SLICE_SCALE="1.0"

: "${BIG_VAE_CHECKPOINT:?Edit BIG_VAE_CHECKPOINT in this script before running}"
if [ "$BIG_VAE_LATENT_INIT" = "diffusion_prior" ]; then
  : "${BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT:?Edit BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT in this script before running}"
fi

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

set -- \
  "$PROJECT_ROOT/experiments/diagnose_latent_landscape_tinyvit.py" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --seeds $SEEDS \
  --train_subset "$TRAIN_SUBSET" \
  --test_subset "$TEST_SUBSET" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --diag_batch_size "$DIAG_BATCH_SIZE" \
  --diagnostic_split "$DIAGNOSTIC_SPLIT" \
  --steps "$STEPS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip_norm "$GRAD_CLIP_NORM" \
  --log_every "$LOG_EVERY" \
  --eval_every "$EVAL_EVERY" \
  --checkpoint_steps $CHECKPOINT_STEPS \
  --vae_ckpt "$BIG_VAE_CHECKPOINT" \
  --big_vae_latent_init "$BIG_VAE_LATENT_INIT" \
  --big_vae_latent_parameterization "$BIG_VAE_LATENT_PARAMETERIZATION" \
  --prior_ckpt "$BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT" \
  --big_vae_diffusion_prior_steps "$BIG_VAE_DIFFUSION_PRIOR_STEPS" \
  --big_vae_diffusion_prior_sampler "$BIG_VAE_DIFFUSION_PRIOR_SAMPLER" \
  --big_vae_diffusion_prior_eta "$BIG_VAE_DIFFUSION_PRIOR_ETA" \
  --big_vae_decode "$BIG_VAE_DECODE" \
  --big_vae_tile_T_patches "$BIG_VAE_TILE_T_PATCHES" \
  --big_vae_tile_d_out "$BIG_VAE_TILE_D_OUT" \
  --big_vae_encoder_context_rows "$BIG_VAE_ENCODER_CONTEXT_ROWS" \
  --big_vae_encoder_context_std "$BIG_VAE_ENCODER_CONTEXT_STD" \
  --big_vae_encoder_batch_size "$BIG_VAE_ENCODER_BATCH_SIZE" \
  --big_vae_init_calibration_batches "$BIG_VAE_INIT_CALIBRATION_BATCHES" \
  --image_size "$IMAGE_SIZE" \
  --patch_size "$PATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --depth "$DEPTH" \
  --num_heads "$NUM_HEADS" \
  --mlp_ratio "$MLP_RATIO" \
  --dropout "$DROPOUT" \
  --attention_dropout "$ATTENTION_DROPOUT" \
  --num_classes "$NUM_CLASSES" \
  --in_channels "$IN_CHANNELS" \
  --radial_scales $RADIAL_SCALES \
  --theta_grid $THETA_GRID \
  --angular_random_dirs "$ANGULAR_RANDOM_DIRS" \
  --jacobian_eps_rel "$JACOBIAN_EPS_REL" \
  --jacobian_tangent_dirs "$JACOBIAN_TANGENT_DIRS" \
  --hessian_subspace_dim "$HESSIAN_SUBSPACE_DIM" \
  --hessian_topk "$HESSIAN_TOPK" \
  --hessian_trace_probes "$HESSIAN_TRACE_PROBES" \
  --slice_grid_size "$SLICE_GRID_SIZE" \
  --slice_scale "$SLICE_SCALE"

if [ "$COMPUTE_ACCESSIBILITY" = "true" ]; then
  set -- "$@" --compute_accessibility
fi
if [ "$COMPUTE_HESSIAN" = "true" ]; then
  set -- "$@" --compute_hessian
fi
if [ "$COMPUTE_2D_SLICES" = "true" ]; then
  set -- "$@" --compute_2d_slices
fi

run_python "$@"
