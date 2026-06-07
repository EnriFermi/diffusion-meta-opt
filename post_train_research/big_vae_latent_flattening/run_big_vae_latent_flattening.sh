#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
ARTIFACT_ROOT="${BIG_VAE_ARTIFACT_ROOT:-./artifacts/big_vae}"

RUN_LABEL="${RUN_LABEL:-big_vae_latent_flattening}"
STORAGE_ROOT="${STORAGE_ROOT:-$ARTIFACT_ROOT/eval/latent_flattening}"

BIG_VAE_CHECKPOINT="${BIG_VAE_CHECKPOINT:-$ARTIFACT_ROOT/checkpoints/train/default/stage_1/latest.pt}"
OFFLINE_ROOT="${OFFLINE_ROOT:-$ARTIFACT_ROOT/datasets/offline/big_vae/stage_1/offline_dataset}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

BATCH_SIZE="${BATCH_SIZE:-2}"
SOURCE_POOL_SIZE="${SOURCE_POOL_SIZE:-$BATCH_SIZE}"
MAX_T_PATCHES="${MAX_T_PATCHES:-4}"
MAX_D_OUT="${MAX_D_OUT:-16}"
MAX_X_ROWS="${MAX_X_ROWS:-0}"

FLOW_LAYERS="${FLOW_LAYERS:-8}"
FLOW_HIDDEN_DIM="${FLOW_HIDDEN_DIM:-512}"
FLOW_NETWORK_DEPTH="${FLOW_NETWORK_DEPTH:-2}"
FLOW_LOG_SCALE_CLAMP="${FLOW_LOG_SCALE_CLAMP:-2.0}"

ETA="${ETA:-0.2}"
PROBES="${PROBES:-1}"
LOSS_SCALE="${LOSS_SCALE:-readme}"
ISO_COEF="${ISO_COEF:-1.0}"
Z_NORM_COEF="${Z_NORM_COEF:-1e-6}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-250}"
AMP_ENCODE="${AMP_ENCODE:-true}"
FORCE_MATH_ATTENTION="${FORCE_MATH_ATTENTION:-true}"
SKIP_NONFINITE_UPDATES="${SKIP_NONFINITE_UPDATES:-true}"
MAX_CONSECUTIVE_NONFINITE_STEPS="${MAX_CONSECUTIVE_NONFINITE_STEPS:-20}"
NONFINITE_DEBUG_TOPK="${NONFINITE_DEBUG_TOPK:-8}"

run_python() {
  if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV_NAME" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python "$@"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "$CONDA_ENV_NAME" sh -c '
      export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      exec python "$@"
    ' _ "$@"
    return
  fi
  python "$@"
}

if [ -z "$BIG_VAE_CHECKPOINT" ]; then
  echo "Set BIG_VAE_CHECKPOINT to a trained BigVAE checkpoint." >&2
  exit 1
fi

if [ -z "$OFFLINE_ROOT" ]; then
  echo "Set OFFLINE_ROOT to a built BigVAE offline dataset root." >&2
  exit 1
fi

run_python \
  "$SCRIPT_DIR/main.py" \
  "experiment.run_label=$RUN_LABEL" \
  "storage.root_dir=$STORAGE_ROOT" \
  "storage.checkpoint_every_steps=$CHECKPOINT_EVERY_STEPS" \
  "big_vae.checkpoint=$BIG_VAE_CHECKPOINT" \
  "data.offline_root=$OFFLINE_ROOT" \
  "data.batch_size=$BATCH_SIZE" \
  "data.source_pool_size=$SOURCE_POOL_SIZE" \
  "data.max_T_patches=$MAX_T_PATCHES" \
  "data.max_d_out=$MAX_D_OUT" \
  "data.max_x_rows=$MAX_X_ROWS" \
  "flow.num_layers=$FLOW_LAYERS" \
  "flow.hidden_dim=$FLOW_HIDDEN_DIM" \
  "flow.network_depth=$FLOW_NETWORK_DEPTH" \
  "flow.log_scale_clamp=$FLOW_LOG_SCALE_CLAMP" \
  "train.device=$DEVICE" \
  "train.seed=$SEED" \
  "train.max_steps=$MAX_STEPS" \
  "train.lr=$LR" \
  "train.weight_decay=$WEIGHT_DECAY" \
  "train.grad_clip_norm=$GRAD_CLIP_NORM" \
  "train.eta=$ETA" \
  "train.probes=$PROBES" \
  "train.loss_scale=$LOSS_SCALE" \
  "train.iso_coef=$ISO_COEF" \
  "train.z_norm_coef=$Z_NORM_COEF" \
  "train.log_every_steps=$LOG_EVERY_STEPS" \
  "train.amp_encode=$AMP_ENCODE" \
  "train.force_math_attention=$FORCE_MATH_ATTENTION" \
  "train.skip_nonfinite_updates=$SKIP_NONFINITE_UPDATES" \
  "train.max_consecutive_nonfinite_steps=$MAX_CONSECUTIVE_NONFINITE_STEPS" \
  "train.nonfinite_debug_topk=$NONFINITE_DEBUG_TOPK"
