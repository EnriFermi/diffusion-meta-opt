#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="onerec"

RUN_LABEL="tinyvit_h1"
OUTPUT_ROOT="./post_train_research/tinyvit_latent_h1/artifacts"

SOURCE_VIT_SCALING_ROOT="./post_train_research/vit_latent_scaling/artifacts"
SOURCE_RUN_DIR=""
SOURCE_CHECKPOINT="best"
USE_CHECKPOINT_VIT_CONFIG="true"
USE_CHECKPOINT_SETUP_CONFIG="true"
TRAIN_ANCHOR="false"
ANCHOR_STEPS="0"
ANCHOR_LR="0.0"
ANCHOR_WEIGHT_DECAY="0.0"

DATA_DIR="./data/cifar10"
DOWNLOAD="true"
TRAIN_SUBSET="10000"
TEST_SUBSET="2000"
TRAIN_BATCH_SIZE="128"
EVAL_BATCH_SIZE="512"
NUM_WORKERS="4"

IMAGE_SIZE="32"
PATCH_SIZE="4"
IN_CHANNELS="3"
NUM_CLASSES="10"
HIDDEN_DIM="128"
DEPTH="3"
NUM_HEADS="4"
MLP_RATIO="2.0"
DROPOUT="0.0"
ATTENTION_DROPOUT="0.0"

BIG_VAE_CHECKPOINT=""
BIG_VAE_DECODE=""
BIG_VAE_TILE_T_PATCHES="0"
BIG_VAE_TILE_D_OUT="0"
BIG_VAE_LATENT_PARAMETERIZATION=""

EPSILONS="0.05,0.10"
RANDOM_DIRECTIONS="4"
ALPHA_MIN="1e-5"
ALPHA_MAX="10.0"
BRACKET_MULTIPLIER="1.8"
BINARY_SEARCH_STEPS="18"

DEVICE="${DEVICE:-auto}"
SEED="42"
STEPS="300"
RAW_LRS="1e-3,3e-4"
LATENT_LRS="1e-2,3e-3"
RAW_WEIGHT_DECAY="0.0"
LATENT_WEIGHT_DECAY="0.0"
ADAM_BETA1="0.9"
ADAM_BETA2="0.95"
ADAM_EPS="1e-9"
GRAD_CLIP_NORM="0.0"
AMP="true"
TF32="true"
LOG_EVERY_STEPS="10"
EVAL_EVERY_STEPS="25"
RECOVER_EPS="0.01"

COMET_ENABLED="false"
COMET_PROJECT_NAME="tinyvit_latent_h1"
COMET_EXPERIMENT_NAME=""
COMET_OFFLINE_DIRECTORY=""
COMET_LOG_CODE="false"

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

: "${SOURCE_RUN_DIR:?Edit SOURCE_RUN_DIR before running}"

run_python \
  "$SCRIPT_DIR/main.py" \
  "experiment.run_label=$RUN_LABEL" \
  "storage.root_dir=$OUTPUT_ROOT" \
  "source.vit_scaling_artifacts_root=$SOURCE_VIT_SCALING_ROOT" \
  "source.run_dir=$SOURCE_RUN_DIR" \
  "source.checkpoint_name=$SOURCE_CHECKPOINT" \
  "source.use_checkpoint_vit_config=$USE_CHECKPOINT_VIT_CONFIG" \
  "source.use_checkpoint_setup_config=$USE_CHECKPOINT_SETUP_CONFIG" \
  "source.train_anchor=$TRAIN_ANCHOR" \
  "source.anchor_steps=$ANCHOR_STEPS" \
  "source.anchor_lr=$ANCHOR_LR" \
  "source.anchor_weight_decay=$ANCHOR_WEIGHT_DECAY" \
  "data.data_dir=$DATA_DIR" \
  "data.download=$DOWNLOAD" \
  "data.train_subset=$TRAIN_SUBSET" \
  "data.test_subset=$TEST_SUBSET" \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.eval_batch_size=$EVAL_BATCH_SIZE" \
  "data.num_workers=$NUM_WORKERS" \
  "model.image_size=$IMAGE_SIZE" \
  "model.patch_size=$PATCH_SIZE" \
  "model.in_channels=$IN_CHANNELS" \
  "model.num_classes=$NUM_CLASSES" \
  "model.hidden_dim=$HIDDEN_DIM" \
  "model.depth=$DEPTH" \
  "model.num_heads=$NUM_HEADS" \
  "model.mlp_ratio=$MLP_RATIO" \
  "model.dropout=$DROPOUT" \
  "model.attention_dropout=$ATTENTION_DROPOUT" \
  "setup.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "setup.big_vae_decode=$BIG_VAE_DECODE" \
  "setup.big_vae_tile_T_patches=$BIG_VAE_TILE_T_PATCHES" \
  "setup.big_vae_tile_d_out=$BIG_VAE_TILE_D_OUT" \
  "setup.big_vae_latent_parameterization=$BIG_VAE_LATENT_PARAMETERIZATION" \
  "search.epsilons=[$EPSILONS]" \
  "search.random_directions=$RANDOM_DIRECTIONS" \
  "search.alpha_min=$ALPHA_MIN" \
  "search.alpha_max=$ALPHA_MAX" \
  "search.bracket_multiplier=$BRACKET_MULTIPLIER" \
  "search.binary_search_steps=$BINARY_SEARCH_STEPS" \
  "train.device=$DEVICE" \
  "train.seed=$SEED" \
  "train.steps=$STEPS" \
  "train.raw_lrs=[$RAW_LRS]" \
  "train.latent_lrs=[$LATENT_LRS]" \
  "train.raw_weight_decay=$RAW_WEIGHT_DECAY" \
  "train.latent_weight_decay=$LATENT_WEIGHT_DECAY" \
  "train.adam_beta1=$ADAM_BETA1" \
  "train.adam_beta2=$ADAM_BETA2" \
  "train.adam_eps=$ADAM_EPS" \
  "train.grad_clip_norm=$GRAD_CLIP_NORM" \
  "train.amp=$AMP" \
  "train.tf32=$TF32" \
  "train.log_every_steps=$LOG_EVERY_STEPS" \
  "train.eval_every_steps=$EVAL_EVERY_STEPS" \
  "train.recover_eps=$RECOVER_EPS" \
  "telemetry.comet.enabled=$COMET_ENABLED" \
  "telemetry.comet.project_name=$COMET_PROJECT_NAME" \
  "telemetry.comet.experiment_name=$COMET_EXPERIMENT_NAME" \
  "telemetry.comet.offline_directory=$COMET_OFFLINE_DIRECTORY" \
  "telemetry.comet.log_code=$COMET_LOG_CODE"
