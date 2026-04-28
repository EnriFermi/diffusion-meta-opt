#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="onerec"

RUN_LABEL="cifar10_50k_latent_prior"
PROFILE="cifar10_50k"
NOTES=""

STORAGE_ROOT="./post_train_research/vit_latent_scaling/artifacts"
SHARED_CHECKPOINT_ROOT=""
SHARED_CHECKPOINT_LABEL=""

SETUP_KIND="latent"                 # raw | latent
INIT_KIND="diffusion_prior"         # fresh | source | diffusion_prior
FRESH_LATENT_MODE="random"          # base | random
SOURCE_RUN_DIR=""                   # path to old run dir or run_id under artifacts/runs
SOURCE_CHECKPOINT="best"            # best | latest | final | step_000250 | /abs/path/to.ckpt.pt
SOURCE_PREFER_DIRECT_LATENT="true"  # exact latent restore when compatible

BIG_VAE_CHECKPOINT=""
BIG_VAE_DECODE="weights"            # weights | all
BIG_VAE_TILE_T_PATCHES="4"
BIG_VAE_TILE_D_OUT="64"
BIG_VAE_LATENT_PARAMETERIZATION="sphere"
BIG_VAE_LATENT_NOISE_STD="0.0"

DIFFUSION_PRIOR_CHECKPOINT=""
DIFFUSION_PRIOR_STEPS="50"
DIFFUSION_PRIOR_SAMPLER="ddim"
DIFFUSION_PRIOR_ETA="0.0"
CALIBRATION_BATCHES="1"

RANDOM_INIT_STD="0.02"

DATA_DIR=""
TRAIN_SUBSET="0"
TEST_SUBSET="0"
BATCH_SIZE="0"
EVAL_BATCH_SIZE="0"
NUM_WORKERS="4"
DOWNLOAD="true"

DEVICE="${DEVICE:-auto}"
SEED="42"
EPOCHS="1000"
MAX_STEPS="3000"
OPTIMIZER_NAME="AdamW"
LR="1e-3"
WEIGHT_DECAY="0.0"
ADAM_BETA1="0.9"
ADAM_BETA2="0.95"
ADAM_EPS="1e-9"
GRAD_CLIP_NORM="0.0"
LABEL_SMOOTHING="0.0"
AMP="true"
TF32="true"
COMPILE="false"

LATENT_LR_SCHEDULER="constant"      # constant | cosine_decay_to_floor
LATENT_LR_FLOOR_RATIO="0.1"
LATENT_LR_DECAY_STEPS="0"

LOG_EVERY_STEPS="50"
EVAL_EVERY_STEPS="150"
CHECKPOINT_EVERY_STEPS="250"
LATENT_DEBUG_METRICS="true"
LATENT_JACOBIAN_EPS="1e-3"
LATENT_JACOBIAN_PROBES="16"

COMET_ENABLED="false"
COMET_PROJECT_NAME="vit_latent_scaling"
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

if [ "$SETUP_KIND" = "latent" ] && [ -z "$BIG_VAE_CHECKPOINT" ]; then
  echo "Edit BIG_VAE_CHECKPOINT before running latent setup." >&2
  exit 1
fi

if [ "$INIT_KIND" = "source" ] && [ -z "$SOURCE_RUN_DIR" ]; then
  echo "Edit SOURCE_RUN_DIR before running init.kind=source." >&2
  exit 1
fi

if [ "$INIT_KIND" = "diffusion_prior" ] && [ -z "$DIFFUSION_PRIOR_CHECKPOINT" ]; then
  echo "Edit DIFFUSION_PRIOR_CHECKPOINT before running init.kind=diffusion_prior." >&2
  exit 1
fi

run_python \
  "$SCRIPT_DIR/main.py" \
  "experiment.run_label=$RUN_LABEL" \
  "experiment.profile=$PROFILE" \
  "experiment.notes=$NOTES" \
  "storage.root_dir=$STORAGE_ROOT" \
  "storage.shared_checkpoint_root_dir=$SHARED_CHECKPOINT_ROOT" \
  "storage.shared_checkpoint_label=$SHARED_CHECKPOINT_LABEL" \
  "storage.checkpoint_every_steps=$CHECKPOINT_EVERY_STEPS" \
  "setup.kind=$SETUP_KIND" \
  "setup.big_vae_checkpoint=$BIG_VAE_CHECKPOINT" \
  "setup.big_vae_decode=$BIG_VAE_DECODE" \
  "setup.big_vae_tile_T_patches=$BIG_VAE_TILE_T_PATCHES" \
  "setup.big_vae_tile_d_out=$BIG_VAE_TILE_D_OUT" \
  "setup.big_vae_latent_parameterization=$BIG_VAE_LATENT_PARAMETERIZATION" \
  "setup.big_vae_latent_noise_std=$BIG_VAE_LATENT_NOISE_STD" \
  "init.kind=$INIT_KIND" \
  "init.fresh_latent_mode=$FRESH_LATENT_MODE" \
  "init.random_init_std=$RANDOM_INIT_STD" \
  "init.source_run_dir=$SOURCE_RUN_DIR" \
  "init.source_checkpoint=$SOURCE_CHECKPOINT" \
  "init.source_prefer_direct_latent=$SOURCE_PREFER_DIRECT_LATENT" \
  "init.diffusion_prior_checkpoint=$DIFFUSION_PRIOR_CHECKPOINT" \
  "init.diffusion_prior_steps=$DIFFUSION_PRIOR_STEPS" \
  "init.diffusion_prior_sampler=$DIFFUSION_PRIOR_SAMPLER" \
  "init.diffusion_prior_eta=$DIFFUSION_PRIOR_ETA" \
  "init.calibration_batches=$CALIBRATION_BATCHES" \
  "data.data_dir=$DATA_DIR" \
  "data.train_subset=$TRAIN_SUBSET" \
  "data.test_subset=$TEST_SUBSET" \
  "data.batch_size=$BATCH_SIZE" \
  "data.eval_batch_size=$EVAL_BATCH_SIZE" \
  "data.num_workers=$NUM_WORKERS" \
  "data.download=$DOWNLOAD" \
  "train.device=$DEVICE" \
  "train.seed=$SEED" \
  "train.epochs=$EPOCHS" \
  "train.max_steps=$MAX_STEPS" \
  "train.optimizer_name=$OPTIMIZER_NAME" \
  "train.lr=$LR" \
  "train.weight_decay=$WEIGHT_DECAY" \
  "train.adam_beta1=$ADAM_BETA1" \
  "train.adam_beta2=$ADAM_BETA2" \
  "train.adam_eps=$ADAM_EPS" \
  "train.grad_clip_norm=$GRAD_CLIP_NORM" \
  "train.label_smoothing=$LABEL_SMOOTHING" \
  "train.amp=$AMP" \
  "train.tf32=$TF32" \
  "train.compile=$COMPILE" \
  "train.latent_lr_scheduler=$LATENT_LR_SCHEDULER" \
  "train.latent_lr_floor_ratio=$LATENT_LR_FLOOR_RATIO" \
  "train.latent_lr_decay_steps=$LATENT_LR_DECAY_STEPS" \
  "logging.log_every_steps=$LOG_EVERY_STEPS" \
  "logging.eval_every_steps=$EVAL_EVERY_STEPS" \
  "logging.latent_debug_metrics=$LATENT_DEBUG_METRICS" \
  "logging.latent_jacobian_eps=$LATENT_JACOBIAN_EPS" \
  "logging.latent_jacobian_probes=$LATENT_JACOBIAN_PROBES" \
  "telemetry.comet.enabled=$COMET_ENABLED" \
  "telemetry.comet.project_name=$COMET_PROJECT_NAME" \
  "telemetry.comet.experiment_name=$COMET_EXPERIMENT_NAME" \
  "telemetry.comet.offline_directory=$COMET_OFFLINE_DIRECTORY" \
  "telemetry.comet.log_code=$COMET_LOG_CODE"
