#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/mom.env" ]; then
  set -a
  . "$PROJECT_ROOT/mom.env"
  set +a
fi

MODE=${1:-all}

OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/artifacts}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}
MNIST_DATA_DIR=${MNIST_DATA_DIR:-$DATA_ROOT/mnist}
CIFAR10_DATA_DIR=${CIFAR10_DATA_DIR:-$DATA_ROOT/cifar10}
IMAGENET_DATA_DIR=${IMAGENET_DATA_DIR:-$DATA_ROOT/imagenet}

CONDA_ENV_NAME=${CONDA_ENV_NAME:-onerec}
DEVICE=${DEVICE:-auto}
SEED=${SEED:-42}
LR_GRID=${LR_GRID:-"1e-4 1e-3 1e-2 1e-1 1e0"}
RAW_INIT_LR=${RAW_INIT_LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
LATENT_WEIGHT_DECAY=${LATENT_WEIGHT_DECAY:-0.0}
BIG_VAE_DECODE=${BIG_VAE_DECODE:-all}
BIG_VAE_LATENT_INIT=${BIG_VAE_LATENT_INIT:-encoded}
BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT=${BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT:-}
BIG_VAE_DIFFUSION_PRIOR_STEPS=${BIG_VAE_DIFFUSION_PRIOR_STEPS:-50}
BIG_VAE_DIFFUSION_PRIOR_SAMPLER=${BIG_VAE_DIFFUSION_PRIOR_SAMPLER:-ddim}
BIG_VAE_DIFFUSION_PRIOR_ETA=${BIG_VAE_DIFFUSION_PRIOR_ETA:-0.0}
BIG_VAE_TILE_T_PATCHES=${BIG_VAE_TILE_T_PATCHES:-16}
BIG_VAE_TILE_D_OUT=${BIG_VAE_TILE_D_OUT:-8}
BIG_VAE_ENCODER_CONTEXT_ROWS=${BIG_VAE_ENCODER_CONTEXT_ROWS:-64}
BIG_VAE_ENCODER_CONTEXT_STD=${BIG_VAE_ENCODER_CONTEXT_STD:-1.0}
BIG_VAE_ENCODER_BATCH_SIZE=${BIG_VAE_ENCODER_BATCH_SIZE:-16}
FORCE=${FORCE:-false}

export PYTHONUNBUFFERED=1

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

  if command -v pipenv >/dev/null 2>&1; then
    pipenv run python "$@"
    return
  fi

  python "$@"
}

lr_tag() {
  printf '%s' "$1" | sed 's/+//g; s/-/m/g; s/\./p/g'
}

require_big_vae() {
  : "${BIG_VAE_CHECKPOINT:?Set BIG_VAE_CHECKPOINT=/path/to/stage_N/latest.pt before latent modes}"
}

require_big_vae_diffusion_prior() {
  if [ "$BIG_VAE_LATENT_INIT" = "diffusion_prior" ]; then
    : "${BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT:?Set BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT=/path/to/prior.pt when BIG_VAE_LATENT_INIT=diffusion_prior}"
  fi
}

latent_mode_tag() {
  if [ "$BIG_VAE_LATENT_INIT" = "encoded" ]; then
    printf '%s' "latent"
    return
  fi
  printf 'latent_%s' "$BIG_VAE_LATENT_INIT"
}

run_one() {
  RUN_DATASET=$1
  RUN_SIZE=$2
  RUN_SETUP=$3
  RUN_LR=$4
  RUN_OUT=$5
  RUN_DATA_DIR=$6
  RUN_EPOCHS=$7
  RUN_MAX_STEPS=$8
  RUN_BATCH_SIZE=$9
  shift 9
  RUN_EVAL_BATCH_SIZE=$1
  RUN_NUM_WORKERS=$2
  RUN_EVAL_EVERY=$3
  RUN_IMAGE_SIZE=$4
  RUN_PATCH_SIZE=$5
  RUN_IN_CHANNELS=$6
  RUN_NUM_CLASSES=$7
  RUN_HIDDEN_DIM=$8
  RUN_DEPTH=$9
  shift 9
  RUN_NUM_HEADS=$1
  RUN_MLP_RATIO=$2
  RUN_RAW_CHECKPOINT=$3

  if [ -f "$RUN_OUT/summary.json" ] && [ "$FORCE" != "true" ]; then
    echo "skip existing: $RUN_OUT"
    return
  fi

  if [ "$RUN_SETUP" = "latent" ]; then
    require_big_vae
    require_big_vae_diffusion_prior
  fi

  set -- "$SCRIPT_DIR/run_vit_latent_scaling.py" \
    --output-dir "$RUN_OUT" \
    --dataset "$RUN_DATASET" \
    --model-size "$RUN_SIZE" \
    --setup "$RUN_SETUP" \
    --data-dir "$RUN_DATA_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --epochs "$RUN_EPOCHS" \
    --max-steps "$RUN_MAX_STEPS" \
    --batch-size "$RUN_BATCH_SIZE" \
    --eval-batch-size "$RUN_EVAL_BATCH_SIZE" \
    --num-workers "$RUN_NUM_WORKERS" \
    --lr "$RUN_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --latent-weight-decay "$LATENT_WEIGHT_DECAY" \
    --log-every-steps 50 \
    --eval-every-steps "$RUN_EVAL_EVERY" \
    --raw-checkpoint "$RUN_RAW_CHECKPOINT" \
    --big-vae-latent-init "$BIG_VAE_LATENT_INIT" \
    --big-vae-decode "$BIG_VAE_DECODE" \
    --big-vae-tile-T-patches "$BIG_VAE_TILE_T_PATCHES" \
    --big-vae-tile-d-out "$BIG_VAE_TILE_D_OUT" \
    --big-vae-encoder-context-rows "$BIG_VAE_ENCODER_CONTEXT_ROWS" \
    --big-vae-encoder-context-std "$BIG_VAE_ENCODER_CONTEXT_STD" \
    --big-vae-encoder-batch-size "$BIG_VAE_ENCODER_BATCH_SIZE" \
    --image-size "$RUN_IMAGE_SIZE" \
    --patch-size "$RUN_PATCH_SIZE" \
    --in-channels "$RUN_IN_CHANNELS" \
    --num-classes "$RUN_NUM_CLASSES" \
    --hidden-dim "$RUN_HIDDEN_DIM" \
    --depth "$RUN_DEPTH" \
    --num-heads "$RUN_NUM_HEADS" \
    --mlp-ratio "$RUN_MLP_RATIO"

  if [ "$RUN_SETUP" = "latent" ]; then
    set -- "$@" --big-vae-checkpoint "$BIG_VAE_CHECKPOINT"
    if [ "$BIG_VAE_LATENT_INIT" = "diffusion_prior" ]; then
      set -- "$@" \
        --big-vae-diffusion-prior-checkpoint "$BIG_VAE_DIFFUSION_PRIOR_CHECKPOINT" \
        --big-vae-diffusion-prior-steps "$BIG_VAE_DIFFUSION_PRIOR_STEPS" \
        --big-vae-diffusion-prior-sampler "$BIG_VAE_DIFFUSION_PRIOR_SAMPLER" \
        --big-vae-diffusion-prior-eta "$BIG_VAE_DIFFUSION_PRIOR_ETA"
    fi
  fi

  run_python "$@"
}

raw_checkpoint_path() {
  RUN_DATASET=$1
  RUN_SIZE=$2
  RUN_TAG=$(lr_tag "$RAW_INIT_LR")
  printf '%s/%s/%s/raw_lr_%s/checkpoints/raw_final.pt' "$OUTPUT_ROOT" "$RUN_DATASET" "$RUN_SIZE" "$RUN_TAG"
}

run_config_raw() {
  CFG_DATASET=$1
  CFG_SIZE=$2
  CFG_DATA_DIR=$3
  CFG_EPOCHS=$4
  CFG_MAX_STEPS=$5
  CFG_BATCH_SIZE=$6
  CFG_EVAL_BATCH_SIZE=$7
  CFG_NUM_WORKERS=$8
  CFG_EVAL_EVERY=$9
  shift 9
  CFG_IMAGE_SIZE=$1
  CFG_PATCH_SIZE=$2
  CFG_IN_CHANNELS=$3
  CFG_NUM_CLASSES=$4
  CFG_HIDDEN_DIM=$5
  CFG_DEPTH=$6
  CFG_NUM_HEADS=$7
  CFG_MLP_RATIO=$8

  for LR in $LR_GRID; do
    TAG=$(lr_tag "$LR")
    OUT="$OUTPUT_ROOT/$CFG_DATASET/$CFG_SIZE/raw_lr_$TAG"
    run_one "$CFG_DATASET" "$CFG_SIZE" raw "$LR" "$OUT" "$CFG_DATA_DIR" "$CFG_EPOCHS" "$CFG_MAX_STEPS" \
      "$CFG_BATCH_SIZE" "$CFG_EVAL_BATCH_SIZE" "$CFG_NUM_WORKERS" "$CFG_EVAL_EVERY" \
      "$CFG_IMAGE_SIZE" "$CFG_PATCH_SIZE" "$CFG_IN_CHANNELS" "$CFG_NUM_CLASSES" \
      "$CFG_HIDDEN_DIM" "$CFG_DEPTH" "$CFG_NUM_HEADS" "$CFG_MLP_RATIO" ""
  done
}

run_config_latent() {
  CFG_DATASET=$1
  CFG_SIZE=$2
  CFG_DATA_DIR=$3
  CFG_EPOCHS=$4
  CFG_MAX_STEPS=$5
  CFG_BATCH_SIZE=$6
  CFG_EVAL_BATCH_SIZE=$7
  CFG_NUM_WORKERS=$8
  CFG_EVAL_EVERY=$9
  shift 9
  CFG_IMAGE_SIZE=$1
  CFG_PATCH_SIZE=$2
  CFG_IN_CHANNELS=$3
  CFG_NUM_CLASSES=$4
  CFG_HIDDEN_DIM=$5
  CFG_DEPTH=$6
  CFG_NUM_HEADS=$7
  CFG_MLP_RATIO=$8

  RAW_CKPT=$(raw_checkpoint_path "$CFG_DATASET" "$CFG_SIZE")
  if [ ! -f "$RAW_CKPT" ]; then
    RAW_TAG=$(lr_tag "$RAW_INIT_LR")
    RAW_OUT="$OUTPUT_ROOT/$CFG_DATASET/$CFG_SIZE/raw_lr_$RAW_TAG"
    run_one "$CFG_DATASET" "$CFG_SIZE" raw "$RAW_INIT_LR" "$RAW_OUT" "$CFG_DATA_DIR" "$CFG_EPOCHS" "$CFG_MAX_STEPS" \
      "$CFG_BATCH_SIZE" "$CFG_EVAL_BATCH_SIZE" "$CFG_NUM_WORKERS" "$CFG_EVAL_EVERY" \
      "$CFG_IMAGE_SIZE" "$CFG_PATCH_SIZE" "$CFG_IN_CHANNELS" "$CFG_NUM_CLASSES" \
      "$CFG_HIDDEN_DIM" "$CFG_DEPTH" "$CFG_NUM_HEADS" "$CFG_MLP_RATIO" ""
  fi

  for LR in $LR_GRID; do
    TAG=$(lr_tag "$LR")
    MODE_TAG=$(latent_mode_tag)
    OUT="$OUTPUT_ROOT/$CFG_DATASET/$CFG_SIZE/${MODE_TAG}_lr_$TAG"
    run_one "$CFG_DATASET" "$CFG_SIZE" latent "$LR" "$OUT" "$CFG_DATA_DIR" "$CFG_EPOCHS" "$CFG_MAX_STEPS" \
      "$CFG_BATCH_SIZE" "$CFG_EVAL_BATCH_SIZE" "$CFG_NUM_WORKERS" "$CFG_EVAL_EVERY" \
      "$CFG_IMAGE_SIZE" "$CFG_PATCH_SIZE" "$CFG_IN_CHANNELS" "$CFG_NUM_CLASSES" \
      "$CFG_HIDDEN_DIM" "$CFG_DEPTH" "$CFG_NUM_HEADS" "$CFG_MLP_RATIO" "$RAW_CKPT"
  done
}

run_config_pipeline() {
  run_config_raw "$@"
  run_config_latent "$@"
}

run_named_config() {
  NAME=$1
  PART=$2
  case "$NAME" in
    mnist_tiny)
      run_config_"$PART" mnist tiny "$MNIST_DATA_DIR" 20 2000 256 512 4 100 28 4 1 10 64 4 4 2.0
      ;;
    mnist_small)
      run_config_"$PART" mnist small "$MNIST_DATA_DIR" 20 3000 256 512 4 150 28 4 1 10 128 6 4 4.0
      ;;
    mnist_medium)
      run_config_"$PART" mnist medium "$MNIST_DATA_DIR" 20 4000 256 512 4 200 28 4 1 10 192 8 6 4.0
      ;;
    cifar10_tiny)
      run_config_"$PART" cifar10 tiny "$CIFAR10_DATA_DIR" 30 5000 128 256 4 250 32 4 3 10 192 6 3 4.0
      ;;
    cifar10_small)
      run_config_"$PART" cifar10 small "$CIFAR10_DATA_DIR" 30 7500 128 256 4 375 32 4 3 10 256 8 4 4.0
      ;;
    cifar10_medium)
      run_config_"$PART" cifar10 medium "$CIFAR10_DATA_DIR" 30 10000 128 256 4 500 32 4 3 10 384 12 6 4.0
      ;;
    imagenet_tiny)
      run_config_"$PART" imagenet tiny "$IMAGENET_DATA_DIR" 90 20000 128 256 8 1000 224 16 3 1000 192 12 3 4.0
      ;;
    imagenet_small)
      run_config_"$PART" imagenet small "$IMAGENET_DATA_DIR" 90 30000 96 192 8 1500 224 16 3 1000 384 12 6 4.0
      ;;
    imagenet_base)
      run_config_"$PART" imagenet base "$IMAGENET_DATA_DIR" 90 50000 64 128 8 2500 224 16 3 1000 768 12 12 4.0
      ;;
    *)
      echo "Unknown config: $NAME" >&2
      exit 2
      ;;
  esac
}

run_mnist() {
  PART=$1
  run_named_config mnist_tiny "$PART"
  run_named_config mnist_small "$PART"
  run_named_config mnist_medium "$PART"
}

run_cifar10() {
  PART=$1
  run_named_config cifar10_tiny "$PART"
  run_named_config cifar10_small "$PART"
  run_named_config cifar10_medium "$PART"
}

run_imagenet() {
  PART=$1
  run_named_config imagenet_tiny "$PART"
  run_named_config imagenet_small "$PART"
  run_named_config imagenet_base "$PART"
}

case "$MODE" in
  all)
    run_mnist pipeline
    run_cifar10 pipeline
    run_imagenet pipeline
    ;;
  mnist)
    run_mnist pipeline
    ;;
  cifar10)
    run_cifar10 pipeline
    ;;
  imagenet)
    run_imagenet pipeline
    ;;
  mnist_raw)
    run_mnist raw
    ;;
  cifar10_raw)
    run_cifar10 raw
    ;;
  imagenet_raw)
    run_imagenet raw
    ;;
  mnist_latent)
    run_mnist latent
    ;;
  cifar10_latent)
    run_cifar10 latent
    ;;
  imagenet_latent)
    run_imagenet latent
    ;;
  *_raw)
    CONFIG_NAME=${MODE%_raw}
    run_named_config "$CONFIG_NAME" raw
    ;;
  *_latent)
    CONFIG_NAME=${MODE%_latent}
    run_named_config "$CONFIG_NAME" latent
    ;;
  *_pipeline)
    CONFIG_NAME=${MODE%_pipeline}
    run_named_config "$CONFIG_NAME" pipeline
    ;;
  *)
    run_named_config "$MODE" pipeline
    ;;
esac
