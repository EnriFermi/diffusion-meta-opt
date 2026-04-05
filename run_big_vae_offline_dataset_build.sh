#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Примеры:
#   ./run_big_vae_offline_dataset_build.sh \
#     train.offline_dataset.root_dir=./artifacts/training/checkpoints/weight_quantile_vae/stage_1/offline_dataset \
#     train.offline_dataset.builder.target_size_gb=300 \
#     train.offline_dataset.builder.overwrite_existing=true
#
#   ./run_big_vae_offline_dataset_build.sh \
#     train.offline_dataset.root_dir=./artifacts/training/checkpoints/weight_quantile_vae/stage_1/offline_dataset_500gb \
#     train.offline_dataset.builder.target_size_gb=500 \
#     train.offline_dataset.builder.max_samples_per_source=512 \
#     train.offline_dataset.builder.overwrite_existing=true
#
# Entrypoint: python -m experiments.build_big_vae_offline_dataset
# Корневой конфиг: conf/config.yaml (run_profiles/train_big_vae).
# Все аргументы передаются как обычные Hydra-overrides.

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m experiments.build_big_vae_offline_dataset "$@"
fi

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m experiments.build_big_vae_offline_dataset "$@"
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-onerec}"
  exec conda run -n "$CONDA_ENV_NAME" python -m experiments.build_big_vae_offline_dataset "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m experiments.build_big_vae_offline_dataset "$@"
fi

exec python -m experiments.build_big_vae_offline_dataset "$@"
