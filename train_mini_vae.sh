#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Примеры:
#   ./train_mini_vae.sh
#   HF_TOKEN=hf_xxx ./train_mini_vae.sh mini_train.device=cuda:0
#   ./train_mini_vae.sh mini_train.max_steps=5000 mini_train.patches_per_sample=24
#
# Entrypoint: python -m experiments.train_mini_vae
# Корневой конфиг: conf/mini_vae_train.yaml (run_profiles/train_mini_vae).
# Все аргументы передаются как обычные Hydra-overrides.

if [ -n "${PIPENV_ACTIVE:-}" ]; then
  exec python -m experiments.train_mini_vae "$@"
fi

if command -v pipenv >/dev/null 2>&1; then
  exec pipenv run python -m experiments.train_mini_vae "$@"
fi

exec python -m experiments.train_mini_vae "$@"
