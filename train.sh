#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Примеры:
#   ./train.sh
#   HF_TOKEN=hf_xxx ./train.sh train.device=cuda:0 collector.device=cuda:1
#   ./train.sh train.max_steps=2000 streaming.mode=local_disk
#
# Entrypoint: python -m experiments.train_big_vae
# Корневой конфиг: conf/config.yaml (run_profiles/train_big_vae).
# Все аргументы передаются как обычные Hydra-overrides.

HYDRA_ARGS=("$@")

if [[ -n "${PIPENV_ACTIVE:-}" ]]; then
  python -m experiments.train_big_vae "${HYDRA_ARGS[@]}"
  exit 0
fi

if command -v pipenv >/dev/null 2>&1; then
  pipenv run python -m experiments.train_big_vae "${HYDRA_ARGS[@]}"
  exit 0
fi

python -m experiments.train_big_vae "${HYDRA_ARGS[@]}"
