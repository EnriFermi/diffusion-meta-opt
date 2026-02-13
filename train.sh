#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Пример (с указанием data-конфига обучения):
#   HF_TOKEN=hf_xxx ./train.sh test_dataset \
#     streaming.mode=local_disk \
#     collector.mode=auto \
#     train.device=cuda:0 \
#     collector.device=cuda:1
#
# Первый позиционный аргумент (если не override вида key=value) трактуется
# как имя data-конфига из conf/data/<name>.yaml.
# Можно также передать через env:
#   TRAIN_DATA_CONFIG=test_dataset ./train.sh ...

DATA_CONFIG="${TRAIN_DATA_CONFIG:-test_dataset}"
ARGS=("$@")

if [[ ${#ARGS[@]} -gt 0 ]]; then
  FIRST_ARG="${ARGS[0]}"
  if [[ "$FIRST_ARG" != -* && "$FIRST_ARG" != *=* ]]; then
    DATA_CONFIG="$FIRST_ARG"
    ARGS=("${ARGS[@]:1}")
  fi
fi

HAS_DATA_OVERRIDE=0
for ARG in "${ARGS[@]}"; do
  if [[ "$ARG" == data=* ]]; then
    HAS_DATA_OVERRIDE=1
    break
  fi
done

if [[ "$HAS_DATA_OVERRIDE" -eq 1 ]]; then
  HYDRA_ARGS=("${ARGS[@]}")
  echo "[train.sh] Using explicit data override from args"
else
  HYDRA_ARGS=("data=${DATA_CONFIG}" "${ARGS[@]}")
  echo "[train.sh] Data config: ${DATA_CONFIG}"
fi

if [[ -n "${PIPENV_ACTIVE:-}" ]]; then
  python train.py "${HYDRA_ARGS[@]}"
  exit 0
fi

if command -v pipenv >/dev/null 2>&1; then
  pipenv run python train.py "${HYDRA_ARGS[@]}"
  exit 0
fi

python train.py "${HYDRA_ARGS[@]}"
