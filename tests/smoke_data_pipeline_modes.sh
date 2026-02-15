#!/usr/bin/env bash
set -euo pipefail

# Запускайте из корня репозитория.
# В этом файле 3 готовых smoke-вызова:
# 1) none (in-memory)
# 2) local_disk
# 3) s3_bridge
#
# По умолчанию активен только (1).
# Чтобы протестировать другой режим: раскомментируйте нужный блок,
# а остальные оставьте закомментированными.
#
# Скрипт python дополнительно выполняет turnover-probe:
# после базового smoke-прохода потребляет ещё объекты и проверяет,
# что ready-чанки действительно обновляются (как у нормального streaming dataset).

# Hugging Face token: задайте через переменную окружения перед запуском:
#   export HF_TOKEN="hf_..."
# или передайте inline:
#   HF_TOKEN="hf_..." bash tests/smoke_data_pipeline_modes.sh
: "${HF_TOKEN:?Set HF_TOKEN before running this script (export HF_TOKEN=hf_...)}"

SCRIPT="tests/smoke_data_pipeline_modes.py"
COMMON_ARGS=(
  --data-profile "all_datasets_no_flickr30k"
  --data-root "./data"
  --predownload
  --chunk-size-samples 8
  --raw-chunk-size-images 16
  --raw-num-chunks-kept 2
  --xy-samples-random-slice 32
  --target-samples 20
  --turnover-probe-samples 24
  --turnover-probe-timeout-seconds 240
  --timeout-seconds 300
  --hf-token "$HF_TOKEN"
)

# -----------------------------------------------------------------------------
# (1) MODE=none (in-memory) [АКТИВЕН ПО УМОЛЧАНИЮ]
# -----------------------------------------------------------------------------
# pipenv run python "$SCRIPT" \
#   --mode none \
#   --train-device "cuda:0" \
#   --collector-device "null" \
#   --collector-mode "auto" \
#   "${COMMON_ARGS[@]}"

# -----------------------------------------------------------------------------
# (2) MODE=local_disk (chunk streaming на локальном диске)
# -----------------------------------------------------------------------------
python "$SCRIPT" \
  --mode local_disk \
  --train-device "cuda:0" \
  --collector-device "cuda:1" \
  --collector-mode "auto" \
  --chunk-size-samples 64 \
  --local-ready-store-max-chunks 40 \
  --local-refill-after-consumed-chunks 20 \
  "${COMMON_ARGS[@]}"

# -----------------------------------------------------------------------------
# (3) MODE=s3_bridge (producer/consumer через S3)
# -----------------------------------------------------------------------------
# Перед запуском выставьте креды окружения, например:
# export AWS_ACCESS_KEY_ID="..."
# export AWS_SECRET_ACCESS_KEY="..."
# export AWS_DEFAULT_REGION="us-east-1"
#
# Для S3-compatible (MinIO и т.п.) можно добавить endpoint:
# --s3-endpoint-url "http://127.0.0.1:9000"
#
# pipenv run python "$SCRIPT" \
#   --mode s3_bridge \
#   --train-device "cuda:0" \
#   --collector-device "cuda:1" \
#   --collector-mode "auto" \
#   --s3-bucket "YOUR_BUCKET" \
#   --s3-prefix "diffusion-meta-opt/streaming-smoke" \
#   --s3-region "us-east-1" \
#   --s3-max-remote-chunks 100 \
#   --delete-remote-after consume \
#   "${COMMON_ARGS[@]}"
