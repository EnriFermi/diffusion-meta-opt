#!/bin/sh
set -eu

# Запускайте из корня репозитория.
# В этом файле 3 готовых smoke-вызова:
# 1) none (in-memory)
# 2) local_disk
# 3) s3_bridge
#
# По умолчанию активен блок (2) с deterministic pair-audit.
# Чтобы протестировать другой режим: раскомментируйте нужный блок,
# а остальные оставьте закомментированными.
#
# В pair-audit режиме скрипт делает детерминированный one-shot по всем
# совместимым парам dataset+model и пишет подробный JSON-отчет.

# Hugging Face token: задайте через переменную окружения перед запуском:
export HF_TOKEN="hf_..."
# или передайте inline:
#   HF_TOKEN="hf_..." bash tests/smoke_data_pipeline_modes.sh
: "${HF_TOKEN:?Set HF_TOKEN before running this script (export HF_TOKEN=hf_...)}"

SCRIPT="tests/smoke_data_pipeline_modes.py"

# -----------------------------------------------------------------------------
# (1) MODE=none (in-memory)
# -----------------------------------------------------------------------------
# pipenv run python "$SCRIPT" \
#   --mode none \
#   --train-device "cuda:0" \
#   --collector-device "null" \
#   --collector-mode "auto" \
#   --data-profile "kaggle_smoke" \
#   --datasets "cord_v2,docvqa_1200,oxford_pets,stanford_cars" \
#   --data-root "./data" \
#   --chunk-size-samples 4 \
#   --raw-chunk-size-images 8 \
#   --raw-num-chunks-kept 2 \
#   --xy-samples-random-slice 16 \
#   --target-samples 8 \
#   --turnover-probe-samples 8 \
#   --turnover-probe-timeout-seconds 120 \
#   --timeout-seconds 180 \
#   --dataset-model cord_v2=clip_vit_b32 \
#   --dataset-model docvqa_1200=clip_vit_b32 \
#   --dataset-model oxford_pets=clip_vit_b32 \
#   --dataset-model stanford_cars=clip_vit_b32 \
#   --hf-token "$HF_TOKEN"

# -----------------------------------------------------------------------------
# (2) MODE=local_disk + deterministic pair-audit [АКТИВЕН ПО УМОЛЧАНИЮ]
# -----------------------------------------------------------------------------
# Проверяет ВСЕ совместимые пары dataset+model из выбранного data-profile.
# Для каждой пары в отчете будет ровно одна запись.
# Общий hard-time budget: 10 минут.
exec python "$SCRIPT" \
  --mode local_disk \
  --train-device "cuda:0" \
  --collector-device "cuda:1" \
  --collector-mode "auto" \
  --pair-audit \
  --pair-audit-max-seconds 600 \
  --pair-audit-sample-timeout-seconds 8 \
  --pair-audit-report-path "./data/reports/pair_audit_all_pairs.json" \
  --data-profile "all_datasets_no_flickr30k" \
  --data-root "./data" \
  --chunk-size-samples 8 \
  --local-ready-store-max-chunks 48 \
  --local-refill-after-consumed-chunks 24 \
  --raw-chunk-size-images 16 \
  --raw-num-chunks-kept 2 \
  --xy-samples-random-slice 64 \
  --timeout-seconds 600 \
  --hf-token "$HF_TOKEN"

# -----------------------------------------------------------------------------
# (2b) MODE=local_disk, ПОЛНОЕ покрытие всех допустимых пар dataset+model
# -----------------------------------------------------------------------------
# ВАЖНО:
# - Не задаём --datasets (берём весь профиль)
# - Не задаём --dataset-model (не сужаем модели вручную)
# - Используем более щедрые таймауты, т.к. комбинаций существенно больше
#
# pipenv run python "$SCRIPT" \
#   --mode local_disk \
#   --train-device "cuda:0" \
#   --collector-device "cuda:1" \
#   --collector-mode "auto" \
#   --data-profile "all_datasets_no_flickr30k" \
#   --data-root "./data" \
#   --chunk-size-samples 8 \
#   --local-ready-store-max-chunks 48 \
#   --local-refill-after-consumed-chunks 24 \
#   --raw-chunk-size-images 16 \
#   --raw-num-chunks-kept 2 \
#   --xy-samples-random-slice 64 \
#   --target-samples 256 \
#   --turnover-probe-samples 64 \
#   --turnover-probe-timeout-seconds 600 \
#   --require-all-dataset-model-pairs \
#   --pair-coverage-timeout-seconds 5400 \
#   --timeout-seconds 1800 \
#   --hf-token "$HF_TOKEN"

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
#   --data-profile "kaggle_smoke" \
#   --datasets "cord_v2,docvqa_1200,oxford_pets,stanford_cars" \
#   --data-root "./data" \
#   --chunk-size-samples 4 \
#   --raw-chunk-size-images 8 \
#   --raw-num-chunks-kept 2 \
#   --xy-samples-random-slice 16 \
#   --target-samples 8 \
#   --turnover-probe-samples 8 \
#   --turnover-probe-timeout-seconds 120 \
#   --timeout-seconds 180 \
#   --dataset-model cord_v2=clip_vit_b32 \
#   --dataset-model docvqa_1200=clip_vit_b32 \
#   --dataset-model oxford_pets=clip_vit_b32 \
#   --dataset-model stanford_cars=clip_vit_b32 \
#   --hf-token "$HF_TOKEN"
