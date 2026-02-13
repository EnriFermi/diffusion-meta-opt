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

# Hugging Face token: задайте через переменную окружения перед запуском:
#   export HF_TOKEN="hf_..."
# или передайте inline:
#   HF_TOKEN="hf_..." bash tests/smoke_data_pipeline_modes.sh
: "${HF_TOKEN:?Set HF_TOKEN before running this script (export HF_TOKEN=hf_...)}"

SCRIPT="tests/smoke_data_pipeline_modes.py"
COMMON_ARGS=(
  --data-root "./data"
  --datasets "coco2017,scene_parse_150"
  --dataset-model "coco2017=clip_vit_b32"
  --dataset-model "scene_parse_150=clip_vit_b32"
  --predownload
  --target-samples 20
  --timeout-seconds 300
  --hf-token "$HF_TOKEN"
)

# -----------------------------------------------------------------------------
# (1) MODE=none (in-memory) [АКТИВЕН ПО УМОЛЧАНИЮ]
# -----------------------------------------------------------------------------
# python "$SCRIPT" \
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
  --local-max-ready-chunks 40 \
  --local-low-watermark-chunks 20 \
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
# python "$SCRIPT" \
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
