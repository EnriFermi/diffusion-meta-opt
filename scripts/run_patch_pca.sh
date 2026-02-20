#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(
    deit_base
    dinov2_base
    swinv2_base
    donut_base
    grounding_dino_tiny
  )
fi

python -m experiments.patch_pca_analysis \
  --models "${MODELS[@]}" \
  --config-name "${PCA_CONFIG_NAME:-mini_vae_train}" \
  --output-dir "${PCA_OUTPUT_DIR:-data/reports/pca_patches}" \
  --patch-size "${PCA_PATCH_SIZE:-64}" \
  --patches-per-sample "${PCA_PATCHES_PER_SAMPLE:-16}" \
  --components "${PCA_COMPONENTS:-8}" \
  --max-samples-per-model "${PCA_MAX_SAMPLES_PER_MODEL:-64}" \
  --max-total-samples "${PCA_MAX_TOTAL_SAMPLES:-1024}" \
  --max-patches-per-group "${PCA_MAX_PATCHES_PER_GROUP:-8192}" \
  --time-limit-seconds "${PCA_TIME_LIMIT_SECONDS:-1800}" \
  --seed "${PCA_SEED:-42}" \
  --streaming-mode "${PCA_STREAMING_MODE:-none}" \
  --collector-mode "${PCA_COLLECTOR_MODE:-async}" \
  ${PCA_COLLECTOR_DEVICE:+--collector-device "${PCA_COLLECTOR_DEVICE}"} \
  ${PCA_DATASETS:+--datasets ${PCA_DATASETS}} \
  ${PCA_PREDOWNLOAD_MODELS:+--predownload-models}

