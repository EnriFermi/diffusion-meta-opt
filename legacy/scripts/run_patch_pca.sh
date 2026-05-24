#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -gt 0 ]; then
  MODELS="$*"
else
  MODELS="deit_base dinov2_base swinv2_base donut_base grounding_dino_tiny"
fi

set -- \
  python -m legacy.analysis.patch_pca_analysis \
  --config-name "${PCA_CONFIG_NAME:-mini_vae_train}" \
  --output-dir "${PCA_OUTPUT_DIR:-data/reports/pca_patches}" \
  --patch-size "${PCA_PATCH_SIZE:-64}" \
  --patches-per-sample "${PCA_PATCHES_PER_SAMPLE:-16}" \
  --components "${PCA_COMPONENTS:-8}" \
  --tsne-max-points "${PCA_TSNE_MAX_POINTS:-5000}" \
  --tsne-perplexity "${PCA_TSNE_PERPLEXITY:-30}" \
  --tsne-learning-rate "${PCA_TSNE_LEARNING_RATE:-200}" \
  --tsne-n-iter "${PCA_TSNE_N_ITER:-1000}" \
  --max-samples-per-model "${PCA_MAX_SAMPLES_PER_MODEL:-64}" \
  --max-total-samples "${PCA_MAX_TOTAL_SAMPLES:-1024}" \
  --max-patches-per-group "${PCA_MAX_PATCHES_PER_GROUP:-8192}" \
  --time-limit-seconds "${PCA_TIME_LIMIT_SECONDS:-1800}" \
  --seed "${PCA_SEED:-42}" \
  --streaming-mode "${PCA_STREAMING_MODE:-none}" \
  --collector-mode "${PCA_COLLECTOR_MODE:-async}" \
  --models

for model_name in $MODELS; do
  set -- "$@" "$model_name"
done

if [ -n "${PCA_COLLECTOR_DEVICE-}" ]; then
  set -- "$@" --collector-device "$PCA_COLLECTOR_DEVICE"
fi

if [ -n "${PCA_DATASETS-}" ]; then
  set -- "$@" --datasets
  set -f
  for dataset_name in $PCA_DATASETS; do
    set -- "$@" "$dataset_name"
  done
  set +f
fi

if [ -n "${PCA_PREDOWNLOAD_MODELS-}" ]; then
  set -- "$@" --predownload-models
fi

exec "$@"
