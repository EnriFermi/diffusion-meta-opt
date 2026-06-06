#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-diff-meta-opt312}"

BIG_VAE_CHECKPOINT="${BIG_VAE_CHECKPOINT:-}"
LATENT_DIFFUSION_PRIOR_CHECKPOINT="${LATENT_DIFFUSION_PRIOR_CHECKPOINT:-}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-}"
RUN_LABEL="${RUN_LABEL:-}"
ARTIFACT_ROOT="${BIG_VAE_ARTIFACT_ROOT:-./artifacts/big_vae}"
SUITE_ROOT="${SUITE_ROOT:-${BIG_VAE_EVAL_SUITE_ROOT:-$ARTIFACT_ROOT/eval/suite}}"
DRY_RUN="${DRY_RUN:-false}"
SUITE_CONDA_ENV="${SUITE_CONDA_ENV:-}"
HELDOUT_CONDA_ENV="${HELDOUT_CONDA_ENV:-}"
SCALING_CONDA_ENV="${SCALING_CONDA_ENV:-}"
LANDSCAPE_CONDA_ENV="${LANDSCAPE_CONDA_ENV:-}"

run_python() {
  CURRENT_CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
  if [ -n "${CONDA_PREFIX:-}" ] && [ -n "$CURRENT_CONDA_ENV" ] && [ "$CURRENT_CONDA_ENV" = "$CONDA_ENV_NAME" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python "$@"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$@"
    return
  fi
  python "$@"
}

if [ -z "$BIG_VAE_CHECKPOINT" ]; then
  echo "Set BIG_VAE_CHECKPOINT=/path/to/big_vae_checkpoint.pt" >&2
  exit 1
fi

run_python \
  -m big_vae.entrypoints.post_train_eval_suite \
  "checkpoint.big_vae=$BIG_VAE_CHECKPOINT" \
  "checkpoint.latent_diffusion_prior=$LATENT_DIFFUSION_PRIOR_CHECKPOINT" \
  "checkpoint.label=$CHECKPOINT_LABEL" \
  "suite.run_label=$RUN_LABEL" \
  "suite.root_dir=$SUITE_ROOT" \
  "suite.dry_run=$DRY_RUN" \
  "suite.conda_env=$SUITE_CONDA_ENV" \
  "stages.heldout_eval.conda_env=$HELDOUT_CONDA_ENV" \
  "stages.scaling_check.conda_env=$SCALING_CONDA_ENV" \
  "stages.landscape_ablation.conda_env=$LANDSCAPE_CONDA_ENV" \
  "$@"
