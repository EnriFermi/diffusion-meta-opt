#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Conda environment ───────────────────────────────────────────────
CONDA_ENV="onerec"
PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/${CONDA_ENV}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "       Make sure conda env '${CONDA_ENV}' exists."
    exit 1
fi

# ── Install missing deps if needed ──────────────────────────────────
"$PYTHON" -c "import hydra" 2>/dev/null || {
    echo "Installing hydra-core..."
    "${PYTHON%/python}/pip" install hydra-core
}

# ── Launch training ─────────────────────────────────────────────────
echo "=== Big VAE Training ==="
echo "Python:  $PYTHON"
echo "Workdir: $SCRIPT_DIR"
echo ""

exec "$PYTHON" experiments/train_big_vae.py "$@"
