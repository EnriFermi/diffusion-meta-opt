#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m MetaOpt.meta_learning --config MetaOpt/experiments/cifar10_fast.yaml "$@"

