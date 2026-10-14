#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POPE_GPU="${POPE_GPU:-7}"
export CUDA_VISIBLE_DEVICES="$POPE_GPU"

exec python "${PROJECT_ROOT}/llava/eval/pope_baseline.py" \
    --model-path "${PROJECT_ROOT}/llava-v1.5-7b" \
    --pope-root "${PROJECT_ROOT}/DATA/POPE" \
    --image-folder "${PROJECT_ROOT}/DATA/val2014" \
    --output-dir "${PROJECT_ROOT}/outputs/pope_llava_pure" \
    --max-new-tokens 4 \
    --num-beams 1 \
    "$@"
