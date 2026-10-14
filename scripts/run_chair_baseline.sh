#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-/mnt/data2/zhangwei/conda_env/pti}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/llava-v1.5-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/chair}"
RUN_NAME="${RUN_NAME:-llava15_7b_pure}"
CHAIR_GPU="${CHAIR_GPU:-7}"
CHAIR_CODE_DIR="${CHAIR_CODE_DIR:-}"
CAP_JSONL="${OUTPUT_DIR}/chair_captions_${RUN_NAME}.jsonl"
CHAIR_JSON="${OUTPUT_DIR}/chair_captions_${RUN_NAME}_chair_result.json"
CHAIR_CACHE="${CHAIR_CACHE:-${OUTPUT_DIR}/chair_coco2014.pkl}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export NLTK_DATA="${NLTK_DATA:-/home/wq/nltk_data}"
export CUDA_VISIBLE_DEVICES="${CHAIR_GPU}"

if [ "${FRESH_RUN:-0}" = "1" ]; then
  rm -f "${CAP_JSONL}" "${CHAIR_JSON}"
fi

python "${PROJECT_ROOT}/llava/eval/chair_baseline.py" \
  --model-path "${MODEL_PATH}" \
  --data-root "${PROJECT_ROOT}/DATA" \
  --output-file "${CAP_JSONL}" \
  --split val \
  --sampling random \
  --num-samples 500 \
  --seed 0 \
  --max-new-tokens 64 \
  --num-beams 3 \
  "$@"

CHAIR_ARGS=()
if [ -n "${CHAIR_CODE_DIR}" ]; then
  CHAIR_ARGS+=(--chair-code-dir "${CHAIR_CODE_DIR}")
fi

python -m hulluedit.eval.eval_chair \
  --input "${CAP_JSONL}" \
  --coco-annotations "${PROJECT_ROOT}/DATA/annotations" \
  --output "${CHAIR_JSON}" \
  --cache "${CHAIR_CACHE}" \
  "${CHAIR_ARGS[@]}"

echo "Captions: ${CAP_JSONL}"
echo "CHAIR: ${CHAIR_JSON}"
