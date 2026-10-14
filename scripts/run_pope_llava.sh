#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

POPE_GPU="${POPE_GPU:-7}"
export CUDA_VISIBLE_DEVICES="$POPE_GPU"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/pope_llava_run.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/pope_llava}"
mkdir -p "$OUTPUT_DIR"

# 运行 random
python hulluedit/eval/pope_eval.py \
    --config "$CONFIG_FILE" \
    --split random \
    --output "$OUTPUT_DIR/pope_random.json" \
    --model-name "LLaVA-1.5"

# 运行 popular
python hulluedit/eval/pope_eval.py \
    --config "$CONFIG_FILE" \
    --split popular \
    --output "$OUTPUT_DIR/pope_popular.json" \
    --model-name "LLaVA-1.5"

# 运行 adversarial
python hulluedit/eval/pope_eval.py \
    --config "$CONFIG_FILE" \
    --split adversarial \
    --output "$OUTPUT_DIR/pope_adversarial.json" \
    --model-name "LLaVA-1.5"

# 聚合结果
python hulluedit/eval/aggregate_pope.py \
    --files "$OUTPUT_DIR/pope_random.json" "$OUTPUT_DIR/pope_popular.json" "$OUTPUT_DIR/pope_adversarial.json" \
    --output "$OUTPUT_DIR/pope_all_metrics.json"
