#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export NLTK_DATA="${NLTK_DATA:-/home/wq/nltk_data}"
export CUDA_VISIBLE_DEVICES="${PEM_GPU:-7}"
case "${1:-smoke}" in
  test) exec /mnt/data2/zhangwei/conda_env/pti/bin/python -m unittest pem.test_pem -v ;;
  smoke) exec /mnt/data2/zhangwei/conda_env/pti/bin/python -u -m pem.run_smoke ;;
  screen) exec /mnt/data2/zhangwei/conda_env/pti/bin/python -u -m pem.run_screen ;;
  *) echo "usage: $0 {test|smoke|screen}" >&2; exit 2 ;;
esac
