#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-start}"
args=(
  --run-name llama_40m_v1
  --max-steps 60000
  --warmup-steps 400
  --micro-batch-size 4
  --accumulation-steps 4
  --workers 0
  --lr 0.0002
  --weight-decay 0.1
  --precision bf16
  --log-every 20
  --eval-every 500
  --checkpoint-every 1000
  --generate-every 1000
  --generation-max-tokens 64
  --validation-windows 256
)
case "$mode" in
  start) ;;
  resume) args+=(--resume) ;;
  *) echo "Usage: bash scripts/run_main.sh [start|resume]" >&2; exit 2 ;;
esac
PYTHONPATH=src python scripts/train_ggpt.py "${args[@]}"
