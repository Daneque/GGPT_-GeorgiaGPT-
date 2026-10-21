#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-start}"
args=(
  --run-name llama_40m_smoke
  --max-steps 40
  --warmup-steps 10
  --micro-batch-size 4
  --accumulation-steps 4
  --workers 0
  --lr 0.0002
  --precision bf16
  --log-every 5
  --eval-every 10
  --checkpoint-every 10
  --generate-every 20
  --generation-max-tokens 32
  --validation-windows 128
)
case "$mode" in
  start) args+=(--stop-after 12) ;;
  resume) args+=(--resume) ;;
  *) echo "Usage: bash scripts/run_smoke.sh [start|resume]" >&2; exit 2 ;;
esac
PYTHONPATH=src python scripts/train_ggpt.py "${args[@]}"
