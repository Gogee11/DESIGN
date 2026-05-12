#!/usr/bin/env bash
# Launch a sweep on remote DCU box. Run from project root: bash scripts/launch_sweep.sh <grid>
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

GRID=${1:-pareto}
DATASETS=${2:-digital_music}
WORKERS=${3:-3}
RUN_NAME=${4:-${GRID}_$(date +%m%d_%H%M)}

mkdir -p log/lightgcn log/lightgcn_agr checkpoint/lightgcn checkpoint/lightgcn_agr results/utility

echo "[launch] grid=$GRID datasets=$DATASETS workers=$WORKERS name=$RUN_NAME"
exec python3 -m scripts.sweep \
    --grid "$GRID" \
    --datasets $DATASETS \
    --workers "$WORKERS" \
    --run_name "$RUN_NAME" \
    --out_jsonl "results/${RUN_NAME}.jsonl"
