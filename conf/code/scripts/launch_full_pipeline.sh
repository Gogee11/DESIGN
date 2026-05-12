#!/usr/bin/env bash
# Run the whole study end-to-end on remote.
# Phase 1: full-budget baselines (lightgcn + lightgcn_agr default).
# Phase 2: bounded-budget hyperparameter sweep (3-way parallel).
# Phase 3: collate + plot.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

DATASETS=${1:-digital_music}
SWEEP_GRID=${2:-pareto}
SWEEP_EPOCH=${3:-150}
WORKERS=${4:-3}

mkdir -p log/lightgcn log/lightgcn_agr checkpoint/lightgcn checkpoint/lightgcn_agr results/utility

stamp() { date +%H:%M:%S; }
echo "[$(stamp)] phase 1: full-budget baselines"
python3 -m scripts.sweep \
    --grid all_baselines \
    --datasets $DATASETS \
    --workers 2 \
    --run_name "phase1_baselines" \
    --out_jsonl results/phase1_baselines.jsonl

echo "[$(stamp)] phase 2: HP sweep (epoch=$SWEEP_EPOCH workers=$WORKERS grid=$SWEEP_GRID)"
python3 -m scripts.sweep \
    --grid "$SWEEP_GRID" \
    --datasets $DATASETS \
    --workers "$WORKERS" \
    --epoch "$SWEEP_EPOCH" \
    --patience 3 \
    --run_name "phase2_${SWEEP_GRID}" \
    --out_jsonl "results/phase2_${SWEEP_GRID}.jsonl"

echo "[$(stamp)] phase 3: lightgcn_agr_priv (Plan B) -- norm + noise"
mkdir -p log/lightgcn_agr_priv checkpoint/lightgcn_agr_priv
RESULTS_PRIV=results/phase3_planb.jsonl
> "$RESULTS_PRIV"
for noise in 0.0 0.05 0.15; do
  tag="ag_priv_n${noise}"
  echo "[$(stamp)]   cell $tag"
  python3 -m scripts.run_experiment \
    --model lightgcn_agr_priv --dataset $DATASETS --seed 2025 --cuda 0 \
    --tag "$tag" --results_jsonl "$RESULTS_PRIV" \
    --priv_noise_std "$noise" --epoch "$SWEEP_EPOCH" --patience 4 \
    || echo "[planb] $tag failed but continuing"
done

echo "[$(stamp)] phase 4: collate + plot + report"
cat results/phase1_baselines.jsonl results/phase2_${SWEEP_GRID}.jsonl results/phase3_planb.jsonl > results/all_combined.jsonl
python3 -m scripts.analyze --in results/all_combined.jsonl --out_dir results/figs || true
python3 -m scripts.write_thesis_report --in results/all_combined.jsonl --out_md results/THESIS_REPORT.md || true

echo "[$(stamp)] done. summary:"
ls -la results/
ls -la results/figs/ 2>/dev/null
