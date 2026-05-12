#!/usr/bin/env bash
# Plan B: run LightGCN_AGR_Priv (with explicit privacy hooks) and a small
# noise-std sweep, then re-run analysis. To be invoked AFTER phase 1+2.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

DATASET=${1:-digital_music}
WORKERS=${2:-3}

mkdir -p log/lightgcn_agr_priv checkpoint/lightgcn_agr_priv results/utility

# Build inline grid via run_experiment for noise sweep.
RESULTS=results/phase3_planb.jsonl
> "$RESULTS"

# Default-priv baseline + noise sweep
declare -A CELLS
CELLS[ag_priv_default]="--priv_noise_std 0.05"
CELLS[ag_priv_n0.0]="--priv_noise_std 0.0"
CELLS[ag_priv_n0.1]="--priv_noise_std 0.1"
CELLS[ag_priv_n0.2]="--priv_noise_std 0.2"

# We do them serially since main sweep already took the GPU; user said single-card
for tag in "${!CELLS[@]}"; do
  echo "[planb] running $tag with extra=${CELLS[$tag]}"
  python3 -m scripts.run_experiment \
    --model lightgcn_agr_priv --dataset "$DATASET" --seed 2025 \
    --cuda 0 --tag "$tag" --results_jsonl "$RESULTS" --epoch 200 --patience 3
done

# Combine with main results
cat results/phase1_baselines.jsonl results/phase2_pareto.jsonl "$RESULTS" > results/all_combined.jsonl

# Re-run analyze
python3 -m scripts.analyze --in results/all_combined.jsonl || true
python3 -m scripts.write_thesis_report --in results/all_combined.jsonl || true

echo "[planb] done"
