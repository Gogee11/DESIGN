#!/usr/bin/env bash
# Wave 3: combination defenses (heavy dropout + L2 norm + noise) plus the
# minimum-defense Pareto winner (PRF-only + noise). To run after waves 1-2.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn_agr_priv checkpoint/lightgcn_agr_priv results/utility

RESULTS=results/phase4_pushauc.jsonl

run_cell() {
  local cuda=$1 model=$2 tag=$3
  shift 3
  echo "[push3] $(date +%H:%M:%S) start cuda=$cuda model=$model tag=$tag args=$*"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$RESULTS" \
    --epoch 200 --patience 4 \
    "$@" \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[push3] $(date +%H:%M:%S) done cuda=$cuda tag=$tag" \
    || echo "[push3] $(date +%H:%M:%S) FAIL cuda=$cuda tag=$tag"
}

# Combos: dropout + L2 norm + noise; and PRF-only-priv (best non-priv config + noise)
run_cell 1 lightgcn_agr_priv ag_priv_kr0.5_n0.15  --keep_rate 0.5 --priv_noise_std 0.15 &
P1=$!
run_cell 2 lightgcn_agr_priv ag_priv_kr0.3_n0.30  --keep_rate 0.3 --priv_noise_std 0.30 &
P2=$!
run_cell 3 lightgcn_agr_priv ag_priv_b0_s0_n0.15  --beta 0.0 --str_weight 0.0 --priv_noise_std 0.15 &
P3=$!
wait $P1 $P2 $P3

echo "[push3] wave 3 done"
wc -l "$RESULTS"
