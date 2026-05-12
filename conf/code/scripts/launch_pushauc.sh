#!/usr/bin/env bash
# AUC-pushing defenses, launched in parallel on cuda=1/2/3 to coexist with the
# phase-2 sweep on cuda=0. Each cell runs in its own subshell (background).
#
# Defenses tested:
#   1. lightgcn_agr_priv with noise = 0.05 / 0.15 / 0.30 / 0.50
#       (L2-normalize + Gaussian noise on inference embeddings)
#   2. lightgcn_agr with heavy edge dropout (keep_rate = 0.5 / 0.3)
#   3. lightgcn_agr with small embedding (size=16) -- reduces capacity to memorize
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn_agr_priv checkpoint/lightgcn_agr_priv results/utility
mkdir -p log/lightgcn_agr checkpoint/lightgcn_agr

RESULTS=results/phase4_pushauc.jsonl
> "$RESULTS"

run_cell() {
  local cuda=$1 model=$2 tag=$3
  shift 3
  echo "[push] $(date +%H:%M:%S) start cuda=$cuda model=$model tag=$tag args=$*"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$RESULTS" \
    --epoch 200 --patience 4 \
    "$@" \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[push] $(date +%H:%M:%S) done cuda=$cuda tag=$tag" \
    || echo "[push] $(date +%H:%M:%S) FAIL cuda=$cuda tag=$tag"
}

# Wave 1 (3 cells, cuda 1/2/3)
run_cell 1 lightgcn_agr_priv ag_priv_n0.05  --priv_noise_std 0.05 &
P1=$!
run_cell 2 lightgcn_agr_priv ag_priv_n0.15  --priv_noise_std 0.15 &
P2=$!
run_cell 3 lightgcn_agr_priv ag_priv_n0.30  --priv_noise_std 0.30 &
P3=$!
wait $P1 $P2 $P3

# Wave 2 (3 cells, cuda 1/2/3) -- heavy edge dropout + small embedding
run_cell 1 lightgcn_agr     ag_kr0.5         --keep_rate 0.5 --beta 0.0 --str_weight 0.0 &
P1=$!
run_cell 2 lightgcn_agr     ag_kr0.3         --keep_rate 0.3 --beta 0.0 --str_weight 0.0 &
P2=$!
run_cell 3 lightgcn_agr_priv ag_priv_n0.50   --priv_noise_std 0.50 &
P3=$!
wait $P1 $P2 $P3

echo "[push] all cells done"
ls -la "$RESULTS"
wc -l "$RESULTS"
