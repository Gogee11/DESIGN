#!/usr/bin/env bash
# Phase 7: confidence-regularized AGR sweep (Plan C).
# Trains lightgcn_agr_conf with different conf_weight values to find the best
# privacy-utility point that beats both LightGCN baseline AND b=0,s=0 AGR.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn_agr_conf checkpoint/lightgcn_agr_conf results/utility

P7=results/phase7_conf_sweep.jsonl
> "$P7"

run_cell() {
  local cuda=$1 tag=$2
  shift 2
  echo "[phase7] $(date +%H:%M:%S) start cuda=$cuda tag=$tag args=$*"
  python3 -m scripts.run_experiment \
    --model lightgcn_agr_conf --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$P7" \
    --epoch 200 --patience 4 \
    "$@" \
    > "log/lightgcn_agr_conf/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[phase7] $(date +%H:%M:%S) done cuda=$cuda tag=$tag" \
    || echo "[phase7] $(date +%H:%M:%S) FAIL cuda=$cuda tag=$tag"
}

# Wave A: 3 cells parallel - conf_weight = {0.1, 0.5, 2.0}, conf_target=0.7
run_cell 1 conf_w0.1_t0.7 --conf_weight 0.1 --conf_target 0.7 &
P1=$!
run_cell 2 conf_w0.5_t0.7 --conf_weight 0.5 --conf_target 0.7 &
P2=$!
run_cell 3 conf_w2.0_t0.7 --conf_weight 2.0 --conf_target 0.7 &
P3=$!
wait $P1 $P2 $P3

# Wave B: 3 more cells - vary conf_target {0.55, 0.85} and one strong combo
run_cell 1 conf_w0.5_t0.55 --conf_weight 0.5 --conf_target 0.55 &
P1=$!
run_cell 2 conf_w0.5_t0.85 --conf_weight 0.5 --conf_target 0.85 &
P2=$!
run_cell 3 conf_w5.0_t0.6  --conf_weight 5.0 --conf_target 0.6 &
P3=$!
wait $P1 $P2 $P3

echo "[phase7] all done"
wc -l "$P7"
