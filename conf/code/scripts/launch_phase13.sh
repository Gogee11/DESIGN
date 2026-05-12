#!/usr/bin/env bash
# Phase 13: Pareto-gap filling + critical ablations.
# Designed to run on either remote (auto-detects DTK version).
# Cells:
#   13a Cross-method combos (M3 emb compress + M4 adv)        4 cells
#   13b aug_top_k ablation (verify LLM kNN graph importance)   4 cells
#   13c keep_rate (edge dropout) sensitivity                   3 cells
#   13d M4 long-patience extra points                          3 cells
# Total = 14 cells across 8 GPU slots = ~2 batches of ~30 min each.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
if [ -f /opt/dtk-25.04.2/env.sh ]; then source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk-25.04.1/env.sh ]; then source /opt/dtk-25.04.1/env.sh
elif [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi

mkdir -p log/lightgcn_agr log/lightgcn_agr_adv log/lightgcn_agr_priv \
         checkpoint/lightgcn_agr checkpoint/lightgcn_agr_adv checkpoint/lightgcn_agr_priv \
         results/utility

JSONL=results/phase13_extensions.jsonl
> "$JSONL"

declare -a CELLS

# 13a: cross-method M3 (emb compress) + M4 (adv) combos
CELLS+=("lightgcn_agr_adv:p13a_emb16_adv0.05:--embedding_size 16 --beta 0.0 --str_weight 0.0 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p13a_emb24_adv0.05:--embedding_size 24 --beta 0.0 --str_weight 0.0 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p13a_emb16_adv0.1:--embedding_size 16 --beta 0.0 --str_weight 0.0 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr:p13a_emb16_b0s0:--embedding_size 16 --beta 0.0 --str_weight 0.0")  # control (re-confirm)

# 13b: aug_top_k ablation (verify LLM kNN augmented graph design)
CELLS+=("lightgcn_agr:p13b_augk0_b0s0:--aug_top_k 0 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p13b_augk3_b0s0:--aug_top_k 3 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p13b_augk10_b0s0:--aug_top_k 10 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p13b_augk20_b0s0:--aug_top_k 20 --beta 0.0 --str_weight 0.0")

# 13c: keep_rate edge dropout sensitivity (with b0,s0 base)
CELLS+=("lightgcn_agr:p13c_kr0.3:--keep_rate 0.3 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p13c_kr0.5:--keep_rate 0.5 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p13c_kr0.6:--keep_rate 0.6 --beta 0.0 --str_weight 0.0")

# 13d: M4 finer + extra patience
CELLS+=("lightgcn_agr_adv:p13d_adv0.15_p15:--adv_weight 0.15 --adv_lambda 1.0 --patience 15")
CELLS+=("lightgcn_agr_adv:p13d_adv0.05_p15:--adv_weight 0.05 --adv_lambda 1.0 --patience 15")
CELLS+=("lightgcn_agr_adv:p13d_adv0.2_lam0.5_p15:--adv_weight 0.2 --adv_lambda 0.5 --patience 15")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p13q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/p13q/queue
QFILE=/tmp/p13q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p13] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 5 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p13] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p13] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
}

worker() {
  local slot_id=$1
  local cuda=$(slot_to_cuda $slot_id)
  while true; do
    local idx
    idx=$( (
      flock -x 200
      head -n1 "$QFILE" || true
      tail -n+2 "$QFILE" > "${QFILE}.new" 2>/dev/null || true
      mv "${QFILE}.new" "$QFILE" 2>/dev/null || true
    ) 200>"${QFILE}.lock" )
    if [ -z "$idx" ]; then echo "[p13] [slot $slot_id] queue empty"; return 0; fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[p13] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"
