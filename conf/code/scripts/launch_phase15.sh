#!/usr/bin/env bash
# Phase 15: fill Pareto gaps + multi-seed for top Pareto winners + M1+M3+M4 triple combo
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
if [ -f /opt/dtk-25.04.2/env.sh ]; then source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk-25.04.1/env.sh ]; then source /opt/dtk-25.04.1/env.sh
elif [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi

mkdir -p log/lightgcn_agr log/lightgcn_agr_adv log/lightgcn_agr_conf log/lightgcn_agr_combo \
         checkpoint/lightgcn_agr checkpoint/lightgcn_agr_adv checkpoint/lightgcn_agr_conf checkpoint/lightgcn_agr_combo \
         results/utility

JSONL=results/phase15_pareto_fill.jsonl
> "$JSONL"

declare -a CELLS

# Fill emb gaps + emb x adv combos
CELLS+=("lightgcn_agr:p15_emb10_b0s0:--embedding_size 10 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p15_emb20_b0s0:--embedding_size 20 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr_adv:p15_emb10_adv0.05:--embedding_size 10 --beta 0.0 --str_weight 0.0 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p15_emb10_adv0.1:--embedding_size 10 --beta 0.0 --str_weight 0.0 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p15_emb20_adv0.05:--embedding_size 20 --beta 0.0 --str_weight 0.0 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p15_emb12_adv0.05:--embedding_size 12 --beta 0.0 --str_weight 0.0 --adv_weight 0.05 --adv_lambda 1.0")

# M1+M3+M4 triple combo (lightgcn_agr_combo has both conf+adv knobs)
CELLS+=("lightgcn_agr_combo:p15_emb16_triple1:--embedding_size 16 --beta 0.0 --str_weight 0.0 --conf_weight 0.3 --conf_target 0.7 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_combo:p15_emb16_triple2:--embedding_size 16 --beta 0.0 --str_weight 0.0 --conf_weight 0.5 --conf_target 0.7 --adv_weight 0.05 --adv_lambda 1.0")

# Multi-seed for s1_combo_c0.5_a0.1 (Pareto winner) and emb=16+adv=0.1 (privacy winner)
CELLS+=("lightgcn_agr_combo:p15_combo_c0.5_a0.1_seed2026:--seed 2026 --conf_weight 0.5 --conf_target 0.7 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_combo:p15_combo_c0.5_a0.1_seed2027:--seed 2027 --conf_weight 0.5 --conf_target 0.7 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p15_emb16_adv0.1_seed2026:--seed 2026 --embedding_size 16 --beta 0.0 --str_weight 0.0 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p15_emb16_adv0.1_seed2027:--seed 2027 --embedding_size 16 --beta 0.0 --str_weight 0.0 --adv_weight 0.1 --adv_lambda 1.0")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p15q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/p15q/queue
QFILE=/tmp/p15q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p15] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 5 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p15] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p15] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
    if [ -z "$idx" ]; then echo "[p15] [slot $slot_id] queue empty"; return 0; fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[p15] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"
