#!/usr/bin/env bash
# Phase 10 stretch experiments:
#  S1 - M1 + M4 combo: lightgcn_agr_combo with both conf reg and adv training
#  S2 - LLM-aware confidence target: lightgcn_agr_conf with conf_target_llm_alpha > 0
# Both built on top of the (b=0, s=0) Pareto-best base config.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn_agr_combo log/lightgcn_agr_conf \
         checkpoint/lightgcn_agr_combo checkpoint/lightgcn_agr_conf \
         results/utility

JSONL=results/phase10_stretch.jsonl
> "$JSONL"

declare -a CELLS

# S1: M1 + M4 combo (5 cells)
CELLS+=("lightgcn_agr_combo:s1_combo_c0.1_a0.05:--conf_weight 0.1 --conf_target 0.7 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_combo:s1_combo_c0.5_a0.05:--conf_weight 0.5 --conf_target 0.7 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_combo:s1_combo_c0.3_a0.02:--conf_weight 0.3 --conf_target 0.7 --adv_weight 0.02 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_combo:s1_combo_c0.5_a0.1:--conf_weight 0.5 --conf_target 0.7 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_combo:s1_combo_c0.2_a0.05_t0.6:--conf_weight 0.2 --conf_target 0.6 --adv_weight 0.05 --adv_lambda 1.0")

# S2: LLM-aware confidence target (5 cells, varying alpha and base target)
CELLS+=("lightgcn_agr_conf:s2_llmconf_a0.1_t0.6:--conf_weight 0.5 --conf_target 0.6 --conf_target_llm_alpha 0.1")
CELLS+=("lightgcn_agr_conf:s2_llmconf_a0.2_t0.5:--conf_weight 0.5 --conf_target 0.5 --conf_target_llm_alpha 0.2")
CELLS+=("lightgcn_agr_conf:s2_llmconf_a0.3_t0.5:--conf_weight 0.5 --conf_target 0.5 --conf_target_llm_alpha 0.3")
CELLS+=("lightgcn_agr_conf:s2_llmconf_a0.2_t0.7:--conf_weight 0.3 --conf_target 0.7 --conf_target_llm_alpha 0.2")
CELLS+=("lightgcn_agr_conf:s2_llmconf_a0.4_t0.4:--conf_weight 0.5 --conf_target 0.4 --conf_target_llm_alpha 0.4")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p10q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/p10q/queue
QFILE=/tmp/p10q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p10] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 8 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p10] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p10] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
    if [ -z "$idx" ]; then
      echo "[p10] [slot $slot_id] queue empty; exiting"; return 0
    fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[p10] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"
