#!/usr/bin/env bash
# Phase 16: extreme Pareto corner, more multi-seed, conf target finer sweep.
# Runs on Remote A in parallel with Phase 15 on main remote.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
if [ -f /opt/dtk-25.04.2/env.sh ]; then source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk-25.04.1/env.sh ]; then source /opt/dtk-25.04.1/env.sh
elif [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi

mkdir -p log/lightgcn_agr log/lightgcn_agr_adv log/lightgcn_agr_conf checkpoint/lightgcn_agr checkpoint/lightgcn_agr_adv checkpoint/lightgcn_agr_conf results/utility

JSONL=results/phase16_extreme_seeds.jsonl
> "$JSONL"

declare -a CELLS

# Extreme privacy corner — push emb even smaller
CELLS+=("lightgcn_agr:p16_emb2_b0s0:--embedding_size 2 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p16_emb3_b0s0:--embedding_size 3 --beta 0.0 --str_weight 0.0")

# Multi-seed for emb=4 and emb=8 (currently single seed)
CELLS+=("lightgcn_agr:p16_emb4_seed2026:--seed 2026 --embedding_size 4 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p16_emb4_seed2027:--seed 2027 --embedding_size 4 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p16_emb8_seed2026:--seed 2026 --embedding_size 8 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p16_emb8_seed2027:--seed 2027 --embedding_size 8 --beta 0.0 --str_weight 0.0")

# Multi-seed for top-3 Pareto winners (extra confidence in error bars)
CELLS+=("lightgcn_agr_adv:p16_emb16adv01_seed2026:--seed 2026 --embedding_size 16 --beta 0.0 --str_weight 0.0 --adv_weight 0.1 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:p16_emb16adv01_seed2027:--seed 2027 --embedding_size 16 --beta 0.0 --str_weight 0.0 --adv_weight 0.1 --adv_lambda 1.0")

# Conf target finer sweep on b0,s0 (test if sigmoid target value matters at fixed weight)
CELLS+=("lightgcn_agr_conf:p16_conf_w0.3_t0.4:--conf_weight 0.3 --conf_target 0.4")
CELLS+=("lightgcn_agr_conf:p16_conf_w0.3_t0.6:--conf_weight 0.3 --conf_target 0.6")
CELLS+=("lightgcn_agr_conf:p16_conf_w0.3_t0.8:--conf_weight 0.3 --conf_target 0.8")
CELLS+=("lightgcn_agr_conf:p16_conf_w0.3_t0.9:--conf_weight 0.3 --conf_target 0.9")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p16q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/p16q/queue
QFILE=/tmp/p16q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p16] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 5 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p16] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p16] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
    if [ -z "$idx" ]; then echo "[p16] [slot $slot_id] queue empty"; return 0; fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[p16] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"
