#!/usr/bin/env bash
# Phase 14: complementary to Phase 13 (Phase 13 runs on remote A).
# Targets:
#   14a Multi-seed extended (AGR-default, m1_conf, m3_emb16) for variance bars   6 cells
#   14b M1 + M3 combo (conf reg + emb compression)                                 4 cells
#   14c Strong privacy corner: emb=4, b0s0 (extreme)                               2 cells
#   14d Post-hoc noise sweep on emb=8 ckpt (eval_only, no retrain)                 0 GPU (CPU eval)
# Total = 12 GPU cells.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
if [ -f /opt/dtk-25.04.2/env.sh ]; then source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk-25.04.1/env.sh ]; then source /opt/dtk-25.04.1/env.sh
elif [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi

mkdir -p log/lightgcn log/lightgcn_agr log/lightgcn_agr_conf log/lightgcn_agr_adv \
         checkpoint/lightgcn checkpoint/lightgcn_agr checkpoint/lightgcn_agr_conf checkpoint/lightgcn_agr_adv \
         results/utility

JSONL=results/phase14_extra.jsonl
> "$JSONL"

declare -a CELLS

# 14a: multi-seed for AGR-default + m1_conf_w0.1 + m3_emb16
CELLS+=("lightgcn_agr:p14a_agrdef_seed2026:--seed 2026")  # AGR default with seed 2026
CELLS+=("lightgcn_agr:p14a_agrdef_seed2027:--seed 2027")
CELLS+=("lightgcn_agr_conf:p14a_m1conf01_seed2026:--seed 2026 --conf_weight 0.1 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:p14a_m1conf01_seed2027:--seed 2027 --conf_weight 0.1 --conf_target 0.7")
CELLS+=("lightgcn_agr:p14a_emb16b0s0_seed2026:--seed 2026 --embedding_size 16 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p14a_emb16b0s0_seed2027:--seed 2027 --embedding_size 16 --beta 0.0 --str_weight 0.0")

# 14b: M1 + M3 combo
CELLS+=("lightgcn_agr_conf:p14b_emb16_conf01:--embedding_size 16 --beta 0.0 --str_weight 0.0 --conf_weight 0.1 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:p14b_emb16_conf05:--embedding_size 16 --beta 0.0 --str_weight 0.0 --conf_weight 0.5 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:p14b_emb12_conf03:--embedding_size 12 --beta 0.0 --str_weight 0.0 --conf_weight 0.3 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:p14b_emb24_conf03:--embedding_size 24 --beta 0.0 --str_weight 0.0 --conf_weight 0.3 --conf_target 0.7")

# 14c: extreme privacy corner
CELLS+=("lightgcn_agr:p14c_emb4_b0s0:--embedding_size 4 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:p14c_emb6_b0s0:--embedding_size 6 --beta 0.0 --str_weight 0.0")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p14q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/p14q/queue
QFILE=/tmp/p14q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p14] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 5 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p14] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p14] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
    if [ -z "$idx" ]; then echo "[p14] [slot $slot_id] queue empty"; return 0; fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[p14] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"
