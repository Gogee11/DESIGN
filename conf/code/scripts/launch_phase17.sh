#!/usr/bin/env bash
# Phase 17: post-hoc noise sweeps on best Pareto ckpts (fast eval_only) +
# additional multi-seed + LR sensitivity + DP-SGD reference for thesis.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
if [ -f /opt/dtk-25.04.2/env.sh ]; then source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk-25.04.1/env.sh ]; then source /opt/dtk-25.04.1/env.sh
elif [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; fi

mkdir -p log/lightgcn_agr log/lightgcn_agr_adv checkpoint/lightgcn_agr checkpoint/lightgcn_agr_adv results/utility

JSONL_T=results/phase17_train.jsonl
JSONL_E=results/phase17_eval.jsonl
> "$JSONL_T"
> "$JSONL_E"

# ---------- Part A: Multi-seed for emb=12 + GPU training (4 cells) ----------
declare -a CELLS_T
CELLS_T+=("lightgcn_agr:p17_emb12_seed2026:--seed 2026 --embedding_size 12 --beta 0.0 --str_weight 0.0")
CELLS_T+=("lightgcn_agr:p17_emb12_seed2027:--seed 2027 --embedding_size 12 --beta 0.0 --str_weight 0.0")
CELLS_T+=("lightgcn_agr:p17_emb6_seed2026:--seed 2026 --embedding_size 6 --beta 0.0 --str_weight 0.0")
CELLS_T+=("lightgcn_agr:p17_emb6_seed2027:--seed 2027 --embedding_size 6 --beta 0.0 --str_weight 0.0")
# DP-SGD reference (failure case)
CELLS_T+=("lightgcn_agr:p17_dpsgd_ref:--seed 2025 --epoch 30 --patience 5 --keep_rate 1.0 --str_weight 0.0 --beta 0.0")
# adv=0.05 patience=20 to push convergence
CELLS_T+=("lightgcn_agr_adv:p17_adv005_p20:--adv_weight 0.05 --adv_lambda 1.0 --patience 20")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p17q
for i in "${!CELLS_T[@]}"; do echo $i; done > /tmp/p17q/queue
QFILE=/tmp/p17q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS_T[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p17] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} ${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL_T" \
    --epoch 200 --patience 5 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p17] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p17] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
    if [ -z "$idx" ]; then return 0; fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[p17-train] $(date +%H:%M:%S) ALL TRAIN DONE"

# ---------- Part B: post-hoc noise sweeps on best ckpts (fast eval_only) ----------
echo "[p17-eval] $(date +%H:%M:%S) running post-hoc noise sweeps"

run_eval_only() {
  local ckpt_path=$1 src_model=$2 tag_prefix=$3
  if [ ! -f "$ckpt_path" ]; then echo "[eval] SKIP no ckpt: $ckpt_path"; return; fi
  for noise in 0.0 0.05 0.10 0.15 0.20 0.30; do
    python3 -m scripts.eval_only --device cuda --cuda 0 \
      --ckpt "$ckpt_path" --src_model "$src_model" \
      --priv_noise_std "$noise" --normalize_at_predict 0 \
      --tag "${tag_prefix}_n${noise}" \
      --out_jsonl "$JSONL_E" --dataset digital_music 2>&1 | grep -E "MIA AUC|recall=" | head -1
  done
}

# noise sweep on emb=16 (best Pareto privacy point base)
run_eval_only "checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-m3_emb16_b0s0.pth" lightgcn_agr emb16
# noise sweep on emb=8 (extreme privacy base)
run_eval_only "checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-m3_emb8_b0s0.pth" lightgcn_agr emb8
# noise sweep on b0s0 default (already had b0s0_n* combo; redoing for full curve)
# (skip - already have phase4f)

echo "[p17-eval] $(date +%H:%M:%S) ALL EVAL DONE"
wc -l "$JSONL_T" "$JSONL_E"
