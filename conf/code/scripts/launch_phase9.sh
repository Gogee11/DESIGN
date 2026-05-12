#!/usr/bin/env bash
# Phase 9: refinements based on Phase 8 findings.
#  - M3 finer embedding grid (24, 12) on top of b0,s0
#  - M4 with longer patience (12) + smaller adv_weight (so the disc doesn't kill recall)
#  - Best Pareto: combine b0,s0 + emb=16 + post-hoc noise sweep (eval-only, no retrain)
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn_agr_adv log/lightgcn_agr checkpoint/lightgcn_agr_adv checkpoint/lightgcn_agr

JSONL=results/phase9_refine.jsonl
> "$JSONL"

declare -a CELLS
# M3 finer
CELLS+=("lightgcn_agr:m3_emb24_b0s0:--embedding_size 24 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:m3_emb12_b0s0:--embedding_size 12 --beta 0.0 --str_weight 0.0")
# M4 with longer patience
CELLS+=("lightgcn_agr_adv:m4_adv0.02_p12:--adv_weight 0.02 --adv_lambda 1.0 --patience 12")
CELLS+=("lightgcn_agr_adv:m4_adv0.1_p12:--adv_weight 0.1 --adv_lambda 1.0 --patience 12")
CELLS+=("lightgcn_agr_adv:m4_adv0.05_lam0.5_p12:--adv_weight 0.05 --adv_lambda 0.5 --patience 12")
# M1 with longer patience to let conf reg converge
CELLS+=("lightgcn_agr_conf:m1_conf_w0.3_p12:--conf_weight 0.3 --conf_target 0.7 --patience 12")
CELLS+=("lightgcn_agr_conf:m1_conf_w1.0_p12:--conf_weight 1.0 --conf_target 0.7 --patience 12")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/p9q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/p9q/queue
QFILE=/tmp/p9q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[p9] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[p9] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[p9] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
      echo "[p9] [slot $slot_id] queue empty; exiting"
      return 0
    fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do
  worker $s &
done
wait

echo "[p9] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"

# Eval-only: combine the just-trained b0,s0 emb=16 with post-hoc noise sweep.
echo "[p9-eval] $(date +%H:%M:%S) eval-only sweep on phase 8 emb=16 ckpt"
P9E=results/phase9_eval_combo.jsonl
> "$P9E"
ckpt=checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-m3_emb16_b0s0.pth
if [ -f "$ckpt" ]; then
  for noise in 0.0 0.05 0.10 0.15 0.20; do
    tag="emb16_b0s0_n${noise}_combo"
    python3 -m scripts.eval_only --device cuda --cuda 0 \
      --ckpt "$ckpt" --src_model lightgcn_agr \
      --priv_noise_std "$noise" --normalize_at_predict 0 \
      --tag "$tag" --out_jsonl "$P9E" --dataset digital_music \
      --embedding_size 16 2>&1 | grep -E "MIA AUC|recall=" | head -2
  done
  echo "[p9-eval] done"
  wc -l "$P9E"
fi
