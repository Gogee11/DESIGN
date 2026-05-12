#!/usr/bin/env bash
# Run on remote A (port 10154), shares NFS with main remote.
# Phase 12: published-AGR baseline (β=4, str=0.234) + multi-seed Pareto winners.
# Optionally fills idle slots with Phase 12b: MLP attacker reruns on existing ckpts.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
# Auto-detect DTK version (remote A has 25.04.1, main remote has 25.04.2)
if [ -f /opt/dtk-25.04.2/env.sh ]; then
    source /opt/dtk-25.04.2/env.sh
elif [ -f /opt/dtk-25.04.1/env.sh ]; then
    source /opt/dtk-25.04.1/env.sh
elif [ -f /opt/dtk/env.sh ]; then
    source /opt/dtk/env.sh
fi

mkdir -p log/lightgcn log/lightgcn_agr log/lightgcn_agr_adv \
         checkpoint/lightgcn checkpoint/lightgcn_agr checkpoint/lightgcn_agr_adv \
         results/utility

JSONL=results/phase12_remoteA.jsonl
> "$JSONL"

declare -a CELLS

# Cell 0: published-AGR baseline (β=4, str=0.234, seed=2025) -- the canonical reference
CELLS+=("lightgcn_agr:published_agr_seed2025:--seed 2025 --beta 4.0 --str_weight 0.234 --prf_weight 0.025")

# Multi-seed for top Pareto winners (seeds 2026 and 2027)
CELLS+=("lightgcn:lgcn_baseline_seed2026:--seed 2026")
CELLS+=("lightgcn:lgcn_baseline_seed2027:--seed 2027")
CELLS+=("lightgcn_agr:b0s0_seed2026:--seed 2026 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:b0s0_seed2027:--seed 2027 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr_adv:m4_adv0.05_seed2026:--seed 2026 --adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:m4_adv0.05_seed2027:--seed 2027 --adv_weight 0.05 --adv_lambda 1.0")

slot_to_cuda() {
  case $1 in 0|1) echo 0 ;; 2|3) echo 1 ;; 4|5) echo 2 ;; 6|7) echo 3 ;; esac
}

mkdir -p /tmp/remoteA_q
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/remoteA_q/queue
QFILE=/tmp/remoteA_q/queue

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[remoteA] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start #${idx} model=${model} tag=${tag}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 5 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[remoteA] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[remoteA] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
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
      echo "[remoteA] [slot $slot_id] queue empty; exiting"; return 0
    fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

# Launch 8 worker slots (2/card x 4 cards). 7 cells, last slot will idle.
for s in 0 1 2 3 4 5 6 7; do worker $s & done
wait
echo "[remoteA] $(date +%H:%M:%S) GPU phase done"
wc -l "$JSONL"

# ---- Phase 12b: MLP attacker reruns (CPU, runs in parallel with idle slots) ----
P12B=results/phase12b_mlp_attacker.jsonl
> "$P12B"
echo "[remoteA-mlp] $(date +%H:%M:%S) MLP attacker on existing ckpts"

run_mlp() {
  local model=$1 ckpt=$2 tag=$3
  if [ ! -f "$ckpt" ]; then echo "[mlp] SKIP $tag (no ckpt)"; return; fi
  python3 -m attack.MIA --model "$model" --dataset digital_music --seed 2025 \
    --device cpu --checkpoint_path "$ckpt" --tag "$tag" --attacker mlp \
    --out_json "$P12B" 2>&1 | tail -3
}

run_mlp lightgcn      checkpoint/lightgcn/lightgcn-digital_music-2025-baseline.pth lgcn_baseline_mlp
run_mlp lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025.pth   agr_default_mlp
run_mlp lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b0.0_s0.0.pth     b0s0_mlp
run_mlp lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b1.0_s0.0.pth     b1s0_mlp
run_mlp lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b4.0_s0.0.pth     b4s0_mlp
run_mlp lightgcn_agr_adv  checkpoint/lightgcn_agr_adv/lightgcn_agr_adv-digital_music-2025-m4_adv0.05.pth  m4_adv0.05_mlp
run_mlp lightgcn_agr_conf checkpoint/lightgcn_agr_conf/lightgcn_agr_conf-digital_music-2025-m1_conf_w0.1_t0.7.pth m1_conf01_mlp
run_mlp lightgcn_agr_conf checkpoint/lightgcn_agr_conf/lightgcn_agr_conf-digital_music-2025-m1_conf_w0.5_t0.7.pth m1_conf05_mlp

echo "[remoteA-mlp] $(date +%H:%M:%S) MLP done"
wc -l "$P12B"
