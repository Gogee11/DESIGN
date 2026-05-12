#!/usr/bin/env bash
# Unified launcher for the four privacy-utility methods, scheduled across all
# 4 DCU cards × 2 slots/card = 8 parallel GPU jobs.
#
# Methods covered:
#   M1 - Confidence regularization     (lightgcn_agr_conf, conf_weight sweep)
#   M2 - Degree-balanced MIA            (post-hoc on existing ckpts, CPU)
#   M3 - Embedding compression          (lightgcn_agr w/ b0,s0 + emb_size sweep)
#   M4 - Adversarial training (GRL)     (lightgcn_agr_adv, adv_weight sweep)
#
# Slot scheduling: maintain a queue and 8 background workers, each pinned to
# (cuda, slot) so two cells per card share a GPU.
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn_agr_conf log/lightgcn_agr_adv \
         checkpoint/lightgcn_agr_conf checkpoint/lightgcn_agr_adv \
         results/utility

JSONL=results/phase8_methods.jsonl
> "$JSONL"

declare -a CELLS

# ---- M1: confidence regularization (5 cells) ----
CELLS+=("lightgcn_agr_conf:m1_conf_w0.1_t0.7:--conf_weight 0.1 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:m1_conf_w0.5_t0.7:--conf_weight 0.5 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:m1_conf_w2.0_t0.7:--conf_weight 2.0 --conf_target 0.7")
CELLS+=("lightgcn_agr_conf:m1_conf_w0.5_t0.55:--conf_weight 0.5 --conf_target 0.55")
CELLS+=("lightgcn_agr_conf:m1_conf_w5.0_t0.6:--conf_weight 5.0 --conf_target 0.6")

# ---- M3: embedding compression (4 cells) ----
CELLS+=("lightgcn_agr:m3_emb16_b0s0:--embedding_size 16 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:m3_emb8_b0s0:--embedding_size 8 --beta 0.0 --str_weight 0.0")
CELLS+=("lightgcn_agr:m3_emb16_default: --embedding_size 16")
CELLS+=("lightgcn_agr:m3_emb8_default: --embedding_size 8")

# ---- M4: adversarial training (5 cells) ----
CELLS+=("lightgcn_agr_adv:m4_adv0.05:--adv_weight 0.05 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:m4_adv0.2:--adv_weight 0.2 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:m4_adv1.0:--adv_weight 1.0 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:m4_adv3.0:--adv_weight 3.0 --adv_lambda 1.0")
CELLS+=("lightgcn_agr_adv:m4_adv1.0_lam2:--adv_weight 1.0 --adv_lambda 2.0")

echo "[unified] $(date +%H:%M:%S) total cells: ${#CELLS[@]}"

run_cell_index() {
  local idx=$1 slot_id=$2 cuda=$3
  local entry=${CELLS[$idx]}
  local model=${entry%%:*}; local rest=${entry#*:}
  local tag=${rest%%:*}; local args=${rest#*:}
  echo "[unified] $(date +%H:%M:%S) [slot ${slot_id}/cuda ${cuda}] start cell #${idx} model=${model} tag=${tag} args=${args}"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed 2025 \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$JSONL" \
    --epoch 200 --patience 4 \
    $args \
    > "log/${model}/${tag}_$(date +%H%M%S).out" 2>&1 \
    && echo "[unified] $(date +%H:%M:%S) [slot ${slot_id}] DONE #${idx} ${tag}" \
    || echo "[unified] $(date +%H:%M:%S) [slot ${slot_id}] FAIL #${idx} ${tag}"
}

# 8 worker slots: 2 per card on cuda 0/1/2/3.
slot_to_cuda() {
  case $1 in
    0) echo 0 ;;  1) echo 0 ;;
    2) echo 1 ;;  3) echo 1 ;;
    4) echo 2 ;;  5) echo 2 ;;
    6) echo 3 ;;  7) echo 3 ;;
  esac
}

# Schedule: round-robin assign cells to slots; worker pulls next and runs.
mkdir -p /tmp/unified_queue
for i in "${!CELLS[@]}"; do echo $i; done > /tmp/unified_queue/cells.txt

# Lockfile-based queue
QFILE=/tmp/unified_queue/queue
cp /tmp/unified_queue/cells.txt "$QFILE"

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
      echo "[unified] [slot $slot_id] queue empty; exiting"
      return 0
    fi
    run_cell_index "$idx" "$slot_id" "$cuda"
  done
}

for s in 0 1 2 3 4 5 6 7; do
  worker $s &
done
wait

echo "[unified] $(date +%H:%M:%S) ALL DONE"
wc -l "$JSONL"

# ---- M2: degree-balanced MIA on existing ckpts (CPU sequential) ----
# Runs after the GPU pipeline so it doesn't fight for resources.
M2=results/phase8_m2_balanced_mia.jsonl
> "$M2"
echo "[unified-m2] $(date +%H:%M:%S) running degree-balanced MIA on existing ckpts"

run_balanced_mia() {
  local model=$1 ckpt=$2 tag_out=$3
  if [ ! -f "$ckpt" ]; then
    echo "[m2] SKIP $tag_out (no ckpt: $ckpt)"
    return
  fi
  python3 -m attack.MIA --model "$model" --dataset digital_music --seed 2025 \
    --device cpu --checkpoint_path "$ckpt" --tag "$tag_out" \
    --balance_by_degree 1 --num_degree_bins 5 \
    --out_json "$M2" 2>&1 | tail -3
}

run_balanced_mia lightgcn      checkpoint/lightgcn/lightgcn-digital_music-2025-baseline.pth lgcn_baseline_balanced
run_balanced_mia lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025.pth   agr_default_balanced
run_balanced_mia lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b0.0_s0.0.pth b0s0_balanced
run_balanced_mia lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b1.0_s0.0.pth b1s0_balanced
run_balanced_mia lightgcn_agr  checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b4.0_s0.0.pth b4s0_balanced

echo "[unified-m2] $(date +%H:%M:%S) all DONE"
wc -l "$M2"
