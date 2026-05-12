#!/usr/bin/env bash
# Follow-up experiments to harden the headline findings.
# Phase 5: MLP attacker re-run on existing ckpts (CPU; runs immediately, parallel-safe)
# Phase 6: multi-seed re-run of the Pareto winners (GPU; waits for wave 3 to finish)
set -eo pipefail
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PATH=${PATH:-}
source /opt/dtk-25.04.2/env.sh

mkdir -p log/lightgcn log/lightgcn_agr log/lightgcn_agr_priv \
         checkpoint/lightgcn checkpoint/lightgcn_agr checkpoint/lightgcn_agr_priv \
         results/utility

# ---------------- Phase 5: MLP attacker re-run on existing ckpts ----------------
P5=results/phase5_mlp_attacker.jsonl
> "$P5"
echo "[phase5] $(date +%H:%M:%S) start: MLP attacker on existing ckpts (CPU)"

run_mia_mlp() {
  local model=$1 tag_train=$2 ckpt=$3 tag_out=$4
  if [ ! -f "$ckpt" ]; then
    echo "[phase5] SKIP $tag_out (no ckpt: $ckpt)"
    return
  fi
  python3 -m attack.MIA --model "$model" --dataset digital_music --seed 2025 \
    --device cpu --checkpoint_path "$ckpt" --tag "$tag_out" --attacker mlp \
    --out_json "$P5" 2>&1 | tail -3
}

run_mia_mlp lightgcn      baseline    checkpoint/lightgcn/lightgcn-digital_music-2025-baseline.pth      lgcn_baseline_mlp
run_mia_mlp lightgcn_agr  default     checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025.pth        agr_default_mlp
run_mia_mlp lightgcn_agr  b0.0_s0.0   checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b0.0_s0.0.pth   agr_b0s0_mlp
run_mia_mlp lightgcn_agr  b1.0_s0.0   checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b1.0_s0.0.pth   agr_b1s0_mlp
run_mia_mlp lightgcn_agr  b4.0_s0.0   checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-b4.0_s0.0.pth   agr_b4s0_mlp
run_mia_mlp lightgcn_agr_priv ag_priv_n0.05 checkpoint/lightgcn_agr_priv/lightgcn_agr_priv-digital_music-2025-ag_priv_n0.05.pth priv_n0.05_mlp
run_mia_mlp lightgcn_agr_priv ag_priv_n0.15 checkpoint/lightgcn_agr_priv/lightgcn_agr_priv-digital_music-2025-ag_priv_n0.15.pth priv_n0.15_mlp
run_mia_mlp lightgcn_agr  ag_kr0.3    checkpoint/lightgcn_agr/lightgcn_agr-digital_music-2025-ag_kr0.3.pth    agr_kr0.3_mlp

echo "[phase5] $(date +%H:%M:%S) done"
wc -l "$P5"

# ---------------- Phase 6: multi-seed re-run of Pareto winners ------------------
echo "[phase6] $(date +%H:%M:%S) waiting for wave3 to finish"
while pgrep -fc "main\.py.*ag_priv_kr\|main\.py.*ag_priv_b0_s0" >/dev/null 2>&1 \
      && [ "$(pgrep -fc 'main\.py.*ag_priv_kr\|main\.py.*ag_priv_b0_s0' 2>/dev/null)" != "0" ]; do
  sleep 30
done
echo "[phase6] $(date +%H:%M:%S) wave 3 free; launching multi-seed grid"

P6=results/phase6_multiseed.jsonl
> "$P6"

run_seed_cell() {
  local cuda=$1 model=$2 tag=$3 seed=$4
  shift 4
  echo "[phase6] $(date +%H:%M:%S) start cuda=$cuda model=$model tag=$tag seed=$seed"
  python3 -m scripts.run_experiment \
    --model "$model" --dataset digital_music --seed "$seed" \
    --cuda "$cuda" --tag "$tag" \
    --results_jsonl "$P6" \
    --epoch 200 --patience 4 \
    "$@" \
    > "log/${model}/${tag}_seed${seed}_$(date +%H%M%S).out" 2>&1 \
    && echo "[phase6] $(date +%H:%M:%S) done cuda=$cuda tag=$tag seed=$seed" \
    || echo "[phase6] $(date +%H:%M:%S) FAIL cuda=$cuda tag=$tag seed=$seed"
}

# Wave A: 3 cells parallel - LGCN baseline + AGR(b=0,s=0) + AGR(b=1,s=0) at seed 2026
run_seed_cell 1 lightgcn      lgcn_seed2026 2026 &
A1=$!
run_seed_cell 2 lightgcn_agr  b0.0_s0.0_seed2026 2026 --beta 0.0 --str_weight 0.0 &
A2=$!
run_seed_cell 3 lightgcn_agr  b1.0_s0.0_seed2026 2026 --beta 1.0 --str_weight 0.0 &
A3=$!
wait $A1 $A2 $A3

# Wave B: same 3 cells at seed 2027
run_seed_cell 1 lightgcn      lgcn_seed2027 2027 &
B1=$!
run_seed_cell 2 lightgcn_agr  b0.0_s0.0_seed2027 2027 --beta 0.0 --str_weight 0.0 &
B2=$!
run_seed_cell 3 lightgcn_agr  b1.0_s0.0_seed2027 2027 --beta 1.0 --str_weight 0.0 &
B3=$!
wait $B1 $B2 $B3

echo "[phase6] $(date +%H:%M:%S) all multi-seed done"
wc -l "$P6"
