#!/usr/bin/env bash
# Quick status snapshot of running experiments.
# Usage: bash scripts/status.sh
cd "$(dirname "$0")/.."

echo "==== $(date +%H:%M:%S) ===="
echo
echo "== running training/MIA processes =="
ps -ef | awk '/(main\.py|attack\.MIA|run_experiment|sweep)/ && !/grep/ && !/awk/ {print substr($0, 1, 200)}' | head -10
echo
echo "== completed utility files =="
ls -1 results/utility/*.json 2>/dev/null | wc -l

echo
echo "== latest log per (model, dataset, tag) =="
for d in lightgcn lightgcn_agr lightgcn_agr_priv; do
  for log in $(ls -t log/$d/*.log 2>/dev/null | head -3); do
    base=$(basename "$log")
    if grep -q "Best Epoch" "$log" 2>/dev/null; then
      best=$(grep "Best Epoch" "$log" | tail -1 | head -c 200)
      echo "  $d/$base: DONE  $(echo "$best" | grep -oE "Best Epoch [0-9]+|recall.: array.[^]]+.")"
    else
      last=$(tail -1 "$log")
      epoch=$(echo "$last" | grep -oE "Epoch [0-9]+" | tail -1)
      rec=$(echo "$last" | grep -oE "recall@20: [0-9.]+")
      [ -n "$epoch" ] && echo "  $d/$base: $epoch $rec"
    fi
  done
done

echo
echo "== completed jsonl rows =="
for f in results/*.jsonl; do
  [ -f "$f" ] || continue
  echo "  $(wc -l < "$f") $f"
done
