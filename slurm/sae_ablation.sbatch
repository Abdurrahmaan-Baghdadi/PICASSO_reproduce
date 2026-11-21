#!/bin/bash
#SBATCH -J picasso-sae
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x-%A_%a.out
# Rows of an ablation table on one GPU node, one GPU per run. Submit with slurm/submit.sh.
# PER_NODE: GPUs used per node. N_TASKS: array size. GPU g of task t trains rows
# t*PER_NODE+g, then that plus N_TASKS*PER_NODE, ... (one row each when N_TASKS*PER_NODE >= rows).
# THEN_COMPARE=1: run scripts/compare_saes.py here afterwards (with COMPARE_ARGS).
set -euo pipefail
source slurm/common.sh
TSV=$1; PER_NODE=${PER_NODE:-1}; N_TASKS=${N_TASKS:-1}
N=$(($(wc -l < "$TSV") - 1)); STRIDE=$((N_TASKS * PER_NODE))
pids=()
for g in $(seq 0 $((PER_NODE - 1))); do
  FIRST=$((${SLURM_ARRAY_TASK_ID:-0} * PER_NODE + g))
  [ "$FIRST" -lt "$N" ] || continue
  ROWS=$(seq "$FIRST" "$STRIDE" $((N - 1)) | tr '\n' ' ')
  CUDA_VISIBLE_DEVICES=$g bash scripts/run_ablation.sh "$TSV" "$ROWS" & pids+=($!)
done
status=0
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || status=1; done
[ "$status" = 0 ] || { echo "some runs failed: see $OUTPUT_ROOT/logs/sae_*.log" >&2; exit 1; }
if [ -n "${THEN_COMPARE:-}" ]; then
  python scripts/compare_saes.py "$TSV" --device cuda ${COMPARE_ARGS:-}
fi
