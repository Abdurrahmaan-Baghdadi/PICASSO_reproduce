#!/bin/bash
#SBATCH -J picasso-sae
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x-%A_%a.out
# One array task = one GPU node running PER_NODE rows of an ablation table at once, one per GPU.
# Submit from the repo root with slurm/submit.sh sae <table.tsv>.
set -euo pipefail
source slurm/common.sh
TSV=$1; PER_NODE=${PER_NODE:-1}
for g in $(seq 0 $((PER_NODE - 1))); do
  i=$((SLURM_ARRAY_TASK_ID * PER_NODE + g))
  CUDA_VISIBLE_DEVICES=$g bash scripts/run_ablation.sh "$TSV" "$i" &
done
wait
