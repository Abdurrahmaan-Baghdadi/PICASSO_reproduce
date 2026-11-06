#!/usr/bin/env bash
# Submit the TACC jobs from the repo root.
#   ALLOCATION=<project> bash slurm/submit.sh data                          # import + stores (CPU node)
#   ALLOCATION=<project> bash slurm/submit.sh sae ablations/sae_brain.tsv   # SAE grid (GPU nodes)
#   ALLOCATION=<project> bash slurm/submit.sh compare ablations/sae_brain.tsv
# SYSTEM picks partitions and GPUs per node: ls6 (Lonestar6, default) or vista.
# DEPEND=<job id> starts the job after that one succeeds. Extra arguments go to sbatch.
set -euo pipefail; cd "$(dirname "$0")/.."
: "${ALLOCATION:?set ALLOCATION to your TACC project (see /usr/local/etc/taccinfo)}"
case "${SYSTEM:-ls6}" in
  ls6)   CPU_PART=${CPU_PART:-normal}; GPU_PART=${GPU_PART:-gpu-a100}; GPUS_PER_NODE=3 ;;
  vista) CPU_PART=${CPU_PART:-gg};     GPU_PART=${GPU_PART:-gh};       GPUS_PER_NODE=1 ;;
  *) echo "SYSTEM must be ls6 or vista" >&2; exit 1 ;;
esac
mkdir -p slurm_logs
DEP=${DEPEND:+--dependency=afterok:$DEPEND}
what=$1; shift
case $what in
  data)
    sbatch -A "$ALLOCATION" -p "$CPU_PART" $DEP "$@" slurm/prepare_data.sbatch ;;
  sae)
    TSV=$1; shift
    n=$(($(wc -l < "$TSV") - 1)); per=${PER_NODE:-$GPUS_PER_NODE}
    PER_NODE=$per sbatch -A "$ALLOCATION" -p "$GPU_PART" --array=0-$(((n + per - 1) / per - 1)) $DEP "$@" \
      slurm/sae_ablation.sbatch "$TSV" ;;
  compare)
    TSV=$1; shift
    sbatch -A "$ALLOCATION" -p "$GPU_PART" $DEP "$@" slurm/compare.sbatch "$TSV" ;;
  *) echo "usage: slurm/submit.sh data | sae <table.tsv> | compare <table.tsv>" >&2; exit 1 ;;
esac
