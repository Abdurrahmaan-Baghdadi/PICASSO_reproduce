#!/bin/bash
#SBATCH -J picasso-compare
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x-%j.out
# Metrics table, seed / encoder concept matching and GBM-vs-LGG probes of an ablation table.
# Submit from the repo root with slurm/submit.sh compare <table.tsv>.
set -euo pipefail
source slurm/common.sh
python scripts/compare_saes.py "$1" --device cuda ${COMPARE_ARGS:-}
