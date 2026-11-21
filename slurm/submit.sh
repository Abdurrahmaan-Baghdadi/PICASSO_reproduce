#!/usr/bin/env bash
# TACC entry point. Fill in tacc.env once (cp tacc.env.example tacc.env), then from the repo root:
#   bash slurm/submit.sh setup    # login node, once: conda env, and checks of data, queues and allocation
#   bash slurm/submit.sh test     # small end-to-end run on the dev queues (60 slides, 9 tiny SAEs, comparison)
#   bash slurm/submit.sh all      # full run: data -> 21 SAEs -> comparison, each job waiting for the one before
# Single stages: data | sae <table.tsv> | compare <table.tsv>. Extra arguments go to sbatch;
# DEPEND=<job id> starts a single stage after that job succeeds.
set -euo pipefail; cd "$(dirname "$0")/.."
if [ -f tacc.env ]; then set -a; source tacc.env; set +a; fi
case "${SYSTEM:-ls6}" in
  ls6)   CPU_PART=${CPU_PART:-normal}; GPU_PART=${GPU_PART:-gpu-a100}; GPUS_PER_NODE=3
         TEST_CPU_PART=${TEST_CPU_PART:-development}; TEST_GPU_PART=${TEST_GPU_PART:-gpu-a100-dev} ;;
  vista) CPU_PART=${CPU_PART:-gg}; GPU_PART=${GPU_PART:-gh}; GPUS_PER_NODE=1
         TEST_CPU_PART=${TEST_CPU_PART:-gg}; TEST_GPU_PART=${TEST_GPU_PART:-gh-dev} ;;
  *) echo "SYSTEM must be ls6 or vista" >&2; exit 1 ;;
esac
export SYSTEM=${SYSTEM:-ls6} CPU_PART GPU_PART TEST_CPU_PART TEST_GPU_PART
export CONDA_SH=${CONDA_SH:-$WORK/miniconda3/etc/profile.d/conda.sh} CONDA_ENV=${CONDA_ENV:-picasso}
mkdir -p slurm_logs

need() {
  for v; do
    [ -n "${!v:-}" ] || { echo "$v is not set: put it in tacc.env (template: tacc.env.example)" >&2; exit 1; }
  done
}
job() { local id; id=$(sbatch --parsable -A "$ALLOCATION" "$@"); echo "${id%%;*}"; }
after() { if [ -n "${1:-}" ]; then echo "--dependency=afterok:$1"; fi; }
rows() { echo $(($(wc -l < "$1") - 1)); }

submit_data() {      # submit_data <after job> [sbatch args]
  local dep=$1; shift
  job -p "$CPU_PART" $(after "$dep") "$@" slurm/prepare_data.sbatch
}
submit_sae() {       # submit_sae <table> <after job> [sbatch args]; TASKS nodes (default: one row per GPU)
  local tsv=$1 dep=$2; shift 2
  local per=${PER_NODE:-$GPUS_PER_NODE} n; n=$(rows "$tsv")
  local tasks=${TASKS:-$(((n + per - 1) / per))}
  PER_NODE=$per N_TASKS=$tasks job -p "$GPU_PART" --array=0-$((tasks - 1)) $(after "$dep") "$@" \
    slurm/sae_ablation.sbatch "$tsv"
}
submit_compare() {   # submit_compare <table> <after job> [sbatch args]
  local tsv=$1 dep=$2; shift 2
  job -p "$GPU_PART" $(after "$dep") "$@" slurm/compare.sbatch "$tsv"
}

what=${1:-}; shift || true
case $what in
  setup)
    exec bash slurm/setup.sh ;;

  test)
    need ALLOCATION TRIDENT_DIR DATA_ROOT OUTPUT_ROOT
    export DATA_ROOT=${DATA_ROOT}_test OUTPUT_ROOT=${OUTPUT_ROOT}_test
    export IMPORT_ARGS="--max-slides-per-project 30 --max-tiles-per-slide 2000"
    # The local smoke grid with CUDA instead of Apple's GPU. One node trains every row (dev queues
    # allow few jobs per user), then runs the comparison itself.
    sed 's/device=mps/device=cuda/g' ablations/smoke.tsv > ablations/smoke_tacc.tsv
    CPU_PART=$TEST_CPU_PART GPU_PART=$TEST_GPU_PART
    D=$(submit_data "" -t 01:00:00)
    S=$(TASKS=1 THEN_COMPARE=1 COMPARE_ARGS="--n-rows 10000 --probe-train-rows 20000" \
        submit_sae ablations/smoke_tacc.tsv "$D" -t 01:30:00)
    cat <<EOF
Test submitted to $CPU_PART / $GPU_PART:
  data  job $D   60 slides x 2,000 tiles -> $DATA_ROOT
  sae   job $S   9 tiny SAEs + comparison, starts after $D -> $OUTPUT_ROOT
Watch with: squeue -u \$USER     Logs: slurm_logs/
Passed when $OUTPUT_ROOT/compare/smoke_tacc/ has metrics.csv, matching.csv and probes.csv.
EOF
    ;;

  all)
    need ALLOCATION TRIDENT_DIR DATA_ROOT OUTPUT_ROOT
    D=$(submit_data "" "$@")
    S=$(submit_sae ablations/sae_brain.tsv "$D" "$@")
    C=$(submit_compare ablations/sae_brain.tsv "$S" "$@")
    cat <<EOF
Full run submitted to $CPU_PART / $GPU_PART:
  data     job $D   every paired slide -> $DATA_ROOT
  sae      job $S   $(rows ablations/sae_brain.tsv) SAEs, starts after $D -> $OUTPUT_ROOT/sae
  compare  job $C   starts after $S -> $OUTPUT_ROOT/compare/sae_brain
Watch with: squeue -u \$USER     Logs: slurm_logs/
If a job fails or times out, rerun the same command: finished slides and SAEs are skipped.
EOF
    ;;

  data)
    need ALLOCATION TRIDENT_DIR DATA_ROOT OUTPUT_ROOT
    echo "data job $(submit_data "${DEPEND:-}" "$@")" ;;
  sae)
    need ALLOCATION DATA_ROOT OUTPUT_ROOT
    TSV=${1:?usage: slurm/submit.sh sae <table.tsv>}; shift
    echo "sae job $(submit_sae "$TSV" "${DEPEND:-}" "$@")" ;;
  compare)
    need ALLOCATION DATA_ROOT OUTPUT_ROOT
    TSV=${1:?usage: slurm/submit.sh compare <table.tsv>}; shift
    echo "compare job $(submit_compare "$TSV" "${DEPEND:-}" "$@")" ;;

  *) echo "usage: slurm/submit.sh setup | test | all | data | sae <table.tsv> | compare <table.tsv>" >&2; exit 1 ;;
esac
