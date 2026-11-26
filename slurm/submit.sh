#!/usr/bin/env bash
# TACC entry point. Fill in tacc.env once (cp tacc.env.example tacc.env), then from the repo root:
#   bash slurm/submit.sh setup    # login node, once: conda env, and checks of data, queues and allocation
#   bash slurm/submit.sh test     # small end-to-end run on the dev queues (60 slides, 9 tiny SAEs, comparison)
#   bash slurm/submit.sh all      # full run: data -> 21 SAEs -> comparison, each job waiting for the one before
# Single stages: data | sae <table.tsv> | compare <table.tsv>. Extra arguments go to sbatch;
# DEPEND=<job id> starts a single stage after that job succeeds.
# Emb2Img (Virchow2 only, after data): emb2img-download (login node, once), emb2img-test, emb2img
# (pairs -> merge -> train + eval), emb2img-train (resume or retrain on existing pairs).
set -euo pipefail; cd "$(dirname "$0")/.."
if [ -f tacc.env ]; then set -a; source tacc.env; set +a; fi
case "${SYSTEM:-ls6}" in
  ls6)   CPU_PART=${CPU_PART:-normal}; GPU_PART=${GPU_PART:-gpu-a100}; GPUS_PER_NODE=3
         TEST_CPU_PART=${TEST_CPU_PART:-development}; TEST_GPU_PART=${TEST_GPU_PART:-gpu-a100-dev} ;;
  vista) CPU_PART=${CPU_PART:-gg}; GPU_PART=${GPU_PART:-gh}; GPUS_PER_NODE=1
         TEST_CPU_PART=${TEST_CPU_PART:-gg}; TEST_GPU_PART=${TEST_GPU_PART:-gh-dev} ;;
  *) echo "SYSTEM must be ls6 or vista" >&2; exit 1 ;;
esac
export SYSTEM=${SYSTEM:-ls6} CPU_PART GPU_PART TEST_CPU_PART TEST_GPU_PART GPUS_PER_NODE
export HF_HOME=${HF_HOME:-$WORK/hf_cache}   # Hugging Face models for Emb2Img ($HOME is too small)
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
submit_emb2img() {   # submit_emb2img <stage> <after job> [sbatch args]
  local stage=$1 dep=$2; shift 2
  job -J "picasso-emb2img-$stage" $(after "$dep") "$@" slurm/emb2img.sbatch "$stage"
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

  emb2img-download)
    # Login node, once: the diffusion, VAE and Virchow2 weights into $HF_HOME (jobs run offline).
    # Virchow2 is gated: accept its terms on huggingface.co/paige-ai/Virchow2, then `hf auth login`.
    set +u; source "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
    python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sd-vae-ft-mse")
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                  allow_patterns=["model_index.json", "scheduler/*", "unet/config.json", "unet/diffusion_pytorch_model.safetensors"])
snapshot_download("paige-ai/Virchow2")
print("models ready")
PY
    python -c "import openslide; print('openslide', openslide.__library_version__)" \
      || echo "openslide is missing: pip install openslide-bin" >&2
    echo "cache: $HF_HOME" ;;

  emb2img-test)
    # 20,000 pairs, 100 steps, 64 samples on the dev queue: checks the stages and times a step.
    need ALLOCATION DATA_ROOT OUTPUT_ROOT
    export PAIRS_DIR=$DATA_ROOT/emb2img_pairs_test EMB2IMG_OUT=$OUTPUT_ROOT/emb2img_test N_TRAIN=20000 \
      TRAIN_ARGS="steps=100 save_every=100 log_every=10" SAMPLE_ARGS="n=64 num_steps=50"
    J=$(submit_emb2img all "${DEPEND:-}" -p "$TEST_GPU_PART" -t 01:30:00 "$@")
    echo "emb2img test job $J -> $EMB2IMG_OUT/eval/grid.jpg; its log ends with the time of the full schedule" ;;

  emb2img)
    need ALLOCATION DATA_ROOT OUTPUT_ROOT
    SHARDS=${SHARDS:-8}
    P=$(submit_emb2img pairs "${DEPEND:-}" -p "$GPU_PART" --array=0-$((SHARDS - 1)) "$@")
    M=$(submit_emb2img merge "$P" -p "$CPU_PART" -t 04:00:00 "$@")
    T=$(submit_emb2img train "$M" -p "$GPU_PART" -t 24:00:00 "$@")
    cat <<EOF
Emb2Img submitted:
  pairs  job $P   $SHARDS shards on $GPU_PART -> ${PAIRS_DIR:-$DATA_ROOT/emb2img_pairs}
  merge  job $M   starts after $P
  train  job $T   starts after $M, then sampling + eval -> ${EMB2IMG_OUT:-$OUTPUT_ROOT/emb2img_sd15_256}
Watch with: squeue -u \$USER     Logs: slurm_logs/picasso-emb2img-*
If training hits its time limit: bash slurm/submit.sh emb2img-train (resumes from the last checkpoint).
EOF
    ;;

  emb2img-train)
    need ALLOCATION DATA_ROOT OUTPUT_ROOT
    echo "emb2img train job $(submit_emb2img train "${DEPEND:-}" -p "$GPU_PART" -t 24:00:00 "$@")" ;;

  *) echo "usage: slurm/submit.sh setup | test | all | data | sae <table.tsv> | compare <table.tsv> | emb2img-download | emb2img-test | emb2img | emb2img-train" >&2; exit 1 ;;
esac
