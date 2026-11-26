#!/bin/bash
#SBATCH -J picasso-emb2img
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x-%j.out
# Emb2Img (Virchow2 CLS -> tile image) on TACC, one stage per job. Submit with slurm/submit.sh
# emb2img | emb2img-test | emb2img-train, which set the stage and the variables below.
#   pairs   (array job) shard $SLURM_ARRAY_TASK_ID of the (VAE latent, CLS) pairs, read from the slides
#   merge   (CPU) join the shards into $PAIRS_DIR/{train,test}_*.npy
#   train   DDP on this node's GPUs (resumes from $EMB2IMG_OUT/ckpt), then sampling + closed-loop eval
#   all     pairs (one shard) + merge + train: the small test on a dev queue
# PAIRS_DIR, EMB2IMG_OUT: pairs and model folders. N_TRAIN: train pairs (default: configs/emb2img/pairs.yaml).
# TRAIN_ARGS / SAMPLE_ARGS: extra key=value overrides of configs/emb2img/{train,sample}.yaml.
# A rerun skips finished shards and resumes training from the last checkpoint.
set -euo pipefail
source slurm/common.sh
export ENCODER=virchow2 STORE_DIR=$DATA_ROOT/store/virchow2
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # models come from $HF_HOME (slurm/submit.sh emb2img-download)
OUT=${EMB2IMG_OUT:-$OUTPUT_ROOT/emb2img_sd15_256}
NGPU=${GPUS_PER_NODE:-1}
echo "PAIRS_DIR=$PAIRS_DIR  EMB2IMG_OUT=$OUT  HF_HOME=${HF_HOME:-}"

pairs() {   # pairs <shard> <num_shards>
  python -c "import openslide" || { echo "openslide is missing: pip install openslide-bin (in the $CONDA_ENV env)" >&2; exit 1; }
  local workers=$(( ${SLURM_CPUS_ON_NODE:-16} > 28 ? 24 : ${SLURM_CPUS_ON_NODE:-16} - 4 ))
  python -m picasso.emb2img.pairs --config configs/emb2img/pairs.yaml \
    --set shard="$1" num_shards="$2" num_workers=$workers ${N_TRAIN:+n_train=$N_TRAIN}
}
merge() { python -m picasso.emb2img.pairs --config configs/emb2img/pairs.yaml --set merge=true; }
train() {
  # Effective batch 128 x GPUs x accumulation, kept near the paper's 1024.
  local accum=$(( (1024 + 128 * NGPU - 1) / (128 * NGPU) ))
  python -m torch.distributed.run --nproc_per_node "$NGPU" --master_port $((29500 + RANDOM % 1000)) \
    -m picasso.emb2img.train --config configs/emb2img/train.yaml \
    --set output_dir="$OUT" grad_accum=$accum ${TRAIN_ARGS:-}
  python - "$OUT/log.jsonl" <<'EOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
s = recs[-1]["s_per_step"]
print(f"speed {s:.2f} s/step: the paper's 16,000 steps take {s * 16000 / 3600:.1f} h on this node")
EOF
  python -m picasso.emb2img.sample_eval --config configs/emb2img/sample.yaml \
    --set checkpoint="$OUT/ema" output_dir="$OUT/eval" ${SAMPLE_ARGS:-}
  echo "done: $OUT/eval/grid.jpg (top row real tiles, bottom row generated) and summary.json"
}

case ${1:?stage: pairs | merge | train | all} in
  pairs) pairs "${SLURM_ARRAY_TASK_ID:-0}" "${SLURM_ARRAY_TASK_COUNT:-1}" ;;
  merge) merge ;;
  train) train ;;
  all)   pairs 0 1; merge; train ;;
  *) echo "unknown stage $1" >&2; exit 1 ;;
esac
