#!/bin/bash
#SBATCH -J picasso-data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o slurm_logs/%x-%j.out
# TRIDENT features -> per-slide embeddings (both encoders) -> one SAE store per encoder.
# CPU only; submit from the repo root with slurm/submit.sh data.
# TRIDENT_DIR: the TRIDENT coords folder (20x_256px_0px_overlap) holding features_virchow2 and features_uni_v2.
# SLIDES_DIR: the .svs folder TRIDENT read (only stages 6-7 need it).
set -euo pipefail
source slurm/common.sh
: "${TRIDENT_DIR:?set TRIDENT_DIR to the TRIDENT 20x_256px_0px_overlap folder}"

python scripts/import_trident.py --coords-dir "$TRIDENT_DIR" --data-root "$DATA_ROOT" \
  --projects TCGA-GBM,TCGA-LGG --encoders virchow2,uni2 --meanpatch \
  ${SLIDES_DIR:+--slides-dir "$SLIDES_DIR"} --workers "${SLURM_CPUS_ON_NODE:-16}"
ENCODER=virchow2 STORE_DIR=$DATA_ROOT/store/virchow2 bash pipeline/04_build_store.sh --set 'fields=[cls]'
ENCODER=uni2 STORE_DIR=$DATA_ROOT/store/uni2 bash pipeline/04_build_store.sh --set 'fields=[cls]'
ENCODER=virchow2 STORE_DIR=$DATA_ROOT/store/virchow2_meanpatch bash pipeline/04_build_store.sh --set 'fields=[meanpatch]'
cmp "$DATA_ROOT/store/virchow2/splits.json" "$DATA_ROOT/store/uni2/splits.json" && echo "virchow2 and uni2 stores share one split"
