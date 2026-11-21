#!/bin/bash
#SBATCH -J picasso-data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x-%j.out
# TRIDENT features -> per-slide embeddings (both encoders) -> one SAE store per encoder.
# CPU only; submit from the repo root with slurm/submit.sh data.
# TRIDENT_DIR: TRIDENT folder(s) (20x_256px_0px_overlap) holding features_virchow2 and features_uni_v2.
#   Several datasets train together: separate the folders with ':' (e.g. TCGA:CCDI).
# MANIFEST: CSV(s) of the slides to import (slide_id, project, patient), ':'-separated; one per
#   dataset. Default: the TCGA manifest.
# PROJECTS: comma-separated projects to keep. Default: TCGA-GBM,TCGA-LGG with the TCGA manifest, all otherwise.
# SLIDES_DIR: the .svs folder TRIDENT read (only stages 6-7 need it).
# IMPORT_ARGS: extra import_trident.py options (slurm/submit.sh test limits slides and tiles with it).
# A rerun skips slides already imported, so a timed-out job can just be resubmitted.
set -euo pipefail
source slurm/common.sh
: "${TRIDENT_DIR:?set TRIDENT_DIR to the TRIDENT 20x_256px_0px_overlap folder(s)}"
PROJ=${PROJECTS:-}
if [ -z "${MANIFEST:-}" ]; then MANIFEST=manifests/tcga_dx_all.csv; PROJ=${PROJ:-TCGA-GBM,TCGA-LGG}; fi
unset PROJECTS   # the import filters; the store builds below take every imported slide
IFS=: read -ra DIRS <<< "$TRIDENT_DIR"
IFS=: read -ra MANS <<< "$MANIFEST"

python scripts/import_trident.py --coords-dir "${DIRS[@]}" --data-root "$DATA_ROOT" \
  --manifest "${MANS[@]}" ${PROJ:+--projects "$PROJ"} --encoders virchow2,uni2 --meanpatch \
  ${SLIDES_DIR:+--slides-dir "$SLIDES_DIR"} --workers "${SLURM_CPUS_ON_NODE:-16}" ${IMPORT_ARGS:-}
ENCODER=virchow2 STORE_DIR=$DATA_ROOT/store/virchow2 bash pipeline/04_build_store.sh --set 'fields=[cls]'
ENCODER=uni2 STORE_DIR=$DATA_ROOT/store/uni2 bash pipeline/04_build_store.sh --set 'fields=[cls]'
ENCODER=virchow2 STORE_DIR=$DATA_ROOT/store/virchow2_meanpatch bash pipeline/04_build_store.sh --set 'fields=[meanpatch]'
cmp "$DATA_ROOT/store/virchow2/splits.json" "$DATA_ROOT/store/uni2/splits.json" && echo "virchow2 and uni2 stores share one split"
