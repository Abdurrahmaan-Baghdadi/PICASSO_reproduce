#!/usr/bin/env bash
# Train + evaluate rows of an ablation table (scripts/make_ablations.py) with pipeline/05_train_sae.sh.
#   bash scripts/run_ablation.sh ablations/sae_brain.tsv 3        # row 3 (0-based)
#   bash scripts/run_ablation.sh ablations/smoke.tsv all          # every row, one after another
# A row whose eval/summary.json exists is skipped, so reruns resume. EXTRA_SET="key=value ..."
# adds training overrides to every row (e.g. EXTRA_SET="device=cpu").
set -euo pipefail; cd "$(dirname "$0")/.."
TSV=$1; WHICH=${2:-all}
N=$(($(wc -l < "$TSV") - 1))
if [ "$WHICH" = all ]; then ROWS=$(seq 0 $((N - 1))); else ROWS=$WHICH; fi

for i in $ROWS; do
  if [ "$i" -ge "$N" ]; then echo "row $i: table has $N rows" >&2; continue; fi
  IFS=$'\t' read -r RUN GROUPS_ ENC STORE SEED_ TRAIN_SET EVAL_SET_ < <(sed -n "$((i + 2))p" "$TSV")
  (
    export ENCODER=$ENC RUN_NAME=$RUN SEED=$SEED_ EVAL_SET=$EVAL_SET_
    unset STORE_DIR; source env.sh; export STORE_DIR=$DATA_ROOT/store/$STORE
    if [ -f "$OUTPUT_ROOT/sae/$RUN/eval/summary.json" ]; then echo "[$i] $RUN: done, skipping"; exit 0; fi
    echo "[$i] $RUN ($GROUPS_) store=$STORE_DIR"
    # shellcheck disable=SC2086
    bash pipeline/05_train_sae.sh --set $TRAIN_SET ${EXTRA_SET:-} > "$OUTPUT_ROOT/logs/sae_$RUN.log" 2>&1 \
      && echo "[$i] $RUN: ok" || { echo "[$i] $RUN: FAILED, see $OUTPUT_ROOT/logs/sae_$RUN.log" >&2; exit 1; }
  ) || STATUS=1
done
exit ${STATUS:-0}
