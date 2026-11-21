#!/usr/bin/env bash
# One-time TACC setup, run on a login node through `bash slurm/submit.sh setup` (which reads tacc.env).
# Creates the conda env if it is missing, then checks everything the jobs need and says what to fix.
set -uo pipefail; cd "$(dirname "$0")/.."
ok() { echo "  ok    $*"; }
bad() { echo "  FIX   $*"; problems=$((problems + 1)); }
problems=0

echo "Settings (tacc.env)"
if [ -f tacc.env ]; then ok "tacc.env found"; else bad "no tacc.env: cp tacc.env.example tacc.env, then fill it in"; fi
for v in ALLOCATION TRIDENT_DIR DATA_ROOT OUTPUT_ROOT; do
  if [ -n "${!v:-}" ]; then ok "$v=${!v}"; else bad "$v is empty in tacc.env"; fi
done

echo "Conda env"
if [ ! -f "$CONDA_SH" ]; then
  bad "no conda at $CONDA_SH. Install Miniconda under \$WORK, then rerun setup:
          curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-\$(uname -m).sh
          bash Miniconda3-latest-Linux-\$(uname -m).sh -b -p \$WORK/miniconda3"
else
  set +u; source "$CONDA_SH"
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "  creating env $CONDA_ENV (several minutes)"
    conda create -n "$CONDA_ENV" python=3.12 -y -q > /dev/null
  fi
  conda activate "$CONDA_ENV"; set -u
  echo "  installing this repo into $CONDA_ENV"
  if pip install -q -e .; then ok "repo installed in $CONDA_ENV"; else bad "pip install -e . failed (see above)"; fi
  if out=$(python -c "import torch, h5py, polars, yaml; print(torch.__version__, torch.version.cuda)" 2>&1); then
    read -r tv cv <<< "$out"
    if [ "$cv" != None ]; then ok "torch $tv, built for CUDA $cv"
    else bad "torch $tv has no CUDA build, so the jobs cannot use the GPUs. Reinstall it, e.g.
          pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126"; fi
  else bad "packages missing: $out"; fi
fi

echo "Data"
if [ -n "${TRIDENT_DIR:-}" ]; then
  for enc in virchow2 uni_v2; do
    d=$TRIDENT_DIR/features_$enc
    if [ -r "$d" ]; then ok "features_$enc: $(find "$d" -maxdepth 1 -name '*.h5' | wc -l | tr -d ' ') slides"
    else bad "cannot read $d (missing, or no permission)"; fi
  done
fi
for d in "${DATA_ROOT:-}" "${OUTPUT_ROOT:-}"; do
  [ -n "$d" ] || continue
  if mkdir -p "$d" 2>/dev/null && [ -w "$d" ]; then ok "can write $d"; else bad "cannot create or write $d"; fi
done

echo "Queues and allocation ($SYSTEM)"
for p in "$CPU_PART" "$GPU_PART" "$TEST_CPU_PART" "$TEST_GPU_PART"; do
  if [ -n "$(sinfo -h -p "$p" -o %P 2>/dev/null)" ]; then ok "partition $p"
  else bad "no partition $p on this system: set CPU_PART / GPU_PART / TEST_CPU_PART / TEST_GPU_PART in tacc.env (sinfo -s lists them)"; fi
done
if [ -n "${ALLOCATION:-}" ]; then
  if out=$(sbatch --test-only -A "$ALLOCATION" -p "$TEST_CPU_PART" -t 00:10:00 --wrap true 2>&1); then
    ok "allocation $ALLOCATION accepted"
  else bad "allocation $ALLOCATION rejected: $out"; fi
fi

echo
if [ "$problems" = 0 ]; then echo "All set. Next: bash slurm/submit.sh test"
else echo "$problems thing(s) to fix above, then rerun: bash slurm/submit.sh setup"; exit 1; fi
