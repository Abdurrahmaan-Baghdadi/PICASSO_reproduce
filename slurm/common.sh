# Sourced by the slurm/*.sbatch jobs: conda env + repo env.
# CONDA_SH: conda's profile script (TACC: install Miniconda under $WORK, not $HOME).
# CONDA_ENV: env with this repo installed (pip install -e .).
source "${CONDA_SH:-$WORK/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-picasso}"
cd "$SLURM_SUBMIT_DIR"
source env.sh
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  DATA_ROOT=$DATA_ROOT  OUTPUT_ROOT=$OUTPUT_ROOT"
nvidia-smi -L 2>/dev/null || true
