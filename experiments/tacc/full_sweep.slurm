#!/bin/bash
#SBATCH -J camps-full
#SBATCH -o camps-full.%j.out
#SBATCH -e camps-full.%j.err
#SBATCH -p gpu-a100-small     # Lonestar6: one A100. Vista: gh. Stampede3: h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A YOUR_ALLOCATION   # your TACC project, or pass it with: sbatch -A <project>

# Full sweep: every experiment of Section 5 at full size and full protocol,
# plus the verification suite, with archived-versus-new comparison. Produces
# a complete second-architecture results set in $CAMPS_RESULTS_DIR.
#
#   cd $WORK/CAMPS && sbatch -A <your-allocation> experiments/tacc/full_sweep.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$WORK/CAMPS}"
module purge 2>/dev/null || true
module load gcc 2>/dev/null || true
if [ -f "${CAMPS_VENV:-$WORK/camps-venv}/bin/activate" ]; then
    source "${CAMPS_VENV:-$WORK/camps-venv}/bin/activate"
elif [ -f "$WORK/miniforge3/bin/activate" ]; then
    source "$WORK/miniforge3/bin/activate" camps
else
    echo "no environment found; run experiments/tacc/setup_env.sh first" >&2; exit 1
fi

export CAMPS_RESULTS_DIR="${CAMPS_RESULTS_DIR:-$WORK/CAMPS/results_$(hostname -s)}"
export PYTHONIOENCODING=utf-8
nvidia-smi
python -m experiments.verify_all --full
python -m experiments.rerun_noboost
python -m experiments.side_checks
echo "report: $CAMPS_RESULTS_DIR/verification_report.md"
