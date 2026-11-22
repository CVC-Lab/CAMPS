#!/bin/bash
#SBATCH -J camps-verify
#SBATCH -o camps-verify.%j.out
#SBATCH -e camps-verify.%j.err
#SBATCH -p gpu-a100-small     # Lonestar6: one A100. Vista: gh. Stampede3: h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A YOUR_ALLOCATION   # your TACC project, or pass it with: sbatch -A <project>

# Verification suite (paper Appendix A.6): recursive-bound check, energy
# contract, fallback/ladder regression, and the pilot sweep, compared
# against the archived results. Writes to $CAMPS_RESULTS_DIR and ends with
# verification_report.md there.
#
#   cd $WORK/CAMPS && sbatch -A <your-allocation> experiments/tacc/verify.slurm

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

export CAMPS_RESULTS_DIR="${CAMPS_RESULTS_DIR:-$WORK/CAMPS/verification_$(hostname -s)}"
export PYTHONIOENCODING=utf-8
nvidia-smi
python -m experiments.verify_all
echo "report: $CAMPS_RESULTS_DIR/verification_report.md"
