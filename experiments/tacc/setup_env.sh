#!/usr/bin/env bash
# One-time environment setup on a TACC system (run on a login node).
#
#   cd $WORK && git clone https://github.com/CVC-Lab/CAMPS.git && cd CAMPS
#   bash experiments/tacc/setup_env.sh
#
# Installs the CUDA build of PyTorch pinned in requirements.txt into
# $WORK/camps-venv. The pip wheel bundles its own CUDA libraries, so no cuda
# module is needed at run time, only a driver new enough for CUDA 12.8 on the
# compute node. torch 2.11 needs Python >= 3.10; if the system python3 module
# is older, a Miniforge installation is created in $WORK/miniforge3 and the
# environment is built there instead (the Slurm scripts find either).
set -euo pipefail

module purge 2>/dev/null || true
module load gcc 2>/dev/null || true
module load python3 2>/dev/null || true

ARCH=$(uname -m)   # x86_64 on Lonestar6/Stampede3, aarch64 on Vista
have_py310=$(python3 -c 'import sys; print(int(sys.version_info >= (3, 10)))' 2>/dev/null || echo 0)

if [ "$have_py310" = "1" ]; then
    VENV=${CAMPS_VENV:-$WORK/camps-venv}
    echo "using $(command -v python3) ($(python3 --version)) -> venv $VENV"
    python3 -m venv "$VENV"
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
    PIP="pip"
else
    echo "system python3 is older than 3.10; installing Miniforge into $WORK/miniforge3"
    cd "$WORK"
    curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh"
    bash "Miniforge3-Linux-${ARCH}.sh" -b -p "$WORK/miniforge3"
    # shellcheck disable=SC1091
    source "$WORK/miniforge3/bin/activate"
    conda create -y -n camps python=3.11
    conda activate camps
    PIP="pip"
    cd - >/dev/null
fi

$PIP install --upgrade pip
$PIP install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install numpy==2.3.1 scipy==1.16.2 matplotlib==3.10.3
python -c "import torch, scipy, numpy; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
echo "environment ready"
