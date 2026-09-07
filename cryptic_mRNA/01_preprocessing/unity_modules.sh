#!/usr/bin/env bash
# Source this on Unity to load the GPU GROMACS + supporting modules.
#   source env/unity_modules.sh
#
# Unity uses Lmod. Module names drift between quarters — CONFIRM with:
#   module spider gromacs
#   module spider cuda
# and edit the versions below to match what `module spider` reports.

# --- GPU GROMACS (production MD) ---------------------------------------------
# Typical Unity names (verify!). GROMACS is often provided with CUDA + MPI.
module load gromacs/2023.3         2>/dev/null || module load gromacs
module load cuda/12.2              2>/dev/null || module load cuda

# --- MPI (needed for REMD -multidir) -----------------------------------------
module load openmpi                2>/dev/null || true

# --- Conda for build/analysis Python -----------------------------------------
# Unity provides miniconda via `module load conda` OR you use your own install.
module load conda                  2>/dev/null || true

echo "[unity_modules] loaded. gmx = $(command -v gmx gmx_mpi 2>/dev/null | head -1)"
echo "[unity_modules] Verify GPU gmx with:  gmx --version | grep -i gpu"
