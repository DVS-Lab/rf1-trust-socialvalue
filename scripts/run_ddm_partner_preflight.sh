#!/usr/bin/env bash
# Start a timed partner-only diagnostic from the new checkout, using the old
# Linux2 environment and the exact archived model input.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
original=${1:?Pass the absolute path of the November Linux2 checkout}
shift
cd "$root"
python_exe=${DDM_PYTHON:-"$original/.venv-linux2/bin/python"}
export MPLCONFIGDIR="$root/work/ddm_hierarchical/mpl"
export NUMBA_CACHE_DIR="$root/work/ddm_hierarchical/numba"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
umask 077
mkdir -p work/ddm_hierarchical/logs
log=$(mktemp work/ddm_hierarchical/logs/partner-preflight.XXXXXXXX)
echo "Private launcher log: $log"
"$python_exe" feasibility/ddm/partner_preflight.py --original-root "$original" "$@" 2>&1 | tee "$log"
