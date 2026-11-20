#!/usr/bin/env bash
# Start a timed partner-only diagnostic from the new checkout, using the old
# Linux2 environment and the exact archived model input.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
publication=(--log)
if [[ ${1:-} == --push ]]; then
  publication=(--push --log)
  shift
fi
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
printf 'Preflight code revision %s\n' "$(git rev-parse HEAD)" > "$log"
result=0
"$python_exe" feasibility/ddm/partner_preflight.py --original-root "$original" "$@" 2>&1 | tee -a "$log" || result=$?
printf 'Preflight runner exit %d\n' "$result" >> "$log"
python3 scripts/share_ddm_preflight.py "${publication[@]}" "$log"
exit "$result"
