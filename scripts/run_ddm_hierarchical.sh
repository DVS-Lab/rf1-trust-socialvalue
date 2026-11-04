#!/usr/bin/env bash
# Full behavioral cohort. No participant subsampling and no Git publication.
set -euo pipefail
DDM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DDM_ROOT"
DDM_STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$DDM_STAGE" in
  prepare|check|prior|fit|recovery|report|all|test) ;;
  *) echo 'Usage: bash scripts/run_ddm_hierarchical.sh {all|prepare|check|prior|fit|recovery|report|test} [Python CLI options]' >&2;exit 2 ;;
esac
DDM_PYTHON="${DDM_PYTHON:-$DDM_ROOT/.venv-linux2/bin/python}"
[[ -x "$DDM_PYTHON" ]] || { echo 'Set DDM_PYTHON to the existing Linux2 scientific Python.' >&2;exit 2; }
export MPLCONFIGDIR="$DDM_ROOT/work/ddm_hierarchical/matplotlib"
export NUMBA_CACHE_DIR="$DDM_ROOT/work/ddm_hierarchical/numba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p work/ddm_hierarchical/logs results/ddm_hierarchical
DDM_STAMP="$(date -u +%Y%m%dT%H%M%SZ)-${DDM_STAGE}-$$"
DDM_LOG="$DDM_ROOT/work/ddm_hierarchical/logs/$DDM_STAMP.log"
record_ddm_exit() {
  local result=$?
  trap - EXIT
  "$DDM_PYTHON" - "$DDM_STAGE" "$result" "$DDM_STAMP" <<'PY'
from pathlib import Path
import json,sys
stage,code,stamp=sys.argv[1:]
Path('results/ddm_hierarchical/last_run.json').write_text(json.dumps(dict(stage=stage,exit_code=int(code),run_id=stamp,
    private_log='work/ddm_hierarchical/logs/'+stamp+'.log'),indent=2)+'\n')
PY
  echo "Hierarchical DDM $DDM_STAGE exit: $result; private log: $DDM_LOG"
  exit "$result"
}
trap record_ddm_exit EXIT
run_ddm_stage() (
  set -e
  echo "Full-cohort hierarchical DDM stage: $DDM_STAGE"
  "$DDM_PYTHON" --version
  git rev-parse HEAD
  if [[ "$DDM_STAGE" != report ]]; then
    "$DDM_PYTHON" -m pytest feasibility/ddm/test_ddm.py feasibility/ddm/test_hierarchical.py -q
  fi
  if [[ "$DDM_STAGE" != test ]]; then
    "$DDM_PYTHON" feasibility/ddm/hierarchical.py "$DDM_STAGE" "$@"
  fi
)
set +e
run_ddm_stage "$@" 2>&1 | tee "$DDM_LOG"
DDM_RESULT="${PIPESTATUS[0]}"
exit "$DDM_RESULT"
