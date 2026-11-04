#!/usr/bin/env bash
# Isolated read-only behavioral workflow. Invoked via bash; parent shell untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-audit}"
case "$STAGE" in
  audit|pilot|report|test) ;;
  *) echo 'Usage: bash scripts/run_ddm_feasibility.sh {audit|pilot|report|test}' >&2; exit 2 ;;
esac
DDM_PYTHON="${DDM_PYTHON:-$ROOT/.venv-linux2/bin/python}"
DDM_UPSTREAM="${DDM_UPSTREAM:-/ZPOOL/data/projects/rf1-sra-linux2}"
if [[ ! -x "$DDM_PYTHON" ]]; then
  echo 'Set DDM_PYTHON to an existing Python with numpy, scipy, pandas, matplotlib, numba, pytest.' >&2
  exit 2
fi
export MPLCONFIGDIR="$ROOT/work/ddm/matplotlib"
export NUMBA_CACHE_DIR="$ROOT/work/ddm/numba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p work/ddm/logs results/ddm_feasibility
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-${STAGE}-$$"
LOG="$ROOT/work/ddm/logs/$STAMP.log"
record_exit() {
  local status=$?
  trap - EXIT
  "$DDM_PYTHON" - "$STAGE" "$status" "$STAMP" <<'PY'
import json,sys
from pathlib import Path
stage,code,stamp=sys.argv[1:]
Path('results/ddm_feasibility/last_run.json').write_text(json.dumps(
    dict(stage=stage,exit_code=int(code),run_id=stamp,private_log='work/ddm/logs/'+stamp+'.log'),indent=2)+'\n')
PY
  echo "Stage $STAGE exit: $status; private log: $LOG"
  exit "$status"
}
trap record_exit EXIT
run_stage() (
set -e
echo "DDM feasibility stage: $STAGE"
"$DDM_PYTHON" --version
git rev-parse HEAD
if [[ "$STAGE" == test || "$STAGE" == audit || "$STAGE" == pilot ]]; then
  "$DDM_PYTHON" -m pytest feasibility/ddm/test_ddm.py -q
fi
if [[ "$STAGE" == audit || "$STAGE" == pilot ]]; then
  "$DDM_PYTHON" feasibility/ddm/audit.py --upstream "$DDM_UPSTREAM"
fi
if [[ "$STAGE" == pilot ]]; then
  "$DDM_PYTHON" feasibility/ddm/pilot.py
fi
if [[ "$STAGE" != test ]]; then
  "$DDM_PYTHON" feasibility/ddm/report.py
fi
)
set +e
run_stage 2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"
exit "$STATUS"
