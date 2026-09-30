#!/usr/bin/env bash
# Resume full-cohort post-fit checks using existing posterior draws. No interactive shell options are changed.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
phase_config="config/full_sample_batch_retry.json"
stage="full-cohort-postfit"
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/run_full_sample_postfit.sh' >&2
  exit 2
fi
record="results/run_logs/$(date -u +%Y%m%dT%H%M%SZ)-$stage"
mkdir -p "$record"
exec > >(tee "$record/console.txt") 2>&1
finish() {
  result=$?
  trap - EXIT
  python3 - "$record/status.json" "$result" "$stage" <<'PY'
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(stage=sys.argv[3],exit_code=int(sys.argv[2]),
 finished_at=datetime.now(timezone.utc).isoformat(),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
PY
  echo "Diagnostic export exit=$result. Preserve results/full_sample/hierarchical and $record when committing."
  exit "$result"
}
trap finish EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/work/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$root/work/numba}"
python3 -c 'import platform,sys; assert platform.system()=="Linux"; assert sys.version_info >= (3,11)'
if [[ -n "$(git status --porcelain --untracked-files=normal -- src scripts tests config stan pyproject.toml)" ]]; then
  echo 'Code/config has local changes; reconcile before diagnostic export.' >&2
  exit 1
fi
python3 -m pytest -q
ppc_exit=0
python3 -m rf1_trust_socialvalue.full_sample_ppc run || ppc_exit=$?
geometry_exit=0
python3 -m rf1_trust_socialvalue.full_sample_geometry || geometry_exit=$?
echo "Post-fit stages: PPC exit=$ppc_exit; geometry export exit=$geometry_exit"
if [[ "$ppc_exit" -ne 0 || "$geometry_exit" -ne 0 ]]; then
  exit 1
fi
