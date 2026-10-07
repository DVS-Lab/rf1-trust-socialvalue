#!/usr/bin/env bash
# Child process only: interactive shell options are never changed.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root" || exit 1
stage="${1:-all}"
case "$stage" in all|analysis|recovery|report) ;; *) echo 'Usage: bash scripts/run_full_sample_closeout.sh [all|analysis|recovery|report]'; exit 2;; esac
if [[ $# -gt 1 ]]; then echo 'Too many arguments'; exit 2; fi
python="$root/.venv-linux2/bin/python"
if [[ ! -x "$python" ]]; then echo "Missing Linux2 environment: $python"; exit 1; fi
record="results/run_logs/$(date -u +%Y%m%dT%H%M%SZ)-full-cohort-closeout-$stage"
mkdir -p "$record" || exit 1
exec > >(tee "$record/console.txt") 2>&1
finish() {
  result=$?
  trap - EXIT
  "$python" - "$record/status.json" "$result" "$stage" <<'PY'
import json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(stage='closeout-'+sys.argv[3],exit_code=int(sys.argv[2]),
 finished_at=datetime.now(timezone.utc).isoformat(),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
PY
  echo "Closeout exit=$result. Commit results/full_sample/hierarchical/closeout and results/run_logs."
  exit "$result"
}
trap finish EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/work/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$root/work/numba}"
"$python" -c 'import platform,sys; assert platform.system()=="Linux"; assert sys.version_info >= (3,11)' || exit $?
if [[ -n "$(git status --porcelain --untracked-files=normal -- src scripts tests config stan pyproject.toml)" ]]; then
  echo 'Code/config has local changes; reconcile before closeout.' >&2
  exit 1
fi
"$python" -m pytest -q || exit $?
"$python" -m rf1_trust_socialvalue.full_sample_closeout "$stage"
exit $?
