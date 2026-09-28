#!/usr/bin/env bash
# Reviewed full/run-1 batch with up to 64 active chains. No interactive shell options are changed.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
phase_config="config/full_sample_batch.json"
stage="full-cohort-batch"
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/run_full_sample_batch.sh' >&2
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
  echo "Batch exit=$result. Preserve results/full_sample/hierarchical and $record when committing."
  exit "$result"
}
trap finish EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/work/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$root/work/numba}"
python3 -c 'import platform,sys; assert platform.system()=="Linux"; assert sys.version_info >= (3,11)'
if [[ -n "$(git status --porcelain --untracked-files=normal -- src scripts tests config stan pyproject.toml)" ]]; then
  echo 'Code/config has local changes; reconcile before sampling.' >&2
  exit 1
fi
python3 -m pytest -q
python3 -m rf1_trust_socialvalue.full_sample_batch run --config "$phase_config"
