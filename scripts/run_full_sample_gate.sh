#!/usr/bin/env bash
# Freeze / verify / describe. No Stan installation or sampling in this gate.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
stamp="$(date -u +%Y%m%dT%H%M%SZ)-full-sample"
record="results/run_logs/$stamp"
mkdir -p "$record"
exec > >(tee "$record/console.txt") 2>&1
finish() {
  result=$?
  trap - EXIT
  python3 - "$record/status.json" "$result" <<'PY'
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(exit_code=int(sys.argv[2]),
    finished_at=datetime.now(timezone.utc).isoformat(),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
PY
  echo "Integration exit=$result. Preserve $record and results/full_sample when committing."
  exit "$result"
}
trap finish EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/work/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$root/work/numba}"
python3 -c 'import sys; assert sys.version_info >= (3,11), "Activate the Linux2 Python environment first"'
if [[ -n "$(git status --porcelain --untracked-files=normal -- src scripts tests config pyproject.toml)" ]]; then
  echo 'Scientific code/config checkout has local changes; reconcile before freezing.' >&2
  exit 1
fi
python3 -m rf1_trust_socialvalue.full_sample preflight
python3 -m pytest -q
if [[ ! -f results/full_sample/provenance.json ]]; then
  python3 -m rf1_trust_socialvalue.full_sample freeze
else
  python3 -m rf1_trust_socialvalue.full_sample verify
fi
python3 -m rf1_trust_socialvalue.full_sample run
