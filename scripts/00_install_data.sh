#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v datalad >/dev/null || { echo 'Install DataLad and git-annex first.' >&2; exit 1; }
mkdir -p data
if [[ ! -d data/ds005123/.git ]]; then
  datalad install -s https://github.com/OpenNeuroDatasets/ds005123.git data/ds005123
fi
[[ -z "$(git -C data/ds005123 status --porcelain --untracked-files=no)" ]] || { echo 'Dataset has modifications; stopping.' >&2; exit 1; }
git -C data/ds005123 fetch origin tag 1.1.3
git -C data/ds005123 checkout --detach 1.1.3
[[ "$(git -C data/ds005123 rev-parse HEAD)" == a3213b56b7bd27d7e3ac10577558eb26bb7c2a61 ]] || exit 1
python - <<'PY'
from pathlib import Path
import csv, subprocess
r=Path('data/ds005123')
# Explicit, extension-allowlisted small files only. Never get a directory.
files=[r/'participants.tsv',r/'dataset_description.json']
base=r/'sourcedata/Scan-Investment_Game'
files += [base/n for n in ['investment_game.py','investment_ratings.py','genTrustDesign.m']]
with (r/'participants.tsv').open() as f:
    ids=[x['participant_id'] for x in csv.DictReader(f,delimiter='\t')]
for sub in ids:
    files += sorted((r/sub/'func').glob('*task-trust*_events.tsv'))
    files += sorted((base/'logs'/sub[4:]).glob('*.csv'))
assert all(p.suffix in {'.tsv','.csv','.json','.py','.m'} for p in files)
subprocess.run(['datalad','get','-J','8',*map(str,files)],check=True)
PY
