#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib"
mkdir -p work results/tables results/figures
if [[ -x .venv/bin/python ]]; then export PATH="$PWD/.venv/bin:$PATH"; fi
[[ -d data/ds005123/.git ]] || { echo 'Run scripts/00_install_data.sh first.' >&2; exit 1; }
python -m pytest -q
for step in 01_build_trial_table 02_behavioral_analysis 03_fit_models 04_robustness 05_recovery 06_make_figures; do
  python "scripts/${step}.py" 2>&1 | tee "work/${step}.log"
done
python scripts/07_validate_outputs.py
