#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib"
if [[ -x .venv/bin/python ]]; then export PATH="$PWD/.venv/bin:$PATH"; fi
python -m pytest -q
python scripts/08_bound_sensitivity.py
python scripts/08_bound_sensitivity.py --recovery
python scripts/08_age_behavior.py
python scripts/check_stan_likelihood.py
python scripts/check_fast_gradient.py
python scripts/check_full_hierarchy_gradient.py
for model in H2 H5 H8 HPreference H7 H4; do
  python scripts/09_fit_hierarchical.py --model "$model" --prior
done
python scripts/09_fit_hierarchical.py --model H5 --prior --bounded
python scripts/09_fit_hierarchical.py --model H5 --prior --age-terms 2
python scripts/09_fit_hierarchical.py --model H5 --prior --prior-scale 1.5
python scripts/12_run_hierarchical_suite.py
python scripts/10_hierarchical_validation.py
python scripts/14_finalize_hierarchical.py
python scripts/11_hierarchical_figures.py
python scripts/07_validate_outputs.py
python scripts/13_validate_second_pass.py --hierarchical
