# N=111 closeout — staged Linux handoff

**Stage A has completed successfully in `d596163`. The next execution is the [stage B zero-option comparison](n111_zero_comparison.md).** The commands below document the completed stage A and are retained for reproducibility.

The [closeout scope](n111_wrapup_scope.md) is fixed. This first command performs **stage A only**: read accepted no-age training posterior CSVs, compute conditional and fully generative predictions, and stop. It never launches Stan, fits models, retries failed fits, changes posterior caches, or accesses another dataset. Four hundred draws balanced across the existing chains are used for each of five models; five independent processes can process the models in parallel. More CPUs are not needed for this stage.

The local preview verifies the motivating zero-option pattern using committed full-age PPC tables. It is intentionally distinct from the new no-age history comparison. The full closeout remains pending until stage A is interpreted and the conditional extension/recovery stages have been completed or explicitly ruled out.

## Run on linux1

From a login on linux1, open or rejoin a persistent session:

```bash
tmux new-session -A -s rf1-n111
```

Inside tmux:

```bash
bash <<'BASH'
set -euo pipefail
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git status --short
git pull --ff-only
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib" MPLBACKEND=Agg
python -m pytest -q
python -u scripts/20_n111_history_diagnostics.py --run --jobs 5 --draws 400
BASH
```

Use the existing working Linux environment; no dependency installation or Python 3.13 command is needed. If Git reports conflicting local edits, preserve them before resolving the pull. Detach with Ctrl-b then d; reattach with `tmux attach -t rf1-n111`.

The reader checks the fixed sample/trial hashes, current completion status, numerical acceptance gate, posterior likelihood/settings fingerprint, chain counts, and all retained transitions for divergences/depth hits. It hashes each posterior file before and after reading. Missing/stale files stop the job; there is no fallback refit. It refuses another active coordinator or live fit lock. A normal sampling process on this checkout should have finished before running it.

The command records combined stdout/stderr and exit status under `results/run_logs/`. Each independent model writes its own files; aggregate tables and figures are written only if all five succeed. `results/n111_wrapup/audit_status.json` is the current execution record. If an error occurs, share the status/logs rather than starting another sampler run.

## Return the results, even if the audit fails

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git add results/n111_wrapup results/run_logs
git diff --cached --stat
git commit -m "Record N111 history diagnostics and run logs"
git push origin main
```

No raw chains or original data are staged by those paths. Do not use `git add work` or `git add data`.

Successful stage A produces:

- `results/n111_wrapup/tables/predictive_residuals.csv`
- `results/n111_wrapup/tables/conditional_vs_generative.csv`
- `results/n111_wrapup/tables/zero_option_residual_contrasts.csv`
- corresponding per-model and participant-level derived summaries;
- figure 1 (observed versus predicted by offer) and figure 2 (history residual comparison), each PNG/PDF/SVG;
- a clearly partial `results/n111_wrapup/README.md`, plus per-model posterior hash evidence.

Main figures show held-out choices only. Tables include all/training/held-out periods separately. Means weight participants equally within cells; previous-feedback labels describe actual information available before the trial. Conditional replicated choices do not feed back into learning. Generative simulations use their own feedback exposure, starting at the beginning of each participant's sequence.

## Reproduce the committed-table preview locally

```bash
.venv/bin/python scripts/20_n111_history_diagnostics.py --published-check
```

This reads no posterior CSVs and creates only the published full-age offer check and its provenance/figure. It does not claim the new history diagnostic has run.

## Subsequent stopping points

After reviewing stage A, implement only the shared gamma0 extension if warranted. Compare matched no-age base/extended models before deciding retention. Then run the fast realistic recovery screen; hierarchical confirmation is conditional on useful discrimination, not automatic. Preserve the strict failed-fit exclusions throughout. The final report and decision record are completed only after these questions are resolved.
