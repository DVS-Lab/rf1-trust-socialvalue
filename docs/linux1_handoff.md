# Linux continuation checkpoint — 24 September 2026

The laptop analysis processes were terminated at the user’s request; their completed disk caches remain available. This commit is a checkpoint, not the completed second-pass publication. Continue in:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git status --short
git pull --ff-only
```

Preserve any existing Linux changes before resolving a pull conflict. Do not force-push or reset either checkout. No connection to linux1 or remote execution was performed from the laptop.

## What is complete

The audited dataset remains **ds005123 v1.1.3**, primary N=111, rating-complete secondary N=103. The committed canonical trial/sample/rating tables are sufficient for the remaining computational analyses; no imaging is needed.

- Bound audit; M4–M7 theta ceilings 5/10/20; phi sensitivity and logistic-prior control.
- Theta≤10 parameter recovery: 50 repetitions per participant/model.
- Behavioral linear, age×time and quadratic GEE; 500 participant bootstraps.
- Validated full-data H2, H5, H8, HPreference, H7 and secondary H4 posteriors.
- Both H2 training-only age/no-age posteriors and their held-out scores.
- All 15 paired H5 hierarchical/MLE recovery datasets.
- Full analytic-gradient checks, posterior summaries, several figure previews and the six full-data trace figures.

Thus **23 of 34 requested posterior runs are complete**: 8 real-data fits and 15 recovery fits. `results/tables/hierarchical_fit_status.csv` is the status inventory; `results/hierarchical_checkpoint.json` records restart settings and exact input hashes.

## Remaining fits

| Run | Warmup / retained per chain | adapt_delta |
| --- | --- | --- |
| H5 full, theta≤10 | 3,000 / 8,000 | .99 |
| H5 full, quadratic age | 2,000 / 2,000 | .95 |
| H5 full, prior SDs ×1.5 | 2,000 / 2,000 | .95 |
| H5 training, age | 3,000 / 16,000 | .99 |
| H5 training, no age | 2,000 / 2,000 | .99 |
| HPreference training, age and no age | 2,000 / 2,000 each | .99 |
| H7 training, age | 3,000 / 8,000 | .99 |
| H7 training, no age | 2,000 / 2,000 | .95 |
| H8 training, age and no age | 2,000 / 2,000 each | .95 |

All runs use four chains. These are the settings reached at suspension, including diagnostic retries; do not automatically restart every difficult model at its already-failed shorter setting. The resume command uses these settings for absent caches and runs one fit at a time, with two simultaneous chains by default. Increase `--parallel-chains` only within the server's resource allocation.

The H5 training-age 8,000-draw attempt failed only the alpha–kappa population correlation threshold (R-hat 1.0116; bulk ESS 735; no divergences). Its 16,000-draw retry was unfinished. Existing `heldout_H5_train_age*` files and aggregated prediction tables therefore contain **provisional scores from that failed attempt**, not an accepted final comparison. Use the status inventory before interpreting any checkpoint table. Unsuffixed diagnostics for unfinished runs may likewise refer to their last completed failed attempt. Archived diagnostic attempts remain visible deliberately.

## Git results versus posterior caches

Git contains code, configuration, small result tables and figures. **It does not contain raw posterior chains, CmdStan builds, source data or live process state.** A Mac process suspended in memory cannot move to Linux.

Optional reuse of completed posteriors requires a separate transfer of the 23 completed `work/hierarchical/<run>/` directories from the laptop. Their posterior CSVs total approximately **4.25 GB before compression**. The JSON inventory lists exact manifest and CSV paths. Copy the complete run directories, including the `chain1`…`chain4` layout for HPreference and recovery `mle.csv` files. Do not copy top-level PID locks, the Mac executable, the Mac environment, or unfinished run directories. Keep failed/incomplete laptop attempts intact for provenance.

After copying, the resume command rebases manifest paths to this checkout without modifying CSV contents, priors, settings or target fingerprints. The original HPreference adaptation-source references are retained as provenance; its fresh retained chain files are the inference inputs.

**A Git pull alone preserves the published results but cannot reuse missing posterior draws.** Without the optional cache transfer, the command refits the completed hierarchical runs needed by finalization, including recovery posteriors. It does not rerun the expensive bound/MLE/behavioral analyses. This additional computation is explicit, not a hidden cache hit.

## Environment and execution

Use Python 3.13 if available to match the recorded environment. Create a Linux environment rather than copying the Mac virtual environment:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-hierarchical.lock.txt
python -m pip install -e '.[hierarchical,test]'
python -c 'import cmdstanpy; cmdstanpy.install_cmdstan(version="2.40.0", dir="work/cmdstan", cores=4)'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib"
pytest -q
python scripts/15_resume_checkpoint.py
```

The last command is a read-only plan. It validates hashes and lists present versus absent caches; it never starts sampling by default. Before fitting, compile and check the Linux implementation using `scripts/check_stan_likelihood.py`, `scripts/check_fast_gradient.py`, and `scripts/check_full_hierarchy_gradient.py`.

Launch the actual continuation through the site's normal scheduler or a persistent server session:

```bash
python -u scripts/15_resume_checkpoint.py --run --parallel-chains 2
```

This command is guarded against running on macOS. It prepares optional copied caches, fits/reuses the real-data posteriors, regenerates missing recovery posteriors, finalizes posterior summaries/comparisons/traces, regenerates figures/report/gallery, and runs `13_validate_second_pass.py --hierarchical`. A fit that still fails diagnostics is recorded as failed, and independent fits continue. Finalization stays blocked until every required fit passes. Operational errors still stop the runner. No automatic Git push occurs.

Do not start the old `98_run_hierarchical.sh` from scratch merely to continue: it also repeats already-completed nonhierarchical analyses. Do not start laptop fitting while Linux writes the replacement results.

## Final review and publication

1. Require R-hat<1.01, bulk/tail ESS≥400, zero divergences/depth hits, and every chain BFMI>.3. Report any persistent failure explicitly.
2. Inspect prediction rankings with paired uncertainty; age/no-age gains; H5 bounded, quadratic and broader-prior sensitivity; generative predictive misfit; and theta versus choice-effect uncertainty.
3. Regenerate and visually inspect figures 00 and 14–22 and representative trace/PDF exports. Figures 18–19 and 22 currently include incomplete previews. The existing main report/gallery retain the historical first pass, now labeled with this checkpoint notice.
4. The current recovery gains apply to a low/moderate simulated theta regime (median 1.29, maximum 4.36), below many empirical H5 estimates (median posterior mean 6.48). Keep that limitation and the small five-dataset-per-condition coverage caveat prominent. Rating timing remains unresolved and H4 remains secondary.
5. Run `pytest`, `python scripts/13_validate_second_pass.py --hierarchical`, and the historical `python scripts/07_validate_outputs.py` where the audited dataset checkout is installed. That last audit deliberately requires unfetched imaging placeholders; to reproduce it on Linux, install the pinned behavioral-only snapshot using `00_install_data.sh`. Do not pretend an absent raw-data audit has run.
6. Replace the checkpoint notice with final status only after all required work is complete, update provenance, inspect staged files, commit and push directly to `origin/main` without force. Keep raw data, posterior chains and binaries out of Git.

The requested final interpretation separates behavioral age moderation, computational age effects and mechanism. Theta=10 is inadequate as a scale solution; hierarchy alone does not establish reciprocation-specific social reward. Full 12-model recovery remains deliberately deferred because the raw-theta formulation is unresolved.


## Linux failure on 25 September and restart

Linux commit `fcef2ea` contains successful fresh full-data H2/H5/H8 fits. HPreference's .95 attempt narrowly missed precision thresholds (max R-hat 1.01015, minimum bulk ESS 399.762), with zero divergences and zero maximum-depth hits. The old automatic .99 retry hit maximum depth 3,635 times out of 8,000 retained draws. The runner then intentionally raised a diagnostic failure; this was not a demonstrated hardware or memory crash.

The revised policy lengthens chains when only precision thresholds fail, preserving the integrator setting. `--retry-run HPreference_full_age` explicitly preserves the failed current cache and starts 3,000 warmup / 8,000 retained draws per chain at .95 from the geometry-clean attempt's settings. All four chains adapt from scratch; none of the old retained draws are mixed in. The posterior target, priors and acceptance thresholds are unchanged. This is a proposed sampler remedy, not a claim that the new fit already passes.

On Linux, enter or create a persistent session:

```bash
tmux new-session -A -s rf1-trust
```

Then run this block inside tmux using the environment created earlier:

```bash
bash <<'BASH'
set -euo pipefail
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git pull --ff-only
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib" MPLBACKEND=Agg
python scripts/16_collect_run_logs.py
python -u scripts/15_resume_checkpoint.py --run --parallel-chains 2 --retry-run HPreference_full_age
BASH
```

Detach with Ctrl-b, then d; reconnect with `tmux attach -t rf1-trust`. Use `--retry-run` for this explicit restart only; ordinary later continuation uses the same command without that option. A passing cache cannot be replaced using this flag. Completed H2/H5/H8 posteriors are reused from the server's disk caches.

Every new `--run` invocation now records combined stdout/stderr in `results/run_logs/<UTC-id>/console.txt` and the command, code revision, timestamps and exit code in `run.json`. While running, `results/linux_run_status.json` distinguishes pending/running/complete/diagnostic_failed/error runs. On normal exit or a caught error, sampler diagnostic text and manifest settings are also exported. Sudden machine termination can leave a run record marked running; inspect the process before treating that as a live job.

The read-only collector imports the existing `work/logs/linux-resume-*.log` tails and CmdStan console/diagnostic text into Git-visible `.txt` files. Large imported logs are explicitly marked as truncated; complete originals remain under `work/`. Raw posterior CSVs and binaries remain excluded. Logs are not pushed automatically. After the job ends (whether successful or failed), publish the evidence with:

```bash
python scripts/16_collect_run_logs.py
git add results
git commit -m "Record Linux continuation results and run logs"
git push origin main
```

The original `hierarchical_checkpoint.json` and `hierarchical_fit_status.csv` remain the dated laptop inventory, not the live Linux status. A final report must not be published while the live status records unresolved diagnostic failures.
