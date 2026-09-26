# N=111 realistic mechanism recovery — finite structural screen

**Completed in results commit `46ad9c2`:** all 7,992 cases succeeded. The [final review](../results/n111_wrapup/realistic_recovery_review.md) stops without hierarchical confirmation. The launch below is retained for reproducibility, not a request for another run.

The [stage-B decision](../results/n111_wrapup/zero_option_decision.md) retains gamma0 as a candidate, with explicit remaining PPC errors. All twelve comparison sources are now accepted using the isolated HPreference depth-14 source. No further empirical posterior fits are requested.

## Generators and fixed scope

The primary family is H5/H7/HPreference/H8 with the shared zero-option term. The original four no-age models form a separate secondary family. Each family is fitted only against its corresponding four candidates: this addresses mechanism confusion conditional on the choice specification, not selection of gamma0 itself. All eight source fits are accepted full-data no-age fits. The original failed HPreference fit and H4_full_age/H5_train_age are excluded.

For each source, stream its retained chain CSVs and audit the manifest, target fingerprint, participant order, transitions, BFMI and file SHA-256. Use eight joint hyperparameter draws, two randomly selected retained draws from each of four chains. In each draw generate 111 new multivariate-normal participant effects on the latent scales using the same posterior mu, tau and Cholesky correlation; transform to the natural parameter scales. This is a posterior population-predictive generator, not resampling participant posterior means or estimating recovery under the low-theta prior alone.

Assign the new effects to the existing 111 participant schedules. Partner order, exact offers, programmed outcomes, run boundaries and missed-trial pattern remain unchanged. Simulated positive investments expose the programmed outcome; simulated zero and missed choices do not. No age effects or ratings enter these models, and no additional participants are accessed.

Eight empirical datasets per generator × four generators × two families give 64 complete N=111 datasets. In addition, two high-theta stress datasets for H5 and H7 in each family give eight more, for **72 datasets / 7,992 participant simulations** total. Stress changes only friend theta: 55 participants Uniform(5,10), 56 Uniform(10,20), randomly assigned. This intervention deliberately breaks some posterior correlations involving friend theta, is labeled as stress, and is never pooled with empirical results. Other parameters retain the joint generated values. No generating theta is clipped.

The empirical range table records min, 5th/50th/95th percentiles, max and tail coverage for every generated parameter and replicate. Stress guarantees explicit coverage of 5–10 and 10–20 even when zero adjustment lowers the empirical theta distribution.

## Fast fitting and evidence

Every simulated participant is fitted by all four candidates, with a full-task fit and a training-only temporal fit: **63,936 fast fits**, each with 32 seeded starts. This is finite multistart maximum likelihood, with no Stan/NUTS sampling. It reuses the established four likelihoods and feedback conventions through the validated zero-option engine; the added analytic gradient is checked against that engine, original-model likelihoods and numerical derivatives.

Fitting bounds start at theta [0,20], kappa [0,100], signed preference [-10,10], gamma0 [-10,10], alpha [0,1]. Before fitting, expand each shared upper magnitude to exceed the maximum corresponding realized generating parameter by 10% if needed. The preference magnitude cap is also at least the final theta cap, so it is not artificially narrower than a constant-p value contribution. The same resulting bounds apply to all candidates, regimes and replicates. This avoids silently truncating the generator or fitting below its upper tail. Upper-bound and lower-bound hits are exported. Wide bounds do not themselves establish parameter identification.

No fit receives the true generating parameters as an initialization. H7 receives an H5-derived nested start (stranger theta=0) within its 32 starts, and a check requires its optimized likelihood to be no worse than H5. L-BFGS-B convergence and near-best-start counts are recorded; a single bounded Powell fallback is allowed for an unresolved best solution. An unresolved optimizer failure is saved explicitly and excluded; it is not silently replaced by a worse converged solution.

AICc and BIC are reported both for individual participants and complete simulated datasets (sum of the individual criteria). Temporal log loss and Brier use training-fitted parameters held fixed while replaying simulated prior feedback through the same run-1→run-2 / single-run 65:35 split. Dataset predictive criteria average participant losses equally. These aggregate MLE comparisons are not hierarchical marginal-likelihood comparisons. Candidate ties within 1e-6 receive fractional selection weight.

Show both participant and dataset confusion. Eight empirical population draws per generator support a structural screen, not precise population-level recovery probabilities; participant cases within a replicate share hyperparameters. Do not treat 888 participant replications as independent population draws. No arbitrary rate cutoff declares identifiability. H7↔HPreference prediction differences and confusion with H8 are the central readouts.

## Run on linux1

Enter/rejoin tmux:

```bash
tmux new-session -A -s rf1-n111
```

Inside tmux:

```bash
bash <<'BASH'
set -euo pipefail
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git pull --ff-only
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib" MPLBACKEND=Agg
python -m pytest -q
python -u scripts/24_n111_realistic_recovery.py --run --jobs 40
BASH
```

Use `--jobs 32` instead if the server is shared. These are single-threaded worker processes, not chains. No large screen is run on the laptop. Detach with Ctrl-b then d. Runtime depends on optimizer difficulty; progress reports completed cases every 20 cases rather than claiming a fixed ETA.

Each participant simulation's eight fits (four full plus four training) are cached atomically under ignored `work/n111_wrapup/realistic_recovery/cases/`. The cache fingerprint binds actual posterior file hashes, generated populations, configuration and implementation. The same command resumes completed cases without refitting, including recorded error cases. An interruption can lose only currently unfinished cases. Changed fingerprints stop instead of overwriting results; partial case files are not treated as accepted results.

The initial stage streams the existing posterior files and draws new population parameters; it starts no sampling. `results/n111_wrapup/realistic_recovery_status.json` records preparation, case counts, errors and final state. All console output and exit status are logged under `results/run_logs/`. Error cases are preserved in `realistic_recovery_errors.json`. If some cases fail, partial inference is labeled and incomplete 111-person datasets are excluded from dataset-level confusion.

After the command exits (complete or partial):

```bash
git add results/n111_wrapup results/run_logs
git commit -m "Record N111 realistic recovery screen and logs"
git push origin main
```

## Outputs and stopping point

- `tables/realistic_model_recovery.csv`: accepted participant/candidate full and temporal fits, fitted parameters and optimizer diagnostics.
- `tables/realistic_model_recovery_confusion.csv`: primary/secondary, empirical/stress, participant/dataset, AICc/BIC/log-loss/Brier selection rates.
- `tables/realistic_recovery_dataset_scores.csv` and `realistic_recovery_H7_vs_HPreference.csv`: per-replicate scores and paired prediction differences.
- `tables/realistic_recovery_generating_ranges.csv` and `realistic_recovery_optimizer_audit.csv`: range coverage, near-best-start counts and boundary rates.
- `realistic_recovery_provenance.json`, `realistic_recovery_status.json`, `realistic_recovery_report.md`, and Figure 4 in PNG/PDF/SVG.

The batch **stops after the fast screen**. If H7 and HPreference are strongly confused, record that finding and avoid unnecessary hierarchical fits. If discrimination appears useful across criteria, parameter regimes and optimizer checks, review whether a limited 3–5-dataset hierarchical confirmation is informative. No automatic hierarchical confirmation or new model extension is implemented. Final scientific review and the compact N=111 synthesis follow the returned results.
