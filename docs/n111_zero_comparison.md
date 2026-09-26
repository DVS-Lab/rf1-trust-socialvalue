# N=111 stage B — one generic zero-option feature

**Initial batch received:** 11/12 fits passed in `e6eb3b3`. See the [scientific review](../results/n111_wrapup/zero_option_initial_review.md) and [one-fit retry command](n111_zero_retry.md). The original launch below is retained for reproducibility; it is not the next command for the existing Linux checkout. A separate explicitly partial Figure 3 is now available; the full figure still requires accepted sources for all comparisons.

Stage A completed on linux1 in commit `d596163` with all five audits successful. In held-out data, observed-history zero-minus-positive residual contrasts are approximately .18 for friends, .29–.30 for strangers, and .20 for computers across the focal models. All corresponding participant-bootstrap intervals exclude zero. The largest absolute conditional-versus-generative prediction difference across held-out partner × offer cells is .005252. These are aggregate descriptive contrasts; they do not identify a psychological mechanism, match offer amounts, or exclude individual/history-specific learning errors.

This evidence supports **testing** the single feature in the user brief. It does not establish retention. The decision is recorded in `config/n111_zero_decision.json`, bound to the exact stage-A evidence hashes.

## Exact change and priors

For each participant, add `gamma0 * I(low_option == 0)` to the existing choice logit. Gamma is common across partners and is not multiplied by kappa. Positive gamma favors a positive investment over $0. It is described as zero-option avoidance / positive-investment bias, without a stronger mechanistic interpretation.

Append one untransformed participant parameter to the existing correlated non-centered hierarchy. Its population mean has Normal(0,1) prior; its population SD has the existing half-Normal(0,.8) prior. The augmented correlation matrix retains LKJ(2). Other parameters retain their original population-location priors and transforms. This is one feature applied to H5, H7, HPreference and H8; no H2 extension, partner-specific gamma, age term, or rating model is added.

The new Stan, analytic-gradient header, Python simulator, outputs and caches are separate from the accepted implementations. Sixteen full-target/gradient checks cover each model, gamma absent, and negative/zero/positive gamma locations. A reference-autodiff implementation and Python likelihood are checked before Linux sampling. Unit tests also confirm reduction to the old models at gamma=0, an unscaled common logit increment, no future feedback, correct simulated feedback exposure, preservation of the original model specifications, and exclusion of stale failed-fit outputs.

## Why twelve new fits

- Four full-data **no-age baselines**, one per focal model. Existing full-data fits have age covariates and cannot serve as a matched no-age comparison.
- Four full-data no-age zero-option fits for PPCs and parameter stability.
- Four training no-age zero-option fits for temporal prediction. Existing accepted training no-age baselines are reused.

Each new fit uses four chains, 3,000 warmup and 4,000 retained draws per chain, adapt_delta=.99, maximum depth 12, diagonal metric and fixed deterministic seeds. Acceptance is unchanged: Rhat<1.01, bulk/tail ESS≥400, no divergences/depth hits and BFMI>.3 in every chain. Settings are fixed in `config/n111_zero.json`; there are no automatic retries. Passing caches are reused on rerun. An incomplete cache without its manifest stops rather than being silently replaced. A changed target/settings fingerprint also stops.

The default launch runs eight fits at a time, four simultaneous chains each (up to 32 sampling CPUs). New caches are ignored under `work/n111_wrapup/`; the two previously excluded fits are never touched. This batch stops after the matched comparison and does not launch realistic recovery.

## Linux launch

Enter/rejoin tmux:

```bash
tmux new-session -A -s rf1-n111
```

Then:

```bash
bash <<'BASH'
set -euo pipefail
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git pull --ff-only
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib" MPLBACKEND=Agg
python -m pytest -q
python -u scripts/21_n111_zero_option.py --run --jobs 8 --parallel-chains 4
BASH
```

Detach with Ctrl-b then d. The command first compiles and checks the new implementation on Linux, then starts the finite batch. `results/n111_wrapup/zero_option_status.json` records per-fit status. Per-fit live console logs are in `results/n111_wrapup/logs/zero-*/`; bounded sampler diagnostic evidence is exported under `results/n111_wrapup/sampler_evidence/`. The outer invocation also uses the established tracked run-log wrapper.

A diagnostically failed fit is excluded, independent fits continue, and the batch exits 2 after recording partial comparisons. There is no automatic escalation to achieve a passing count. Operational errors retain their logs. After the command exits, whether complete or partial, push the evidence:

```bash
git add results/n111_wrapup results/run_logs
git commit -m "Record N111 zero-option comparison and logs"
git push origin main
```

## Outputs and interpretation

`zero_option_report.md` summarizes current coverage and comparisons. `zero_option_model_comparison.csv` contains paired participant differences in log loss, Brier and secondary accuracy; negative loss/Brier differences favor the extension. Existing and extended training posteriors both predict through actual prior feedback with fixed training parameters and the same 4,017 held-out choices. All retained draws are used for predictive scoring. Bootstrap intervals (5,000 paired participant resamples) describe participant sampling variability conditional on the fitted models.

`zero_option_predictive_summary.csv` and `zero_option_ppc_comparison.csv` compare matched full-data no-age fits by partner, exact offer and zero-option status, in probability and investment units. Conditional intervals are posterior expected-probability intervals; generative intervals include replicated choices. Generative simulations use 400 posterior draws and matched random uniforms across base/extended counterparts to reduce simulation noise in their comparison. Observed means weight participants equally within cells.

`zero_option_parameter_summary.csv` reports subject-level posterior summaries including gamma. `zero_option_parameter_changes.csv` gives paired changes in existing parameter posterior means. These mean changes are descriptive comparisons of fits, not posterior distributions of a jointly estimated contrast. All diagnostics and source run labels are retained.

Figure 3 is generated only when all matched fits pass. It summarizes zero/positive-positive PPCs and paired prediction; exact-offer results remain in the tables. Retention requires joint scientific review of PPC repair, no material damage to positive-positive offers, held-out log loss/Brier, and parameter changes. No new feature or automatic recovery batch follows. Since this extension was motivated using N=111 diagnostics including held-out trials, its evaluation remains exploratory; the held-out data are not an untouched confirmatory test.
