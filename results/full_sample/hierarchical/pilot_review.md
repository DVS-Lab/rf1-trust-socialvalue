# Full-cohort pilot review, 2026-09-27

All four fits completed sampling for N=343 (26,560 valid choices). Three pass the unchanged diagnostic gate. HPreference fails only bulk ESS for two unique population correlations.

| Model | Sampling minutes | Max Rhat | Min bulk ESS | Min tail ESS | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Full_H2_base_noage | 20.5 | 1.00488 | 466.6 | 697.1 | complete |
| Full_H5_zero_noage | 87.0 | 1.00946 | 477.7 | 770.5 | complete |
| Full_H7_zero_noage | 149.7 | 1.00590 | 597.1 | 1241.8 | complete |
| Full_HPreference_zero_noage | 173.1 | 1.00877 | 329.4 | 587.0 | diagnostic_failed |

All fits have zero divergences and zero maximum-treedepth hits; minimum BFMI ranges from 0.623 to 0.775. All 20 target/gradient and Python-likelihood checks passed. Diagnostic and accepted-summary hashes were checked against the pushed statuses.

## The one diagnostic failure

`Omega[1,3]` (alpha versus friend preference) has bulk ESS 329.406; `Omega[1,4]` (alpha versus stranger preference) has 358.074. Their mirrored matrix entries repeat the same two correlations. Every other checked quantity passes. The required minimum is 400, even though CmdStan’s general diagnose report says no problems detected.

A single reviewed retry doubles retained draws from 2,000 to 4,000 per chain. It preserves four chains, 2,000 warmup, the effective sampler seed, priors, metric, adapt_delta, treedepth and acceptance criteria. New outputs use `Full_HPreference_zero_noage_draws4000`; no failed posterior is merged with the retry or exported as accepted inference. The original attempt and three accepted fits remain unchanged.

The configuration pins the failed attempt’s status, diagnostic and manifest hashes. It rejects additional attempts, settings/prior drift and evidence drift. Retry setup evidence is written separately under `hierarchical/retries/Full_HPreference_zero_noage_draws4000/`.

## Runtime and next stage

The slowest HPreference chain took about 87.7 minutes warmup and 85.3 minutes sampling. Keeping warmup and doubling sampling suggests about 4.3 hours at the same per-iteration speed; this is an estimate, not a deadline or guarantee of adequate ESS. Peak Python worker RSS was roughly 0.6–1.2 GiB; these per-process peaks are not a measurement of total concurrent memory.

Retain H2, H5z and H7z as accepted full-cohort fits. Inspect the single retry before expanding the matched base/zero full-data and run-1→run-2 batch. The cohort freeze and N111 endpoint do not change. Numerical convergence does not establish predictive adequacy or select a psychological mechanism; heldout scoring and PPCs remain pending.
