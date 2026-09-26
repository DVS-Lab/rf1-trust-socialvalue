# N=111 zero-option comparison

Partial results: one or more new fits failed or are unavailable. Failed fits are excluded; no automatic retries were attempted.

This is one common zero-option logit term, partially pooled across participants and shared across partners. New full-data no-age base fits provide the matched PPC/parameter comparison; existing accepted no-age training fits provide the predictive baseline. Negative log-loss/Brier differences favor the extension; positive accuracy differences favor it. Intervals are paired participant bootstrap intervals, conditional on the fitted posteriors.

| model | metric | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| H5 | log_loss | -0.08563 | -0.10409 | -0.06675 |
| H5 | brier | -0.03664 | -0.04386 | -0.02938 |
| H5 | accuracy | 0.12324 | 0.10182 | 0.14516 |
| H7 | log_loss | -0.07017 | -0.08662 | -0.05392 |
| H7 | brier | -0.03027 | -0.03685 | -0.02384 |
| H7 | accuracy | 0.08980 | 0.07275 | 0.10762 |
| HPreference | log_loss | -0.06694 | -0.08425 | -0.05001 |
| HPreference | brier | -0.02898 | -0.03552 | -0.02244 |
| HPreference | accuracy | 0.08807 | 0.06914 | 0.10667 |
| H8 | log_loss | -0.06071 | -0.07699 | -0.04511 |
| H8 | brier | -0.02489 | -0.03188 | -0.01834 |
| H8 | accuracy | 0.05103 | 0.02994 | 0.07386 |

## Parameter and model-fit checks

[Parameter summaries](tables/zero_option_parameter_summary.csv), [comparison status](tables/zero_option_model_comparison.csv), and per-fit diagnostics are available in tables/. Full-data conditional intervals describe posterior expected probabilities; generative intervals include replicated choice variation. Comparisons remain exploratory because the zero feature was motivated by this dataset, including held-out diagnostics; these scores are not an untouched confirmatory test.

## Pending scientific decision

Review exact-offer and zero/positive-positive PPC changes alongside held-out log loss/Brier and parameter shifts before deciding retention. No automatic retention rule or additional model extension is applied. The targeted realistic recovery screen is the next stage after that decision; it has not run. Age and rating conclusions, N=111 scope, and exclusion of H4_full_age/H5_train_age remain unchanged.
