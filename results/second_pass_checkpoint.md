# Second-pass checkpoint — audit reviewed; accepted-fit report available

**32 of 34 Linux posterior runs passed diagnostics.** The targeted retry resolved HPreference training no-age. H4 full age and H5 training age still have one divergent transition each. Their other diagnostic checks pass. The targeted batch ran from 16:27 to 18:11 UTC on 25 September (approximately 104 minutes).

| Remaining fit | Total retained draws | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Depth hits | Min BFMI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H4 full age | 16,000 | 1.00322 | 1656.8 | 3325.1 | 1 | 0 | .673 |
| H5 training age | 64,000 | 1.00497 | 1776.7 | 2740.2 | 1 | 0 | .759 |

Both fits used adapt_delta=.995, four chains and 4,000 warmup iterations per chain. Their zero-divergence acceptance requirement remains unmet. Good mixing metrics and small divergence counts do not establish unbiased inference. The larger integrator target did not eliminate the problem, so another automatic tuning escalation is not justified by the current evidence.

- [Live Linux status](linux_run_status.json)
- [Latest console](run_logs/20260925T162743Z-ca98702e/console.txt)
- [Latest per-fit logs](run_logs/parallel-20260925T162758Z-4cef5a2d)
- [Linux diagnostic audit instructions](../docs/linux1_handoff.md)

The audit in commit `7e47807` is complete: the four custom-versus-reference gradient checks agree to maximum scaled error 2.81e-14. The largest population-mean changes between attempts are 0.050 posterior SD for H4 and 0.023 SD for H5. No obvious separated chain appears in the checked traces. These checks do not clear the remaining divergences: recorded and preceding states are not the unrecorded intermediate trajectory failure points. See [audit outputs](diagnostic_review/README.md).

A [scoped accepted-fit review](review/README.md) now provides six figure sets, paired temporal prediction, joint H5 age effects and sensitivities, posterior predictive checks, θ-bound comparisons, and recovery recomputed from all 15 accepted per-run files. H4 inference and H5 age-model held-out scores are explicitly excluded. All 32 accepted fits contribute to the review; the original 34-fit finalizer remains blocked. No automatic sampling retry is recommended from this audit alone.

No priors, likelihoods, seeds, or acceptance thresholds were changed in response to these latest failures. The 32 passing caches are preserved. H4 remains a secondary analysis with unresolved rating timing. H5 training-age predictive results remain excluded from the new review. The full second-pass overview/report/gallery are still blocked; the new review is clearly labeled as partial, and old reports must not be presented as final.

The [original fit inventory](tables/hierarchical_fit_status.csv) and [original restart checkpoint](hierarchical_checkpoint.json) describe the 24 September laptop snapshot, not live Linux completion. Laptop fitting has been terminated; raw posterior chains remain outside Git.
