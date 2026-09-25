# Second-pass checkpoint — latest results add7e92

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

The next command, `scripts/18_audit_remaining.py`, reads the current failed posterior caches without changing them or starting MCMC. It exports current flagged and previous states, population-parameter locations, chain diagnostics, plots, comparisons against the archived .99 attempt, and exact analytic-gradient versus reference-autodiff checks at the recorded states. The failing intermediate trajectory states are not available in ordinary CmdStan CSV output; the audit cannot directly test those unavailable points or prove that a divergence is harmless.

The existing `divergence_locations_linux_divergences_20260925_*` tables refer to the earlier .99 attempts. They must not be mistaken for locations from the latest .995 fits. The new audit output under `results/diagnostic_review` has not yet been generated on Linux.

No priors, likelihoods, seeds, or acceptance thresholds were changed in response to these latest failures. The 32 passing caches are preserved. H4 remains a secondary analysis with unresolved rating timing. H5 training-age predictive results remain provisional. The final second-pass overview/report/gallery are still blocked; old report/gallery and figure previews must not be presented as final.

The [original fit inventory](tables/hierarchical_fit_status.csv) and [original restart checkpoint](hierarchical_checkpoint.json) describe the 24 September laptop snapshot, not live Linux completion. Laptop fitting has been terminated; raw posterior chains remain outside Git.
