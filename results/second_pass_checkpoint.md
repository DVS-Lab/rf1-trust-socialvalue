# Second-pass checkpoint — 25 September 2026

**31 of 34 Linux posterior runs passed diagnostics.** The parallel batch finished in approximately 68 minutes (15:07–16:15 UTC). It completed its queue and deliberately blocked final reporting because the following fits each retained one divergent transition:

| Fit | Retained draws, all chains | Max R-hat | Min bulk ESS | Min tail ESS | Divergences |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4 full age | 8,000 | 1.00258 | 958.5 | 1600.8 | 1 |
| H5 training age | 64,000 | 1.00438 | 1571.1 | 2073.3 | 1 |
| HPreference training no age | 8,000 | 1.00987 | 425.2 | 493.5 | 1 |

All three passed the tree-depth and BFMI requirements and every recorded R-hat/ESS threshold. They remain unaccepted under the zero-divergence criterion. This is not evidence of an out-of-memory or worker crash. Evidence is committed in `71504f8`.

- [Live Linux run status](linux_run_status.json)
- [Parallel run console and exit record](run_logs/20260925T142200Z-b6bf059e/console.txt)
- [Per-fit logs](run_logs/parallel-20260925T150717Z-825cb3b8)
- [Linux continuation instructions](../docs/linux1_handoff.md)
- [Explicit three-fit retry plan](../config/linux_divergence_retry.json)

The targeted retry preserves the original chains and settings, retains the seed, priors and model, and uses adapt_delta=.995 with 4,000 warmup iterations. H4 and preference no-age use 4,000 retained draws per chain; H5 training age keeps 16,000. Passing fits are reused. Divergent recorded states and population-parameter percentiles will be exported from the archived attempts on Linux for follow-up inspection. The revised sampler has not yet been run or validated on Linux; persistent divergences require further investigation, not repeated seed changes or a relaxed acceptance rule.

The original [fit inventory](tables/hierarchical_fit_status.csv) and [restart checkpoint](hierarchical_checkpoint.json) describe the 24 September laptop snapshot, not current Linux completion. The laptop processes were terminated; completed caches remain on disk. Raw posterior draws are excluded from Git.

The final second-pass overview, report and gallery remain outstanding. H5 training-age prediction files from an earlier failed attempt remain provisional, and the preference no-age comparison is incomplete. Existing figure previews must not be presented as the completed second-pass analysis. Full 12-model recovery remains deliberately deferred pending resolution of raw-theta scaling. Rating timing is unresolved and H4 remains secondary.
