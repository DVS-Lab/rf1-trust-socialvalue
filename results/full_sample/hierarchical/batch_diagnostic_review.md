# Full-cohort batch diagnostic review

Evidence: Linux2 results pushed in `f08e77f`. All 16 new samplers completed; seven passed and nine failed diagnostic thresholds. With the four accepted pilots, 11 of 20 targets are accepted. No fit hit maximum treedepth; minimum BFMI across this batch exceeded .61. Failed posteriors have not been used for scientific comparisons.

| Fit | Rhat max | Bulk ESS min | Tail ESS min | Divergences | Status |
|---|---:|---:|---:|---:|---|
| Full_H2_zero_noage | 1.03741 | 186 | 507 | 0 | diagnostic_failed |
| Full_H5_base_noage | 1.00431 | 488 | 265 | 0 | diagnostic_failed |
| Full_H7_base_noage | 1.00665 | 607 | 1467 | 0 | complete |
| Full_H8_base_noage | 1.00629 | 618 | 1184 | 0 | complete |
| Full_H8_zero_noage | 1.00970 | 603 | 1120 | 0 | complete |
| Full_HPreference_base_noage | 1.00487 | 765 | 1651 | 1 | diagnostic_failed |
| Train_H2_base_noage | 1.01234 | 397 | 472 | 0 | diagnostic_failed |
| Train_H2_zero_noage | 1.04153 | 182 | 634 | 0 | diagnostic_failed |
| Train_H5_base_noage | 1.00397 | 580 | 1025 | 1 | diagnostic_failed |
| Train_H5_zero_noage | 1.01358 | 352 | 896 | 0 | diagnostic_failed |
| Train_H7_base_noage | 1.00935 | 502 | 1300 | 0 | complete |
| Train_H7_zero_noage | 1.00668 | 796 | 1383 | 0 | complete |
| Train_H8_base_noage | 1.00808 | 466 | 1022 | 0 | complete |
| Train_H8_zero_noage | 1.01263 | 522 | 1050 | 0 | diagnostic_failed |
| Train_HPreference_base_noage | 1.01098 | 685 | 1370 | 1 | diagnostic_failed |
| Train_HPreference_zero_noage | 1.00478 | 1063 | 2526 | 0 | complete |

## Reviewed response

- Mixing: Full H2 zero; Train H2 base/zero, H5 zero, and H8 zero. Full/Train H2 zero have the largest Rhat (1.037/1.042), concentrated in population correlations involving learning rate. These two receive 8,000 retained draws per chain; other retries receive 4,000.
- Tail ESS: Full H5 base has two failing natural parameters for one participant. It receives more draws; priors and parameter definitions stay fixed.
- Divergences: Full HPreference base, Train H5 base and Train HPreference base each have exactly one divergence. Train HPreference base also has Rhat 1.01098. These three use adapt_delta .995 (previously .99).
- Every retry uses eight chains and 4,000 warmup, up to 72 active sampling cores. Keep depth14, diag_e, seeds, data, model, priors and all diagnostic thresholds. One explicit retry under a new name per failed fit; do not combine attempts or rerun the 11 accepted fits.

The plan follows [Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html) and the [CmdStan diagnose guide](https://mc-stan.org/docs/cmdstan-guide/diagnose_utility.html). Longer chains do not guarantee resolved mixing, and stricter adaptation does not guarantee elimination of divergences. Persistent failures require chain/geometry inspection rather than repeated blind extensions.

For a failed retry, portable chain means/SDs and trace excerpts (200 evenly spaced retained draws per chain plus every divergent draw) are saved with hashes. Full posterior CSVs remain in ignored work storage. The excerpts are diagnostic aids, not a replacement for full-draw diagnostics.

The historical `batch/status.json` remains unchanged. Retry status, implementation checks, and eventual combined accepted/replacement comparisons go under `batch_retry/`.
