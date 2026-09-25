# Linux failure evidence received 25 September 2026

Source: user-pasted terminal excerpt and committed diagnostic tables in `fcef2ea`. The original full console log has not yet been transferred from Linux. This file is a diagnostic note, not a reconstructed raw log.

The terminal reported non-fatal `lkj_corr_cholesky_lpdf` proposal exceptions and then raised `RuntimeError: Final diagnostic thresholds still fail: HPreference_full_age; inspect before interpreting` in the checkpoint runner.

| HPreference full-data attempt | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Depth hits | Min BFMI |
| --- | --- | --- | --- | --- | --- | --- |
| adapt_delta=.95, 2,000 retained/chain | 1.01015 | 399.762 | 922.32 | 0 | 0 | .60867 |
| adapt_delta=.99, 2,000 retained/chain | 1.00744 | 503.047 | 926.132 | 0 | 3,635 | .63074 |

The latest fit failed the predeclared zero-depth-hit requirement. H2, H5 and H8 fresh Linux full-data diagnostics passed. Later fit results in the repository can still be older laptop results and must not be treated as evidence that Linux completed those fits.

The revised runner keeps scientific gates unchanged, tries longer chains before tightening the integrator for precision-only failures, records tracked logs and status, and continues independent fits after diagnostic failures. The explicit preference retry starts at .95 with 3,000 warmup and 8,000 retained iterations per chain. Its actual outcome still requires execution on Linux.

Stan distinguishes maximum-depth warnings (an efficiency concern) from divergences and poor mixing: https://mc-stan.org/learn-stan/diagnostics-warnings.html . This project's stricter zero-hit acceptance requirement has not been relaxed.
