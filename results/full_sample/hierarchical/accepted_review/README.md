# Accepted full-cohort results after the targeted retry

18/20 fits pass all predeclared diagnostics; 9/10 training models can be compared on 304 participants and 12,494 run-2 choices.

Two fits remain unaccepted: `Train_H5_base_noage_retry1` and `Full_HPreference_base_noage_retry1`. Each has exactly one divergent transition among 32,000 retained draws; all Rhat, ESS, treedepth and BFMI checks pass. The zero-divergence threshold remains unchanged. No scores or scientific parameter summaries from these fits enter this report.

| Model | Variant | Mean log loss | Difference from H2 base | 95% paired interval |
|---|---|---:|---:|---|
| H7 | zero | 0.5055 | -0.1753 | [-0.1961, -0.1541] |
| HPreference | zero | 0.5118 | -0.1690 | [-0.1896, -0.1488] |
| H5 | zero | 0.5295 | -0.1513 | [-0.1684, -0.1341] |
| H7 | base | 0.5567 | -0.1241 | [-0.1427, -0.1051] |
| H8 | zero | 0.5577 | -0.1231 | [-0.1433, -0.1025] |
| HPreference | base | 0.5629 | -0.1179 | [-0.1366, -0.0988] |
| H8 | base | 0.5961 | -0.0847 | [-0.1035, -0.0662] |
| H2 | zero | 0.6014 | -0.0794 | [-0.0910, -0.0686] |
| H2 | base | 0.6808 | 0.0000 | [0.0000, 0.0000] |

## Adding the zero-option term

| Model | Zero minus base log loss | 95% paired interval |
|---|---:|---|
| H2 | -0.0794 | [-0.0910, -0.0686] |
| H7 | -0.0512 | [-0.0601, -0.0426] |
| HPreference | -0.0512 | [-0.0602, -0.0423] |
| H8 | -0.0383 | [-0.0459, -0.0309] |

H7 zero minus HPreference zero: -0.00628, 95% paired interval [-0.00887, -0.00364].

Intervals describe participant sampling variation and are not adjusted for multiple comparisons. Prediction uses observed feedback online, with run-1 posterior parameters fixed; it is not a joint run-2 marginal likelihood. Similar predictive scores cannot establish equivalent or distinguishable psychological mechanisms.

The two failed fits need targeted geometry inspection before another sampling change. The saved thinned global traces show similar chain locations but cannot rule out local geometry problems. Inspect all participant-level states at the divergent iterations and their chain context from existing Linux2 posterior files; this requires no new sampling.

This results-only review verifies published diagnostics, manifests, summaries and heldout-score hashes. It does not claim to revalidate the ignored Linux2 canonical data or raw posterior CSVs locally.
