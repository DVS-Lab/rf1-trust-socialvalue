# Full-cohort hierarchical Trust DDM assessment

The November 4–18, 2026 Linux2 run retained all 343 behavioral participants and ended on November 18 at 2:00 p.m. Eastern with exit code 2. All five observed-data fits either raised an execution error or failed sampler diagnostics. Recovery stopped before the first simulated dataset. [Launcher exit](last_run.json) · [Observed fits](fit_status.json) · [Recovery status](recovery_status.json)

**The individual friend-minus-stranger starting-point contrast remains unvalidated.** All 85 people who always chose the higher or lower investment for friend or stranger were included. All 343 people contributed at least one modeled decision, with 26,494 decisions across 646 retained runs. There is no hierarchical recovery correlation, RMSE, or interval coverage to report for either the full cohort or the constant responders. These results give no basis for attributing the failure to those 85 participants or excluding them. [Cohort counts](cohort_summary.json)

| Observed-data specification | Result |
|---|---|
| Shared starting point, partner-specific starting points (primary), wider priors, imaging sensitivity | Each recorded an execution failure. The aggregate records identify the exception class as `RuntimeError`. The private Linux2 launcher log contains the error messages. |
| Partner-specific starting points with feedback history | 10,571 divergent transitions across 12,000 post-warmup draws (88.1%), 1,429 depth-limit hits, maximum R-hat 4.263, minimum bulk ESS 4.25, and minimum BFMI 0.107. The chains did not yield a usable posterior. |

The runner needed an accepted primary partner-model posterior to generate the 16 recovery datasets. The 100 simulations in `prior_predictive.tsv` check what the priors predict and provide no estimates of recovery. The earlier independent-fit pilot remains the available recovery evidence, and its results apply to that pilot specification. [Pilot assessment](../ddm_feasibility/ddm_feasibility_report.md)

The generated report labels the four execution errors `stale_prior_run` because their short status records lack fit fingerprints. The stage and individual fit JSON files agree on `execution_failed`. The history fit's code and cohort hashes match the run provenance. Diagnosis of the primary error requires the launcher log identified in `last_run.json`. [Generated report](ddm_hierarchical_report.md) · [Data provenance](data_provenance.json) · [Run instructions](../../feasibility/ddm/HIERARCHICAL.md)
