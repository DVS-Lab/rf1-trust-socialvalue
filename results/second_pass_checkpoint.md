# Second-pass checkpoint — analysis incomplete

The laptop analysis is paused for continuation on linux1. **23 posterior runs are validated; 11 remain unfinished.** Completed code and small outputs are saved in this commit. This is not the final second-pass report.

- [Linux continuation instructions](../docs/linux1_handoff.md)
- [Fit status and diagnostic history](tables/hierarchical_fit_status.csv)
- [Machine-readable restart settings](hierarchical_checkpoint.json)
- [Theta-bound sensitivity](tables/theta_bound_sensitivity.csv)
- [Behavioral age contrasts](tables/age_partner_change_25_to_75.csv)
- [Hierarchical age effects](tables/hierarchical_age_effects.csv)
- [Paired recovery](tables/hierarchical_recovery_summary.csv)

All six full-data model families and all 15 recovery datasets passed diagnostics. Both H2 training hierarchies are complete. Other prediction and sensitivity fits are unfinished or retrying. H5 training-age prediction files currently describe a failed 8,000-draw attempt and must not be treated as a validated comparison. Some figure files are previews of incomplete comparisons; the final overview/report/gallery are still outstanding.

Completed findings include continued theta-ceiling dependence, uncertain behavioral/H5 age effects, and improved paired hierarchical recovery in the tested low/moderate-theta regime. See the handoff for scope and interpretation limits.

Large posterior caches are intentionally excluded from Git. The optional completed-cache transfer is about 4.25 GB uncompressed; without it, Linux must regenerate the hierarchical posteriors needed by the finalizer. The laptop jobs remain paused.
