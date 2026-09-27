# Full-sample Linux2 analysis

Status: **implementation prepared; live Linux2 integration has not run**. No cohort
N is frozen yet. The September 1 QC snapshot is background information only.

The completed N=111 analysis remains fixed at
`beac6f4d48421b546aa0c7011d4f60adec21ec56`. Its reports, historical tables,
posterior summaries, and recovery results are not regenerated here.

Follow [the Linux2 runbook](../../docs/linux2_full_sample_handoff.md). The first
milestone certifies the upstream schema, refreshes live QC, freezes the cohort,
checks overlap, and produces descriptive behavior and continuous-age GEE results.
It stops before any hierarchical fitting. A parity discrepancy is a review gate,
not permission to waive the disagreement or start sampling.

`tables/cohort_manifest.tsv` is run-level; `tables/data_audit.tsv` describes primary
participants and prespecified sensitivity eligibility. The canonical decision
table is kept in ignored `work/full_sample/canonical_trials.tsv`, with its checksum
in the tracked provenance. The downstream repo never reads private source logs.
