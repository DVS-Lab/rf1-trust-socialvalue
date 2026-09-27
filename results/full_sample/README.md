# Linux2 full-sample integration

N=111 remains frozen at `beac6f4d48421b546aa0c7011d4f60adec21ec56`.

Status: **parity review required**. No hierarchical models launched.

## Cohort and migration

- total_canonical_participants: 343
- primary_n: 343
- sensitivity_n: 338
- two_run_primary_n: 304
- total_valid_choices: 26560
- age_n: 343
- age_missing_n: 0
- age_min: 20.0
- age_max: 89.0
- age_mean: 47.07871720116618
- age_sd: 17.807028856948275
- source_excluded_trust_n: 2
- source_exclusion_inventory_n: 12
- qc_review_runs: 8
- qc_review_participants: 6
- n111_overlap_n: 109
- unexplained_parity_fields: 168
- expected_canonical_corrections: 0
- expected_public_omissions: 84

## Behavior

Estimates weight participants equally; intervals resample participants. Zero-versus-positive-positive contrasts are descriptive and confounded with offer amounts. Cohort tags are internal heterogeneity checks, not separate primary samples.

- chose_high, friend - computer: 0.342 (95% CI 0.308, 0.377; N=343).
- chose_high, friend - stranger: 0.250 (95% CI 0.222, 0.279; N=343).
- chose_high, stranger - computer: 0.092 (95% CI 0.069, 0.115; N=343).
- chosen_amount, friend - computer: 1.599 (95% CI 1.440, 1.760; N=343).
- chosen_amount, friend - stranger: 1.205 (95% CI 1.068, 1.338; N=343).
- chosen_amount, stranger - computer: 0.393 (95% CI 0.289, 0.502; N=343).
- chose_high, computer: zero - positive-positive: 0.147 (95% CI 0.119, 0.175; N=342).
- chose_high, friend: zero - positive-positive: 0.110 (95% CI 0.088, 0.133; N=343).
- chose_high, stranger: zero - positive-positive: 0.205 (95% CI 0.175, 0.234; N=342).

Age model status: blocked_by_parity.

Review `tables/cohort_heterogeneity.tsv`, `tables/behavior_contrasts.tsv`, and `figures/02_zero_option_cohort_check.png` before pooling for computational interpretation.

## Provenance

- Upstream SHA at freeze: `c44dd97dfe0a778fb48113546a3e60d776be38c2`
- Analysis SHA at execution: `e098fda830ff8ea4d7658c69379298ab371b80d0`
- Refreshed QC hash: `0024397bb42f80739b04a3832531a1157d00d65c0731c20d09aff7be4f143f57`

Ratings inventory/export is deferred to upstream work and does not gate this milestone.
Stop for review here. The planned next phase includes H2/H5/H7/HPreference/H8, each with and without generic gamma0; representative no-age fits first, then run-1 → run-2 prediction.
