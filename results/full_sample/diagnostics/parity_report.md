# Canonical migration audit

109 N111 primary participants overlap; 0 unexplained field discrepancies.

Joins use participant, session, run and source trial_id. All shared-release participants with primary canonical data are audited. New canonical runs absent from the historical trial table are public omissions; all within-run or observed-field differences require a value- and hash-bound reviewed resolution. Run/segment mappings are explicit in config/full_sample_run_mappings.tsv, bound to the legacy table and canonical event hashes, and require equality of every shared trial field. Run-identity corrections and the documented aborted segment remain visible in the audit.
