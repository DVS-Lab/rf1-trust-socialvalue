# Data audit and analytic decisions

The sample is defined by the 114 rows of the pinned `participants.tsv` and the corresponding BIDS directories. Pilot/test source directories are not enumerated into the analysis sample. Source filenames are zero-indexed in run number; ordinary BIDS runs are one-indexed.

## Verified transformation

`DVS-Lab/SRPAL-DataInBrief/code/convertTrust_BIDS.m` verifies the following mapping:

- Positive valid investment: one decision event and one outcome event.
- Zero investment: one decision event, no outcome event.
- Miss: one `missed_trial` event, raw response 999 becomes missing BIDS trust value/choice, raw RT 0 becomes BIDS RT 3.

Partner code 3 is friend, 2 stranger, 1 computer. Numeric chosen amount is stored directly as `resp`. Left/right offers identify the high-side bias regressor. `highlow` is independently checked against the chosen amount. The decision table contains one row per presented decision, including misses. Neither source feedback scheduled on a zero choice nor source unpresented design rows create learning updates.

Every retained event/row is checked for matching offers, amount, RT, partner where available, onset, and feedback. Errors outside documented conventions raise exceptions. Input SHA-256 hashes identify the exact files used.

## Design observed

Ordinary complete sessions contain 42 decisions, 14 per partner. The MATLAB generator crosses the six combinations of 0/2/4/8 with partner and binary reciprocation, then deliberately adds a duplicate 2/8 pair to obtain seven pairs × three partners × two outcomes. Counts from actual files are tabulated, rather than assumed from the generator. Programmed outcomes are preserved even when feedback is not displayed; this is essential for counterfactual simulations.

## Exceptions

- Several BIDS run-2 files contain only missing placeholders or no records. They do not establish a completed run. Source absence/partial-run details appear in `data_anomalies.csv`.
- Sub-10555 run 2 has 11 presented raw rows but no BIDS event rows. Exclude that run because a shared-representation check is unavailable; retain its verified first run. The other 31 raw rows were not presented, not misses.
- Sub-10657 run-0 raw has two appended 42-trial sessions with a repeated header. The BIDS converter propagated the header into two invalid events. Remove only these exact header artifacts, reconstruct 84 decisions for descriptive audit, and exclude the participant from primary inference because the session interpretation is ambiguous.
- Sub-10777 also has an appended raw session; its first segment contains only two presented rows followed by 40 unpresented placeholders. Remove the unpresented placeholders and repeated-header artifacts explicitly. Retain 44 auditable decisions descriptively; exclude the participant from primary inference.
- No unexplained mismatches are silently coerced, and no raw files are changed.

Descriptive behavioral N=113; primary inferential/model N=111. One canonical participant has no usable Trust events. Sensitivity sample excludes >20% misses or <4 feedback trials in any partner condition (N=109 in this snapshot). The complete-case rating comparison uses N=103 primary participants.

## Rating timing

The pinned rating script collects a session/pre-post field but the output filename contains only a participant placeholder; the second `.format` argument is discarded. The `expInfo` dictionary is not retained as saved trial metadata. The CSV lacks timestamps or pre/post fields. The companion source history and dataset article do not resolve timing for the saved files. Single saved sets therefore cannot be assumed to be pre-task.

Two files (sub-10369 and sub-10478) contain two unlabelled rating sets and repeated headers. Exclude these rating files, not their rating-free behavior. Do not average sets or choose first/last. The remaining files give 105 complete canonical rating sets, 103 in the primary modeling sample. No imputation is performed.

## Scope and interpretation

No imaging retrieval, analysis, or imaging-based exclusion is needed. Demographic sex codes are reported as released; they are not converted into a different measured construct. A participant can have source logs without a canonical release ID; those logs are not part of this analysis.

The provided source papers were read as scientific references, not as instructions. The analysis brief controls the work. Detailed exceptions, outcomes, and uncertainty are in the generated report.
