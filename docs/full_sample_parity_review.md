# Linux2 parity review, 2026-09-27

The first full-sample freeze is retained in commit `5a1a819`. Its live run stopped
on 168 unexplained audit entries and reported 84 public-run omissions. This
review does not change canonical trials, source exclusions, or frozen N111 data.

## What the pushed audit establishes

| Participant | Unexplained entries | Cause to test | Proposed correspondence |
| --- | ---: | --- | --- |
| sub-10657 | 42 | Historical appended complete segment has synthetic run label 1.1; canonical run 2 was counted separately as 42 public omissions. | Legacy 1.1 → canonical 2 |
| sub-10777 | 84 | Historical run 1 contains a two-trial first attempt, followed by a complete segment labeled 1.1. Comparing that first attempt against canonical run 1 produces two RT mismatches and 40 extra rows, plus 42 missing rows for segment 1.1. | Legacy 1.1 → canonical 1; retain an explicit audit of the two omitted first-attempt trials |
| sub-10836 | 42 | The sole historical run is labeled 2; canonical run 1 was counted as 42 public omissions. | Legacy 2 → canonical 1 |

These are proposed correspondences, not a claim that the live cross-run trial
comparison has already passed. The pushed audit contains row-presence markers,
not all values needed to check these correspondences locally.

`docs/data_audit.md` documents the two appended-session cases. Those two
participants were excluded from the N111 primary analysis but remain in the
historical descriptive table, which the full-sample audit also checks.
Upstream `rf1-sra-linux2/docs/behavior-source-repairs.md` documents the
team-confirmed August 31 relabeling for sub-10836. Current upstream certification
provides the canonical run identities and hashes; the downstream code does not
read private source logs or infer acquisition identity from filenames alone.

The N111 overlap is 109 because sub-10555 and sub-10951 are in the current
upstream source-exclusion inventory. They are not lost through trial joining.
The remaining 108 overlapping N111 participants have no flagged field
differences in the original audit; sub-10836 has the run-label difference above.

## Acceptance checks

`config/full_sample_run_mappings.tsv` records explicit candidate mappings and
binds each to the frozen legacy-table hash and current canonical event hash.
There is no automatic search for a convenient matching run. Before accepting a
mapping, the code requires identical complete trial-ID sets and checks all 12
shared fields: partner, offers/sides, chosen amount, high choice, RT, observed
feedback, observed and scheduled reciprocity, and trial order. Numeric tolerance
remains 1.1e-6, as in the original audit.

Any value mismatch, changed hash, absent run, duplicate mapping, or collision
stops execution. The omitted two-trial segment is allowed only under its
specific reviewed rule, with its old-table hash and trial count, and only when
the full replacement segment passes every comparison. It is not silently
reassigned or allowed to overwrite the complete run's first two responses.

The audit retains each mapped trial as a `run_identity` correction, and each
omitted first-attempt trial as an `aborted_segment_presence` correction, alongside
ordinary trial comparisons. If all three live mappings pass, the expected result
is zero unexplained fields, 126 run-identity corrections, two aborted-segment
corrections, and no remaining public-run omissions among these cases. These
counts are predictions to verify, not thresholds that waive discrepancies.

Structural certification and trial equivalence do not settle separate scientific
questions about historical stimulus validity. This patch does not change the
inclusion policy or adjudicate those upstream questions.

## Resume on Linux2

This changes the frozen audit configuration and implementation. Use the explicit
refresh option once. The wrapper records all output; the old full-sample outputs
are backed up under ignored `work/full_sample/previous_freeze_*` before replacing
them. Proposed mappings are validated before the existing freeze is replaced.
The original pushed audit remains available at `5a1a819`.

Inside the existing tmux session:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_gate.sh --refresh
```

The first gate then generates the age analysis if parity passes and stops for
review. No hierarchical fits are launched. After success or failure, push
`results/full_sample` and `results/run_logs` for review. Do not rerun conversion
or erase the frozen N111 analysis.
