# Upstream change investigation: sub-10668

## Finding

The most likely explanation is the deliberate task-label repair for sub-10668:
its second acquisition labeled Trust is being reassigned to Shared Reward run 1.
The repair code removes the old acquisition's event companions and regenerates
Shared Reward events. That would remove exactly the two Trust-labeled imaging
templates reported by the analysis verifier. The available pushed log confirms
a successful preview, **not a completed live apply**; receipt/archive inspection
on Linux2 is still needed to confirm the exact operation.

## Evidence and timeline

1. The Trust cohort was frozen at `2026-09-27T17:30:23.570209+00:00` against
   upstream commit `c44dd97d`. Both run-2 imaging event templates had SHA-256
   `f49141d568d2bfebb7456958a33a9f3d67f923ec542a560f3c7cb6b2de599423`,
   the empty-template content recorded in the freeze. Neither was a canonical
   behavioral run.
2. Upstream commit [`57b2daaa`](https://github.com/DVS-Lab/rf1-sra-linux2/commit/57b2daaa)
   added the preserved reassignment/rebuild workflow. The
   [repair implementation](https://github.com/DVS-Lab/rf1-sra-linux2/blob/be6cc69f/code/repair_10668.py)
   maps old Trust run 2 to Shared Reward run 1 and old Shared Reward run 1 to
   Shared Reward run 2. `build_stage()` removes the old companions from staging,
   skips copying their event files, and generates replacement Shared Reward
   events from the reviewed sources. The original session is archived during apply.
3. The [September 30 00:29 preview record](https://github.com/DVS-Lab/rf1-sra-linux2/blob/be6cc69f/logs/records/20260930-002914_10668-repair-preview-20260930-002914.md)
   passed and printed the exact reassignment plan, but also `DRY RUN: no files
   changed`. It is the latest relevant pushed record at upstream `be6cc69f`.
   The dry run itself cannot explain the changed live inventory. A subsequent
   live apply is a strong possibility, not a confirmed fact from this record.
4. The failed geometry export at `2026-09-30T05:10:29Z` reported only these two
   differences in its Trust input inventory:
   - `sub-10668_ses-01_task-trust_run-2_part-mag_events.tsv`
   - `sub-10668_ses-01_task-trust_run-2_part-phase_events.tsv`
   That verifier message lists changed paths but does not say whether their
   bytes changed or the files disappeared. Renaming the surrounding acquisition
   can therefore present as a missing old input, without editing file contents.

The documented approved mapping preserves Trust run 1, crops the reassigned
acquisition to its first 255 volumes, and retains the existing Shared Reward
acquisition as corrected run 2. See the
[repair handoff](https://github.com/DVS-Lab/rf1-sra-linux2/blob/be6cc69f/docs/10668-task-reconstruction.md).
This describes upstream scope; this analysis task has not executed that repair.

## Impact on this Trust analysis

The frozen cohort manifest already excluded sub-10668 run 2:
`include_primary=False`, zero presented/valid choices, unresolved BOLD-only run
with a missing behavioral source. Sub-10668 contributed **only run 1: 42 presented
trials, 37 valid choices**. This participant is absent from the 304-person actual
run-1/run-2 prediction cohort.

At the failed check, no canonical behavioral event file, participant table or
other recorded Trust input was reported as changed. The two listed files were
not used to build modeled decisions. Thus this specific inventory difference
does not indicate changed modeled Trust choices, and does not by itself justify
rerunning the fitted models. Imaging/Shared Reward products have their separate
rebuild and provenance requirements.

The retry batch finished and verified live inputs on September 29, before this
September 30 export failure. This subsequent file drift does not explain the
two pre-existing sampling divergences.

## Verification and fix

The diagnostic exporter now authenticates the historical integration status,
provenance, saved canonical table, frozen parser/configuration, reconstructed
model targets and raw posterior hashes. Its separate before/after live-source
audits report additions, removals, changed hashes and current template row counts.
It also checks these known repair records, exporting only relevant status/hash
information and the two implicated archived event files:

- `/ZPOOL/data/projects/rf1-sra-linux2/bids/sub-10668/ses-01/.rf1-10668-repair.json`
- `/ZPOOL/data/projects/rf1-sra-linux2/derivatives/source_repairs/10668-sharedreward-v1/receipt.json`
- The two matching files under that archive's `original_session/func/`.

A complete BIDS repair receipt does not itself establish successful derivative
rebuilding; the status is reported as recorded. The exported archive hashes can
confirm that the old empty templates match the original Trust freeze.

No cohort refresh, input deletion, source repair or new sampling is performed.
New fits retain the strict live-input gate. A modified saved analysis snapshot
or posterior still blocks diagnostic export. Rerun the existing
`scripts/export_full_sample_geometry.sh` command after pulling the fix and push
its outputs/logs so the live receipt and changed-file audit can be reviewed.
