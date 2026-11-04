# Full-cohort hierarchical DDM: execution pending

The full behavioral workflow is implemented for all 343 certified participants,
including constant responders. Full-cohort scientific fitting and recovery have
**not been run** in this checkout; they are intended for Linux2.

Software validation passed 15 synthetic tests, compiled native Wiener
probability/likelihood/gradient comparisons, and a short four-chain synthetic
execution check. The short check is not a converged scientific fit and supplies
no RF1 scientific inference. See `software_validation.json` and
`implementation_status.json` for provenance and numerical errors.

A separate metadata-only reconciliation of upstream canonical Trust QC gives a
**provisional 320-person imaging intersection**, with 597 passing certified
runs. It preserves settled run decisions visible in the repository and good
single runs. Private Linux2 run dispositions and completed contrast availability
still need verification. See `imaging_metadata_preview.json`; no imaging outcomes
were examined, and imaging eligibility does not restrict the full behavioral fit.

Run instructions and the disclosed numerical evaluation safeguard are in
[HIERARCHICAL.md](../../feasibility/ddm/HIERARCHICAL.md). Scientific jobs write
aggregate reports here; participant data, posterior estimates and chains remain
in ignored `work/ddm_hierarchical/`. The earlier independent-DDM pilot and RL
outputs are preserved. Git publication remains manual.
