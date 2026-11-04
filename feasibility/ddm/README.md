# Trust Game DDM feasibility — isolated Linux2 workflow

This repository is the record for both the learning/social-value models and the
DDM feasibility assessment. This workflow does not alter either the completed
N=111 analysis or existing expanded-cohort fits. It reads canonical Linux2 BIDS
events and upstream certification, never raw task logs or demographics, ratings,
FEVS, closeness, or imaging. No association analysis or MCMC launch is reachable.

## Current status

Only published upstream QC metadata and synthetic numerical/software checks have
been run locally. The user will execute canonical audit and recovery on Linux2.
Read [the current report](../../results/ddm_feasibility/ddm_feasibility_report.md).
No scientific GREEN/YELLOW/RED verdict is justified before those outputs exist.

## Run on Linux2

Use the existing production analysis checkout and its Python environment. Do not
pull or switch its branch while another workflow is using that checkout. If it
is busy, wait until idle; no production reset, stash, or overwrite is needed.

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git status --short
git fetch origin
# Once the new branch is merged, use: git pull --ff-only origin main
# To review/run it before merge, from a clean idle checkout:
git switch --track origin/codex/ddm-feasibility

# First: canonical data readiness only. Stops if certification is inconsistent.
bash scripts/run_ddm_feasibility.sh audit
```

The default Python is `.venv-linux2/bin/python`. To use another existing
environment, set `DDM_PYTHON` to its absolute executable. It needs NumPy, SciPy,
pandas, Matplotlib, Numba and pytest. The existing scientific environment should
already provide these; no CmdStan, HDDM or new Conda environment is necessary.
`requirements.txt` lists portable minimum versions. Exact runtime versions are
recorded with the recovery outputs. Set `DDM_UPSTREAM` only for a deliberately
different canonical upstream checkout. Session 01 is explicit in `config.json`.

Inspect the audit before running the bounded pilot:

```bash
cat results/ddm_feasibility/ddm_feasibility_report.md
# Confirm trial deadline, exclusions and data coding; then run the requested pilot:
bash scripts/run_ddm_feasibility.sh pilot
```

`pilot` repeats the live audit, then fits at most 360 simulated datasets and
16 observed pilot models, serially with four optimization starts each. Only
eight observed participants are fitted. Simulation parameters are fresh draws,
not fitted point estimates. Trial-count stratification includes short schedules.
This is intentionally a screen; no automatic full-cohort or hierarchical phase
follows a favorable result. Runtime is not yet benchmarked on Linux2.

Use `tmux` if desired; the launcher sets fail-fast options only in its child
shell and leaves the interactive shell alone. Raw console logs stay private
under `work/ddm/logs`. `last_run.json` records the exit status. After a failure,
inspect the private log; do not interpret older success figures as new results.
The latest `audit_status.json` and `pilot_status.json` determine completion.
Re-running stages overwrites only this new isolated workflow's generated files.
Archive any DDM results you want to compare before changing configuration.

## Outputs and publication

**Ignored private outputs under `work/ddm/`:**

- `participant_available_trials.tsv`: QC counts, with eligibility unassessed
  until the direct event audit completes.
- `participant_eligibility.tsv`: audited counts and separate choice-rule,
  missing-partner, post-RT and final eligibility flags.
- `participant_partner_run.tsv`: presented, recorded-choice, miss, fast-RT and
  DDM-valid counts for each participant, partner and run.
- `canonical_trials.tsv`, `rt_offer_choice_cells.tsv`, input-event manifest,
  selected IDs, individual simulated truths/fits and observed pilot estimates.

**Compact public outputs under `results/ddm_feasibility/`:**

- `ddm_feasibility_report.md`, aggregate audit and recovery tables/provenance.
- Trial-count/miss figures, direct choice/RT figures, recovery and confounding
  figures, and explicitly labeled plug-in predictive checks.

Commit only code/documentation and aggregate outputs, after reviewing the diff:

```bash
git add feasibility/ddm scripts/run_ddm_feasibility.sh results/ddm_feasibility
git diff --cached --stat
git diff --cached --name-only
git commit -m "Record canonical Trust DDM audit and bounded recovery pilot"
git push
```

Never stage `work/`, BIDS, individual fitted values, participant IDs, raw logs,
or source events. The source repository already ignores `work/`; this directory
adds no exception. The publishable tables must remain aggregate even when an
upstream public QC table happens to expose IDs.

## Model and scope

`audit.py` reuses only the existing strict `load_run` function. It checks sorted
offers against left/right amounts, selected amount and high/low labels, partner,
trial order, and observed feedback versus latent scheduled reciprocation. A
response side is inferred from selected amount and displayed offers, rather
than claiming an independent button-log validation. Missing-event rows do not
become decisions because an older converter used a 3-s placeholder. RT 999/-999
is invalid; valid zero investments are retained. The choice-variability rule is
computed separately before and after the 200-ms/deadline filter; missing
friend/stranger cells are reported separately from constant responders.

`model.py` implements an analytic joint Wiener choice/RT density, with absorbing
bounds 0 and a, z as proportion of a, and noise SD 1. An image expansion handles
short decision times, a spectral expansion handles longer times, and a spectral
CDF normalizes the entire contaminant mixture for RT selection. Simulation uses
independent small-step Euler paths; analytic choice probabilities and moments
test it, and a half-step-size stress screen checks discretization sensitivity.
This custom prototype still requires comparison with an established Wiener
implementation before any final confirmatory analysis.

Shared-z and partner-z models have 7 and 9 parameters respectively: common a
and t0, partner drift intercepts, and a common offer-gap drift coefficient.
The upper boundary always means higher investment. The estimand is high-choice
bias within a partner, with friend-minus-stranger as the paired contrast.
"Bias toward strangers" does not mean choosing a stranger as a response.

The prototype fixes 1% unconditional uniform contaminants, applies the actual
response/RT rejection mask, and also censors newly simulated fast/late RTs.
Unobserved programmed feedback never enters learning updates. The learning
stress resets beliefs each run and uses fixed alpha=.3 with drift gain 2;
this deliberately misspecified generator is not a full joint RL-DDM fit.
Simple-model fitting is deterministic conditional on its multistart seed.

Eight stress schedules get a paired baseline, wider z bounds, two explicit
natural-z MAP penalties, a null-z drift contrast, true shared-z generation,
unmodeled learning and half-step-size simulation. These are small screens,
not precise calibration or null false-positive studies. MAP penalties are not
hierarchical pooling. Recovery-error correlations are not posterior correlations.
Shared-z estimates under heterogeneous partner truths have no unique true z;
the code does not label an arithmetic partner mean as its recoverable truth.

**Not completed by this pilot:** full hierarchical fitting, posterior uncertainty
or predictive checks, interval coverage, calibrated group-effect inference,
full drift/z model confusion, selection sensitivity after variability exclusion,
or associations. Passing the moderate-variance gate permits discussion of a
bounded next stage, not a GREEN decision. Failure argues against individual-z
association tests under the tested specification. Read all stress and low-SD
results; convergence alone does not establish identification.

The OSF amendment itself was inaccessible locally. Before confirmation, verify
its wording and the 3-s deadline. Operational pilot thresholds and priors in
`config.json` are explicitly not claimed preregistered.

## Verification

```bash
bash scripts/run_ddm_feasibility.sh test
# Or in an activated scientific Python:
python -m pytest feasibility/ddm/test_ddm.py -q
```

Tests include normalization against analytic boundary probabilities, CDF
derivatives, simulator moments, placeholder/missing-response exclusion,
condition-specific variability and upstream tamper detection. Synthetic tests
do not establish recovery on RF1's actual schedules.
