# Full-cohort hierarchical Trust DDM

For the November primary-model failure, start with the [timed partner retry](PARTNER_RETRY.md). It uses the archived Linux2 input and writes a fresh attempt.

This is the full behavioral-cohort extension authorized after review of the
independent-fit pilot. It fits every participant in the certified session-01
Trust roster (currently 343), including constant responders and valid single-run
participants. There is no participant subsampling, participant miss-rate cutoff,
or variability-rule exclusion from the primary fit. Required upstream source
exclusions and settled scientific run decisions remain binding. The prior pilot,
its results, N=111 analyses and existing RL outputs remain unchanged.

## Linux2 execution

Use the existing analysis environment and CmdStan 2.40.0 installation. Stop using
the checkout for other jobs before updating code. Do not pull or edit the source
while this workflow is running: inputs, source and caches are hash-bound.

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git status --short
git pull --ff-only origin codex/ddm-feasibility

# Full preparation, prior checks, observed fits, posterior checks and recovery:
bash scripts/run_ddm_hierarchical.sh all
```

Run in `tmux` for a persistent session. Scientific sampling is restricted to
Linux; the launcher sets shell options only inside its child shell. It uses
`.venv-linux2/bin/python` unless `DDM_PYTHON` names another existing executable.
No environment or CmdStan installation is silently changed. If necessary,
activate the existing environment and install `requirements-hierarchical.txt`.
Set `CMDSTAN` to the existing CmdStan directory if it is outside the recognized
project-local installations. ArviZ is not required by this workflow.

The task source is checked before RT adjustment. Default discovery looks for
`/ZPOOL/data/projects/rf1-sra/stimuli/Scan-Investment_Game/investment_game.py`,
then the same stimulus path in the upstream checkout. If the current task script
is elsewhere, supply its actual path:

```bash
export DDM_TASK_SOURCE=/actual/path/to/Scan-Investment_Game/investment_game.py
bash scripts/run_ddm_hierarchical.sh all
```

This reads task code, not private raw trial logs. Verification establishes what
the current script does; it does not certify historical per-run executable
versions. A timing mismatch stops preparation rather than applying an assumed
clock correction. The released reference script SHA-256 is recorded in code.

Stages can also run separately, without changing the full cohort:

```bash
bash scripts/run_ddm_hierarchical.sh prepare
bash scripts/run_ddm_hierarchical.sh check
bash scripts/run_ddm_hierarchical.sh prior
bash scripts/run_ddm_hierarchical.sh fit
bash scripts/run_ddm_hierarchical.sh recovery
bash scripts/run_ddm_hierarchical.sh report
```

`fit` runs four observed-cohort specifications: shared z, partner z, partner z
with history, and partner z with wider population priors. If canonical imaging
QC is available, it adds the derived imaging-member sensitivity fit. The first
four always use the complete behavioral roster. Each specification uses four
chains × (3,000 warmup + 3,000 retained draws), with four threads per chain:
16 CPU threads while a fit samples. Fits run sequentially. Runtime has not been
benchmarked on Linux2 and may be substantial; this is not the old small pilot.

`recovery` uses **the entire cohort in every simulated dataset**. It fits four
datasets at each of three z-variation levels and two null-z-contrast/drift-effect
datasets, plus two misspecification stresses: 16 complete-cohort hierarchical
refits. The stresses set the true starting-point contrast to zero while changing
partner-specific boundary separation or nondecision time by 10%. They test
whether an omitted partner effect is incorrectly absorbed into z. The correctly
specified common-a/t0 recovery and these misspecified generators are reported
separately. It generates fresh participant
effects from accepted population draws rather than recycling fitted participant
means. No simulated constant responder is discarded. Posterior rank recovery,
RMSE, bias, interval calibration, null contrast exclusions and subgroup results
are recorded. Four population draws per regime and two null datasets are still
a limited Monte Carlo study, not precise coverage or false-positive estimates.

## Behavioral and imaging eligibility

Primary behavior comes from Linux2 BIDS events and the existing certified
Trust eligibility/QC contract. The original choice-variability rule is retained
as an annotation and comparison subset, not a participant filter. No survey,
demographic, FEVS, closeness or neuroimaging outcomes are read.

Preparation reads upstream `qc/run_disposition.tsv` when available and
`code/task_availability_holds.tsv`. It preserves exclude/review/unavailable run
decisions, retaining usable companion runs. A narrowly parsed settled
task-stimulus decision in `docs/open-decisions.md` is also honored if no later
structured adjudication covers that run. Participant-specific records remain
private; the public report gives aggregate counts and input hashes. Historical
unresolved allegations are not converted into automatic new exclusions.

The imaging intersection is derived separately from `qc/run_qc.tsv`,
`qc/thresholds.tsv`, `qc/qc_policy.json` and the reviewed holds. The Trust/session
FD and tSNR fences are reproduced from the **full canonical Trust/session QC
population**, before behavioral subsetting. The established rule is coverage
≥95%, high-FD and low-tSNR 1.5-IQR exclusion, applied once within task/session,
with all settled decisions preserved. Good single runs remain available. No
age-55 restriction, TEDANA-count rule, aggregate-outlier rule, scanner rule or
effect-size exclusion is introduced. Missing QC is not a pass.

This produces a Trust-specific QC intersection rather than borrowing the
Social Doors/UGR participant lists. It is not certification of completed
activation/PPI contrasts. Its sensitivity fit uses all scientifically retained
behavior of the imaging-eligible participants. The full behavioral fit continues
to use all participants. An already-approved Trust imaging run manifest can
override the derived intersection explicitly:

```bash
bash scripts/run_ddm_hierarchical.sh prepare \
  --approved-trust-imaging-manifest /actual/approved/trust_runs.tsv
```

That file needs `subject` or `participant_id`, plus `session` and `run`; optional
`task` must equal `trust`. If it contains `include`, values must be true/false
or 1/0. Input hashes and membership changes are recorded, without republishing IDs.

## Task clock and likelihood

The released task starts logging RT at the partner cue. Offers appear after a
0.5-s partner-only interval; the nominal response window ends 3 s from cue
onset. The hierarchy subtracts this known cue interval, producing a 2.5-s
offer-based window. Remaining nondecision time is estimated explicitly. A
response under 200 ms *after offers appear* is flagged and retained with the
contaminant model; the original cue-based 200-ms flag did not measure this.
Missed rows and placeholder RTs never become decisions. Responses outside the
nominal window are counted explicitly; they do not trigger participant removal.

Stan's native four-parameter Wiener density and unnormalized CDF are used. Both
response boundaries enter the conditional RT normalization. A fixed 1% uniform
contaminant is mixed before conditioning, so its fraction among retained trials
is not incorrectly fixed at 1%. Nondecision time can exceed a fast observed
response when that response is handled by the contaminant component. Diffusion
noise SD is fixed to 1.

Each person has boundary separation, remaining nondecision time, three drift
intercepts, an offer-gap slope, and shared or partner-specific z. Parameters are
partially pooled through a correlated noncentered hierarchy with an LKJ(2)
correlation prior. Boundary separation is log-transformed without an arbitrary
upper ceiling; z uses its intrinsic open (0,1) support. Nondecision time uses
the available response window, without the pilot's 0.8-s cap. Priors are explicit
in `hierarchical.json` and `hier_data.py`; wider-prior sensitivity doubles the
population location/scale prior widths.

There is one explicit numerical evaluation safeguard: reject proposals with
`a² / (deadline − t0) > 1000`, `a < 1e-6`, or `|drift| > 1e6` before calling
the native CDF. A synthetic MCMC check exposed a nonterminating native-series
loop for extreme warmup proposals in Stan Math 2.40. This safeguard avoids
that failure; it is a computational support restriction and must be disclosed.
It is much wider than the pilot's a≤3 bound. Posterior proximity to the guard
is exported, and joint prior draws are conditioned on its evaluable domain.
A fit near this boundary needs numerical-domain sensitivity before interpretation.

A common zero-offer drift coefficient and a motor-side effect on logit z are
included in both models. The history sensitivity adds lagged observed partner
feedback and within-run trial position to drift; it is not a joint RL-DDM.
The friend-minus-stranger natural-z contrast refers to content bias at the
balanced motor-side reference. It does not mean choosing between partners.

Timing, priors and nuisance structure differ from the old pilot. Consequently,
improvement cannot be attributed to hierarchical pooling alone. A matched
nonhierarchical comparison and more extensive missingness/lapse sensitivities
would require separate implementation if scientifically necessary.

## Validation, diagnosis and interpretation

`check` compiles the real model and verifies native Wiener density/CDF values
against an independent series implementation, plus full cell-cached joint
likelihoods and finite-difference gradients for both z specifications and the
history model. The Euler simulator is independent of the fitting likelihood.
Python tests verify full-roster retention, missingness, clock origin, no feedback
leakage, run-specific holds, QC intersections and constant-responder reporting.
Local tests are synthetic engineering checks, not RF1 model fits.

Accepted MCMC fits require zero divergences and depth-limit hits, R-hat <1.01,
bulk/tail ESS ≥400 and BFMI >.3 across stochastic parameters and participant
natural parameters. Failed fits retain their chains/status and do not supply
scientific summaries. No unchanged automatic retry or exclusion of hard-to-fit
participants is performed. A completed compatible cache is rechecked and reused;
partial chain files are preserved and stop for explicit review. Changed targets
use separate fingerprinted caches. Never delete old CSVs to make a check pass.

Posterior predictive checks integrate parameter uncertainty, include every
participant, and report partner/exact-offer/choice RT cells for the full cohort
and preregistered subset. A separate unconditional simulation measures predicted
RT retention. Recovery reports both actual-roster eligibility groups and
*realized simulated* constant responders, rather than calling every simulated
member of an originally constant-responder roster constant again.

Weak individual information can remain after pooling. In particular, a person
with no retained observations has a population-based latent estimate rather
than evidence of a measured individual parameter. Small posterior uncertainty,
population significance or successful sampling alone does not establish useful
individual-difference measurement. Inspect recovery, drift/z dependence,
predictive tails and sensitivity before a scientific go/no-go decision. No
association analysis or automatic GREEN recommendation is implemented.

## Outputs, resumption and Git

- `results/ddm_hierarchical/`: aggregate cohort/QC reconciliation, diagnostics,
  posterior predictive figures/tables, recovery, sensitivity and report.
- `work/ddm_hierarchical/`: ignored participant tables, trial data, masks,
  timing provenance, individual posteriors, simulation truths, raw chains and logs.
- `last_run.json`: launcher exit status; `run_status.json` and stage-specific
  status files distinguish incomplete/failed stages from accepted fits.

Rerun the same command to reuse compatible completed fits. It does not silently
resample a diagnostic failure or incomplete chain set. After an interruption,
inspect the private log and status before changing any settings. The report
checks source/cohort/sampler fingerprints so older accepted fits are not treated
as current after a specification change.

Publication is manual and stays on the working branch until separately merged:

```bash
git add feasibility/ddm scripts/run_ddm_hierarchical.sh results/ddm_hierarchical
git diff --cached --stat
git diff --cached --name-only
git commit -m "Record full-cohort hierarchical Trust DDM results"
git push origin codex/ddm-feasibility
```

Never add `work/`, BIDS, participant parameters, raw task logs or chain CSVs.
No command here merges into main, changes imaging preprocessing/eligibility
upstream, or launches an external-outcome analysis.
