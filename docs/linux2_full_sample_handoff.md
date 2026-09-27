# Linux2 full-sample handoff

The N=111 endpoint is frozen at `beac6f4d48421b546aa0c7011d4f60adec21ec56`.
This is a new integration phase, not a rerun of the Linux1 analyses. The first
milestone stops after upstream certification, cohort freeze, parity, behavior,
and a lightweight continuous-age GEE. **No hierarchical fits are launched.**

The code has been tested locally with synthetic data. The live Linux2 cohort,
source exclusions, demographics, and overlap have not yet been certified.
No N≈350 assumption is used in the code. Do not publish the old QC count as the
new analysis N.

## 1. Open a persistent Linux2 session

On **Linux2**, not Linux1 or the laptop:

```bash
tmux new-session -A -s rf1-trust-full 'bash --noprofile --norc -i'
```

Detach using **Ctrl-b**, then **d**. Reattach with:

```bash
tmux attach -t rf1-trust-full
```

The explicit Bash command bypasses login/startup files that could immediately
close the session. Keep fail-fast options out of the interactive shell:

```bash
set +e
set +u
set +o pipefail
```

The child workflow scripts stop on failures and save their exit status; the
interactive shell should remain available to inspect those failures.

Inside that session, synchronize both repositories. These commands stop if a
checkout has uncommitted work; they do not reset, stash, or force-push anything.

```bash
(
set -e
cd /ZPOOL/data/projects/rf1-sra-linux2
git status --short
test -z "$(git status --porcelain)"
git pull --ff-only
git rev-parse HEAD

cd /ZPOOL/data/projects
if [ ! -d rf1-trust-socialvalue ]; then
    git clone https://github.com/DVS-Lab/rf1-trust-socialvalue.git
fi
cd rf1-trust-socialvalue
git status --short
test -z "$(git status --porcelain)"
git pull --ff-only
)
```

The parentheses confine `set -e` to this setup subprocess. If either clean-check fails, preserve the local changes and reconcile them before
continuing. Do not use a force push to resolve a rejected push.

## 2. Create an isolated environment

Use Conda to supply a known Python version; this avoids depending on a system
`python3.13` binary. The following assumes the Linux2 shell has Conda available,
as did the Linux1 base shell. It installs Python packages only, **not CmdStan**.
The optional hierarchical Python dependencies are needed by existing regression
tests; installing them does not start sampling.

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
if [ ! -x .venv-linux2/bin/python ]; then
    conda create -y -p "$PWD/.venv-linux2" python=3.12 pip
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.venv-linux2"
python --version
python -m pip install -e '.[test,hierarchical]' \
    -r /ZPOOL/data/projects/rf1-sra-linux2/requirements-dev.txt
```

After detaching/reconnecting, activate the same environment again before running
any stage. All entry points use `python3` from that environment. The upstream
wrapper uses four behavior-conversion jobs; downstream numerical libraries use
one thread. This milestone does not need dozens of processors.

## 3. Validate the small upstream cohort first

```bash
cd /ZPOOL/data/projects/rf1-sra-linux2
bash code/run_trust_handoff.sh validation
```

This runs the upstream tests, selects 10317 and 10953 plus a participant with an
actual zero investment, snapshots their current events, runs the required
**dry-run**, then behavior-only conversion, then source/BIDS validation. It checks:

- schedule retained on zero and missed trials without inventing observed feedback;
- positive-investment outcomes equal the programmed schedule;
- left/right offers reproduce low/high and selected amount;
- all historical columns and row order remain unchanged;
- imaging file size/mtime/ctime/inode inventory is unchanged.

Image bytes are not rehashed. This command never invokes imaging conversion.
Its original snapshot stays in ignored `work/trust_schema` and is reused on retry.
Do not delete that snapshot to make a failure disappear.

**Review the validation console before running the cohort stage.** If it fails,
stop and commit the stage log as shown below so it can be investigated. Do not
proceed to backfill on a failed validation.

## 4. Backfill and certify the live cohort

After the validation stage passes:

```bash
cd /ZPOOL/data/projects/rf1-sra-linux2
bash code/run_trust_handoff.sh cohort
```

This discovers the live Trust cohort from BIDS, honors authoritative source
exclusions, repeats dry-run/backfill/checks, refreshes all canonical events
response QC, runs its checker, and exports a deidentified Trust eligibility
contract. No private source logs are copied into the analysis repository.

If conversion/checking fails for an unresolved source/run, the wrapper stops.
Those failures must be inspected upstream; do not add blanket curation approvals.
The downstream manifest supports source-excluded/unresolved runs as exclusions,
but the migration does not silently certify a failed backfill.

Inspect and commit **upstream** products (including failure logs, if applicable):

```bash
cd /ZPOOL/data/projects/rf1-sra-linux2
git status --short
git add qc/events/results qc/trust_analysis
git diff --cached --stat
if ! git diff --cached --quiet; then
    git commit -m "Record Linux2 Trust schema validation and refreshed QC"
fi
git pull --ff-only
git push origin main
```

An empty commit is skipped. If `pull --ff-only` reports divergence, stop for
reconciliation; do not force. Do not `git add bids` or private source paths.
The upstream converter change is already supplied in a separate code commit.

## 5. Freeze the cohort and run the scientific integration gate

Only after the upstream cohort stage succeeds:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
bash scripts/run_full_sample_gate.sh
```

The wrapper runs the scientific regression suite and creates the manifest only
if one does not exist. It then verifies the frozen inputs, audits overlap, and
writes behavior tables/figures. The primary age model uses continuous standardized
age, partner interactions, exact offer pair, run, trial position, and participant-
clustered repeated-measures inference. A redundant run term can be removed only
following a recorded rank check. No hierarchical code is reachable from this
entry point.

The canonical BIDS `participants.tsv` must contain `participant_id`, `age` in
years, and `sex`. Missing age is reported and excluded only from age inference;
missing required metadata columns stop the loader rather than guessing a private
source. If Linux2 lacks this canonical demographics export, create it upstream.

The primary cohort retains response-QC review flags and curated short runs.
Sensitivity rules are fixed in `config/full_sample_linux2.json`: at most 20%
missed choices and at least four observed feedback trials for **every** partner.
A separate complete-run flag is also recorded. The rules use live data, not old
excluded IDs.

Expected outputs include:

- `results/full_sample/README.md` and `cohort_summary.json`;
- `tables/cohort_manifest.tsv`, `data_audit.tsv`, `openneuro_linux2_parity.tsv`;
- participant-weighted `behavior_summary.tsv`, paired `behavior_contrasts.tsv`,
  and `cohort_heterogeneity.tsv`;
- age coefficients, three age interaction contrasts, marginal predictions, and
  age-25-to-75 changes when the parity gate passes and GEE is estimable;
- three compact figure sets (PNG/PDF/SVG): partner/offer behavior, zero-option
  internal cohort check, and age × partner;
- tracked input/provenance/output hashes and `milestone_status.json`;
- ignored `work/full_sample/canonical_trials.tsv`, hash-bound to the freeze.

The original-release/later tags are for internal heterogeneity checks. Full
available primary data remain the scientific sample. A zero-versus-positive-
positive descriptive contrast is confounded with the exact offers and is not a
causal estimate of the model's gamma0.

Commit scientific output **even when parity stops for review**:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git status --short
git add results/full_sample results/run_logs
git diff --cached --stat
if ! git diff --cached --quiet; then
    git commit -m "Record Linux2 full-cohort integration gate and logs"
fi
git pull --ff-only
git push origin main
```

## Logs, retries, and explicit refreshes

Each wrapper writes a unique tracked console plus JSON exit code, including on
failure. Upstream logs live under `qc/trust_analysis/run_logs/`; scientific logs
live under `results/run_logs/`. No need to rely on scrollback or pipe output into
an untracked `nohup.out`. `tmux` keeps either wrapper running when disconnected.

An unchanged scientific rerun verifies and reuses its frozen inputs. If BIDS,
QC, the resolution ledger, config, or implementation changes, verification
stops. After explicitly reviewing those changes, refresh with:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
python -m rf1_trust_socialvalue.full_sample freeze --refresh
bash scripts/run_full_sample_gate.sh
```

The previous full-sample output directory is copied into ignored
`work/full_sample/previous_freeze_*` before replacement. Historical N111 outputs
are never touched. Commit the existing output/logs before an intentional refresh.

Unexplained partner/choice/offer/outcome/order/RT differences block the milestone.
The audit includes shared-release participants outside the historical primary
sample, while reporting the N111 primary overlap separately. New canonical runs
absent from the frozen public trial table are labeled expected public omissions;
within-run discrepancies require an individually reviewed resolution in
`config/full_sample_parity_resolutions.tsv`, bound to both event and historical
trial hashes and the exact old/new values. The empty ledger grants no waivers.
Do not blanket-label discrepancies as canonical corrections.

## First scientific review and future work

Return the pushed upstream and scientific results for review. We will inspect:
actual primary/sensitivity N, two-run N, valid choices, ages, source exclusions,
QC flags, overlap and corrections, friend contrasts, descriptive zero-option
patterns, and old-release/later heterogeneity. A significant friend effect is not
a software pass/fail criterion; a different observed result must be reported.

**Stop here before launching the expensive hierarchical suite.** The configuration
records that future phase but marks launch authorization false. The planned set
is H2/H5/H7/HPreference/H8, each base and zero-adjusted, with representative no-age
fits first and the established diagnostic gates. Run 1 → run 2 is the primary
prediction task. Hierarchical age effects and whole-dataset recovery follow only
after accepted fits. Ratings inventory/timing remains upstream and does not block
this integration milestone. No unique H7-versus-HPreference claim is inherited
from the N111 screen.


## Recover an unexpectedly closed tmux session

`[exited]` means the session ended; it does not identify the failing command.
Read the latest upstream stage log from the ordinary login shell first. This
command does not restart conversion, need Conda activation, or enable fail-fast
shell options:

```bash
/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin/python - <<'PYLOG'
from pathlib import Path
root = Path('/ZPOOL/data/projects/rf1-sra-linux2/qc/trust_analysis/run_logs')
logs = sorted(root.glob('*/console.txt'), key=lambda p: p.stat().st_mtime)
if not logs:
    print('No validation log found; the workflow may not have started.')
else:
    log = logs[-1]
    print(f'Latest log: {log}')
    print('\n'.join(log.read_text(errors='replace').splitlines()[-100:]))
    status = log.with_name('status.json')
    print(status.read_text() if status.exists() else 'No saved exit status.')
PYLOG
```

The production workflow repository is `rf1-sra-linux2`. A clean Git status in
`rf1-sra` (the separate source-data project) does not report the workflow's status.


## Empty imaging-template validation fix

Canonical behavioral files end in `_run-N_events.tsv`. Empty
`_run-N_part-mag_events.tsv` and `_run-N_part-phase_events.tsv` files are imaging
conversion templates, not extra behavioral runs. The upstream validator and
scientific inventory distinguish those files explicitly. Nonempty templates are
still a review error; nothing is deleted. The scientific freeze hashes the empty
templates separately so later changes are detected.

To resume the initial validation after this discovery fix, keep the original
`work/trust_schema/validation.json` and the failed run log. Pull both repositories,
then use the existing Python environment and the check-only entry point:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue && git pull --ff-only
cd /ZPOOL/data/projects/rf1-sra-linux2 && git pull --ff-only
bash code/run_trust_handoff.sh validation --check-only
```

The retry records a new console/status, tests the updated upstream code, and
checks the existing converted files against the original pre-conversion snapshot.
It never recreates the snapshot or reruns conversion. A successful report counts
canonical behavioral runs only and lists ignored empty imaging templates
separately. Review `Stage validation exit=0` before proceeding to the cohort stage.
