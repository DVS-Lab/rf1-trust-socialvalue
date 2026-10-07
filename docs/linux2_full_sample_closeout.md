# Linux2 closeout: diagnostics, hierarchical age, predictive checks and recovery

This stage retains the frozen 343-person cohort, all old source files and fit caches, and the accepted `Train_HPreference_zero_bias_v1` result. It creates `results/full_sample/hierarchical/closeout` and `work/full_sample/hierarchical/closeout`. No laptop sampling and no cohort refresh. Live-source exceptions are limited to the authenticated removal of two empty sub-10668 imaging templates and the exact reviewed SharedReward-only global QC update described below.

## Scientific scope

The N=111 study already included hierarchical age models. The expanded cohort had behavioral age regressions and age residual checks, but its previous Bayesian fits had **no age predictors**. The present stage addresses that gap.

Eight real-data fits, all with eight chains:

1. One retry each for `Train_H7_zero_bias_v1`, `Train_H7_zero_amount_v1`, and `Train_HPreference_zero_amount_v1`. These have respectively 1, 3, and 4 divergences and pass all other diagnostics. Before retrying, verify raw posterior hashes, recompute diagnostics, and export divergent saved-state context including extension hyperparameters and participant latents. The original executable, equations, priors, data, seed, and 4,000 retained draws per chain are retained. Warmup increases from 4,000 to 6,000, target acceptance from .995 to .9995, and maximum tree depth from 14 to 16. Separate caches preserve every original draw. This is one bounded numerical retry, not evidence that the posterior geometry is fixed; persistent divergence requires review/reparameterization, not another unchanged run. See [Stan's diagnostic guidance](https://mc-stan.org/docs/cmdstan-guide/diagnose_utility.html).
2. Preference + zero + general choice bias, trained on run 1 of 304 participants, **with age**. Compare run-2 scores against the already accepted no-age bias fit on exactly the same 12,494 choices. These are predictions for the same people's second run, not new-person predictions.
3. The same model, using all available runs from **343 participants**, with and without age (two fits).
4. The same full age model in the frozen **338-person QC sensitivity sample**.
5. The same full age model with a wider age-slope prior (SD 1 instead of .5).

Age is `(years - 50)/20`. Independently regularized population slopes enter all six latent parameter locations: logit learning rate, log inverse temperature, friend preference, stranger preference, zero-option coefficient and general choice bias. The existing correlated five-parameter random-effect block and independent bias random effect are preserved. Age is cross-sectional; no causal aging claim or residual sex/cohort adjustment is implied.

The bias model is a fixed reference for this age analysis, chosen from the currently accepted evidence. If the amount extension clearly improves heldout prediction, fit an age version of that extension before attributing effects to a particular parameter. This stage does not automatically choose a new scientific model after seeing results.

## Checks and outputs

Before sampling, fast/reference Stan target and gradient comparisons, Python likelihood checks, and **full no-age target nesting** must pass. A joint prior predictive screen draws every hierarchy component (including LKJ correlations), simulates on real task schedules and preserves missingness. It covers both age priors. Finite prior output alone does not establish a scientifically adequate prior: inspect its extreme-choice fractions.

The original strict diagnostic thresholds apply to every base parameter, extension, age slope and participant natural parameter. Zero divergences and zero depth-limit hits are required; no failed fit contributes scientific summaries. All eight chains and all retained draws enter accepted analyses.

Full fits produce 200-replicate conditional and generative predictive checks, exact partner/offer cells, between-person variability, extreme responders and age residual checks. Generative feedback follows simulated choices; scheduled outcomes remain hidden after zero investments or missing responses. Population age curves are transformed locations at zero random effect, not population-marginal averages.

The age variance fraction is `beta² Var(age_z)/(beta² Var(age_z)+tau²)`, separately for each latent parameter and each posterior draw. It is **not behavioral R²**, a causal effect, or variance explained in new people. Wider-prior and QC-sensitivity estimates must be inspected alongside primary estimates.

After an accepted full age fit is available, a second eight-fit batch generates fresh participant effects and choices on the real schedules. Four datasets have zero age slopes; four have known slopes of ±.3 latent units per 20 years, with alternating signs. Population locations, scales and correlations are anchored to the accepted full fit. The refits export slope bias, interval coverage, zero exclusion, participant rank correlations, RMSE and coverage. Eight datasets give a **targeted recovery screen**, not precise coverage/false-positive estimates or SBC. They do not establish discrimination between H7 and HPreference mechanisms. Weak recovery limits parameter interpretation; it does not authorize repeated refitting until favorable.

The `all` command runs both batches, with up to **8 concurrent fits × 8 chains = 64 cores**, leaving headroom on Linux2. BLAS/OpenMP threading is limited to one per chain. Actual CPU use declines as fits finish. Allow several hours, potentially overnight; conservative adaptation and full-cohort fits can take longer than the prior training batch. There is no verified runtime benchmark for this stage.

## Run in tmux

Start an interactive shell inside tmux. Detach with Ctrl-b, then d.

```bash
tmux new-session -A -s rf1-trust-closeout 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_closeout.sh all
```

The launcher invokes `.venv-linux2/bin/python` directly; activation/PATH changes are unnecessary. It runs the test suite and writes console/exit-status logs. It sets no options in your interactive shell. `analysis`, `recovery`, and `report` can also be invoked separately. `all` attempts recovery when the primary full-age fit passes even if another real-data fit fails. It exits 2 if any required fit fails or recovery is blocked. `report` regenerates available summaries; its successful exit does not mean all fits passed.

Existing compatible completed draws are reloaded, never resampled. A rerun does **not** fix failed diagnostics. Partial or mismatched caches stop for inspection. The old amount launcher should not be used to request a new retry.

Inspect progress from another terminal:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
.venv-linux2/bin/python - <<'PY'
import json
from pathlib import Path
root=Path('results/full_sample/hierarchical/closeout')
for file in sorted((root/'fits').glob('*/status.json')):
    state=json.loads(file.read_text())
    print(file.parent.name, state['status'], 'divergences=', state.get('divergences','pending'))
PY
```

When it stops, push results and logs even if a fit failed:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical/closeout results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record full-cohort age closeout and recovery results"
fi
git pull --rebase && git push origin main
```

Raw posterior CSVs remain on Linux2 under ignored `work/`. Portable manifests record their hashes. Preserve them for diagnostic review.

## Stopping rule

Review sampler diagnostics, amount-versus-bias contrasts, age/no-age prediction, full-data predictive checks, QC/prior robustness and recovery together. An accepted sampler is not proof of adequate model fit. If the age results survive these checks, write the final figures and report. If recovery or exact-offer checks remain weak, report the descriptive/predictive findings and limit mechanistic claims. An improved amount model may justify one age robustness fit; this is not an open-ended model search. Existing N=111 results remain an immutable historical analysis.

## October 7 restart: global QC changed after the SharedReward repair

The first closeout attempt stopped before compilation or sampling. Its 282 tests passed; pandas/Matplotlib FutureWarnings were not the failure. The strict amount-stage snapshot gate rejected the updated `qc/events/results/provenance.json` before it could save a live-source audit.

Upstream commit `5e8bbd62f2ed97bf21141f9995dbfa6b97e718f1` (September 30) added precisely one global response-QC row: **sub-10668 / session 01 / SharedReward / run 2**. The table grew from 2,751 to 2,752 rows. Removing that exact added line reproduces the frozen table byte for byte, including all **647 Trust QC rows**. The global provenance changed only its timestamp, run count and events-manifest hash; QC policy and review counts remained unchanged. Trust run eligibility, source exclusions, Trust handoff provenance, converter and curation files match their frozen hashes in the reviewed upstream revision. The added row is not Trust run 2 and does not enter modeling.

`config/full_sample_closeout_qc_review.json` records the before/after hashes, added row, both metadata versions and upstream commits. The closeout-only snapshot adapter requires this exact reviewed pair of files and independently verifies exact reconstruction of the original QC table. Every other live input still passes the existing check, including the earlier empty-template repair receipts. Any further QC update, Trust event change, policy change, eligibility change or incomplete QC update stops the run. Original amount-stage source files, accepted-fit fingerprints, frozen cohort configuration and model equations are unchanged.

The full audit is now written **before** validation and retains all accepted and rejected changes. If a future check stops the run, push `results/full_sample/hierarchical/closeout/live_source_audit.json` along with the normal logs so all differences can be reviewed together.

Restart using the same `all` command above after pulling `main`. No cache deletion, cohort refresh or upstream checkout/reset is required; this failed attempt created no new posterior draws.
