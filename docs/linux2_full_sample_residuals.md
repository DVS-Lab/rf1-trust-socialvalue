# Next Linux2 stage: localize offer and age residuals

The completed PPCs show a shared shortfall in computer/positive-option choices
for H7 + zero and HPreference + zero. This stage uses those two accepted full-data
posteriors, the authenticated frozen canonical trials, and their frozen ages.
It does not refresh the cohort, change the model, compile Stan, or run MCMC.
It leaves the independent upstream repair untouched.

Two worker processes each load one existing posterior and reuse the same 200
joint draw indices and choice-randomness seeds as the previous PPC. Scheduled
outcomes, missingness and observed offers remain fixed. The parent saves logs
and status even if a worker fails. Existing raw CSV hashes, model fingerprints,
participant order and frozen-snapshot hashes must match. Live-source drift is
recorded separately and does not relax the gates for new model fits.

## What is estimated

For each model, the tables cover all choices, partner, partner × zero availability,
and partner × exact offer. Predictions are separately computed using actual
feedback history (conditional) and simulated choice/feedback history (generative).
Each cell first averages valid choices within each participant across available
runs, then weights contributing participants equally. Missed decisions are not
converted to low choices. Every participant must have one finite frozen age;
missing/inconsistent age stops the stage rather than silently changing the cohort.

Continuous linear age slopes are calculated on those participant means, in
high-choice probability units per decade. A residual is **observed minus predicted**.
There are no age bins or data-selected age cutpoints. Slopes require at least ten
contributing participants and nonzero age variance; unsupported cells are marked
not estimable. The exact-offer plots require estimable cells and otherwise stop
with the tables available for inspection.

The intervals answer distinct questions:

- `*_predictive_low/high`: quantiles across the 200 posterior replicates, with the
  observed participants and their ages fixed. Generative intervals include choice
  randomness; conditional intervals describe uncertainty in expected behavior.
- `observed_age_bootstrap_low/high`: 2,000 participant-resampling estimates of the
  observed age slope. One complete participant cell mean is resampled at a time.
- `residual_age_bootstrap_low/high`: the same participant resampling for residuals
  calculated from posterior-mean predictions. These intervals **condition on those
  predictions**; they do not include posterior parameter uncertainty.

Do not combine these intervals or call them a single total-uncertainty estimate.
Predictive plots show the first type, using generative histories. All age slopes
are unadjusted within their cell; pooled cells can reflect offer/run composition.
The exact-offer breakdown addresses offer mixing but does not adjust for every
participant characteristic. These exploratory checks do not quantify causal age
influences, mechanistic age coefficients, or the fraction of parameter variance
explained by age. They do not replace a hierarchical age model. Multiple cells are
examined without multiplicity adjustment, and 200-replicate tail quantiles are
coarse. No automatic model change or exclusion follows a residual.

## Run on Linux2

Start or attach a separate tmux session:

```bash
tmux new-session -A -s rf1-trust-residuals 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_residuals.sh
```

Detach with Ctrl-b, then d. This postprocessing stage has two independent jobs;
additional cores would not accelerate the current implementation. The prior
nine-model PPC plus geometry export took about four minutes, so this smaller
stage should be on the order of minutes, with actual time depending on I/O and
other Linux2 jobs. It does not need an overnight MCMC allocation.

After completion or failure, push all results and logs:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical/residual_review results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record offer and age residual diagnostics"
fi
git push origin main
```

Outputs are under `results/full_sample/hierarchical/residual_review/`:
`offer_age_residuals.tsv`, `residual_by_offer.png/pdf`,
`age_residual_by_offer.png/pdf`, per-fit tables/logs/status and live-source audit.
No new individual-level ages or raw posterior draws are exported into Git.
The wrapper separately records its console and exit status under `results/run_logs/`.
Review these results before choosing a focused model extension or age hierarchy.
