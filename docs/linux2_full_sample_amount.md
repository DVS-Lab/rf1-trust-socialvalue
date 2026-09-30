# Exploratory amount-sensitive model comparison on Linux2

The next run fits four new hierarchical models to the same 304 paired
participants' run-1 data (12,498 valid choices), then scores the same 12,494
run-2 choices. It reuses the two accepted original training models. The complete
comparison therefore has six models, with only four requiring new sampling.

| Family | Original + zero | + general bias | + bias and quadratic amount term |
|---|---|---|---|
| H7 | Reuse accepted training fit | New | New |
| HPreference | Reuse accepted training fit | New | New |

## Decision rule and priors

For participant i and offers low and high, retain the original model logit,
including its zero-option term, and add:

- Bias control: `b_i`.
- Amount model: `b_i + q_i * (high^2 - low^2) / 64`.

This is a smooth amount-dependent correction on the **logit scale**, not a
separate parameter for each offer or partner. Dividing by 64 gives a fixed scale
based on the maximum offered investment of 8. Negative q penalizes larger
squared-amount differences. For example, the feature is 0.1875 for 2/4 and 0.75
for 4/8. Including the bias control tests whether q improves prediction beyond
a general tendency to choose high. It is a candidate quadratic decision-rule
correction, not a claim to have identified a psychological utility function.

Each added coefficient has a population mean `Normal(0,1)` and a population SD
`HalfNormal(0,0.8)`, with noncentered standard-normal participant effects.
The added effects are independent of each other and of the original latent
random-effect block a priori. This preserves the original five-parameter
correlation prior and all original population priors. Posterior dependence can
still arise through the likelihood. Both additions are signed and centered on
no correction. With b=q=0 the original choice likelihood is recovered exactly.

All original learning updates, observed-history feedback rules, zero choices,
missingness, run ordering and partner effects are retained. No age coefficients
are fitted in this step. A full-data extension, parameter/model recovery and age
hierarchies are later decisions, not automatic continuations of this command.
The extension prior screen holds original parameters at transformed prior
locations while drawing only the added hierarchical effects. It is explicitly
not a full joint prior-predictive or recovery analysis.

## Gates and computation

The launcher performs the regression tests, checks the frozen cohort and baseline
training fingerprints, compiles fast and reference Stan models, then runs twelve
comparisons across both families, both extensions and negative/zero/positive
extension effects. It compares complete Stan targets and gradients, compares the
Python likelihood, and checks the zero-effect likelihood against original Stan.
A failed check blocks all new sampling. No local laptop Stan compilation or
sampling was used to prepare this stage; the numerical Stan checks run on Linux2.

Four fits can run together, each with eight chains, 4,000 warmup and 4,000 retained
iterations per chain, adapt_delta .995, diagonal metric and maximum depth 14.
That is **up to 32 active sampling cores**. Compilation and validation initially
use fewer cores. This is an hours-scale job, unlike the short residual export;
wall time is uncertain until the extended models' sampling geometry is observed.
There are no automatic retries or automatic changes to priors/settings.

All original diagnostic thresholds remain: Rhat <1.01, bulk/tail ESS >=400,
zero divergences, zero maximum-depth hits and chain BFMI >.3. Added population
means/SDs and participant parameters are included. A failing model is saved but
excluded from predictive comparisons; the other workers continue. The parent
exports available accepted comparisons and exits nonzero if any worker failed.
Identical completed posterior caches may be reused; partial CSVs or changed
fingerprints block restart rather than being overwritten.

The existing frozen canonical table remains the data source. This new phase
checks every live input against the frozen inventory and admits only the already
reviewed removal of the two empty sub-10668 Trust run-2 imaging templates. Each
removed template must be authenticated by the completed repair receipt and its
original archived hash. Any other changed/added/missing input or read error
blocks fitting. Original sampling gates are unchanged. No cohort refresh or
upstream file writes occur. Live audits are recorded before and after fitting.

## What the comparison means

Primary comparison: amount model minus bias-only model within each family.
Secondary comparison: each new model minus its original counterpart. Lower
participant-averaged run-2 log loss is better. Intervals use 2,000 paired
participant bootstrap samples. All models score the same people and choice
counts. The training posterior stays fixed while observed run-2 feedback updates
beliefs online; this is not a joint marginal likelihood of the entire run.

**This is exploratory:** full-data residuals informed the new model choice,
so run 2 is no longer an untouched confirmatory test set for the extension.
An improvement supports further full-data checks and recovery, not a mechanism
claim or immediate interpretation of age effects. A nonsignificant result is
not proof that amount sensitivity is absent. No selection threshold is applied
automatically to score differences.

## Linux2 commands

```bash
tmux new-session -A -s rf1-trust-amount 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_amount.sh
```

Detach with Ctrl-b, then d. A successful run ends with `Amount comparison exit=0`.
If a worker returns exit 2, it failed the diagnostic thresholds; other fits keep
running and the complete logs should be pushed afterward.

After completion or failure:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical/amount_comparison results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record exploratory amount model comparison and logs"
fi
git push origin main
```

Results are under `results/full_sample/hierarchical/amount_comparison/`:
implementation checks, extension-prior screen, source audits, per-fit status,
diagnostics, parameter summaries, participant run-2 scores, run-2 conditional
partner/offer summaries, `available_model_comparison.tsv/png/pdf`, and
`amount_vs_bias.tsv`. Raw posterior CSVs stay in ignored
`work/full_sample/hierarchical/amount_comparison/`. The outer console and exit
record are under `results/run_logs/`.
