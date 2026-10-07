# Amount-comparison review — 7 October 2026

Results commit: `854b523`. The batch ran on 30 September, 15:36–17:23 UTC
(approximately 1 hour 47 minutes). All four new fits completed sampling. One
passed every diagnostic; three were rejected for retained divergences. The
outer exit 1 was the deliberate diagnostic gate, not an interrupted sampling job.
The Linux2 regression suite passed 268 tests and all twelve Stan/Python
likelihood, target/gradient and zero-effect nesting checks passed.

## Diagnostic disposition

Each fit has eight chains and 32,000 retained draws. No fit hit maximum treedepth.
All Rhat, bulk/tail ESS and BFMI thresholds passed, including the added population
and participant coefficients. The zero-divergence criterion remains unchanged.

| Training model | Status | Divergences | Max Rhat | Min bulk ESS | Min tail ESS | Min BFMI |
|---|---|---:|---:|---:|---:|---:|
| HPreference + zero + bias | Accepted | 0 | 1.00191 | 2,078 | 4,301 | 0.667 |
| H7 + zero + bias | Unaccepted | 1 | 1.00429 | 2,704 | 5,869 | 0.658 |
| H7 + zero + bias + amount | Unaccepted | 3 | 1.00400 | 1,274 | 2,330 | 0.665 |
| HPreference + zero + bias + amount | Unaccepted | 4 | 1.00438 | 1,418 | 3,188 | 0.669 |

The retained divergences occurred in chain 5 for H7 bias; chains 2 and 6 for H7
amount; and chains 3, 4 and 5 for HPreference amount. Console LKJ proposal warnings
are also present. They do not establish a causal explanation for these retained
divergences. No unaccepted model is scored or used for scientific parameter claims.

## Accepted predictive result

All rows below score the same 304 participants and 12,494 valid run-2 choices.
Parameters come only from run 1; observed feedback updates beliefs online in run 2
while posterior parameters remain fixed. Scores average per-trial log loss
within participant and then weight participants equally. Lower is better.

| Model | Run-2 log loss |
|---|---:|
| HPreference + zero + bias | 0.48097 |
| Original H7 + zero | 0.50549 |
| Original HPreference + zero | 0.51177 |

Adding bias to HPreference improves log loss by -0.03080, with a paired
participant-bootstrap 95% interval [-0.03983, -0.02178]. This is a 6.0% relative
reduction in log loss, not a six-percentage-point increase in accuracy. It improves
individual mean log loss for 66.1% of participants.

Against the previous leading original H7 model, the accepted bias extension has
a difference of -0.02452 [-0.03230, -0.01676], improving individual log loss for
63.2% of participants. The intervals use 2,000 paired bootstrap samples with seed
20260930 and condition on the fitted predictive scores. The figure uses a common
original-H7 baseline; the raw batch figure used each family's own original model.

This is encouraging evidence that a general participant-specific tendency to
choose high matters for prediction in this candidate set. It is not a completed
comparison between the H7 and HPreference mechanisms: the corresponding H7 bias
model is still unaccepted. Nor does it answer whether amount sensitivity adds
anything beyond bias: neither amount model passed, so `amount_vs_bias.tsv` has
only a header. An empty contrast table is not a null effect.

The accepted population bias mean is 0.516 logit units, with a 95% posterior
interval [0.335, 0.696], and population SD about 1.20. The coefficient is a general
choice intercept, not a unique measure of trust, generosity, risk preference or
age. Full-data residuals informed this extension, so reused run-2 performance is
exploratory rather than untouched confirmatory evidence.

## What is still unexplained

Run-2 conditional mean predictions from the accepted bias model retain exact-offer
errors. Examples (equal participant averages within each cell):

| Partner / offer | Observed high-choice rate | Predicted |
|---|---:|---:|
| Friend / 2 versus 4 | 78.1% | 70.6% |
| Friend / 4 versus 8 | 66.8% | 75.2% |
| Stranger / 4 versus 8 | 36.8% | 48.8% |
| Computer / 0 versus 8 | 52.0% | 43.3% |

These are run-2 conditional predictions, not the previous N=343 full-data
replicated-choice checks. Do not compare those residual magnitudes directly as
if cohort, training information and prediction type were identical. The current
pattern supports finishing the amount comparison; it does not prove that the
specific quadratic correction will resolve the mismatch.

`accepted_bias_run2_cells.tsv` gives all cells and participant-bootstrap intervals
for observed-minus-predicted means. They condition on posterior-mean predictions,
exclude posterior uncertainty, and are not corrected for multiple comparisons.
No age covariates or mechanistic parameter recovery were estimated in this run.

## Next step

Preserve the accepted HPreference bias fit and all original models. Inspect the
saved divergent-state/chain context for the three unaccepted new fits, then make
one explicitly configured, more conservative sampler retry if that inspection
supports it. Keep the equations, priors, data and acceptance thresholds fixed;
use separate retry caches. More retained draws alone do not address the recorded
failure, because ESS already passes. If divergences recur, address the geometry
rather than repeatedly increasing computation or dropping troublesome draws.

Do not simply rerun the unchanged launcher: its matching caches would reload the
same posteriors and the same divergences. This review adds no sampling job and
changes no accepted/unaccepted designation. The proposed diagnostic export and
retry are subsequent work, not actions already executed by this review.

## Provenance and reproduction

Run `python scripts/review_amount_results.py` in the project environment. It
checks recorded implementation/output hashes, recalculates the diagnostic
acceptance flags, verifies paired participant identities and choice counts, and
writes `diagnostic_inventory.tsv`, `accepted_contrasts.tsv`,
`accepted_bias_run2_cells.tsv`, and `accepted_prediction_comparison.png/pdf`.
`status.json` records the verified evidence and output hashes.

The raw posterior CSVs and ignored canonical trials remain on Linux2. Their
recorded Linux2 verification is distinct from this portable published-results
review. All before/after audits retain the same two authenticated removed imaging
templates for sub-10668; the modeled behavioral inputs did not change in the run.
