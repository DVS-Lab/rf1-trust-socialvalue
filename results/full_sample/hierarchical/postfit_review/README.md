# Full-cohort post-fit review, 30 September 2026

Linux2 result commit: `e18f880`. All nine accepted full-data posterior predictive
checks and both diagnostic exports completed successfully. The wrapper passed
253 tests and finished in approximately four minutes. No new sampling occurred.
The primary cohort remains N=343 with 26,560 valid choices.

## Main finding

H7 + zero and HPreference + zero capture much of the partner/zero-option pattern,
but share a substantial residual in the computer condition with two positive
options. Both also underpredict variation between participants, particularly
for computer choices. These are model-adequacy findings, not new heldout model
rankings or proof of either psychological mechanism.

The earlier accepted run-1 to run-2 comparison still favors H7 + zero slightly:
mean log loss 0.5055 versus 0.5118; paired difference -0.00628, 95% participant
bootstrap interval [-0.00887, -0.00364]. Its prediction cohort is N=304, distinct
from the N=343 full-data checks here. See `../accepted_review/README.md`.

## Observed and generative high-choice rates

These are equal-participant averages within each condition. Zero means the lower
investment option was zero; positive means both options were positive.

| Condition | Observed | H7 + zero, mean [95% interval] | HPreference + zero, mean [95% interval] |
|---|---:|---:|---:|
| Friend / zero | 85.1% | 84.5% [83.2, 85.5] | 84.6% [83.3, 85.8] |
| Friend / positive | 74.1% | 71.4% [70.2, 72.9] | 71.7% [70.4, 73.0] |
| Stranger / zero | 65.5% | 65.2% [63.7, 66.9] | 64.9% [63.6, 66.4] |
| Stranger / positive | 45.1% | 45.3% [44.0, 46.5] | 44.4% [43.0, 45.8] |
| Computer / zero | 53.1% | 54.0% [52.3, 55.4] | 54.3% [52.9, 56.1] |
| Computer / positive | 38.3% | 30.0% [28.6, 31.3] | 30.2% [28.9, 31.6] |

The computer/positive shortfall is 8.3 and 8.1 percentage points. Conditional
predictions using actual observed feedback also predict 29.9% and 30.1% there.
Thus the aggregate shortfall is present under both history treatments. This
does not identify which model component should change.

Both models also miss the offer pattern: for the 2-versus-4 offer, observed
high-choice rate is 58.7%, versus 49.9% and 49.3% predicted, pooling partners.
Offer and partner residuals overlap and should not be counted as independent
pieces of evidence.

## Participant variation and consistent choices

| Summary | Observed | H7 + zero | HPreference + zero |
|---|---:|---:|---:|
| SD of participant overall high-choice rates | 0.211 | 0.178 | 0.174 |
| SD of participant computer high-choice rates | 0.295 | 0.208 | 0.208 |
| Always high across all valid choices | 2.04% (7/343) | 0.000% | 0.0015% |
| Always high for computer choices | 5.25% (18/343) | 0.036% | 0.029% |

Predicted entries are averages across 200 replicated datasets. They are not
probabilities estimated with arbitrary precision: rare-event tails are coarse
at this replication count. Observed missingness and offers are held fixed.
These summaries use existing participants' posterior parameters, so they do not
test generalization to entirely new participants. The models already include
hierarchical participant variation; this stage has no age covariates.

## Remaining diagnostics

| Fit | Divergent retained state | Divergences / draws | Min chain BFMI | Max-depth hits |
|---|---|---:|---:|---:|
| Train H5 base retry1 | Chain 1, draw 2052 | 1 / 32,000 | 0.708 | 0 |
| Full HPreference base retry1 | Chain 5, draw 2670 | 1 / 32,000 | 0.576 | 0 |

Both remain unaccepted under the unchanged zero-divergence criterion. The saved
states show no population scale near zero. This does not rule out difficult
geometry elsewhere along the trajectory. Extreme participant coordinates among
thousands of inspected values do not establish a causal explanation. The export
contains saved iteration states, not the internal leapfrog locations where the
numerical failures arose. No chain/draw is deleted and no automatic retry is
launched. The 18/20 accepted-fit count remains unchanged.

## Upstream repair

Both audits identify only the removed sub-10668 Trust run-2 mag/phase templates.
Their archived hashes match the frozen hashes exactly. The repair receipt says
complete; the BIDS marker separately says derivatives require rebuilding. This
supports the expected source-file change, not completion of every upstream
processing stage. Those empty files supplied no modeled Trust choices. All
post-fit work used the authenticated frozen trial table and existing posteriors.
New-fitting entry points retain their strict live-source verification.

## Next scientific step

1. Localize the shared mismatch by partner × exact offer and by continuous age,
   retaining participant clustering. Compare both leaders under the same checks.
   Do not equate regression on posterior means with a hierarchical age model.
2. Use that evidence to specify a small, explicitly exploratory sensitivity to
   the shared decision rule (for example, a general investment bias). The
   current residuals motivate investigation but do not uniquely support an
   intercept, a new latent group, or a specific learning mechanism.
3. Evaluate any extension with the original run-1 to run-2 comparison and focused
   predictive checks, then parameter/model recovery and hierarchical age effects.
   Keep the original model results as the frozen benchmark.

A further broad retry of all fitted models is not the next priority. Age could
explain some participant variation, but this no-age PPC cannot quantify that.
This review launches no new fit and changes no scientific inclusion threshold.

## Reproduction and provenance

Run `python scripts/review_full_sample_postfit.py` from this repository using the
project environment. It verifies published output hashes and fit-status links,
then writes the focused PNG/PDF, focal table, diagnostic summary and evidence
manifest in this directory. This is a portable results-only review; the raw
Linux2 posterior CSVs and ignored canonical table are not available locally.
Their verification is documented in the Linux2 stage records. The report's
figure shows predictive uncertainty; it does not show corrected hypothesis tests.
