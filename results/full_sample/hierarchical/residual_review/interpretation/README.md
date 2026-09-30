# Offer and age diagnostic interpretation — 30 September 2026

Reviewed Linux2 result commit `604868a`. Both workers completed with exit 0;
257 tests passed on Linux2. Postprocessing took 26 seconds after the test suite,
using the authenticated frozen N=343 cohort and existing accepted posterior draws.
No MCMC was run. Cells contain 342–343 participants because some people have no
valid choice in a particular cell, not because of age exclusions. All age slopes
were estimable. The seven output hashes and 30 implementation/fit links verified;
all 40 overlapping summaries reproduce the previous PPC (counts and means).

## What changed in our understanding

The mismatch is more specific than a general shortfall in computer investment.
Both leading models miss a shared pattern across exact amounts and partners:

| Partner and offer | Observed high-choice rate | H7 + zero | HPreference + zero |
|---|---:|---:|---:|
| Friend, 2 versus 4 | 80.8% | 65.5% | 65.1% |
| Friend, 4 versus 8 | 67.7% | 71.4% | 71.6% |
| Stranger, 4 versus 8 | 38.1% | 45.4% | 44.3% |
| Computer, 0 versus 2 | 57.1% | 63.8% | 63.7% |
| Computer, 2 versus 8 | 37.9% | 26.4% | 26.8% |

Predictions are generative posterior means, with equal participant weighting.
The selected rows illustrate the pattern; the original table and plots contain
all 18 partner/offer cells per model. For friend 2/4 the observed-minus-predicted
residual is +15.3 percentage points for H7 (95% predictive interval +12.6 to +17.6)
and +15.7 for HPreference (+13.5 to +18.3). For computer 2/8 the corresponding
residuals are +11.5 (+9.9 to +13.3) and +11.1 (+9.2 to +12.8).

The signs reverse elsewhere: H7 overpredicts stranger 4/8 by 7.3 points and
computer 0/2 by 6.7 points. Conditional predictions using actual feedback history
show essentially the same mean residual pattern. These mean discrepancies are
therefore not confined to simulated-history behavior.

A uniform upward shift in predicted high choice would improve some cells while
worsening others at the current fitted parameters. This does not rule out a
refitted intercept model, but weakens the earlier suggestion that a general
investment bias alone is the natural first extension.

## Age: behavior varies, but the remaining errors have limited broad age structure

Unadjusted observed high-choice slopes, in percentage points per decade:

| Cell | Observed slope [95% participant bootstrap interval] | H7 predicted slope | HPreference predicted slope |
|---|---:|---:|---:|
| All choices | +1.25 [+0.09, +2.40] | +1.06 | +1.12 |
| Friend | +0.15 [-1.06, +1.38] | +0.18 | +0.30 |
| Stranger | +1.57 [+0.03, +3.10] | +1.30 | +1.42 |
| Computer | +1.96 [+0.32, +3.50] | +1.63 | +1.56 |

These slopes are descriptive linear associations within this cross-sectional
cohort, not adjusted or causal age effects. The models have no population age
regression, but estimate participant-specific parameters from these same
participants' choices. Thus reproducing an age pattern here does not show that a
model predicts age effects for a new person or that age explains a particular
fraction of latent parameter variance.

For the exact-offer residual age slopes, all 36 participant-bootstrap intervals
include zero for each history treatment. For the generative predictive intervals,
35/36 include zero; H7 computer 4/8 is the exception (+1.42 points/decade,
interval +0.10 to +3.07). Its participant-bootstrap interval includes zero
(-0.07 to +3.01), as does the HPreference generative interval for the same cell.
This is an exploratory, unadjusted comparison across many cells, with only 200
replicates; it is not strong evidence for a special age-specific mechanism.

There is an important qualification: 14/36 **conditional expected-behavior**
intervals exclude zero. These condition on the observed people and histories
and omit replicated-choice noise, so they are narrower. The participant-bootstrap
intervals for those residual slopes still include zero. Neither interval type
can be substituted for the other. The complete conditional results remain in the
source table. We should describe age-dependent misfit as uncertain, not absent.

The clearest result is the large, consistent mean error across exact offers.
These diagnostics do not establish that adding age would fix it, and do not
answer how much model-parameter variability age explains.

## Next computational decision

Prioritize a small matched comparison of the shared amount-dependent decision
rule in H7 and HPreference before mechanistic interpretation of their age effects.
Their current likelihood uses the investment difference linearly in the value
contrast. A sensitivity allowing nonlinear amount valuation or amount-dependent
cost is a candidate motivated by these residuals, not an established explanation.
Specify its equation and priors, check likelihood/gradient parity and recovery,
and compare it with the frozen originals under the same run-1 to run-2 task.
Any such extension is exploratory because these full-data diagnostics informed
its selection; the existing run-2 data are not a newly untouched test set.

Do not add a separate arbitrary coefficient for every problematic cell merely
to erase the plot. Keep both leading psychological model families in the focused
comparison. Carry the age question forward to a hierarchical age model with
recovery and model-adequacy checks; regressions on posterior means are not a
substitute. The original 18/20 accepted-fit inventory and the two unaccepted
one-divergence controls remain unchanged. No automatic retry is warranted here.

This review creates no new sampling job and does not change the frozen cohort or
live-source fitting gate. The live audit still records only the two archived
sub-10668 imaging templates affected by the independent upstream repair.

## Files and uncertainty

`selected_offer_checks.tsv` contains the five displayed offer cells under both
history treatments; `pooled_age_slopes.tsv` contains the four pooled age summaries
for both models under generative histories. Both are direct selections from
`../offer_age_residuals.tsv`; probability-scale slopes there are per decade
(multiply by 100 for percentage points). `review_status.json` records evidence
and output hashes. Full figures are `../residual_by_offer.png` and
`../age_residual_by_offer.png`, with PDF counterparts.

Predictive intervals hold the observed cohort fixed. Residual participant
bootstrap intervals condition on posterior-mean predictions and do not include
posterior uncertainty. Neither is a multiplicity-adjusted hypothesis test. See
`docs/linux2_full_sample_residuals.md` for the complete generation method.
