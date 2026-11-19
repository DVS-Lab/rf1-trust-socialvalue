# Full-cohort hierarchical Trust DDM assessment

## Status and interpretation

This analysis retains the complete certified behavioral cohort, including constant responders and valid single-run participants. Its purpose is to assess whether partial pooling supports defensible individual z and friend–stranger contrasts. It runs no age, FEVS, closeness or neural association analysis.

Observed-fit stage: **stale_prior_run**. Recovery stage: **blocked_primary_posterior_or_execution**.

**A scientific GREEN decision is not automatic.** Accepted sampler diagnostics must be considered with full-cohort recovery, uncertainty calibration, RT/choice predictive fit, drift–z confounding and prior/specification sensitivity. Diagnostic failures are execution/estimation findings, not participant exclusion rules.

## Cohort and timing

- Full behavioral roster: **343 participants**, 647 certified runs.
- Scientifically retained runs after upstream settled decisions: **646**; settled run holds applied: 1.
- Retained modeled decisions: **26494**; observed missed decisions: 583.
- Preregistered variability subset: **258**, recorded for comparison; it does not restrict primary hierarchical fitting.
- Participants with no retained RTs: 0; their latent estimates, if any, have no personal likelihood information.
- Responses under 200 ms **from offer availability**: 99; retained with the contaminant model.
- Recorded responses beyond the nominal cue-based response window: 24; withheld from the nominal-window likelihood, without a participant exclusion.

The verified task script has a 0.5-s partner-only cue before offers. Its response window ends 3 s from cue onset. The model subtracts the known cue interval and estimates remaining nondecision time. Current task code is verified; historical per-run executable versions remain a provenance limitation. Previously reported cue-based RT thresholds must not be interpreted as offer-based response thresholds.

## Imaging intersection

Status: `derived_Trust_run_QC_intersection_preserving_upstream_decisions`; overlapping imaging participants: 320.

The separate Trust imaging intersection uses saved Trust/session FD and tSNR thresholds, coverage ≥95%, and recorded upstream holds/source exclusions. Good individual runs remain available. It uses no age restriction, TEDANA-count cutoff or effect-size exclusion. Missing imaging/QC records are not assumed to pass. Imaging eligibility does not discard valid behavior from the full behavioral fit, and this table does not certify completed activation/PPI contrasts. An imaging-member sensitivity fit uses all certified, scientifically retained behavior of those members.

## Model

Both models use a joint choice/RT Wiener likelihood with unit diffusion noise, conditional on the retained RT window, and a fixed 1% unconditional uniform contaminant mixture before conditioning. The upper boundary denotes higher investment. The shared model has one starting point across partners within each person; the partner model has friend/stranger/computer starting points. Every person is estimated jointly through a correlated noncentered population distribution (LKJ(2) correlation prior). Boundary separation is positive without the pilot's upper ceiling; z spans its natural open unit interval. Nondecision time has intrinsic support below the response window, without the pilot's 0.8-s cap. Individual drift intercepts and an offer-gap slope are partially pooled; common zero-offer drift and motor-side bias terms are included. The history sensitivity adds lagged observed partner feedback and trial position to drift; the wider-prior sensitivity doubles population prior scales. These changes mean improvement over the old pilot cannot be attributed to pooling alone.

A disclosed numerical guard rejects extreme proposals before Stan Math's CDF series can become nonterminating: a²/(deadline−t0)≤1000, a≥1e−6 and |drift|≤1e6. Prior-predictive populations are conditioned on this domain, and posterior proximity to the guard is exported. Estimates near it require numerical-domain sensitivity.

Stan's native Wiener density/CDF are used and checked against the independent reference likelihood, analytic probabilities and finite-difference gradients. The independent Euler simulator checks recovery without reusing the fitting density. Missed responses are not imputed; fixed empirical missingness assumes omissions are noninformative. Generative history reveals programmed feedback only after a retained positive investment.

## Sampler diagnostics

| Model | Status | N | Divergences | Max R-hat | Minimum bulk ESS | Minimum tail ESS |
|---|---|---:|---:|---:|---:|---:|
| partner | stale_prior_run | — | — | — | — | — |
| partner_history | diagnostic_failed | 343 | 10571 | 4.26301 | 4.24779 | 27.3205 |
| partner_imaging | stale_prior_run | — | — | — | — | — |
| partner_wide_prior | stale_prior_run | — | — | — | — | — |
| shared | stale_prior_run | — | — | — | — | — |

Thresholds: zero divergences/depth-limit hits, R-hat <1.01, bulk/tail ESS ≥400, BFMI >.3. Raw chains, participant summaries and parameter correlations remain private. Only accepted fits supply scientific summaries.

## Posterior predictive fit

Predictive checks integrate parameter uncertainty and cover all participants. They compare participant-weighted partner, exact-offer and choice-specific RT cells; report the full cohort and preregistered subset separately. Conditional checks preserve the observed response mask; a separate unconditional simulation reports predicted retention.

## Full-cohort recovery

Each recovery dataset uses the complete roster, actual partner/offer/run order and original missingness pattern. New participant effects are generated from accepted population posterior draws. Small/moderate/large latent z variability is controlled explicitly; realized natural-z variability is reported rather than assumed. Additional datasets set the friend–stranger z contrast to zero while allowing drift differences. Two further stress datasets change partner boundary separation or nondecision time by 10% with zero true z contrast, testing whether the simpler fitted structure misattributes those effects to z. No simulated constant responder is dropped. Recovery is stratified by original eligibility, realized simulated constant responding, trial count and absence of observations. Cohort mean/SD contrasts, participant ranks, bias, RMSE, 95% interval coverage and null contrast exclusions are reported. Four population datasets per regime and two null datasets remain a limited Monte Carlo calibration study; participant cases within a dataset are not independent population replications.

Full-cohort hierarchical recovery has not produced accepted summaries yet.

## Review before any subsequent association analysis

Inspect predictive tails, prior/specification dependence and posterior drift–z dependence; verify recovery for constant responders and low-variation populations; inspect uncertainty rather than treating shrinkage as identification. Resolve any upstream scientific-validity reviews still open. A useful cohort effect does not establish precise participant ranking. No automatic report label can authorize downstream association tests.

The earlier independent-fit pilot and completed RL analyses are preserved. Publication includes only aggregate tables/figures/provenance. Participant identifiers, data, fitted values and chains stay under ignored `work/ddm_hierarchical/`. See [the full-cohort workflow](../../feasibility/ddm/HIERARCHICAL.md).
