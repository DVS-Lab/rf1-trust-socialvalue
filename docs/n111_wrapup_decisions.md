# N=111 developmental-analysis decision record

**Status: closeout in progress.** Stage A is complete and reviewed (Linux results `d596163`). Its evidence supports testing the single shared zero-option feature. Stage B initial batch `e6eb3b3` has 11/12 accepted fits, with positive temporal-prediction evidence and mixed positive-positive PPC changes; see the [review](../results/n111_wrapup/zero_option_initial_review.md). Retention, realistic recovery and final synthesis remain pending. This is not a final full-sample analysis plan. See the [fixed scope](n111_wrapup_scope.md) and [closeout status](../results/n111_wrapup/README.md).

## Sample and trial conventions retained

- ds005123 v1.1.3, exact committed sample/trial hashes; primary N=111. Original descriptive N=113 and rating-complete primary N=103 are distinct samples. No additional participants are accessed.
- Preserve exclusions of ambiguously appended sessions and unverified/unpresented source rows, as documented in [data audit](data_audit.md). No new exclusions are introduced by this diagnostic.
- 8,442 presented primary decisions, 8,251 valid choices; misses remain in schedules but contribute neither likelihood nor feedback updates. Ninety participants have two ordinary runs; 21 have one.
- Canonical partner identities and exact offers (0–2, 0–4, 0–8, 2–4, 2–8, 4–8) are retained. Programmed outcomes exist even when unrevealed. Beliefs update only following observed feedback; simulated positive investments expose programmed outcomes, zero or missed choices do not.
- Existing temporal split: first ordinary run for training and second for held-out scoring; single-run participants use the first floor(65% of presented trials) for training. This gives 4,234 training and 4,017 held-out valid choices. Training parameters remain fixed during prediction.

## Models and inference

The accepted H2/H5/H7/HPreference/H8 training no-age fits are the inputs to stage A. Full-age accepted outputs remain valid for their established role; they are not relabeled as no-age fits. H4_full_age and H5_train_age remain excluded, with no resampling requested.

Existing correlated partial pooling uses logit learning rates, log kappa, softplus nonnegative social values, and untransformed signed preference parameters. Population-location priors on these latent scales are Normal(-1,1.25), Normal(-1.2,.8), Normal(1,1.5), and Normal(0,1), respectively. Population SDs have half-Normal(0,.8) priors; the correlation prior is LKJ(2). Passing diagnostics do not establish robustness to these priors. Raw theta and its tradeoff with kappa remain weakly identified; wider bounds and shrinkage have not resolved that scientific limitation.

The zero-option feature is **approved for the scoped comparison**, neither retained nor rejected. See [the fixed experiment](n111_zero_comparison.md). If fitted, gamma0 is a generic logit addition common across partners, with partial pooling and a documented weakly informative prior. No extension beyond this feature is authorized in this closeout.

## Diagnostic evidence so far

The committed full-age PPCs verify substantial underprediction on computer zero-containing offers in H5/H7/HPreference. H8 has a different offer pattern and is retained as a comparator. The stage-A no-age history audit completed successfully. Held-out zero-minus-positive residual contrasts are approximately .18 for friends, .29–.30 for strangers and .20 for computers, with positive bootstrap intervals throughout. The maximum conditional/generative difference across held-out offer cells is .005252. Thus compounding simulated feedback histories do not explain the main aggregate discrepancy; a generic zero-option term merits testing. This is not proof of a mechanism or a claim that every cell has the same sign. Most diagnostic views are partner × exact offer, then zero versus positive-positive, runs, chronological thirds, and pretrial same-partner feedback. Training and held-out data must remain distinguishable.

The stage A history strata use the same actual pretrial labels for both prediction types. This isolates differences in predictions on matched observed subsets; it does not claim to replicate the distribution of simulated feedback-category membership. Sparse cells are flagged. Predictive intervals include replicated-choice variation; participant-bootstrap zero contrasts are separately labeled.

## Recovery interpretation

The existing hierarchical recovery uses a lower theta range than the empirical fits. Its improved RMSE does not establish calibration or mechanism discrimination in the upper empirical range. Realistic targeted H5/H7/HPreference/H8 recovery is pending. The key reportable quantities are H7→H7/HPreference/H8 and HPreference→HPreference/H7/H8 selection rates. No discrimination conclusion is inferred from the present predictive tie alone.

## Ratings and age questions left open

Ratings exist for 103 primary participants, but pre/post timing is not established from saved files. Preserve secondary/descriptive status; no renewed investigation in this stage.

Behavioral age-25-to-75 friend-minus-computer change: +.025 (95% CI −.150 to +.201); friend-minus-stranger: +.002 (−.141 to +.146). Accepted H5 canonical friend-value probability change: −.050 (95% credible interval −.185 to +.078). These intervals allow meaningful effects in either direction. H5 latent variance fractions must not be described as percentages of all behavioral variability. No new N=111 age analyses are planned in this closeout.

## Outstanding closeout decisions

1. Answered: the relative zero-option misfit persists under actual histories; the main offer-cell discrepancy is not explained by simulated-history compounding.
2. Does the single shared zero term improve both PPCs and prospective log loss/Brier without damaging positive-positive fit?
3. Does it change interpretation of existing value, preference, or learning parameters?
4. Can H7 and HPreference be distinguished in realistic parameter ranges, including confusion with H8?
5. Record the answers, finalize the compact synthesis, and stop.

## Stage B initial review and one bounded retry (26 September)

All four training extensions pass and reduce mean participant log loss by .0607–.0856 and Brier score by .0249–.0366; paired bootstrap intervals exclude zero. Matched full-data H5/H7 zero-offer mean absolute cell errors shrink sharply; some positive-positive cells worsen. H5 median participant theta means change 6.33→2.80, H7 3.84→1.88. Thus carry-forward retention is promising but remains pending the missing HPreference PPC/parameter comparison and the mixed no-damage criterion. These exploratory comparisons reuse N=111 diagnostics; they are not confirmatory validation.

`N111_HPreference_zero_full` failed only the zero-depth-hit gate: 8/16000 hits at depth 12, no divergences, all other thresholds passed. One explicitly reviewed depth-14 attempt is prepared in a separate cache with unchanged target, seed, priors, chains, warmup, draws and diagnostic thresholds. Its scientific purpose is the missing matched comparison and an accepted empirical generator for recovery. The original failed attempt remains excluded and preserved. Eleven accepted runs are reused; neither H4_full_age nor H5_train_age is touched. If this attempt fails, there is no automatic further escalation. See [exact plan and Linux command](n111_zero_retry.md) and `config/n111_zero_retry.json`.

Recovery has not started. The feature decision remains open; no final mechanism-discrimination conclusion or full closeout completion is claimed.
