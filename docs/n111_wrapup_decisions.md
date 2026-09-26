# N=111 developmental-analysis decision record

**Status: N=111 closeout complete.** The zero-option diagnostic, one-feature comparison and realistic structural screen are complete; final recovery evidence is `46ad9c2`. Retain the generic zero term as a candidate with remaining PPC limitations. Do not launch hierarchical confirmation: mechanism separation is asymmetric and criterion-dependent, with weak short-task MLE estimation. The [authoritative final report](../results/n111_wrapup/README.md), [stopping record](../results/n111_wrapup/closeout_status.json) and five figure sets are complete. No additional participants were accessed. Earlier entries below record the historical sequence.


## Sample and trial conventions retained

- ds005123 v1.1.3, exact committed sample/trial hashes; primary N=111. Original descriptive N=113 and rating-complete primary N=103 are distinct samples. No additional participants are accessed.
- Preserve exclusions of ambiguously appended sessions and unverified/unpresented source rows, as documented in [data audit](data_audit.md). No new exclusions are introduced by this diagnostic.
- 8,442 presented primary decisions, 8,251 valid choices; misses remain in schedules but contribute neither likelihood nor feedback updates. Ninety participants have two ordinary runs; 21 have one.
- Canonical partner identities and exact offers (0–2, 0–4, 0–8, 2–4, 2–8, 4–8) are retained. Programmed outcomes exist even when unrevealed. Beliefs update only following observed feedback; simulated positive investments expose programmed outcomes, zero or missed choices do not.
- Existing temporal split: first ordinary run for training and second for held-out scoring; single-run participants use the first floor(65% of presented trials) for training. This gives 4,234 training and 4,017 held-out valid choices. Training parameters remain fixed during prediction.

## Models and inference

The accepted H2/H5/H7/HPreference/H8 training no-age fits are the inputs to stage A. Full-age accepted outputs remain valid for their established role; they are not relabeled as no-age fits. H4_full_age and H5_train_age remain excluded, with no resampling requested.

Existing correlated partial pooling uses logit learning rates, log kappa, softplus nonnegative social values, and untransformed signed preference parameters. Population-location priors on these latent scales are Normal(-1,1.25), Normal(-1.2,.8), Normal(1,1.5), and Normal(0,1), respectively. Population SDs have half-Normal(0,.8) priors; the correlation prior is LKJ(2). Passing diagnostics do not establish robustness to these priors. Raw theta and its tradeoff with kappa remain weakly identified; wider bounds and shrinkage have not resolved that scientific limitation.

The zero-option feature is **retained as a candidate with residual misfit** after the scoped comparison. See [the fixed experiment](n111_zero_comparison.md). Gamma0 is a generic logit addition common across partners, with partial pooling and a documented weakly informative prior. No extension beyond this feature is authorized in this closeout.

## Diagnostic evidence so far

The committed full-age PPCs verify substantial underprediction on computer zero-containing offers in H5/H7/HPreference. H8 has a different offer pattern and is retained as a comparator. The stage-A no-age history audit completed successfully. Held-out zero-minus-positive residual contrasts are approximately .18 for friends, .29–.30 for strangers and .20 for computers, with positive bootstrap intervals throughout. The maximum conditional/generative difference across held-out offer cells is .005252. Thus compounding simulated feedback histories do not explain the main aggregate discrepancy; a generic zero-option term merits testing. This is not proof of a mechanism or a claim that every cell has the same sign. Most diagnostic views are partner × exact offer, then zero versus positive-positive, runs, chronological thirds, and pretrial same-partner feedback. Training and held-out data must remain distinguishable.

The stage A history strata use the same actual pretrial labels for both prediction types. This isolates differences in predictions on matched observed subsets; it does not claim to replicate the distribution of simulated feedback-category membership. Sparse cells are flagged. Predictive intervals include replicated-choice variation; participant-bootstrap zero contrasts are separately labeled.

## Recovery interpretation

The existing hierarchical recovery uses a lower theta range than the empirical fits. Its improved RMSE does not establish calibration or mechanism discrimination in the upper empirical range. Realistic targeted H5/H7/HPreference/H8 recovery is complete; the final review below records criterion-dependent confusion and stops hierarchical confirmation. The key reportable quantities are H7→H7/HPreference/H8 and HPreference→HPreference/H7/H8 selection rates. No discrimination conclusion is inferred from the present predictive tie alone.

## Ratings and age questions left open

Ratings exist for 103 primary participants, but pre/post timing is not established from saved files. Preserve secondary/descriptive status; no renewed investigation in this stage.

Behavioral age-25-to-75 friend-minus-computer change: +.025 (95% CI −.150 to +.201); friend-minus-stranger: +.002 (−.141 to +.146). Accepted H5 canonical friend-value probability change: −.050 (95% credible interval −.185 to +.078). These intervals allow meaningful effects in either direction. H5 latent variance fractions must not be described as percentages of all behavioral variability. No new N=111 age analyses are planned in this closeout.

## Closeout decisions (resolved)

1. Answered: the relative zero-option misfit persists under actual histories; the main offer-cell discrepancy is not explained by simulated-history compounding.
2. Answered with qualification: all four temporal comparisons improve and zero-offer PPCs improve markedly in the partner models, but positive-positive changes are mixed. Retain only as a candidate with residual misfit.
3. Answered: value/preference estimates fall and kappa rises after gamma0, demonstrating sensitivity to the choice specification.
4. Answered with limits: the realistic fast screen shows substantial, asymmetric H7/HPreference confusion, additional H8/H5 confusion and criterion-dependent dataset choices. It does not establish reliable mechanism discrimination or mathematical impossibility.
5. Complete: answers, five figure sets, machine-readable conclusions and the final synthesis are saved. Stop this N=111 phase; no new sampling requested.

## Stage B initial review and one bounded retry (26 September)

Historical entry from `e6eb3b3`; the accepted retry and completed feature decision are recorded below.

All four training extensions pass and reduce mean participant log loss by .0607–.0856 and Brier score by .0249–.0366; paired bootstrap intervals exclude zero. Matched full-data H5/H7 zero-offer mean absolute cell errors shrink sharply; some positive-positive cells worsen. H5 median participant theta means change 6.33→2.80, H7 3.84→1.88. Thus carry-forward retention is promising but remains pending the missing HPreference PPC/parameter comparison and the mixed no-damage criterion. These exploratory comparisons reuse N=111 diagnostics; they are not confirmatory validation.

`N111_HPreference_zero_full` failed only the zero-depth-hit gate: 8/16000 hits at depth 12, no divergences, all other thresholds passed. One explicitly reviewed depth-14 attempt is prepared in a separate cache with unchanged target, seed, priors, chains, warmup, draws and diagnostic thresholds. Its scientific purpose is the missing matched comparison and an accepted empirical generator for recovery. The original failed attempt remains excluded and preserved. Eleven accepted runs are reused; neither H4_full_age nor H5_train_age is touched. If this attempt fails, there is no automatic further escalation. See [exact plan and Linux command](n111_zero_retry.md) and `config/n111_zero_retry.json`.

Recovery has not started. The feature decision remains open; no final mechanism-discrimination conclusion or full closeout completion is claimed.

## Accepted retry and feature decision (results cf9b33d)

The one selected HPreference depth-14 attempt passed: no divergences or depth hits, Rhat max 1.00446, bulk ESS min 727.878, tail ESS min 1888.57, minimum BFMI .6447. All twelve comparison sources are now accepted. Its original failed attempt is preserved and excluded; no additional empirical posterior fits are requested.

Retain the common zero-option term as an important candidate feature, because all four temporal comparisons improve and zero-offer PPC errors shrink sharply in H5/H7/HPreference. The strict no-damage criterion is not uniformly met: some positive-positive cells worsen. This qualified retention is explicitly not a claim of complete PPC repair. The larger sample must evaluate the feature independently; no further feature is added here. Partner/value estimates fall materially after adjustment. See the numerical [feature decision](../results/n111_wrapup/zero_option_decision.md).

Prepare exactly the bounded [realistic recovery screen](n111_realistic_recovery.md), with adjusted models primarily, originals secondarily, eight posterior population draws per generator and two separately labeled high-theta stress datasets for each social-value generator/family. Fit all four competing models to each simulated participant with 32 starts and wide, range-audited bounds; preserve the actual schedules and missingness. Report individual and complete-dataset AICc/BIC and temporal prediction separately. Up to 40 single-threaded workers are allowed on linux1. Caches and logs support resumption. No hierarchical confirmation launches automatically; first inspect H7↔HPreference confusion, H8 confusion, prediction differences and optimizer boundaries.

## Final recovery review and stopping decision (results 46ad9c2)

All 72 simulated N=111 datasets completed: 7,992 participant cases, 63,936 full/training candidate fits, no unresolved optimizer errors. Eight empirical posterior population draws per generator/family and two separately labeled high-theta stress draws per social-value generator/family used the actual schedules and missingness. The generating theta tail reached 22.61; the fitting ceiling expanded to 24.87, so no generator was clipped or placed outside the fitted range.

In the primary zero-adjusted empirical screen, individual AICc selected H7/HPreference/H8 at .1796/.2033/.1233 when H7 generated, and HPreference/H7/H8 at .4234/.1993/.0940 when HPreference generated (remaining selections were H5). Whole-dataset AICc selected H7 in 2/8 H7 datasets and HPreference in 6/8; it selected HPreference in 8/8 HPreference datasets. BIC and temporal criteria differed. This asymmetric performance does not support robust separation of both mechanisms. Whole-dataset comparisons sum nonhierarchical criteria, not hierarchical marginal likelihoods. Eight population draws constrain rate precision.

MLE boundary behavior is common even for the correct generator: H7 69.5% full /78.8% training; HPreference 51.1%/65.5%. Independent short training fits sometimes give extreme test losses (primary empirical maximum about 727.9), making mean-log-loss ranks sensitive to overconfident errors. No losses were clipped or removed to improve rankings. The 32-start near-best audit and nested-likelihood checks passed, but numerical convergence is not statistical identification. Parameter-boundary details and loss tails are exported.

**No hierarchical confirmation is launched.** Under the user's stopping rule, the screen's strong confusion and estimation limitations do not justify expensive confirmation in this closeout. This is a qualified evidence limitation, not proof that a different estimator/design could never distinguish mechanisms. The final report states the zero-option PPC tradeoff, parameter shifts, existing age uncertainty, unresolved rating timing and preserved exclusions. No more models, age forms, ratings work or additional participants are introduced.

Rebuild the final audit and synthesis with `python scripts/26_finalize_n111.py`; it starts no fits and requires only committed tables. It reconstructs all confusion rates and checks coverage, original splits, likelihood criteria, optimizer summaries, source fingerprints, generating ranges and paired predictive scores. The earlier accepted-fit report remains preserved.
