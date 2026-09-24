# Model definitions and estimation

## Common decision process

The participant keeps `8-x`. Reciprocation returns `1.5x`; defection returns zero. Therefore `V(x)=8-x+1.5xP`. For two investments, `ΔV=(xH-xL)(1.5P-1)`, making P=2/3 the risk-neutral indifference point. Use the logistic choice rule with inverse temperature kappa. Exact negative log likelihood is `logaddexp(0,z)-choice*z` for logit z, avoiding overflow. With a lapse parameter, the Bernoulli probability is `lapse/2+(1-lapse)*logistic(z)`.

Partner beliefs start at .5 and update only after displayed feedback. The selected partner alone updates, using `P <- P+alpha*(y-P)`. No counterfactual learning follows zero investments; misses neither contribute likelihood nor update beliefs. Beliefs carry across ordinary runs by default. The run-reset robustness model resets to its model-specific initial prior at each boundary.

## Core candidate set

| ID | Free parameters | Change from common process |
| --- | --- | --- |
| M0 | none | Probability .5; no learning |
| M1 | kappa | All beliefs fixed at .5 |
| M2 | alpha, kappa | Monetary RL |
| M3 | alpha, kappa, phi | Initial P=min(phi*S,1−1e−9), S=(rating+5)/10 |
| M4 | alpha, kappa, theta | Extra value P*x*theta*S |
| M5 | alpha, kappa, theta | Extra value P*x*theta*I(friend) |
| M6 | alpha, kappa, theta | Extra value P*x*theta*I(human) |
| M7 | alpha, kappa, theta, theta_stranger | Separate friend/stranger extra values; computer is zero reference |
| M8 | alpha, alpha_negative, kappa | alpha applies for positive error, alpha_negative for negative error |

M3/M4 are secondary because rating timing is unverified. The model IDs reflect the brief; their numerical parameter values are not a direct replication of the original temperature notation. The exact prior clipping has flat regions once ratings saturate; near-best-start counts and recovery expose the resulting weak identification.

Bounds: alpha [0,1], kappa [1e−5,20], theta/phi [0,5]. Direct bounded optimization is used instead of transforms, allowing exact zero learning/social value. One start uses a common moderate initialization; the remaining starts use uniform parameters except log-uniform initial kappa on [.01,5]. All parameters subsequently optimize over their full bounds. L-BFGS-B uses a compiled central-difference gradient; Powell provides a fallback only if unresolved failure remains. A converged alternative can replace a failed line-search endpoint only if its NLL differs by at most 1e−4. Unresolved failures stop execution. Best matches are counted within 1e−4 NLL; boundary flags mark 1e−4 of a parameter's range. Boundary flags are numerical diagnostics, not significance tests.

## Robustness controls

- `*_signed`: permit signed social bonuses on [−5,5]. Negative bonus is an aversion term; it need not keep total utility positive.
- `*_lapse`: lapse ∈[0,.2], allowing a small random-choice component.
- `*_side`: side_bias ∈[−5,5], added to the high-choice logit as `side_bias*(2*high_is_right−1)`; positive bias favors the right button independently of amount.
- `*_reset`: restart latent beliefs at each run instead of carrying them.
- `M5_wide`: kappa up to 100.
- `M4_ratingcentered`: S=rating/5 on [−1,1], with nonnegative theta. Negative ratings can reduce reciprocation utility; this is not used as a probability prior.
- `M2_power`: `EU(x)=(1−P)*(8−x)^rho + P*(8+.5x)^rho`, rho ∈[.2,3]. Zero wealth has utility zero. No additive wealth background is assumed.
- `M5_power`: the previous monetary EU plus `P*x*theta*I(friend)`. Theta is in utility units. At rho=1 the equation exactly equals M5. Kappa and rho can trade off, so curvature is a misspecification check, not a uniquely identified risk trait.
- `preference`: monetary RL plus `x*(theta*I(friend)+preference_stranger*I(stranger))`, with two signed slopes [−5,5]. These slopes are independent of reciprocation probability, representing partner-specific investment preferences. This control is required before assigning a uniquely reciprocation-reward interpretation.

The full model-recovery set contains M0–M8 plus preference, M2_power, M5_power. Richer three-bonus and partner-specific asymmetric-learning models are omitted because simpler models already have substantial boundary/identification issues.

## Comparison and prospective prediction

All within-participant model comparisons use the identical choice mask. Primary candidates are M0, M1, M2, M5, M6, M7, M8. Secondary comparisons restrict all candidates to rating-complete participants. AIC=2k−2LL; AICc adds `2k(k+1)/(n−k−1)`; BIC=k*log(n)−2LL. Fareri pseudo-R²=(AIC_random−AIC_model)/AIC_random; McFadden=1−LL_model/LL_random. Negative indices are retained. Paired Wilcoxon p values use Holm correction separately for each metric/sample family; participant bootstrap means use 5,000 draws.

For ordinary two-run records, fit run 1 and prospectively score run 2. For single runs, fit the first floor(.65*n_presented) decisions and score the rest. Freeze parameters but keep learning online. Latent beliefs at the training endpoint carry into prediction. Probabilities on trial t never depend on feedback at or after t. Report participant-mean log loss, Brier score, accuracy, and calibration. Deterministic accuracy calls ties high; therefore M0 accuracy need not equal .5. The proper scoring rules are the primary measures.

## Recovery and predictive checks

Each non-null core model is simulated 50 times per fitted participant, then refit with 50 starts. Actual partner/order/offers/programmed outcomes are preserved, the observed miss mask is fixed, and choice determines whether feedback is seen. Choices are Bernoulli draws, never deterministic thresholding. Correlation, RMSE, bias, and boundary rates are computed across participants within each repetition; the summary averages these across repetitions.

Model recovery generates two trajectories per participant/generator for every rating-complete participant and fits all 12 candidate models to each trajectory with 50 starts. Recovery is conditional on empirically fitted parameters, not uniform coverage of parameter space. Exact criterion ties share credit.

Simulation checks use 200 trajectories per participant/core model. Aggregate by partner, offered pair, time, and last actually observed same-partner feedback in each simulated history. Intervals are conditional simulation ranges, without parameter or population uncertainty. They are not Bayesian posterior predictive intervals.

Age regressions use standardized continuous age and HC3 OLS covariance for M5 alpha, kappa, theta, with linear and quadratic forms; Holm corrects all nine slope tests. These are exploratory descriptive parameter associations, with interpretation subordinated to recovery.


## Second pass: explicit bounds, age, and joint estimation

Historical unqualified M0–M8 fits retain theta bounds [0,5]. Revised primary M4–M7 fits are explicitly `M4_theta10` through `M7_theta10` and are published under base labels in `model_fits_theta10.csv`, with a `theta_upper=10` column. `theta_bound_sensitivity.csv` records 5/10/20 fits and boundary flags without discarding the originals. `M5_kappa100` is the renamed first-pass kappa sensitivity, still theta≤5. Parameter recovery for the four revised models uses 50 datasets per participant/model and 50 optimizer starts.

M3_phi10 retains the clipped Fareri-style prior and enlarges phi to [0,10]. Partner-wise saturation begins at (1−1e−9)/S for S>0; S=0 is constant everywhere. The logistic-prior generalization instead uses sigmoid(phi×(2S−1)), phi∈[0,10]. It is not an exact replication. Both rating analyses remain secondary because pre/post timing is unknown.

The behavioral GEE is `chose_high ~ C(partner)*age_z + C(offer_pair) + C(partner)*trial_scaled`, clustered by participant with exchangeable working dependence and robust covariance. Age mean/SD use one observation per participant (ddof=0). Three linear age×partner contrasts receive Holm correction within each specification. Age×time and quadratic-age additions are secondary. Probability curves use equal-participant marginal standardization over each person's observed offer/time distribution. Bootstrap replicates resample whole participants, assign new cluster identifiers to duplicates, and retain fixed age coding and standardization targets.

### Hierarchical model and priors

H2, H5, H8, HPreference, H7 and secondary H4 share the original sequential likelihood and differ only in psychological value/learning terms. All participants are fitted jointly. For latent parameter vector eta_i, `eta_i = mu + beta*age_i + diag(tau)*L*z_i`, with non-centered standard-normal z_i and LKJ(2) Cholesky correlation L. Correlations apply to all parameter blocks, including the kappa/theta tradeoff. No independent-effects fallback is silently substituted.

Natural transforms: logistic for learning rates; exp for kappa; softplus for nonnegative social values; identity for generic partner preferences. H5's bounded sensitivity uses `10*logistic(eta_theta)`.

Priors on latent population means: alpha and alpha_negative N(−1,1.25); log kappa N(−1.2,.8); softplus theta N(1,1.5); signed preference N(0,1). Latent SDs are half-normal(0,.8), age coefficients N(0,.5). Bounded H5 replaces the theta mean prior center with logit(.13), giving a comparable central value near 1.3; its induced prior is nevertheless different. The labeled prior-scale sensitivity multiplies mean-prior SDs, random-effect SD-prior scale and age-prior SDs by 1.5. Linear age is primary. Quadratic H5 is run only after the linear model passes convergence checks.

Before data fitting, 500 exact prior draws per model were generated using Stan's LKJ RNG, and choices were simulated on the real schedules. Prior predictive tables report extreme trial probabilities and the fraction of participants with >95% extreme probabilities. These priors spread mass across varied behavior without concentrating on universally deterministic choices.

### Sampler, likelihood verification and diagnostics

CmdStan 2.40.0 via CmdStanPy; four chains, 2,000 warmup and 2,000 retained draws per chain, initial adapt_delta=.95, max_treedepth=12. Runs failing diagnostics are archived and repeated at adapt_delta=.99. Final diagnostics require R-hat<1.01, bulk/tail ESS≥400, no divergences or maximum-depth hits, and minimum chain BFMI>.3. Exceptions from rejected warmup proposals are recorded in raw logs; they are distinguished from retained-sample divergences. Trace figures accompany full-data fits. Raw chains/builds remain in ignored work/hierarchical.

`hierarchical_shared.stan` is the pure-Stan reference. `hierarchical_fast.stan` calls an exact sequential likelihood with analytically propagated first derivatives in `rl_fast.hpp`; Stan's reverse-mode tape handles the transformations and hierarchy. This changes computational cost, not the likelihood or NUTS algorithm. The original H5 reference fit is retained; remaining fits use the accelerated implementation. `check_stan_likelihood.py` checks choice probabilities against the audited engine; `check_fast_gradient.py` checks likelihood values and derivatives against finite differences across six models and actual schedules. Source hashes are recorded, and incompatible cached analytic fits are rejected.

Learning remains partner-specific, begins at P=.5, updates only after observed feedback, skips zero-investment and missed-trial feedback, and carries across ordinary run boundaries. Removing missed trials from the Stan input is likelihood-equivalent because they have neither a choice likelihood nor a belief update. Simulations use the full presented schedule and programmed outcomes, with simulated choices determining feedback visibility.

### Age endpoints, prediction and recovery

Age coefficients are estimated inside the hierarchy. Report latent slopes, posterior sign probabilities, natural-scale values at ages 25/40/60/75, and changes from 25 to 75. Curves represent a typical participant with zero latent residual. `latent_age_variance_fraction` divides age-associated latent variance across the sampled ages by that variance plus residual tau²; it is not behavioral R² and has a nonnegative posterior even under a null effect.

The H5 `friend_value_probability_effect` is the change in high-choice probability when turning the friend bonus on versus off, holding P=.5 and averaging over the empirical offer-gap distribution with equal participant weights. Compute it jointly from each draw's kappa and theta. The canonical $2/$8 logit contribution is 3*kappa*theta. H7 also reports theta_friend−theta_stranger and its age change.

Posterior predictive datasets preserve participants, actual partner/offer/outcome schedules and missing choices. They include parameter uncertainty and simulated choice-dependent feedback. Summaries cover partner, investment, offer, time, previous same-partner feedback, and continuous-age slope discrepancy statistics.

Temporal holdout refits the hierarchy on training choices only (run 1; or chronological first 65% for one-run participants). Parameters and their posterior weights remain fixed during scoring; observed test feedback updates beliefs. Per-trial predictive probabilities are averaged over all retained training-posterior draws before log loss/Brier/calibration. Subject means receive equal weight. Main H2/H5/H8/HPreference and secondary H7 each have age/no-age training fits. Raw AIC and Bayesian model summaries are not substituted for this prospective test.

H5 recovery uses five complete datasets each at theta-age slopes 0,+.35,−.35 with actual ages and schedules. Generating mu=(−1,−1.2,1), tau=(.65,.65,1), latent kappa/theta correlation −.45. Every synthetic dataset receives both a full four-chain hierarchical fit and 100-start independent MLE at theta≤10. Compare participant RMSE/bias, age slopes, population means/SDs, and probability effects. Five datasets per condition yield descriptive recovery evidence, not precise coverage calibration or formal SBC.
