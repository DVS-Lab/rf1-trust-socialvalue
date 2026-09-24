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
