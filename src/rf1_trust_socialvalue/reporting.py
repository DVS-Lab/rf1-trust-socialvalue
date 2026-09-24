"""Generate a numeric, limitation-forward report and machine-readable provenance."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime,timezone
import numpy as np
import pandas as pd


def markdown_table(df,columns=None,digits=3):
    d=df[columns].copy() if columns else df.copy()
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}' if np.isfinite(x) else 'NA'
        return str(x)
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))


def report(config):
    root=Path('results');p=root/'tables';audit=json.loads((root/'audit.json').read_text());s=pd.read_csv(p/'sample_audit.csv')
    f=pd.read_csv(p/'model_fits.csv');m=pd.read_csv(p/'model_comparison.csv');h=pd.read_csv(p/'heldout_summary.csv');r=pd.read_csv(p/'recovery_summary.csv');rob=pd.read_csv(p/'robustness_comparisons.csv')
    b=pd.read_csv(p/'behavior_summary.csv');age=pd.read_csv(p/'age_exploratory.csv');t=pd.read_csv(p/'trial_table.csv');paired=pd.read_csv(p/'model_pairwise.csv');hp=pd.read_csv(p/'heldout_pairwise.csv')
    primary=m[(m['sample']=='primary')&(m.metric=='AICc')].sort_values('mean')
    rating=m[(m['sample']=='rating_complete_secondary')&(m.metric=='AICc')].sort_values('mean')
    diagnostics=f.groupby('model').agg(n=('participant_id','size'),converged=('converged','sum'),median_near_best=('n_near_best','median'),any_boundary=('boundary_parameters',lambda x:x.notna().mean())).reset_index()
    diagnostics.to_csv(p/'fit_diagnostics.csv',index=False)
    recovery=r[r.model.isin(['M5','M7','M8'])]
    pair=paired[(paired['sample']=='primary')&(paired.metric=='AICc')&(paired.model_a=='M2')&(paired.model_b=='M5')].iloc[0]
    bestmetrics=m[m['sample'].eq('primary')].sort_values('mean').groupby('metric').first()
    heldfits=pd.read_csv(p/'heldout_fits.csv')
    worst=heldfits.nlargest(5,'heldout_log_loss')[['participant_id','model','heldout_n','heldout_log_loss','boundary_parameters']]
    worst.to_csv(p/'heldout_extreme_errors.csv',index=False)
    nratings=int(f[f.model.eq('M4')].shape[0]);nprimary=int(s.primary_include.sum());n2=int((s.loc[s.primary_include,'n_runs']==2).sum())
    recm5=r[(r.model=='M5')&(r.parameter=='theta')].iloc[0]
    confusion=pd.read_csv(p/'model_recovery_confusion.csv')
    diagonal=confusion[confusion.generating.eq(confusion.fitted)][['metric','generating','selection_probability']]
    dem=json.loads((root/'demographics.json').read_text())
    schedule=t.groupby('partner').agg(programmed=('scheduled_reciprocation','mean'),experienced=('observed_reciprocation','mean'),feedback_trials=('feedback_observed','sum')).reset_index()
    text=f'''# Trust, learning, and social value

## Findings at a glance

Friends elicit substantially more investment and higher-offer choices. In the primary rating-free analysis, M7 (separate friend/stranger reciprocation bonuses) has the lowest mean AIC, AICc and BIC. However, M5 wins AICc for the largest single share of participants, while asymmetric learning M8 has the lowest mean held-out log loss among primary candidates. These results support partner-dependent behavior, but **do not establish a unique social-reward mechanism**. A generic partner preference control fits at least as well as M7 on average. Ratings are **not verified as pre-task**, so M3/M4 cannot serve as a primary replication of Fareri's pre-task-rating models.

**Individual parameter interpretation requires caution.** M5's theta recovery has mean Pearson r={recm5.pearson:.3f}, RMSE={recm5.rmse:.3f}, and boundary-hit rate={recm5.boundary_rate:.1%}. See all recovery results below; optimizer convergence does not establish identification. In the expanded 12-candidate model-recovery analysis, AICc reselects M7 for only {float(diagonal[(diagonal.metric=='AICc') & (diagonal.generating=='M7')].selection_probability.iloc[0]):.1%} of M7-generated datasets. **This candidate set poorly distinguishes M7's mechanism**, even though its mean in-sample criterion is favorable.

## Dataset/version and provenance

OpenNeuro **ds005123 v1.1.3**, DOI [10.18112/openneuro.ds005123.v1.1.3](https://doi.org/10.18112/openneuro.ds005123.v1.1.3), Git commit `{audit['commit']}`. The snapshot is pinned and checked at runtime. Only Trust CSVs, task scripts, and small BIDS metadata were retrieved. **No imaging content was retrieved or analyzed.** Dataset inputs are SHA-256 hashed in `input_manifest.json`; software, seed, source-code hashes and analysis commit are in `provenance.json`.

## Task structure discovered from actual files

The canonical sample contains {audit['n_canonical']} participants. There are {audit['n_behavior']} participants with at least one auditable Trust session, {audit['n_trials']} presented decision rows, {audit['n_valid']} valid choices, and {audit['n_missed']} missed decisions ({audit['n_missed']/audit['n_trials']:.2%}). Decisions are reconstructed from raw logs and cross-checked against BIDS for partner, offers, choice, RT, onset, and observed feedback. Misses are represented as `missed_trial`, with 3-second BIDS RT versus 0 in raw; zero choices have no BIDS outcome. This transformation is explicitly verified, not treated as a discrepancy.

Normal sessions contain 42 trials, 14 per partner. Six distinct pairs use amounts 0, 2, 4, 8; the 2–8 pair is duplicated by the generator and occurs twice as frequently. Programmed reciprocation is balanced in complete schedules. Experienced feedback is choice-censored and need not be balanced. Exact counts/rates are in `run_audit.csv` and `schedule_audit.csv`; missingness and run counts are in `sample_audit.csv`.

## Behavioral sample and exclusions

Descriptive figures include all {audit['n_behavior']} participants with auditable decisions. Primary inferential/model analyses include **{nprimary} participants**, of whom {n2} have two ordinary runs. `sub-10657` and `sub-10777` have appended raw sessions and header artifacts propagated into BIDS; file order alone does not establish their session interpretation. Their reconstructable presented rows are retained for descriptive audit, but both participants are excluded from primary inference. Forty unpresented design rows in sub-10777 are removed rather than counted as missed choices. Empty/placeholder BIDS runs are excluded. Sub-10555 has 11 presented raw rows in an empty BIDS run; this run is excluded because cross-validation is unavailable, and the observed other run remains included. Every exception is listed in `data_anomalies.csv`.

The prespecified sensitivity filter removes participants with >20% missed choices or fewer than four observed feedback trials for any partner; N={int(s.sensitivity_include.sum())}. No participant is excluded based on an apparent model effect, age, fitted parameter value, or imaging quality. Sex is reported using the dataset's `sex` field without relabelling it as measured gender.

## Rating-data audit

Trait 2 is trustworthiness; 0 is approachable and 1 is likeable. Ratings are on −5…+5 and normalized as (rating+5)/10. There are {audit['n_rating_complete']} canonical participants with one complete unambiguous rating set and {nratings} in the primary computational sample. No ratings are imputed.

In `investment_ratings.py`, a Session (Pre/Post) value is collected and entered in an `expInfo` dictionary, but the filename has only one replacement field: `sub-{{}}_Trust-Ratings.csv`; passing the session as a second format argument discards it. Saved columns do not include that dictionary/session indicator. Two participants (sub-10369 and sub-10478) have appended, unlabelled rating sets; their ratings are excluded. The associated Data in Brief article does not establish timing for these specific files, and the companion repository history supplies no resolving session metadata. Therefore timing remains **unverified**; M3/M4 and rating associations are secondary/descriptive, potentially reflecting post-task judgments.

## Behavioral results

The following participant-weighted estimates use 95% participant-bootstrap intervals (5,000 draws):

{markdown_table(b[b.measure.isin(['investment','high_probability'])],['partner','measure','n','mean','ci_low','ci_high'])}

The trial-level benchmark is a binomial GEE with exchangeable within-participant dependence and robust covariance: `chose_high ~ partner * trial_scaled + offer_pair`. It uses N={nprimary} participant clusters, with computer as partner reference. Trial is scaled by 83 decisions. `behavior_gee.csv` contains coefficients/odds ratios and `behavior_gee.txt` the complete summary. Offer-adjusted participant contrasts independently verify the behavioral pattern:

{markdown_table(pd.read_csv(p/'behavior_paired_bootstrap.csv'))}

The behavioral sample spans age {dem['age_min']:.0f}–{dem['age_max']:.0f} years (mean {dem['age_mean']:.2f}, SD {dem['age_sd']:.2f}). Released sex counts: {dem['sex_as_released']}. Programmed versus experienced feedback is:

{markdown_table(schedule)}

RT by partner/choice, exact offers, time bins, recent feedback, rating associations, programmed and experienced reciprocity, and sample demographics are provided in the tables. Recent-feedback relationships are descriptive, since feedback exposure depends on earlier choices.

## Computational model definitions

For each partner, beliefs start at .5 unless the model specifies a rating prior. Monetary EV is `8 − x + 1.5 x P`. Social-value models add `P x theta S`, where S is normalized rating (M4), friend indicator (M5), human indicator (M6), or separate friend/stranger bonuses (M7; computer reference). Choice probability is `logistic(kappa * (V_high − V_low))`. M0 is random, M1 fixed P=.5, M2 ordinary RL, M3 rating-initialized RL with `P0=min(phi*S,1−1e−9)`, and M8 separate positive/negative learning rates. Learning updates only the current partner after **displayed feedback**, never after $0 or missed choices. Under linear money-only utility, the indifference belief is 2/3.

Parameters are bounded directly for exact boundary solutions: learning rates [0,1], kappa [0.00001,20], nonnegative bonuses/phi [0,5]. Kappa up to 100 is tested separately. Details and all robustness equations are in `docs/models.md`.

## Parameter-fitting diagnostics

Every actual-data fit uses 100 deterministic randomized starts and stable Bernoulli log likelihoods. The best converged solution is required to match the lowest solution within 0.0001 NLL; unresolved optimizer failures stop the pipeline. Near-best means within 0.0001 NLL. A boundary flag means within 0.01% of the allowed range. All models within a comparison sample use identical valid decisions; missed rows stay in chronological sequences but contribute neither likelihood nor learning.

{markdown_table(diagnostics)}

Frequent bounds and near-equal optima signal flat likelihoods and parameter tradeoffs. Natural-scale optimization is used in preference to transforms so the no-learning boundary is represented exactly. Kappa is not numerically comparable to Fareri's temperature parameter.

## Model comparisons

Primary AICc results (lower is better; best shares split ties):

{markdown_table(primary,['model','n','mean','median','mean_delta','best_share'])}

Mean-criterion minima: {', '.join(str(metric)+'='+row['model'] for metric,row in bestmetrics.iterrows())}. M5 improves on M2 by a mean {pair.mean_difference:.3f} AICc units (paired 95% bootstrap interval {pair.ci_low:.3f} to {pair.ci_high:.3f}; Holm-adjusted Wilcoxon p={pair.p_holm:.3g}). This comparison establishes improvement over simple money-only RL; it does not discriminate all alternatives. `model_deltas.csv` retains participant heterogeneity; `model_pairwise.csv` provides paired bootstrap intervals and Holm correction separately within metric/sample families.

Secondary rating-complete AICc comparison uses the same {nratings} participants for every candidate:

{markdown_table(rating,['model','n','mean','median','best_share'])}

Fareri pseudo-R² is `(AIC_random − AIC_model)/AIC_random`; conventional McFadden is `1−LL_model/LL_random`. Negative values are retained. These are fit indices, not literal proportions of behavioral variance explained.

## Prospective and simulation predictive checks

Train on the first ordinary run and evaluate the second; participants with one run use a 65%/35% chronological split. All training parameters are frozen. Held-out feedback updates latent beliefs online, starting from the training endpoint. Ratings have unverified timing and can contain future information, so M3/M4 prediction is not a verified prospective rating-based test.

{markdown_table(h)}

Accuracy uses the deterministic p≥.5 choice rule, so M0's tied prediction defaults to high and its accuracy reflects the held-out high-choice base rate; its log loss is exactly log(2). Log loss is the main predictive measure. Brier scores and binned calibration are supplied. Large individual log losses expose overconfident errors in short-training, unregularized MLE fits. These errors are retained, not winsorized. The five largest participant/model losses are reported explicitly:

{markdown_table(worst)}

These are pathological out-of-sample predictions despite converged training fits, not optimizer failures. This is a practical reason not to endorse the in-sample mean winner unconditionally. Paired bootstrap log-loss comparisons are in `heldout_pairwise.csv`.

Simulation checks use {config['predictive_iterations']} stochastic trajectories per participant/model at fitted parameters, on actual offers/order/outcome schedules, holding the observed missing-choice mask fixed. Simulated $0 choices conceal feedback, positive investments reveal programmed outcomes. Checks cover partner, investment, offer pairs, time, and response to most recent same-partner feedback. These are **conditional simulation checks, not Bayesian posterior checks**; their intervals omit parameter uncertainty and are not confidence intervals for a population effect.

## Parameter recovery

Each fitted non-null core model is simulated and refit {config['recovery_iterations']} times per participant, with {config['recovery_starts']} starts per refit. Recovery correlations are computed across participants within each repetition, then averaged; SD describes variation across simulations. Simulations use actual latent programmed outcomes and endogenous feedback censoring, not deterministic choice thresholds.

{markdown_table(recovery,['model','parameter','n_subjects','iterations','pearson','spearman','rmse','bias','boundary_rate'])}

Full metrics for all models, iteration-level estimates, and generating/recovered pairs are in `recovery_summary.csv`, `recovery_by_iteration.csv`, and `parameter_recovery.csv`. Correlations below .7 or boundary rates above 20% are treated as cautionary descriptive screens, not validated identification thresholds. Do not treat high correlation alone as proof of accurate absolute estimates; assess bias/RMSE too. Recovery samples the empirical fitted parameter distribution, not the entire parameter space. Unrecoverable quantities are not interpreted as individual psychological traits.

## Model recovery

Each generating model is simulated on every rating-complete participant's schedule for {config['model_recovery_iterations']} repetitions. All candidates are refit to each generated dataset with {config['recovery_starts']} starts. Candidate controls include generic partner preferences and monetary-power utility with and without friend value. AICc/BIC confusion matrices retain off-diagonal selections and split exact ties. This is empirical-distribution recovery, so indistinguishable null/boundary generators can legitimately select a simpler model. It measures discrimination under these fitted parameters rather than universal model identifiability. Correct-generator selection rates are:

{markdown_table(diagonal)}

## Robustness checks

Negative differences favor the first model. Improvement fractions require a difference below −0.0001 to avoid counting optimizer roundoff as substantive improvement:

{markdown_table(rob[(rob['sample']=='primary')&(rob.metric=='AICc')],['comparison','n','mean_difference','median_difference','fraction_improved'])}

Power utility allows `u(m)=m^rho`, rho∈[.2,3], and adds the same `P*x*theta*I(friend)` in utility units. At rho=1 it exactly reduces to the linear model. Friend value still improves on power-utility RL on average, although allowing nonlinear money improves fit. Thus monetary curvature matters and theta's absolute utility scale changes with rho. The generic partner preference model instead adds a partner-specific investment slope independent of reciprocation probability. Its fit prevents attributing all partner effects uniquely to rewarded reciprocation.

Signed bonuses, lapse (0…0.2), left/right logit bias (−5…5), run resetting, wider kappa, and signed rating normalization are reported separately. Complete-case and sensitivity-sample contrasts are in `robustness_comparisons.csv`. The three-bonus model and partner-specific asymmetric learning are not added: substantial boundary/identification issues already affect simpler candidates, so richer participant-level parameters would not support a stronger interpretation.

## Secondary exploratory age analysis

Continuous standardized age predicts M5 theta, alpha, and kappa using linear and quadratic OLS with HC3 covariance. Holm correction covers all nine reported coefficient tests. No corrected age association is supported (minimum adjusted p={age.p_holm.min():.3f}). This does not establish absence of lifespan differences: individual MLEs are noisy/bounded, parameters may be confounded, and age is observational. Theta is the primary descriptive age parameter, but inferential interpretation is subordinate to recovery.

## Main limitations

Short individual sequences (typically 42 or 84 trials), frequent optimization boundaries, partially observed outcome histories, timing-ambiguous ratings, and nonidentifiability among partner-value, preference, prior, and asymmetric-learning accounts limit mechanism claims. Feedback is endogenous to investment; experienced rates must not be interpreted as independently assigned reinforcement probabilities. Model comparisons are conditional on this candidate set and maximum-likelihood fitting. Held-out estimates can be unstable because only the first run informs parameters. Bootstrap resampling addresses participant sampling for behavioral/fit contrasts, not latent parameter uncertainty. Excluding ambiguous source sessions improves auditability but changes the analysis population. The dataset is an interim convenience sample; results are not causal evidence about aging.

## Differences from Fareri et al. (2015)

Fareri used a binary $1 keep/share decision and verified pre-task trustworthiness. Here choices compare two varying investments. A social bonus constant across both positive options would cancel; scaling additional value by investment magnitude preserves a meaningful comparison. This implementation uses an inverse temperature kappa, explicit random and plain-RL baselines, AICc/BIC, prospective prediction, and stochastic rather than probability-thresholded recovery. The original report's pseudo-R² is AIC-based, while McFadden's is additionally reported. Parameter and model recovery are stronger safeguards than assuming a fit improvement uniquely establishes social reward.

## References and outputs

- Fareri, Chang & Delgado (2015). [Computational Substrates of Social Value in Interpersonal Collaboration](https://doi.org/10.1523/JNEUROSCI.4775-14.2015).
- Smith et al. (2024). [Social reward and nonsocial reward processing across the adult lifespan](https://doi.org/10.1016/j.dib.2024.110810).
- [Companion task/conversion code](https://github.com/DVS-Lab/SRPAL-DataInBrief), inspected along with the pinned dataset's task scripts.
- Figures: `figures/00_results_overview.png` through the numbered figure suite, each also in vector PDF/SVG. Tables: `tables/`. Run: `bash scripts/99_run_all.sh`.
'''
    (root/'analysis_summary.md').write_text(text)
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','statsmodels','matplotlib','numba','joblib','pytest']}
    source={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for directory in ['src','scripts','config','tests'] for path in sorted(Path(directory).rglob('*')) if path.is_file() and '__pycache__' not in str(path)}
    def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
    companion=Path('data/task-source')
    provenance=dict(dataset='ds005123',dataset_version='1.1.3',dataset_doi='10.18112/openneuro.ds005123.v1.1.3',dataset_commit=audit['commit'],
        analysis_commit=git('rev-parse','HEAD'),analysis_commit_note='Commit of analysis source before generated-results commit; source hashes identify the exact executed files.',
        source_dirty=bool(git('status','--porcelain','--','src','scripts','config','tests','pyproject.toml','requirements.lock.txt')),
        source_sha256=source,configuration=config,software=versions,python=platform.python_version(),platform=platform.platform(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),random_seed_scheme='SHA256(base seed | purpose | participant | model | iteration), first 4 little-endian bytes',
        companion_commit=git('-C',str(companion),'rev-parse','HEAD') if companion.exists() else None)
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Report and provenance generated',flush=True)
