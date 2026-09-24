"""Report second-pass results without silently relabeling historical artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .hierarchical import TABLE,collect


def md_table(d,digits=4):
    def cell(x):
        if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}' if np.isfinite(x) else '—'
        return str(x)
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'


def findings():
    def interval(row):return f"{row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    required=['hierarchical_heldout_summary','hierarchical_age_prediction','hierarchical_vs_mle_heldout','hierarchical_age_effects']
    if not all((TABLE/(x+'.csv')).exists() for x in required):return ''
    h=pd.read_csv(TABLE/'hierarchical_heldout_summary.csv')
    if len(h)!=10:return ''
    a=pd.read_csv(TABLE/'hierarchical_age_effects.csv');a=a[a.run.eq('H5_full_age')]
    theta=a[(a.parameter=='theta')&(a.quantity=='latent_slope')].iloc[0]
    effect=a[(a.parameter=='friend_value_probability_effect')&(a.quantity=='natural_change_25_to_75')].iloc[0]
    best=h[h.run.str.endswith('_age')].sort_values('log_loss').iloc[0]
    age=pd.read_csv(TABLE/'hierarchical_age_prediction.csv');gain=age[age.ci_high<0]
    age_text=('No model has an age-associated improvement whose paired-bootstrap interval excludes zero.' if not len(gain)
              else 'Age-associated predictive gains with intervals excluding zero occur for '+', '.join(gain.model)+'.')
    improved=pd.read_csv(TABLE/'hierarchical_vs_mle_heldout.csv');strong=improved[improved.ci_high<0]
    mle_text=('No paired hierarchy-minus-MLE improvement has a bootstrap interval wholly below zero.' if not len(strong)
              else 'Paired improvements over the corresponding MLE have bootstrap intervals wholly below zero for '+', '.join(strong.model)+'.')
    return '\n'.join(['## Main findings\n',
      '- **Theta remains weakly scaled.** M5 upper-bound counts are 62/111 at 5, 58/111 at 10, and 52/111 at 20. Ten is a reference bound, not a demonstrated adequate ceiling.',
      '- **Behavioral age moderation is uncertain.** The friend-minus-computer probability advantage changes by +0.025 from age 25 to 75 (95% robust CI −0.150 to +0.201); all three primary interaction tests have Holm p=1.',
      f'- **Joint H5 age effects remain uncertain.** The theta slope per age SD is {interval(theta)}; its model-implied friend-value probability effect changes by {interval(effect)} from age 25 to 75 (95% credible intervals). The two endpoints need not have the same direction because kappa also changes.',
      f"- **Temporal prediction:** {best['model']} has the lowest mean log loss among the age-enabled hierarchies ({best['log_loss']:.4f}). {mle_text} See the paired model comparisons before treating a mean ranking as established.",
      '- **Does age help prediction?** '+age_text,
      '- **Paired recovery improves measurement in the tested regime.** Across age conditions, theta RMSE falls from about 2.7–2.8 for MLE to 0.58–0.60 for hierarchy. Simulated theta is substantially below many real-data estimates, and some interval coverage is below nominal; these results do not validate the largest empirical theta values.',
      '- **Mechanism remains a comparison question.** Social-value parameters, generic partner preferences and asymmetric learning must be judged through their predictive checks and held-out differences. Hierarchical shrinkage alone does not establish reciprocation-specific social reward.\n'])


def report():
    collect();parts=['# Second pass: bounds, age and hierarchical modeling\n',
    'Dataset: OpenNeuro ds005123 v1.1.3, unchanged audited sample. Primary N=111; rating-complete secondary N=103. No imaging was fetched.\n',
    '**Artifact provenance.** Unqualified M0–M8 tables and figures 01–13 retain the first-pass fits (theta≤5). The revised nonhierarchical primary estimates are in `tables/model_fits_theta10.csv`; `theta_bound_sensitivity.csv` explicitly records each ceiling. Hierarchical estimates use H-prefixed names. Historical `M5_wide` is now named `M5_kappa100`: it widened kappa, not theta.\n']
    parts.append(findings())
    d=pd.read_csv(TABLE/'theta_bound_sensitivity.csv')
    rows=[]
    for (m,b),g in d.groupby(['base_model','theta_upper']):
        for p in ['theta','theta_stranger'] if m=='M7' else ['theta']:
            rows.append(dict(model=m,parameter=p,bound=b,n=len(g),median=g[p].median(),at_upper=int(g[p+'_at_upper'].sum()),near_upper_percent=100*g[p+'_near_upper'].mean(),mean_log_likelihood=g.log_likelihood.mean()))
    parts+=['## Theta bounds\n',md_table(pd.DataFrame(rows)),
            'Theta=10 is the requested revised reference fit, **not an adequate resolution of the scale problem**. Many estimates migrate with the ceiling, and likelihood continues to improve at 20. Do not interpret raw theta magnitudes as well-measured individual traits.\n']
    gains=[]
    for m,g in d.groupby('base_model'):
        ll=g.pivot(index='participant_id',columns='theta_upper',values='log_likelihood');delta=ll[20]-ll[10]
        k=g.pivot(index='participant_id',columns='theta_upper',values='kappa');th=g.pivot(index='participant_id',columns='theta_upper',values='theta')
        gains.append(dict(model=m,mean_LL_gain_10_to_20=delta.mean(),max_LL_gain=delta.max(),n_gain_gt_001=int((delta>.01).sum()),n_theta_rises_kappa_falls=int(((th[20]>th[10]+.01)&(k[20]<k[10]-.001)).sum())))
    parts +=[md_table(pd.DataFrame(gains)),
             'New fits use 100 seeded starts plus the optimum from the next smaller bound. Numerical nesting is validated. This distinguishes bound effects from missed local optima.\n']
    if (TABLE/'theta_recovery_summary.csv').exists():
        rec=pd.read_csv(TABLE/'theta_recovery_summary.csv');parts+=['### Recovery at theta≤10 versus theta≤5\n',md_table(rec[['model','parameter','n','pearson','spearman','rmse','bias','upper_rate']]),
        '50 simulations per participant/model with 50-start refits. The generating parameter distributions differ between ceilings; these comparisons alone cannot isolate estimation quality. The paired hierarchical/MLE simulations below hold the generated data fixed. The full 12-model recovery suite remains the historical theta≤5 analysis: the raw-theta parameterization is still unresolved, so its previous confusion matrix must not be relabeled as validation of the new fits.\n']
    old=pd.read_csv(TABLE/'model_fits.csv');new=pd.read_csv(TABLE/'bound_fits.csv');o=old[old.model.eq('M3')].set_index('participant_id');n=new[new.model.eq('M3_phi10')].set_index('participant_id');gain=n.log_likelihood-o.log_likelihood
    sat=pd.read_csv(TABLE/'phi_saturation.csv')
    parts+=['## Phi and rating-prior sensitivity\n',
            f'Widening phi from 5 to 10 improves log likelihood by {gain.mean():.4f} on average; {(gain>.001).sum()} of {len(gain)} participants improve by more than .001. Maximum gain: {gain.max():.4f}. The mean fraction of each participant/partner prior\'s 0–10 domain that is constant is {sat.flat_fraction_0_10.mean():.1%}. This is a partner-specific calculation, not the flat fraction of the complete likelihood. Ratings of zero imply a constant zero initial prior throughout.\n',
            '`M3_logitprior` uses logistic(phi × rating/5), phi∈[0,10]. It removes the hard clipping operation, but it is a different psychological model and is secondary. Both formulations retain the unresolved timing of the rating files; neither establishes pre-task beliefs.\n']
    a=pd.read_csv(TABLE/'age_partner_gee.csv');a=a[a.kind.eq('age_interaction_contrast')]
    parts+=['## Behavioral age effects\n',md_table(a[['specification','term','estimate','ci_low','ci_high','odds_ratio','p_holm']]),
            'Age is standardized using one observation per primary participant. Estimates are changes in partner log-odds contrasts per age SD. The primary model is linear age with partner×age, offer-pair effects, trial time and partner×time. The secondary age×time and quadratic specifications are labeled separately. For the quadratic model, displayed linear interactions are local slopes at the mean age, not an omnibus test of nonlinear age differences.\n',
            'The three primary age×partner contrasts do not establish a detectable linear age dependence. This is not evidence of age invariance. Figure 15 and `age_partner_marginal_effects.csv` show the size and uncertainty of probability-scale differences. Marginal standardization holds a common offer/time distribution fixed with equal participant weights. Shading is a robust-covariance delta-method 95% interval; dotted limits use 500 whole-participant bootstrap replicates.\n']
    if (TABLE/'age_partner_change_25_to_75.csv').exists():
        parts+=['### Behavioral contrast change from age 25 to 75\n',md_table(pd.read_csv(TABLE/'age_partner_change_25_to_75.csv')), 'Intervals use the joint robust coefficient covariance; they are not obtained by subtracting two independent age-specific intervals.\n']
    if (TABLE/'hierarchical_diagnostics.csv').exists():
        d=pd.read_csv(TABLE/'hierarchical_diagnostics.csv');diag=d.groupby('run').agg(n=('n_subjects','first'),max_rhat=('R_hat','max'),min_bulk_ess=('ESS_bulk','min'),min_tail_ess=('ESS_tail','min'),divergences=('divergences','first'),min_bfmi=('min_bfmi','first'),max_depth_hits=('max_depth_hits','first'),all_pass=('passed','all')).reset_index()
        parts+=['## Hierarchical inference and diagnostics\n',
                'Stan/CmdStanPy fits participants jointly with correlated non-centered latent effects, LKJ(2) correlations, and linear age coefficients inside the hierarchy. Alpha uses logistic, kappa exponential, positive social value softplus, and generic preference identity transforms. See `docs/models.md` for priors, sampling settings and validation. H4 remains secondary and uses ratings of uncertain temporal provenance.\n',md_table(diag),
                'The correlated H8 fit required longer sampling for a population correlation. Archived attempts remain visible in this diagnostic table; final runs use their unsuffixed names. The final full-data preference fit continues four previously adapted diagonal chains, discarding their pilot retained draws and collecting 2,000 fresh draws per chain. The slower dense attempt and a discarded output-collision attempt are retained as unsuccessful runtime attempts, without parameter conclusions. Actual settings and sampling segments are recorded in provenance.\n',
                'Acceptance targets: R-hat<1.01; bulk/tail ESS≥400; no divergences or maximum-depth hits; every chain BFMI>.3. Any failed run is explicitly marked and cannot support a definitive parameter interpretation. All reported intervals are equal-tailed 95% credible intervals.\n']
    if (TABLE/'hierarchical_age_effects.csv').exists():
        a=pd.read_csv(TABLE/'hierarchical_age_effects.csv');a=a[a.run.isin(['H5_full_age','H8_full_age','H5_full_age_bounded','H5_full_quadratic','H5_full_age_prior1.5','H7_full_age','HPreference_full_age','H2_full_age'])]
        parts+=['## Computational age effects\n',md_table(a[['run','parameter','quantity','term','mean','ci_low','ci_high','probability_positive']]),
                'The latent age variance fraction is age-associated latent variance divided by that variance plus residual tau², evaluated over the actual ages. It is not behavioral R², and its posterior is nonnegative even when an age slope is uncertain.\n',
                'Natural-scale curves refer to a participant at zero latent deviation from the age-dependent population mean. They are not an average over the entire population distribution. Age associations are cross-sectional and do not identify within-person aging trajectories. A posterior probability above zero is not a multiplicity-adjusted p value.\n',
                'The H5 probability endpoint turns the friend term on versus off at a fixed belief P=.5, averaged over the empirical offer-gap distribution with equal participant weights. Its canonical logit counterpart is 3×kappa×theta for the $2/$8 offer. These are model-implied, standardized endpoints; they are not causal effects estimated by an experimental manipulation. The same posterior draw supplies kappa and theta, preserving their dependence.\n']
    if (TABLE/'hierarchical_h5_sensitivity.csv').exists():
        z=pd.read_csv(TABLE/'hierarchical_h5_sensitivity.csv')
        parts+=['### H5 parameterization and prior sensitivity\n',md_table(z.drop(columns=['choice_effect_age_change_probability_positive'])),
                'The parameter columns summarize individual posterior means across participants, whereas the age-change columns describe the typical-participant probability endpoint. The bounded and broader-prior models change the regularization as well as the numerical parameterization. Quadratic age is secondary. Compare the choice effect and its age uncertainty alongside raw theta; stable probabilities do not make the underlying scale uniquely identified.\n']
    if (TABLE/'hierarchical_predictive_summary.csv').exists():
        pp=pd.read_csv(TABLE/'hierarchical_predictive_summary.csv')
        pp=pp[pp.run.str.endswith('_full_age') & pp.category.eq('partner') & pp.measure.eq('high_probability')].copy()
        pp['observed_outside_95_interval']=(pp.observed<pp.ci_low)|(pp.observed>pp.ci_high)
        parts+=['## Generative posterior predictive checks\n',md_table(pp[['model','label','n_subjects','observed','mean','ci_low','ci_high','observed_outside_95_interval']]),
                'H8 predicts a friend high-choice rate near .53 versus .78 observed; asymmetric learning alone does not reproduce the friend advantage in complete simulated datasets. H5 gets closer to friend behavior but underpredicts stranger and computer investment choices; H7 improves the stranger fit while still underpredicting computer choices. The generic preference model also approximates friend and stranger choices but underpredicts computer choices (about .38 predicted versus .48 observed). These discrepancies remain despite acceptable sampler diagnostics. See Figure 19 for all models and the predictive tables for investment, offer, time, previous-feedback and age-pattern checks.\n',
                'Temporal scoring conditions on observed feedback histories, whereas complete predictive simulations generate feedback visibility through their own choices. A model can predict later choices reasonably after observed histories have separated partners yet fail to generate their differences from the common initial belief. Prediction rankings and generative checks therefore answer complementary questions. Observed values outside predictive intervals are discrepancy flags, not multiple-testing-adjusted hypothesis tests.\n']
    if (TABLE/'hierarchical_heldout_summary.csv').exists():
        h=pd.read_csv(TABLE/'hierarchical_heldout_summary.csv');parts+=['## Temporal held-out prediction\n',md_table(h),
        'Each hierarchy is re-estimated using training choices only: run 1 for ordinary two-run participants, or the chronological first 65% for one-run participants. The training posterior is held fixed while scoring later choices; observed held-out feedback updates beliefs only. Posterior predictive log loss averages conditional choice probabilities over draws before taking the log. Brier scores and calibration use the same mixture probabilities. Summaries weight participants equally. The prediction target is later choices in these same participants.\n']
        p=TABLE/'hierarchical_age_prediction.csv'
        if p.exists():parts +=['### Does age improve prediction?\n',md_table(pd.read_csv(p)),
                                'Negative log-loss differences favor including age. Intervals are paired participant-bootstrap intervals with the fitted models held fixed; they describe variation across this participant sample and do not include variability from refitting a new training sample.\n']
    if (TABLE/'hierarchical_recovery_summary.csv').exists():
        rec=pd.read_csv(TABLE/'hierarchical_recovery_summary.csv');parts+=['## Paired hierarchical versus MLE recovery\n',md_table(rec),
        'Five complete simulated datasets per condition (zero, +.35, −.35 latent theta slope per age SD), each on all 111 real ages and schedules. Generating latent means are (−1,−1.2,1), SDs (.65,.65,1); kappa/theta latent correlation is −.45. Hierarchical posterior means and 100-start theta≤10 MLEs are scored against the same truth on identical simulated choices. Population mean/SD/age slopes, individual parameters and friend-probability effects are all recovered. Five replicates give very imprecise coverage estimates and do not constitute simulation-based calibration. The simulated theta values have median 1.29 and maximum 4.36; real-data H5 posterior means have median 6.48. Thus the RMSE gains characterize the tested low-to-moderate theta regime and do not validate the high-theta real-data estimates. The fitted-parameter theta≤10 recovery and posterior sensitivity analyses provide complementary evidence about that harder regime.\n']
    if (TABLE/'hierarchical_model_pairwise.csv').exists():
        parts+=['### Paired comparisons between hierarchical models\n',md_table(pd.read_csv(TABLE/'hierarchical_model_pairwise.csv')),
                '### Hierarchy versus the corresponding individual MLE\n',md_table(pd.read_csv(TABLE/'hierarchical_vs_mle_heldout.csv')),
                'Differences are paired within participant. Confidence intervals bootstrap participants; Wilcoxon tests have Holm adjustment within each comparison family. Predictive superiority does not uniquely establish a psychological mechanism.\n']
    if (TABLE/'hierarchical_identifiability.csv').exists():
        ident=pd.read_csv(TABLE/'hierarchical_identifiability.csv')
        cols=['theta_kappa_posterior_correlation','theta_cv','kappa_cv','product_cv','probability_effect_cv','probability_effect_ci_width','probability_theta_gt10','probability_theta_gt20']
        parts+=['## Choice effect versus raw-parameter identification\n',md_table(ident.groupby('run')[cols].median().reset_index()),
                'Values above are medians across participants. Coefficients of variation compare relative posterior uncertainty; interval width for the probability effect is in probability units. Stability reflects both information in the choices and the hierarchical prior. A tighter derived effect does not establish that theta itself is separately identified.\n']
    if (TABLE/'hierarchical_age_recovery.csv').exists():
        parts+=['### Recovery of modest age slopes\n',md_table(pd.read_csv(TABLE/'hierarchical_age_recovery.csv')),
                'Intervals excluded zero in 4/5 datasets under each nonzero theta slope, but also in 1/5 zero-slope datasets. This supports recoverability under these generating conditions while showing substantial uncertainty; five replicates cannot estimate power or false-positive rates precisely. Parameter coverage also falls below nominal in some conditions (for example, positive-slope individual theta), so better RMSE is not equivalent to fully calibrated uncertainty.\n']
    parts+=['## Interpretation\n',
            'Behavioral age moderation, computational age coefficients and psychological mechanism are separate questions. The large friend advantage does not identify reciprocation-specific social reward. Generic partner preferences and asymmetric learning remain substantive alternatives. Hierarchy regularizes estimation but cannot itself establish mechanism or eliminate weak identification. Judge changes using temporal prediction, recovery, posterior uncertainty and prior/parameterization sensitivity together.\n',
            '## Reproduction\n\nRun `bash scripts/98_run_hierarchical.sh` after the audited first-pass outputs exist. This runs explicit bound fits, behavioral age analysis, prior predictive checks, four-chain full/train fits, targeted recovery, figures and validation. Builds and raw chains remain under ignored `work/`. `bash scripts/99_run_all.sh --hierarchical` additionally regenerates the historical first pass. Run the Python validator as `python scripts/07_validate_outputs.py`, not `bash`: its extension and contents are Python.\n',
            '## Sources\n\nXia et al. (2021), [Modeling changes in probabilistic reinforcement learning during adolescence](https://doi.org/10.1371/journal.pcbi.1008524), motivates joint age regression; its task and age range differ from this study. [Stan documentation](https://mc-stan.org/docs/stan-users-guide/multivariate-hierarchical-priors.html) describes multivariate hierarchical priors and non-centered parameterizations. Existing audited dataset and Fareri references are retained in the first-pass report.\n']
    p=Path('results/analysis_summary.md');old=Path('results/first_pass_analysis_summary.md')
    if not old.exists():old.write_text(p.read_text())
    p.write_text('\n'.join(parts))
