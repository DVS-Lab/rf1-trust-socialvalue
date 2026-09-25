"""Refresh cached results, paired prediction comparisons and publication provenance."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from .hierarchical import sample,summaries,predictive,heldout,run_name,collect,SPECS,TABLE,WORK,offer_distribution,friend_effect,diagnostics
from .fitting import stable_seed
from .hierarchical_figures import trace_figure,figures
from .hierarchical_validation import aggregate


def comparison(a,b,label,model):
    difference=(a-b).dropna();x=difference.to_numpy();rng=np.random.default_rng(stable_seed(20260924,label,model))
    boot=x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1)
    return dict(comparison=label,model=model,n=len(x),mean_difference=x.mean(),median_difference=np.median(x),
                ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),fraction_improved=np.mean(x<0),
                p=wilcoxon(x).pvalue if abs(x).max()>1e-12 else 1.)


def predictive_comparisons():
    h=pd.read_csv(TABLE/'hierarchical_heldout_participant.csv');age=h[h.run.str.endswith('_age')]
    wide=age.pivot(index='participant_id',columns='model',values='log_loss');rows=[]
    for j,a in enumerate(wide.columns):
        for b in wide.columns[j+1:]:rows.append(comparison(wide[a],wide[b],a+' minus '+b,'between_hierarchies'))
    d=pd.DataFrame(rows);d['p_holm']=multipletests(d.p,method='holm')[1];d.to_csv(TABLE/'hierarchical_model_pairwise.csv',index=False)
    mle=pd.read_csv(TABLE/'heldout_fits.csv');mle=mle[~mle.model.isin(['M4','M5','M6','M7'])]
    new=pd.read_csv(TABLE/'heldout_theta10.csv');new['model']=new.model.str.split('_').str[0]
    mle=pd.concat([mle,new,pd.read_csv(TABLE/'heldout_preference.csv')],ignore_index=True)
    rows=[];individual=[]
    for model,base in [('H2','M2'),('H5','M5'),('H8','M8'),('HPreference','preference'),('H7','M7')]:
        a=age[age.model.eq(model)].set_index('participant_id').log_loss;b=mle[mle.model.eq(base)].set_index('participant_id').heldout_log_loss
        rows.append(comparison(a,b,'hierarchical minus MLE',model))
        for sub,value in (a-b).items():individual.append(dict(model=model,participant_id=sub,log_loss_difference=value))
    d=pd.DataFrame(rows);d['p_holm']=multipletests(d.p,method='holm')[1];d.to_csv(TABLE/'hierarchical_vs_mle_heldout.csv',index=False)
    pd.DataFrame(individual).to_csv(TABLE/'hierarchical_vs_mle_heldout_participant.csv',index=False)


def identifiability(fit,meta,frames,name):
    natural=fit.stan_variable('natural');g,w=offer_distribution(frames);rows=[]
    for i,sub in enumerate(meta['ids']):
        theta=natural[:,i,2];kappa=natural[:,i,1];effect=friend_effect(natural[:,i],g,w);product=theta*kappa
        rows.append(dict(run=name,participant_id=sub,theta_kappa_posterior_correlation=np.corrcoef(theta,kappa)[0,1],
                         theta_cv=theta.std()/theta.mean(),kappa_cv=kappa.std()/kappa.mean(),product_cv=product.std()/product.mean(),
                         probability_effect_cv=effect.std()/effect.mean(),theta_ci_width=np.diff(np.quantile(theta,[.025,.975]))[0],
                         probability_effect_ci_width=np.diff(np.quantile(effect,[.025,.975]))[0],
                         probability_theta_gt10=np.mean(theta>10),probability_theta_gt20=np.mean(theta>20)))
    return rows


def sensitivity_comparison():
    p=pd.read_csv(TABLE/'hierarchical_parameter_summary.csv');a=pd.read_csv(TABLE/'hierarchical_age_effects.csv');rows=[]
    for run in ['H5_full_age','H5_full_age_bounded','H5_full_quadratic','H5_full_age_prior1.5']:
        q=p[p.run.eq(run)];x=a[a.run.eq(run)&a.parameter.eq('friend_value_probability_effect')&a.quantity.eq('natural_change_25_to_75')].iloc[0]
        rows.append(dict(run=run,median_individual_theta_mean=q.loc[q.parameter.eq('theta'),'mean'].median(),
            median_individual_kappa_mean=q.loc[q.parameter.eq('kappa'),'mean'].median(),
            median_individual_choice_effect_mean=q.loc[q.parameter.eq('friend_value_probability_effect'),'mean'].median(),
            choice_effect_age_change_mean=x['mean'],choice_effect_age_change_low=x.ci_low,choice_effect_age_change_high=x.ci_high,
            choice_effect_age_change_probability_positive=x.probability_positive))
    pd.DataFrame(rows).to_csv(TABLE/'hierarchical_h5_sensitivity.csv',index=False)


def finish():
    jobs=[(m,False,1,False,1) for m in ['H2','H5','H8','HPreference','H7','H4']]
    jobs += [('H5',False,1,True,1),('H5',False,2,False,1),('H5',False,1,False,1.5)]
    jobs += [(m,True,a,False,1) for m in ['H2','H5','H8','HPreference','H7'] for a in [1,0]]
    ident=[];provenance=[]
    for model,train,age,bounded,scale in jobs:
        name=run_name(model,train,age,bounded,prior_scale=scale)
        fit,meta,arrays,ratings,frames=sample(model,train,age,bounded,prior_scale=scale)
        if train:heldout(fit,meta,arrays,ratings,name);trace_figure(fit,model,name)
        else:
            summaries(fit,meta,frames,name);predictive(fit,meta,arrays,ratings,frames,name);trace_figure(fit,model,name)
            if model=='H5':ident+=identifiability(fit,meta,frames,name)
        manifest=json.loads((WORK/name/'manifest.json').read_text())
        provenance.append(dict(run=name,input_model_settings_sha256=manifest['fingerprint'],seconds=manifest['seconds'],
                               settings=manifest['settings'],execution=manifest.get('execution',{'parallel_chains':manifest['settings']['parallel_chains']}),n_subjects=len(meta['ids']),n_fitted_trials=meta['n_trials'],training_only=meta['training'],
                               seconds_scope=manifest.get('seconds_scope','Wall time for the final sampling attempt, including warmup'),
                               likelihood_implementation=manifest.get('implementation','Stan reference autodiff'),
                               implementation_sha256=manifest.get('implementation_sha256'),
                               sampling_segments=({**manifest['sampling_segments'],
                                   'adaptation_source':[{**entry,'csv':str(Path(entry['csv']).relative_to(Path.cwd()))} for entry in manifest['sampling_segments']['adaptation_source']]}
                                   if 'sampling_segments' in manifest else None)))
        print('Refreshed '+name,flush=True)
    # Refresh diagnostic settings and provenance for all cached recovery datasets.
    from cmdstanpy import from_csv
    for condition in ['zero','positive','negative']:
        names=[f'H5_recovery_{condition}_{i}' for i in range(5)]
        worst=max(names,key=lambda n:pd.read_csv(TABLE/f'diagnostics_{n}.csv').R_hat.max())
        for name in names:
            manifest=json.loads((WORK/name/'manifest.json').read_text());fit=from_csv(manifest['csv_files'])
            meta=manifest['meta'];diagnostics(fit,name,meta,manifest['settings'])
            if name==worst:trace_figure(fit,'H5',name)
            provenance.append(dict(run=name,input_model_settings_sha256=manifest['fingerprint'],seconds=manifest['seconds'],
                settings=manifest['settings'],execution=manifest.get('execution',{'parallel_chains':manifest['settings']['parallel_chains']}),n_subjects=len(meta['ids']),n_fitted_trials=meta['n_trials'],training_only=False,
                likelihood_implementation=manifest.get('implementation','Stan reference autodiff'),
                implementation_sha256=manifest.get('implementation_sha256'),sampling_segments=None))
    pd.DataFrame(ident).to_csv(TABLE/'hierarchical_identifiability.csv',index=False)
    aggregate();collect();predictive_comparisons();sensitivity_comparison()
    Path('results/hierarchical_provenance.json').write_text(json.dumps(dict(dataset='ds005123 v1.1.3',cmdstan_version='2.40.0',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('stan').glob('*')},runs=provenance),indent=2)+'\n')
