"""Continuous age x partner GEE and marginal standardization with robust covariance."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import dmatrix
from joblib import Parallel,delayed
from .fitting import load_inputs,stable_seed

FORMULA='chose_high ~ C(partner)*age_z + C(offer_pair) + C(partner)*trial_scaled'
OUT=Path('results/tables')


def prepare():
    t,_,_=load_inputs();ages=t.groupby('participant_id').age.first()
    center=ages.mean();scale=ages.std(ddof=0)
    d=t[t.valid_choice].copy();d['chose_high']=d.chose_high.astype(float)
    d['partner']=pd.Categorical(d.partner,categories=['computer','stranger','friend'])
    d['age_z']=(d.age-center)/scale;d['age_z_squared']=d.age_z**2
    return d,float(center),float(scale)


def fit_gee(d,formula):
    return smf.gee(formula,groups='participant_id',data=d,family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=100)


def reference(d):
    # Equal participant weighting, preserving each participant's empirical offer/time distribution.
    q=d[['offer_pair','trial_scaled','participant_id']].copy()
    q['weight']=1/d.groupby('participant_id').participant_id.transform('size').to_numpy()/d.participant_id.nunique()
    return q.groupby(['offer_pair','trial_scaled'],as_index=False).weight.sum()


def predictions(model,ref,ages,center,scale,covariance=True):
    rows=[];cache={}
    for age in ages:
        for partner in ['computer','stranger','friend']:
            q=ref.copy();q['partner']=partner;q['age_z']=(age-center)/scale;q['age_z_squared']=q.age_z**2
            q['partner']=pd.Categorical(q.partner,categories=['computer','stranger','friend'])
            x=np.asarray(dmatrix(model.model.formula.split('~')[1],q,return_type='dataframe').reindex(columns=model.params.index))
            p=expit(x@model.params);estimate=ref.weight@p
            grad=(ref.weight.to_numpy()*p*(1-p))@x
            cache[partner]=(estimate,grad)
            se=np.sqrt(max(0,grad@model.cov_params()@grad)) if covariance else 0
            rows.append(dict(age=age,contrast=partner,estimate=estimate,se=se,ci_low=max(0,estimate-1.96*se),ci_high=min(1,estimate+1.96*se)))
        for partner in ['computer','stranger']:
            delta=cache['friend'][0]-cache[partner][0];grad=cache['friend'][1]-cache[partner][1]
            se=np.sqrt(max(0,grad@model.cov_params()@grad)) if covariance else 0
            rows.append(dict(age=age,contrast='friend - '+partner,estimate=delta,se=se,ci_low=max(-1,delta-1.96*se),ci_high=min(1,delta+1.96*se)))
    return pd.DataFrame(rows)


def bootstrap_one(d,ref,ages,center,scale,it,seed):
    rng=np.random.default_rng(stable_seed(seed,'age_bootstrap',it));ids=d.participant_id.unique()
    chunks=[]
    for j,sub in enumerate(rng.choice(ids,len(ids),replace=True)):
        q=d[d.participant_id.eq(sub)].copy();q['participant_id']=str(j);chunks.append(q)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');m=fit_gee(pd.concat(chunks),FORMULA)
        if not m.converged or not np.isfinite(m.params).all():return None
        q=predictions(m,ref,ages,center,scale,False);q['iteration']=it;return q
    except (ValueError,np.linalg.LinAlgError):return None


def run(config,bootstrap=500):
    d,center,scale=prepare();rows=[]
    forms={'linear':FORMULA,'age_time':FORMULA+' + age_z:trial_scaled',
           'quadratic':FORMULA+' + C(partner)*age_z_squared'}
    ref=reference(d);ages=np.unique(np.r_[np.linspace(d.age.min(),d.age.max(),61),[25,40,60,75]])
    for form,formula in forms.items():
        m=fit_gee(d,formula);assert m.converged
        for term in m.params.index:
            lo,hi=m.conf_int().loc[term]
            rows.append(dict(specification=form,kind='coefficient',term=term,estimate=m.params[term],se=m.bse[term],
                             ci_low=lo,ci_high=hi,p=m.pvalues[term],n_subjects=d.participant_id.nunique(),n_trials=len(d)))
        contrast_rows=[]
        for a,b in [('friend','computer'),('stranger','computer'),('friend','stranger')]:
            w=pd.Series(0.,index=m.params.index)
            if a!='computer':w[f'C(partner)[T.{a}]:age_z']=1
            if b!='computer':w[f'C(partner)[T.{b}]:age_z']=-1
            estimate=w@m.params;se=np.sqrt(w@m.cov_params()@w)
            contrast_rows.append(dict(specification=form,kind='age_interaction_contrast',term=f'{a} - {b}',estimate=estimate,se=se,
                                     ci_low=estimate-1.96*se,ci_high=estimate+1.96*se,p=2*norm.sf(abs(estimate/se)),
                                     n_subjects=d.participant_id.nunique(),n_trials=len(d)))
        for row,padj in zip(contrast_rows,multipletests([r['p'] for r in contrast_rows],method='holm')[1]):row['p_holm']=padj
        rows+=contrast_rows
        if form=='linear':
            marg=predictions(m,ref,ages,center,scale);marg.to_csv(OUT/'age_partner_marginal_effects.csv',index=False)
            Path('results/age_partner_gee.txt').write_text(str(m.summary()))
    tab=pd.DataFrame(rows);tab['odds_ratio']=np.exp(tab.estimate);tab['or_ci_low']=np.exp(tab.ci_low);tab['or_ci_high']=np.exp(tab.ci_high)
    tab.to_csv(OUT/'age_partner_gee.csv',index=False)
    # Fixed standardization target and age coding; resample whole participant clusters.
    boot=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(bootstrap_one)(d,ref,ages,center,scale,it,config['seed']) for it in range(bootstrap))
    good=[x for x in boot if x is not None]
    if len(good)<.95*bootstrap:raise RuntimeError('More than 5% of age bootstrap fits failed')
    b=pd.concat(good).groupby(['age','contrast']).estimate.agg(bootstrap_mean='mean',bootstrap_low=lambda x:x.quantile(.025),bootstrap_high=lambda x:x.quantile(.975)).reset_index()
    b['successful_replicates']=len(good);b['requested_replicates']=bootstrap;b.to_csv(OUT/'age_partner_bootstrap.csv',index=False)
    Path('results/age_standardization.json').write_text(json.dumps(dict(n=111,mean=center,sd=scale,weighting='equal participants; empirical offers and trial time'),indent=2)+'\n')
    figures()


def figures():
    import matplotlib.pyplot as plt
    from .plotting import style,save,heading,COLORS
    style();d,_,_=prepare();m=pd.read_csv(OUT/'age_partner_marginal_effects.csv');b=pd.read_csv(OUT/'age_partner_bootstrap.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
    for partner in ['friend','stranger','computer']:
        q=m[m.contrast.eq(partner)];axes[0].plot(q.age,q.estimate,color=COLORS[partner],label=partner.title());axes[0].fill_between(q.age,q.ci_low,q.ci_high,color=COLORS[partner],alpha=.13)
        raw=d[d.partner.eq(partner)].groupby('participant_id').agg(age=('age','first'),choice=('chose_high','mean'))
        axes[0].scatter(raw.age,raw.choice,s=9,alpha=.16,color=COLORS[partner])
    axes[0].set(xlabel='Age (years)',ylabel='Probability of choosing the higher offer',ylim=(0,1));axes[0].legend()
    for contrast,color in [('friend - computer','#157F86'),('friend - stranger','#D59437')]:
        q=m[m.contrast.eq(contrast)];axes[1].plot(q.age,q.estimate,color=color,label=contrast.title());axes[1].fill_between(q.age,q.ci_low,q.ci_high,color=color,alpha=.15)
        q=b[b.contrast.eq(contrast)];axes[1].plot(q.age,q.bootstrap_low,color=color,ls=':',lw=.9);axes[1].plot(q.age,q.bootstrap_high,color=color,ls=':',lw=.9)
    axes[1].axhline(0,color='#73829A',lw=.8);axes[1].set(xlabel='Age (years)',ylabel='Difference in high-choice probability');axes[1].legend()
    heading(fig,'Does the friend advantage vary with age?','Continuous-age GEE, 111 participants. Equal participant standardization over offers/time; shaded robust 95% CIs; dotted cluster-bootstrap limits.')
    save(fig,'15_age_partner_behavior')
