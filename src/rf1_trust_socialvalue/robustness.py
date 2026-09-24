"""Competing explanations, sample restrictions, and exploratory age associations."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from .fitting import run_fits,load_inputs

VARIANTS=['M2_reset','M5_reset','M5_signed','M7_signed','M4_signed','M2_lapse','M5_lapse','M2_side','M5_side','M2_power','M5_power','M5_wide','M4_ratingcentered','preference']


def run_robustness(config):
    f=run_fits(config,models=VARIANTS,filename='robustness_fits')
    primary=pd.read_csv('results/tables/model_fits.csv');_,sample,_=load_inputs()
    rows=[]
    comparisons=[(m,'M5' if m.startswith('M5') else 'M4' if m.startswith('M4') else 'M7' if m.startswith('M7') else 'M2') for m in VARIANTS]
    comparisons += [('M5_power','M2_power'),('M5_side','M2_side'),('M5_lapse','M2_lapse'),('M7','preference')]
    combined=pd.concat([primary,f],ignore_index=True)
    for a,b in comparisons:
        x=combined[combined.model.eq(a)].set_index('participant_id');y=combined[combined.model.eq(b)].set_index('participant_id');ids=x.index.intersection(y.index)
        for metric in ['AIC','AICc','BIC']:
            diff=x.loc[ids,metric]-y.loc[ids,metric]
            rows.append(dict(comparison=f'{a} - {b}',sample='primary',metric=metric,n=len(ids),mean_difference=diff.mean(),median_difference=diff.median(),fraction_improved=(diff<0).mean()))
    for label,ids in [('sensitivity',sample.loc[sample.sensitivity_include,'participant_id']),('complete_ratings',sample.loc[sample.primary_include&sample.complete_ratings,'participant_id'])]:
        sub=primary[primary.participant_id.isin(ids)]
        for model in config['primary_models']:
            x=sub[sub.model.eq(model)].set_index('participant_id');base=sub[sub.model.eq('M2')].set_index('participant_id')
            for metric in ['AIC','AICc','BIC']:
                diff=x[metric]-base[metric]
                rows.append(dict(comparison=f'{model} - M2',sample=label,metric=metric,n=len(diff),mean_difference=diff.mean(),median_difference=diff.median(),fraction_improved=(diff<0).mean()))
    pd.DataFrame(rows).to_csv('results/tables/robustness_comparisons.csv',index=False)
    age_analysis(primary,sample)


def age_analysis(f,sample):
    rows=[];data=f[f.model.eq('M5')].merge(sample[['participant_id','age']],on='participant_id')
    data['age_z']=(data.age-data.age.mean())/data.age.std(ddof=0)
    for param in ['theta','alpha','kappa']:
        for form in ['linear','quadratic']:
            x=pd.DataFrame({'age_z':data.age_z})
            if form=='quadratic':x['age_z_squared']=data.age_z**2
            model=sm.OLS(data[param],sm.add_constant(x)).fit(cov_type='HC3')
            for term in x:
                ci=model.conf_int().loc[term]
                rows.append(dict(model='M5',parameter=param,form=form,term=term,n=len(data),estimate=model.params[term],se=model.bse[term],p=model.pvalues[term],ci_low=ci.iloc[0],ci_high=ci.iloc[1],r_squared=model.rsquared))
    tab=pd.DataFrame(rows);tab['p_holm']=multipletests(tab.p,method='holm')[1];tab.to_csv('results/tables/age_exploratory.csv',index=False)
