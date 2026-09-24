"""Offer-controlled clustered behavioral inference and subject bootstrap summaries."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from .data import PARTNERS
from .fitting import load_inputs,stable_seed


def boot_ci(x,rng,n=5000):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    draws=x[rng.integers(0,len(x),(n,len(x)))].mean(axis=1)
    return np.quantile(draws,[.025,.975])


def run_behavior(config):
    t=pd.read_csv('results/tables/trial_table.csv');primary,s,r=load_inputs()
    v=t[t.valid_choice].copy();v['chose_high']=v.chose_high.astype(float)
    out=Path('results/tables');rng=np.random.default_rng(config['seed'])
    b=v.groupby(['participant_id','partner']).agg(investment=('chosen_amount','mean'),high_probability=('chose_high','mean'),rt=('response_time','median'),age=('age','first')).reset_index()
    b.to_csv(out/'behavior_participant.csv',index=False)
    rows=[]
    for partner,g in b.groupby('partner'):
        for measure in ['investment','high_probability','rt']:
            low,high=boot_ci(g[measure],rng,config['bootstrap_iterations'])
            rows.append(dict(partner=partner,measure=measure,n=len(g),mean=g[measure].mean(),ci_low=low,ci_high=high))
    pd.DataFrame(rows).to_csv(out/'behavior_summary.csv',index=False)
    for keys,name in [(['participant_id','partner','offer_pair'],'behavior_offer'),(['participant_id','partner','trial_bin'],'behavior_time'),(['participant_id','partner','chose_high'],'response_times'),(['participant_id','partner','last_observed_reciprocation'],'behavior_recent_feedback')]:
        v.groupby(keys).agg(high_probability=('chose_high','mean'),investment=('chosen_amount','mean'),rt=('response_time','median'),n=('chose_high','size')).to_csv(out/f'{name}.csv')
    pv=primary[primary.valid_choice].copy();pv['chose_high']=pv.chose_high.astype(float)
    pv['partner']=pd.Categorical(pv.partner,categories=['computer','stranger','friend'])
    model=smf.gee('chose_high ~ C(partner) * trial_scaled + C(offer_pair)',groups='participant_id',data=pv,
        family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    coef=pd.DataFrame(dict(term=model.params.index,estimate=model.params.values,se=model.bse.values,p=model.pvalues.values,
        ci_low=model.conf_int().iloc[:,0].values,ci_high=model.conf_int().iloc[:,1].values))
    coef['odds_ratio']=np.exp(coef.estimate);coef.to_csv(out/'behavior_gee.csv',index=False)
    (Path('results')/'behavior_gee.txt').write_text(str(model.summary()))
    # Participant differences average offer-specific high probabilities equally.
    cell=pv.groupby(['participant_id','partner','offer_pair'],observed=True).chose_high.mean().groupby(['participant_id','partner'],observed=True).mean().unstack()
    rows=[]
    for a,c in combinations(PARTNERS,2):
        d=(cell[a]-cell[c]).dropna();lo,hi=boot_ci(d,rng,config['bootstrap_iterations'])
        rows.append(dict(contrast=f'{a} - {c}',n=len(d),difference=d.mean(),ci_low=lo,ci_high=hi,p=stats.wilcoxon(d).pvalue))
    df=pd.DataFrame(rows);df['p_holm']=multipletests(df.p,method='holm')[1];df.to_csv(out/'behavior_paired_bootstrap.csv',index=False)
    ratings=pd.read_csv(out/'ratings.csv');ratings=ratings[ratings.trait.eq(2)]
    joined=b.merge(ratings,on=['participant_id','partner'])
    corr=[]
    for partner,g in joined.groupby('partner'):
        z=stats.spearmanr(g.rating,g.high_probability)
        corr.append(dict(partner=partner,n=len(g),spearman=z.statistic,p=z.pvalue))
    d=pd.DataFrame(corr);d['p_holm']=multipletests(d.p,method='holm')[1];d.to_csv(out/'rating_behavior_correlations.csv',index=False)
    demographics=s[s.n_valid.gt(0)]
    summary={'n':len(demographics),'age_min':float(demographics.age.min()),'age_max':float(demographics.age.max()),
        'age_mean':float(demographics.age.mean()),'age_sd':float(demographics.age.std()),'sex_as_released':demographics.sex.value_counts().to_dict(),
        'gee_converged':bool(model.converged),'n_inference':pv.participant_id.nunique()}
    Path('results/demographics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Behavioral analysis complete',flush=True)


def compare_models(config):
    out=Path('results/tables');f=pd.read_csv(out/'model_fits.csv');rng=np.random.default_rng(config['seed']+1)
    primary=config['primary_models'];complete=f.groupby('participant_id').model.nunique().eq(len(primary)+2)
    samples={'primary':f[f.model.isin(primary)],'rating_complete_secondary':f[f.participant_id.isin(complete[complete].index)]}
    summaries=[];pairs=[];deltas=[]
    for sample,data in samples.items():
        assert data.groupby('participant_id').n.nunique().eq(1).all()
        for metric in ['AIC','AICc','BIC']:
            mat=data.pivot(index='participant_id',columns='model',values=metric)
            assert not mat.isna().any().any()
            delta=mat.sub(mat.min(axis=1),axis=0)
            # Split credit for exact criterion ties.
            wins=(delta<1e-6);credit=wins.div(wins.sum(axis=1),axis=0)
            for name in mat:
                summaries.append(dict(sample=sample,metric=metric,model=name,n=len(mat),mean=mat[name].mean(),median=mat[name].median(),mean_delta=delta[name].mean(),best_share=credit[name].mean()))
                for sub,value in delta[name].items():deltas.append(dict(sample=sample,metric=metric,model=name,participant_id=sub,delta=value))
            for a,b in combinations(mat.columns,2):
                diff=mat[a]-mat[b];low,high=boot_ci(diff,rng,config['bootstrap_iterations'])
                p=stats.wilcoxon(diff).pvalue if np.any(abs(diff)>1e-10) else 1.
                pairs.append(dict(sample=sample,metric=metric,model_a=a,model_b=b,n=len(diff),mean_difference=diff.mean(),median_difference=diff.median(),ci_low=low,ci_high=high,p=p))
    pd.DataFrame(summaries).to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'model_deltas.csv',index=False)
    p=pd.DataFrame(pairs);p['p_holm']=p.groupby(['sample','metric']).p.transform(lambda x:multipletests(x,method='holm')[1]);p.to_csv(out/'model_pairwise.csv',index=False)
    h=pd.read_csv(out/'heldout_fits.csv')
    h.groupby('model').agg(n=('participant_id','size'),log_loss=('heldout_log_loss','mean'),accuracy=('heldout_accuracy','mean'),brier=('heldout_brier','mean')).to_csv(out/'heldout_summary.csv')
    pp=pd.read_csv(out/'heldout_fits_predictions.csv');pp['probability_bin']=pd.cut(pp.probability,np.linspace(0,1,11),include_lowest=True)
    pp.groupby(['model','probability_bin'],observed=True).agg(mean_prediction=('probability','mean'),observed_frequency=('choice','mean'),n=('choice','size')).to_csv(out/'heldout_calibration.csv')
    comparisons=[]
    for sample,data in {'primary':h[h.model.isin(primary)],'rating_complete_secondary':h[h.participant_id.isin(complete[complete].index)]}.items():
        mat=data.pivot(index='participant_id',columns='model',values='heldout_log_loss')
        for a,b in combinations(mat.columns,2):
            diff=(mat[a]-mat[b]).dropna();lo,hi=boot_ci(diff,rng,config['bootstrap_iterations'])
            comparisons.append(dict(sample=sample,model_a=a,model_b=b,n=len(diff),mean_difference=diff.mean(),ci_low=lo,ci_high=hi,p=stats.wilcoxon(diff).pvalue if abs(diff).max()>1e-10 else 1))
    q=pd.DataFrame(comparisons);q['p_holm']=q.groupby('sample').p.transform(lambda x:multipletests(x,method='holm')[1]);q.to_csv(out/'heldout_pairwise.csv',index=False)
