"""Paired MLE/hierarchy recovery on identical simulated participants and schedules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from joblib import Parallel,delayed
from .hierarchical import inputs,sample,transform,friend_effect,offer_distribution,summarize,configuration,collect,TABLE
from .models import simulate
from .fitting import fit,stable_seed


def recover(condition,replicate):
    cfg=configuration();data,meta,arrays,ratings,frames=inputs('H5')
    seed=stable_seed(cfg['seed'],'hierarchical_recovery',condition,replicate)
    rng=np.random.default_rng(seed)
    mu=np.array([-1.,-1.2,1.]);tau=np.array([.65,.65,1.])
    beta=np.array([0.,0.,{'zero':0.,'positive':.35,'negative':-.35}[condition]])
    corr=np.array([[1,0,0],[0,1,-.45],[0,-.45,1.]])
    age=np.array(data['age'])[:,0]
    eta=mu+age[:,None]*beta+rng.normal(size=(data['N'],3))@np.linalg.cholesky(corr).T*tau
    truth=transform(eta,[1,2,3]);names=['alpha','kappa','theta'];synthetic={}
    for i,(sub,a) in enumerate(arrays.items()):
        synthetic[sub],_=simulate(a,'M5',dict(zip(names,truth[i])),ratings.get(sub,np.zeros(3)),rng)
    label=f'H5_recovery_{condition}_{replicate}'
    posterior,pm,_,_,_=sample('H5',override=synthetic,label=label)
    natural=posterior.stan_variable('natural');gaps,weights=offer_distribution(frames)
    folder=Path('work/hierarchical')/label;mle_file=folder/'mle.csv'
    if mle_file.exists():mle=pd.read_csv(mle_file)
    else:
        fits=Parallel(n_jobs=4,verbose=5)(delayed(fit)(a,'M5_theta10',ratings.get(sub,np.zeros(3)),cfg['recovery_starts'],stable_seed(seed,sub,'mle')) for sub,a in synthetic.items())
        mle=pd.DataFrame(fits);mle['participant_id']=list(synthetic);mle.to_csv(mle_file,index=False)
    mlepars=mle[names].to_numpy();estimate=natural.mean(axis=0)
    truths=np.column_stack([truth,friend_effect(truth,gaps,weights)])
    hier_draws=np.concatenate([natural,friend_effect(natural,gaps,weights)[...,None]],axis=-1)
    hier=hier_draws.mean(axis=0);mle_est=np.column_stack([mlepars,friend_effect(mlepars,gaps,weights)])
    rows=[];individual=[]
    for j,p in enumerate(names+['friend_value_probability_effect']):
        for method,rec in [('hierarchical',hier[:,j]),('MLE_theta10',mle_est[:,j])]:
            err=rec-truths[:,j]
            rows.append(dict(condition=condition,replicate=replicate,method=method,level='participant',parameter=p,n=len(rec),
                             rmse=np.sqrt(np.mean(err**2)),bias=np.mean(err),pearson=pearsonr(rec,truths[:,j]).statistic,
                             spearman=spearmanr(rec,truths[:,j]).statistic,
                             ceiling_rate=float(np.mean(rec>=9.999)) if p=='theta' else np.nan,
                             interval_coverage=np.mean((truths[:,j]>=np.quantile(hier_draws[:,:,j],.025,axis=0))&(truths[:,j]<=np.quantile(hier_draws[:,:,j],.975,axis=0))) if method=='hierarchical' else np.nan))
            for i,sub in enumerate(synthetic):individual.append(dict(condition=condition,replicate=replicate,method=method,participant_id=sub,parameter=p,true=truths[i,j],recovered=rec[i]))
    for variable,true in [('mu',mu),('tau',tau),('beta',beta)]:
        draws=posterior.stan_variable(variable)
        if variable=='beta':draws=draws[:,:,0]
        for j,p in enumerate(names):
            ss=summarize(draws[:,j]);rows.append(dict(condition=condition,replicate=replicate,method='hierarchical',level='population',parameter=variable+'_'+p,
                                                    true=true[j],bias=ss['mean']-true[j],interval_coverage=float(ss['ci_low']<=true[j]<=ss['ci_high']),probability_positive=float(np.mean(draws[:,j]>0)),**ss))
    # Recovery of the typical-participant probability-scale age difference.
    ages=(np.array([25.,75.])-meta['age_mean'])/meta['age_sd']
    truepar=transform(mu+ages[:,None]*beta,[1,2,3]);trueeffect=np.diff(friend_effect(truepar,gaps,weights))[0]
    mu_draw=posterior.stan_variable('mu');b_draw=posterior.stan_variable('beta')[:,:,0]
    effect=[]
    for a in ages:effect.append(friend_effect(transform(mu_draw+a*b_draw,[1,2,3]),gaps,weights))
    ss=summarize(effect[1]-effect[0]);rows.append(dict(condition=condition,replicate=replicate,method='hierarchical',level='population',parameter='friend_probability_change_25_to_75',true=trueeffect,bias=ss['mean']-trueeffect,interval_coverage=float(ss['ci_low']<=trueeffect<=ss['ci_high']),**ss))
    pd.DataFrame(rows).to_csv(TABLE/f'recovery_{label}.csv',index=False)
    pd.DataFrame(individual).to_csv(TABLE/f'recovery_individual_{label}.csv',index=False)
    return rows


def run(replicates=None):
    replicates=replicates or configuration()['recovery_replicates']
    for condition in ['zero','positive','negative']:
        for it in range(replicates):
            if not (TABLE/f'recovery_H5_recovery_{condition}_{it}.csv').exists():recover(condition,it)
            aggregate()
    collect()


def aggregate():
    paths=sorted(TABLE.glob('recovery_H5_recovery_*.csv'))
    if paths:
        d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);d['interval_coverage']=pd.to_numeric(d.interval_coverage.replace({'True':1.,'False':0.}),errors='raise');d.to_csv(TABLE/'hierarchical_recovery_by_dataset.csv',index=False)
        d.groupby(['condition','method','level','parameter']).agg(datasets=('replicate','nunique'),rmse=('rmse','mean'),bias=('bias','mean'),pearson=('pearson','mean'),spearman=('spearman','mean'),coverage=('interval_coverage','mean'),ceiling_rate=('ceiling_rate','mean')).to_csv(TABLE/'hierarchical_recovery_summary.csv')
