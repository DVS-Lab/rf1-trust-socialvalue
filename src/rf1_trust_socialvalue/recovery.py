"""Stochastic, schedule-preserving parameter/model recovery and predictive checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel,delayed
from .models import simulate,specification,pack
from .fitting import fit,stable_seed,load_inputs


def _recover_subject(sub,frame,name,params,ratings,config):
    a=pack(frame);rows=[]
    for iteration in range(config['recovery_iterations']):
        seed=stable_seed(config['seed'],'parameter_recovery',sub,name,iteration)
        sim,_=simulate(a,name,params,ratings,np.random.default_rng(seed))
        rec=fit(sim,name,ratings,config['recovery_starts'],seed+1)
        for p in specification(name)[1]:
            rows.append(dict(participant_id=sub,model=name,iteration=iteration,parameter=p,true=params[p],recovered=rec[p],
                boundary=p in rec['boundary_parameters'].split(';'),n_near_best=rec['n_near_best'],converged=rec['converged']))
    return rows


def _safe_corr(a,b,method):
    if np.std(a)<1e-8 or np.std(b)<1e-8:return np.nan
    return method(a,b).statistic


def parameter_recovery(config):
    t,s,r=load_inputs();fits=pd.read_csv('results/tables/model_fits.csv');tasks=[]
    for row in fits.to_dict('records'):
        sub=row['participant_id'];name=row['model']
        if name=='M0':continue
        tasks.append((sub,t[t.participant_id.eq(sub)],name,row,r.get(sub,np.zeros(3))))
    result=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(_recover_subject)(*task,config) for task in tasks)
    rows=pd.DataFrame([row for batch in result for row in batch]);rows.to_csv('results/tables/parameter_recovery.csv',index=False)
    summaries=[]
    for (model,p,iteration),g in rows.groupby(['model','parameter','iteration']):
        err=g.recovered-g.true
        summaries.append(dict(model=model,parameter=p,iteration=iteration,n=len(g),pearson=_safe_corr(g.true,g.recovered,stats.pearsonr),
            spearman=_safe_corr(g.true,g.recovered,stats.spearmanr),rmse=np.sqrt(np.mean(err**2)),bias=err.mean(),boundary_rate=g.boundary.mean()))
    d=pd.DataFrame(summaries);d.to_csv('results/tables/recovery_by_iteration.csv',index=False)
    d.groupby(['model','parameter']).agg(n_subjects=('n','first'),iterations=('iteration','size'),pearson=('pearson','mean'),pearson_sd=('pearson','std'),
        spearman=('spearman','mean'),rmse=('rmse','mean'),bias=('bias','mean'),boundary_rate=('boundary_rate','mean')).to_csv('results/tables/recovery_summary.csv')
    print('Parameter recovery complete',flush=True)


def _recover_model(sub,frame,generating,params,ratings,models,iteration,config):
    seed=stable_seed(config['seed'],'model_recovery',sub,generating,iteration)
    a,_=simulate(pack(frame),generating,params,ratings,np.random.default_rng(seed))
    fitted=[fit(a,m,ratings,config['recovery_starts'],stable_seed(seed,m)) for m in models]
    rows=[]
    for metric in ['AICc','BIC']:
        best=min(x[metric] for x in fitted);wins=[x['model'] for x in fitted if x[metric]-best<1e-6]
        for rec in fitted:
            rows.append(dict(participant_id=sub,iteration=iteration,generating=generating,fitted=rec['model'],metric=metric,
                criterion=rec[metric],selected_weight=(1/len(wins) if rec['model'] in wins else 0),converged=rec['converged']))
    return rows


def model_recovery(config):
    t,s,r=load_inputs();f=pd.read_csv('results/tables/model_fits.csv')
    extras=['preference','M2_power','M5_power']
    robust=pd.read_csv('results/tables/robustness_fits.csv')
    f=pd.concat([f,robust[robust.model.isin(extras)]],ignore_index=True)
    models=config['primary_models']+config['secondary_models']+extras
    complete=f.groupby('participant_id').model.nunique().eq(len(models));tasks=[]
    for sub in complete[complete].index:
        for row in f[f.participant_id.eq(sub)].to_dict('records'):
            for it in range(config['model_recovery_iterations']):tasks.append((sub,t[t.participant_id.eq(sub)],row['model'],row,r[sub],models,it))
    results=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(_recover_model)(*task,config) for task in tasks)
    d=pd.DataFrame([x for batch in results for x in batch]);d.to_csv('results/tables/model_recovery.csv',index=False)
    d.groupby(['metric','generating','fitted']).selected_weight.mean().rename('selection_probability').to_csv('results/tables/model_recovery_confusion.csv')
    print('Model recovery complete',flush=True)


def _predictive(sub,frame,name,params,ratings,config):
    a=pack(frame);valid=a[:,3]>=0
    categories=[]
    for c in range(3):
        categories.append(('partner',str(c),(a[:,0]==c)&valid))
        for pair in sorted(frame.offer_pair.unique()):categories.append(('offer',f'{c}:{pair}',(a[:,0]==c)&frame.offer_pair.eq(pair).to_numpy()&valid))
        for b in sorted(frame.trial_bin.unique()):categories.append(('time',f'{c}:{b}',(a[:,0]==c)&frame.trial_bin.eq(b).to_numpy()&valid))
        for previous in [0,1]:categories.append(('recent',f'{c}:{previous}',None))
    def recent(arr):
        last=np.full(3,-1.);out=np.zeros(len(arr))
        for i in range(len(arr)):
            c=int(arr[i,0]);out[i]=last[c]
            if arr[i,4]>0:last[c]=arr[i,5]
        return out
    original_recent=recent(a);values=np.full((len(categories),config['predictive_iterations'],2),np.nan);observed=[]
    for k,(category,label,mask) in enumerate(categories):
        if category=='recent':
            c,y=map(int,label.split(':'));mask=(a[:,0]==c)&(original_recent==y)&valid
        amount=np.where(a[:,3]==1,a[:,2],a[:,1]);observed.append([a[mask,3].mean(),amount[mask].mean()] if mask.any() else [np.nan,np.nan])
    for it in range(config['predictive_iterations']):
        seed=stable_seed(config['seed'],'ppc',sub,name,it)
        sim,_=simulate(a,name,params,ratings,np.random.default_rng(seed));last=recent(sim)
        amount=np.where(sim[:,3]==1,sim[:,2],sim[:,1])
        for k,(category,label,mask) in enumerate(categories):
            if category=='recent':
                c,y=map(int,label.split(':'));mask=(a[:,0]==c)&(last==y)&valid
            if mask.any():values[k,it]=[sim[mask,3].mean(),amount[mask].mean()]
    return [(name,sub,cat,label,observed[k],values[k]) for k,(cat,label,_) in enumerate(categories)]


def predictive_checks(config):
    t,s,r=load_inputs();fits=pd.read_csv('results/tables/model_fits.csv')
    tasks=[(row['participant_id'],t[t.participant_id.eq(row['participant_id'])],row['model'],row,r.get(row['participant_id'],np.zeros(3))) for row in fits.to_dict('records')]
    results=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(_predictive)(*task,config) for task in tasks)
    groups={}
    for batch in results:
        for name,sub,cat,label,obs,values in batch:groups.setdefault((name,cat,label),[]).append((obs,values))
    rows=[]
    for (name,cat,label),batch in groups.items():
        obs=np.array([b[0] for b in batch]);sims=np.array([b[1] for b in batch])
        for j,measure in enumerate(['high_probability','investment']):
            estimates=np.nanmean(sims[:,:,j],axis=0)
            rows.append(dict(model=name,category=cat,label=label,measure=measure,observed=np.nanmean(obs[:,j]),predicted=estimates.mean(),
                sim_low=np.quantile(estimates,.025),sim_high=np.quantile(estimates,.975),n_subjects=int(np.isfinite(obs[:,j]).sum())))
    pd.DataFrame(rows).to_csv('results/tables/predictive_checks.csv',index=False)
    print('Simulation predictive checks complete',flush=True)
