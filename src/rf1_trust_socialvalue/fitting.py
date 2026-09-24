"""Seeded multistart MLE, convergence diagnostics, and prospective scoring."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed
from .models import specification, value_gradient, objective, trajectory, pack


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')


def fit(a,name,ratings,n_starts=100,seed=0,initial_params=None):
    code,names,indices,bounds=specification(name);n=int((a[:,3]>=0).sum());k=len(names)
    if n<=k+1:raise ValueError(f'Insufficient valid trials for {name}: {n}')
    if not k:
        best_nll=n*np.log(2);params={};success=True;near=1;converged=1;bound=[];best_message='exact'
    else:
        rng=np.random.default_rng(seed);solutions=[]
        for i in range(n_starts+(initial_params is not None)):
            x=np.array([rng.uniform(l,u) for l,u in bounds])
            for j,key in enumerate(names):
                if key=='kappa':x[j]=np.exp(rng.uniform(np.log(.01),np.log(5.)))
            if i==0:
                x=np.array([.2 if p.startswith('alpha') else 1. if p in ['kappa','rho','phi'] else 0. for p in names])
            if i==n_starts and initial_params is not None:
                x=np.array([np.clip(initial_params[p],*b) for p,b in zip(names,bounds)])
            result=minimize(value_gradient,x,args=(indices,a,code,ratings,'reset' in name),jac=True,
                method='L-BFGS-B',bounds=bounds,options={'maxiter':600,'ftol':1e-11,'gtol':1e-6,'maxls':40})
            solutions.append(result)
        # Do not substitute a worse converged solution if the lowest solution failed.
        best=min(solutions,key=lambda r:r.fun)
        if not best.success:
            successful=[r for r in solutions if r.success and r.fun<=best.fun+1e-4]
            if successful:best=min(successful,key=lambda r:r.fun)
        if not best.success:
            retry=minimize(objective,best.x,args=(indices,a,code,ratings,'reset' in name),
                method='Powell',bounds=bounds,options={'maxiter':2000,'ftol':1e-10,'xtol':1e-8})
            if retry.success and retry.fun<=best.fun+1e-5:best=retry
        best_nll=float(best.fun);params=dict(zip(names,map(float,best.x)));success=bool(best.success)
        converged=sum(r.success for r in solutions);near=sum(r.success and abs(r.fun-best_nll)<1e-4 for r in solutions)
        bound=[p for p,(l,u),v in zip(names,bounds,best.x) if min(v-l,u-v)<1e-4*(u-l)]
        best_message=str(best.message)
        if not np.isfinite(best_nll):raise RuntimeError(f'Nonfinite likelihood: {name}')
        if not success:raise RuntimeError(f'Unresolved optimizer failure: {name}, {best_message}')
    ll=-best_nll;aic=2*k-2*ll;random_aic=2*n*np.log(2)
    return dict(model=name,n=n,k=k,log_likelihood=ll,AIC=aic,AICc=aic+2*k*(k+1)/(n-k-1),BIC=k*np.log(n)-2*ll,
        pseudo_r2_fareri=(random_aic-aic)/random_aic,pseudo_r2_mcfadden=1-ll/(-n*np.log(2)),
        converged=success,n_starts=n_starts+(initial_params is not None) if k else 1,n_converged=converged,n_near_best=near,
        boundary_parameters=';'.join(bound),optimizer_message=best_message,**params)


def load_inputs():
    t=pd.read_csv('results/tables/trial_table.csv');s=pd.read_csv('results/tables/sample_audit.csv')
    t=t[t.participant_id.isin(s.loc[s.primary_include,'participant_id'])].copy()
    r=pd.read_csv('results/tables/ratings.csv');r=r[r.trait.eq(2)]
    rating={sub:g.set_index('partner').normalized.reindex(['friend','stranger','computer']).to_numpy(float) for sub,g in r.groupby('participant_id')}
    return t,s,rating


def one_fit(sub,frame,name,rating,config,heldout=False,initial_params=None):
    a=pack(frame);r=rating.get(sub,np.zeros(3)).copy()
    if 'ratingcentered' in name:r=2*r-1
    seed_name='M5_wide' if name=='M5_kappa100' else name
    seed=stable_seed(config['seed'],sub,seed_name,heldout)
    split=len(a)
    if heldout:
        runs=np.unique(a[:,7])
        split=int(np.flatnonzero(a[:,7]!=runs[0])[0]) if len(runs)>1 else int(np.floor(.65*len(a)))
    result=fit(a[:split],name,r,config['n_starts'],seed,initial_params)
    result['participant_id']=sub
    if heldout:
        tr=trajectory(a,name,result,r);valid=a[split:,3]>=0;pred=tr[split:,1][valid];y=a[split:,3][valid]
        result.update(heldout_n=int(valid.sum()),heldout_log_loss=float(tr[split:,3][valid].mean()),
            heldout_accuracy=float(((pred>=.5)==y).mean()),heldout_brier=float(np.mean((pred-y)**2)),
            split='run_1_to_2' if len(np.unique(a[:,7]))>1 else 'chronological_65_percent')
        preds=pd.DataFrame(dict(participant_id=sub,model=name,global_trial=frame.global_trial.iloc[split:].to_numpy()[valid],probability=pred,choice=y))
    else:
        tr=trajectory(a,name,result,r)
        preds=pd.DataFrame(dict(participant_id=sub,model=name,global_trial=frame.global_trial,belief=tr[:,0],probability=tr[:,1],trial_nll=tr[:,3]))
    return result,preds


def run_fits(config,models=None,heldout=False,filename=None):
    t,s,rating=load_inputs();models=models or config['primary_models']+config['secondary_models']
    tasks=[]
    for sub,frame in t.groupby('participant_id',sort=True):
        for name in models:
            if name.startswith(('M3','M4')) and (sub not in rating or not np.isfinite(rating[sub]).all()):continue
            tasks.append((sub,frame,name))
    start=time.time();print(f'Fitting {len(tasks)} datasets; {config["n_starts"]} starts; heldout={heldout}',flush=True)
    fitted=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(one_fit)(sub,f,name,rating,config,heldout) for sub,f,name in tasks)
    tab=pd.DataFrame([x[0] for x in fitted]);pred=pd.concat([x[1] for x in fitted],ignore_index=True)
    stem=filename or ('heldout_fits' if heldout else 'model_fits')
    tab.to_csv(f'results/tables/{stem}.csv',index=False)
    pred.to_csv(f'results/tables/{stem}_predictions.csv',index=False)
    print(f'{stem}: {time.time()-start:.1f}s; {len(tab)} fits',flush=True)
    return tab
