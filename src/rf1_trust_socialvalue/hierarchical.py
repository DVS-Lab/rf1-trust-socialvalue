"""Joint Stan/NUTS inference, predictive scoring and age summaries.

Full chains live in ignored work/hierarchical. All age standardization uses one
row per primary participant and is fixed across full/training/rating analyses.
"""
from pathlib import Path
import json
import hashlib
import shutil
import time
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from .models import pack,trajectory,simulate
from .fitting import load_inputs,stable_seed

SPECS={'H2':('M2',2,['alpha','kappa'],[1,2]),
       'H5':('M5',5,['alpha','kappa','theta'],[1,2,3]),
       'H8':('M8',8,['alpha','kappa','alpha_negative'],[1,2,1]),
       'HPreference':('preference',9,['alpha','kappa','theta','preference_stranger'],[1,2,4,4]),
       'H7':('M7',7,['alpha','kappa','theta','theta_stranger'],[1,2,3,3]),
       'H4':('M4',4,['alpha','kappa','theta'],[1,2,3])}
TABLE=Path('results/tables');WORK=Path('work/hierarchical')


def configuration():
    return json.loads(Path('config/hierarchical.json').read_text())


def split_index(a):
    runs=np.unique(a[:,7])
    return int(np.flatnonzero(a[:,7]!=runs[0])[0]) if len(runs)>1 else int(np.floor(.65*len(a)))


def inputs(model,training=False,age_terms=1,bounded=False,independent=False,override=None,prior_scale=1):
    t,_,ratings=load_inputs();ages=t.groupby('participant_id').age.first()
    center=float(ages.mean());scale=float(ages.std(ddof=0))
    if model=='H4':t=t[t.participant_id.isin(ratings)]
    frames={sub:f for sub,f in t.groupby('participant_id',sort=True)}
    arrays={sub:pack(f) for sub,f in frames.items()}
    if override is not None:arrays=override
    ids=list(arrays);age=(ages.loc[ids].to_numpy()-center)/scale
    x=np.column_stack([age,age**2])[:,:age_terms]
    first=[];last=[];all_a=[];cursor=1
    for sub,a in arrays.items():
        b=a[:split_index(a)] if training else a
        b=b[b[:,3]>=0];first.append(cursor);last.append(cursor+len(b)-1);cursor+=len(b);all_a.append(b)
    a=np.concatenate(all_a);_,code,names,kinds=SPECS[model]
    loc=[-1. if k==1 else -1.2 if k==2 else 1. if k==3 else 0. for k in kinds]
    if bounded:
        loc=[np.log(1.3/8.7) if k==3 else v for k,v in zip(kinds,loc)]
    data=dict(N=len(ids),T=len(a),K=len(names),A=age_terms,model_code=code,bounded_theta=int(bounded),independent=int(independent),
              kind=kinds,first=first,last=last,partner=(a[:,0]+1).astype(int).tolist(),gap=(a[:,2]-a[:,1]).tolist(),
              y=a[:,3].astype(int).tolist(),feedback=a[:,4].astype(int).tolist(),outcome=a[:,5].tolist(),
              ratings=[ratings.get(sub,np.zeros(3)).tolist() for sub in ids],age=x.tolist(),
              mu_location=loc,mu_scale=[1.25 if k==1 else .8 if k==2 else 1.5 if k==3 else 1. for k in kinds],
              prior_scale=prior_scale,prior_only=0)
    meta=dict(ids=ids,age_mean=center,age_sd=scale,ages=ages.loc[ids].to_list(),model=model,training=training,
              age_terms=age_terms,bounded=bounded,independent=independent,prior_scale=prior_scale,n_trials=len(a))
    return data,meta,arrays,ratings,frames


def stan_model():
    from cmdstanpy import CmdStanModel,set_cmdstan_path
    installations=sorted(Path('work/cmdstan').glob('cmdstan-*'))
    if not installations:raise RuntimeError('Install CmdStan into work/cmdstan first; see README')
    set_cmdstan_path(str(installations[-1].resolve()))
    WORK.mkdir(parents=True,exist_ok=True)
    # Build in ignored work; preserve small Stan source in version control.
    source=Path('stan/hierarchical_fast.stan').read_text()+'\n// Header SHA256: '+hashlib.sha256(Path('stan/rl_fast.hpp').read_bytes()).hexdigest()+'\n'
    dest=WORK/'hierarchical_fast.stan'
    if not dest.exists() or dest.read_text()!=source:dest.write_text(source)
    return CmdStanModel(stan_file=str(dest.resolve()),user_header=str(Path('stan/rl_fast.hpp').resolve()),stanc_options={'allow-undefined':True})


def run_name(model,training=False,age_terms=1,bounded=False,independent=False,prior_scale=1):
    return '_'.join([model,'train' if training else 'full',['noage','age','quadratic'][age_terms]]+
                    (['bounded'] if bounded else [])+(['independent'] if independent else [])+
                    ([f'prior{prior_scale:g}'] if prior_scale!=1 else []))


def load_chains(files):
    """Load unmodified CSVs, including CmdStanPy 1.3's zero-warmup edge case."""
    from cmdstanpy import from_csv,CmdStanMCMC
    from cmdstanpy.cmdstan_args import CmdStanArgs,SamplerArgs
    from cmdstanpy.stanfit.runset import RunSet
    from cmdstanpy.utils import stancsv
    try:return from_csv(files)
    except ValueError as exc:
        if 'Must specify iter_warmup > 0 when adapt_engaged=True.' not in str(exc):raise
    comments,*_=stancsv.parse_comments_header_and_draws(files[0]);cfg=stancsv.parse_config(comments)
    if cfg.get('num_warmup')!=0:raise ValueError('Continuation loader requires zero warmup')
    if cfg.get('engaged')!=0:raise ValueError('Zero-warmup continuation must have adaptation disabled')
    if len(set(files))!=len(files):raise ValueError('Chain CSV paths must be distinct')
    args=SamplerArgs(iter_warmup=0,iter_sampling=cfg['num_samples'],thin=cfg['thin'],save_warmup=False,adapt_engaged=False,max_treedepth=cfg['max_depth'])
    command=CmdStanArgs(model_name=cfg['model'],model_exe=cfg['model'],chain_ids=list(range(1,len(files)+1)),method_args=args)
    runs=RunSet(args=command,chains=len(files));runs._csv_files=files
    for i in range(len(files)):runs._set_retcode(i,0)
    fit=CmdStanMCMC(runs);fit.draws();return fit


def _sample(model,training=False,age_terms=1,bounded=False,independent=False,override=None,label=None,prior_scale=1,settings=None):
    from cmdstanpy import from_csv,set_cmdstan_path
    installed=sorted(Path('work/cmdstan').glob('cmdstan-*'))
    if not installed:raise RuntimeError('Install CmdStan in work/cmdstan before loading or fitting chains')
    set_cmdstan_path(str(installed[-1].resolve()))
    cfg=configuration();cfg.update(settings or {})
    cfg.setdefault('metric','dense_e' if model in ['HPreference','H7'] else 'diag_e')
    data,meta,arrays,ratings,frames=inputs(model,training,age_terms,bounded,independent,override,prior_scale)
    name=label or run_name(model,training,age_terms,bounded,independent,prior_scale)
    folder=WORK/name;folder.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256((json.dumps(data,sort_keys=True)+json.dumps(cfg,sort_keys=True)+Path('stan/hierarchical_shared.stan').read_text()).encode()).hexdigest()
    manifest=folder/'manifest.json'
    if manifest.exists() and settings is None:
        previous=json.loads(manifest.read_text())
        previous_cfg=previous['settings']
        if previous_cfg['seed']==cfg['seed'] and previous_cfg['chains']>=cfg['chains'] and previous_cfg['warmup']>=cfg['warmup'] and previous_cfg['draws']>=cfg['draws']:
            cfg=previous_cfg
            fingerprint=hashlib.sha256((json.dumps(data,sort_keys=True)+json.dumps(cfg,sort_keys=True)+Path('stan/hierarchical_shared.stan').read_text()).encode()).hexdigest()
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        implementation_hash=hashlib.sha256(Path('stan/rl_fast.hpp').read_bytes()+Path('stan/hierarchical_fast.stan').read_bytes()).hexdigest()
        if saved.get('implementation')=='exact_analytic_gradient' and saved.get('implementation_sha256')!=implementation_hash:
            raise RuntimeError(f'Analytic likelihood changed since cached fit: {name}')
        if saved['fingerprint']!=fingerprint:raise RuntimeError(f'Stale chain cache: {name}; archive/remove before refitting')
        fit=load_chains(saved['csv_files'])
    else:
        from .parallel_checkpoint import execution_chains
        parallel=execution_chains(cfg)
        sm=stan_model();start=time.time()
        fit=sm.sample(data=data,chains=cfg['chains'],parallel_chains=parallel,
                      iter_warmup=cfg['warmup'],iter_sampling=cfg['draws'],seed=stable_seed(cfg['seed'],name),
                      adapt_delta=cfg['adapt_delta'],max_treedepth=cfg['max_treedepth'],metric=cfg.get('metric','diag_e'),
                      output_dir=str(folder.resolve()),show_progress=False,show_console=False,refresh=200,
                      sig_figs=10,inits=.15)
        saved=dict(fingerprint=fingerprint,csv_files=fit.runset.csv_files,seconds=time.time()-start,settings=cfg,execution={'parallel_chains':parallel},meta=meta,implementation='exact_analytic_gradient',implementation_sha256=hashlib.sha256(Path('stan/rl_fast.hpp').read_bytes()+Path('stan/hierarchical_fast.stan').read_bytes()).hexdigest())
        manifest.write_text(json.dumps(saved,indent=2)+'\n')
    diag=diagnostics(fit,name,meta,cfg)
    from .sampling_retry import retry_settings,archive_fit
    retry=retry_settings(cfg,passed=bool(diag.passed.all()),divergences=diag.divergences.max(),
                         max_depth_hits=diag.max_depth_hits.max(),min_bfmi=diag.min_bfmi.min())
    if retry is not None:
        reason,settings=retry
        suffix=f"_attempt{round(cfg['adapt_delta']*100)}_{reason}"
        archive=archive_fit(folder,saved,suffix)
        diag['run']=archive.name;diag.to_csv(TABLE/f'diagnostics_{archive.name}.csv',index=False)
        print(f'{name}: retry {reason}, settings={settings}',flush=True)
        return sample(model,training,age_terms,bounded,independent,override,name,prior_scale,settings)
    return fit,meta,arrays,ratings,frames


def diagnostics(fit,name,meta,cfg):
    summary=fit.summary()
    # Include every population parameter, age slope, covariance and participant natural parameter.
    keep=summary.index.str.startswith(('mu[','tau[','beta[','Omega[','natural['))
    summary=summary[keep].copy();summary.index.name='parameter';summary=summary.reset_index()
    # Constant correlation diagonals have undefined diagnostics and are not sampled quantities.
    summary=summary[~summary.parameter.str.match(r'Omega\[(\d+),\1\]')]
    method=fit.method_variables();div=int(method['divergent__'].sum());depth=int((method['treedepth__']>=cfg['max_treedepth']).sum())
    e=method['energy__'];bfmi=np.mean(np.diff(e,axis=0)**2,axis=0)/np.var(e,axis=0)
    summary['run']=name;summary['n_subjects']=len(meta['ids']);summary['divergences']=div
    summary['metric']=cfg.get('metric','diag_e')
    summary['chains']=cfg['chains'];summary['warmup_per_chain']=cfg['warmup'];summary['draws_per_chain']=cfg['draws'];summary['adapt_delta']=cfg['adapt_delta']
    summary['max_depth_hits']=depth;summary['min_bfmi']=bfmi.min()
    summary['passed']=(summary.R_hat<1.01)&(summary.ESS_bulk>=400)&(summary.ESS_tail>=400)&(div==0)&(bfmi.min()>.3)&(depth==0)
    summary.to_csv(TABLE/f'diagnostics_{name}.csv',index=False)
    (WORK/name/'diagnose.txt').write_text(fit.diagnose())
    print(f'{name}: max Rhat={summary.R_hat.max():.4f}, min bulk ESS={summary.ESS_bulk.min():.0f}, min tail ESS={summary.ESS_tail.min():.0f}, divergences={div}, max-depth hits={depth}, BFMI={bfmi.min():.3f}',flush=True)
    return summary


def summarize(x):
    q=np.quantile(x,[.025,.5,.975]);return dict(mean=float(np.mean(x)),median=float(q[1]),ci_low=float(q[0]),ci_high=float(q[2]))


def transform(eta,kinds,bounded=False):
    v=np.array(eta,copy=True)
    for j,k in enumerate(kinds):
        v[...,j]=expit(eta[...,j]) if k==1 else np.exp(eta[...,j]) if k==2 else (10*expit(eta[...,j]) if bounded else np.logaddexp(0,eta[...,j])) if k==3 else eta[...,j]
    return v


def friend_effect(par,gaps,weights):
    """Canonical P=.5; turn friend value on/off, standardize over empirical offers."""
    base=-.25*par[...,1,None]*gaps
    bonus=.5*par[...,1,None]*par[...,2,None]*gaps
    return ((expit(base+bonus)-expit(base))*weights).sum(axis=-1)


def offer_distribution(frames):
    # Match equal-participant standardization in the behavioral analysis.
    values=[]
    for f in frames.values():
        v=f[f.valid_choice].amount_gap.value_counts(normalize=True);values.append(v)
    w=pd.concat(values,axis=1).fillna(0).mean(axis=1).sort_index()
    return w.index.to_numpy(float),w.to_numpy(float)


def summaries(fit,meta,frames,name):
    names=['preference_friend' if meta['model']=='HPreference' and p=='theta' else p for p in SPECS[meta['model']][2]];kinds=SPECS[meta['model']][3]
    natural=fit.stan_variable('natural');rows=[]
    gaps,weights=offer_distribution(frames)
    for i,sub in enumerate(meta['ids']):
        for j,p in enumerate(names):rows.append(dict(run=name,model=meta['model'],participant_id=sub,parameter=p,age=meta['ages'][i],**summarize(natural[:,i,j])))
        if meta['model']=='H5':
            effect=friend_effect(natural[:,i],gaps,weights)
            rows.append(dict(run=name,model=meta['model'],participant_id=sub,parameter='friend_value_probability_effect',age=meta['ages'][i],**summarize(effect)))
            rows.append(dict(run=name,model=meta['model'],participant_id=sub,parameter='canonical_friend_logit',age=meta['ages'][i],**summarize(3*natural[:,i,1]*natural[:,i,2])))
    pd.DataFrame(rows).to_csv(TABLE/f'parameters_{name}.csv',index=False)
    mu=fit.stan_variable('mu');beta=fit.stan_variable('beta');age_rows=[];curves=[]
    if meta['age_terms']:
        for j,p in enumerate(names):
            for a in range(meta['age_terms']):
                x=beta[:,j,a];age_rows.append(dict(run=name,model=meta['model'],parameter=p,quantity='latent_slope',term='age_z' if a==0 else 'age_z_squared',probability_positive=float((x>0).mean()),**summarize(x)))
        az=(np.array(meta['ages'])-meta['age_mean'])/meta['age_sd']
        design=np.column_stack([az,az**2])[:,:meta['age_terms']]
        explained=np.var(np.einsum('dka,na->dnk',beta,design),axis=1)
        tau=fit.stan_variable('tau');fraction=explained/(explained+tau**2)
        for j,p in enumerate(names):
            age_rows.append(dict(run=name,model=meta['model'],parameter=p,quantity='latent_age_variance_fraction',term='age',probability_positive=np.nan,**summarize(fraction[:,j])))
        at={}
        for age in np.unique(np.r_[np.linspace(min(meta['ages']),max(meta['ages']),61),[25,40,60,75]]):
            az=(age-meta['age_mean'])/meta['age_sd'];xx=np.array([az,az*az])[:meta['age_terms']]
            par=transform(mu+np.einsum('dka,a->dk',beta,xx),kinds,meta['bounded'])
            at[float(age)]=par
            for j,p in enumerate(names):curves.append(dict(run=name,model=meta['model'],age=age,parameter=p,**summarize(par[:,j])))
            if meta['model']=='H5':curves.append(dict(run=name,model=meta['model'],age=age,parameter='friend_value_probability_effect',**summarize(friend_effect(par,gaps,weights))))
            if meta['model']=='H7':curves.append(dict(run=name,model=meta['model'],age=age,parameter='theta_friend_minus_stranger',**summarize(par[:,2]-par[:,3])))
        for j,p in enumerate(names):
            x=at[75.][:,j]-at[25.][:,j];age_rows.append(dict(run=name,model=meta['model'],parameter=p,quantity='natural_change_25_to_75',term='age',probability_positive=float((x>0).mean()),**summarize(x)))
        if meta['model']=='H5':
            x=friend_effect(at[75.],gaps,weights)-friend_effect(at[25.],gaps,weights)
            age_rows.append(dict(run=name,model=meta['model'],parameter='friend_value_probability_effect',quantity='natural_change_25_to_75',term='age',probability_positive=float((x>0).mean()),**summarize(x)))
        if meta['model']=='H7':
            x=(at[75.][:,2]-at[75.][:,3])-(at[25.][:,2]-at[25.][:,3]);age_rows.append(dict(run=name,model=meta['model'],parameter='theta_friend_minus_stranger',quantity='natural_change_25_to_75',term='age',probability_positive=float((x>0).mean()),**summarize(x)))
    if age_rows:pd.DataFrame(age_rows).to_csv(TABLE/f'age_{name}.csv',index=False)
    if curves:pd.DataFrame(curves).to_csv(TABLE/f'curves_{name}.csv',index=False)


def heldout(fit,meta,arrays,ratings,name,ndraws=None):
    natural=fit.stan_variable('natural');count=len(natural) if ndraws is None else min(ndraws,len(natural));idx=np.linspace(0,len(natural)-1,count).astype(int)
    base,_,names,_=SPECS[meta['model']];rows=[];trialrows=[]
    for i,(sub,a) in enumerate(arrays.items()):
        split=split_index(a);valid=a[split:,3]>=0;y=a[split:,3][valid]
        nll=[];pred=[]
        for draw in idx:
            tr=trajectory(a,base,dict(zip(names,natural[draw,i])),ratings.get(sub,np.zeros(3)))
            pred.append(tr[split:,1][valid]);nll.append(tr[split:,3][valid])
        # Average conditional probabilities over the fixed TRAINING posterior.
        # No reweighting/resampling the parameter posterior with held-out choices.
        probability=np.mean(pred,axis=0);loss=-logsumexp(-np.array(nll),axis=0)+np.log(len(idx))
        rows.append(dict(run=name,model=meta['model'],participant_id=sub,n=len(y),posterior_draws=len(idx),log_loss=loss.mean(),brier=np.mean((probability-y)**2),accuracy=np.mean((probability>=.5)==y)))
        for j,(p,yy,l) in enumerate(zip(probability,y,loss)):trialrows.append(dict(run=name,participant_id=sub,heldout_index=j,probability=p,choice=yy,log_loss=l))
    pd.DataFrame(rows).to_csv(TABLE/f'heldout_{name}.csv',index=False)
    pd.DataFrame(trialrows).to_csv(TABLE/f'heldout_predictions_{name}.csv',index=False)


def recent(a):
    last=np.full(3,-1.);out=np.zeros(len(a))
    for j in range(len(a)):
        c=int(a[j,0]);out[j]=last[c]
        if a[j,4]>0:last[c]=a[j,5]
    return out


def predictive(fit,meta,arrays,ratings,frames,name,draws=300):
    natural=fit.stan_variable('natural');idx=np.linspace(0,len(natural)-1,min(draws,len(natural))).astype(int)
    base,_,names,_=SPECS[meta['model']];groups={};age_values=[]
    for i,(sub,a) in enumerate(arrays.items()):
        valid=a[:,3]>=0;f=frames[sub];cats=[];obs_recent=recent(a)
        for c,p in enumerate(['friend','stranger','computer']):
            cats.append(('partner',p,(a[:,0]==c)&valid))
            for pair in sorted(f.offer_pair.unique()):cats.append(('offer',p+':'+pair,(a[:,0]==c)&f.offer_pair.eq(pair).to_numpy()&valid))
            for b in sorted(f.trial_bin.unique()):cats.append(('time',f'{p}:{b}',(a[:,0]==c)&f.trial_bin.eq(b).to_numpy()&valid))
            for y in [0,1]:cats.append(('recent',f'{p}:{y}',None))
        values=np.full((len(cats),len(idx),2),np.nan);observed=[]
        for k,(cat,label,mask) in enumerate(cats):
            if cat=='recent':p,y=label.split(':');mask=(a[:,0]==['friend','stranger','computer'].index(p))&(obs_recent==int(y))&valid
            amounts=np.where(a[:,3]==1,a[:,2],a[:,1]);observed.append([a[mask,3].mean(),amounts[mask].mean()] if mask.any() else [np.nan,np.nan])
        for d,draw in enumerate(idx):
            sim,_=simulate(a,base,dict(zip(names,natural[draw,i])),ratings.get(sub,np.zeros(3)),np.random.default_rng(stable_seed(20260924,name,sub,d,'ppc')))
            previous=recent(sim);amounts=np.where(sim[:,3]==1,sim[:,2],sim[:,1])
            for k,(cat,label,mask) in enumerate(cats):
                if cat=='recent':p,y=label.split(':');mask=(a[:,0]==['friend','stranger','computer'].index(p))&(previous==int(y))&valid
                if mask.any():values[k,d]=[sim[mask,3].mean(),amounts[mask].mean()]
        for k,(cat,label,_) in enumerate(cats):groups.setdefault((cat,label),[]).append((observed[k],values[k]))
        for c,p in enumerate(['friend','stranger','computer']):
            k=next(k for k,x in enumerate(cats) if x[:2]==('partner',p))
            age_values.append((meta['ages'][i],p,observed[k][0],values[k,:,0]))
    rows=[]
    for (cat,label),batch in groups.items():
        obs=np.array([x[0] for x in batch]);vals=np.array([x[1] for x in batch])
        for j,measure in enumerate(['high_probability','investment']):
            estimates=np.nanmean(vals[:,:,j],axis=0)
            rows.append(dict(run=name,model=meta['model'],category=cat,label=label,measure=measure,n_subjects=int(np.isfinite(obs[:,j]).sum()),observed=np.nanmean(obs[:,j]),**summarize(estimates)))
    # Continuous age slopes of participant means are a PPC discrepancy statistic, not the primary GEE.
    for p in ['friend','stranger','computer']:
        batch=[r for r in age_values if r[1]==p];x=np.array([r[0] for r in batch]);x=(x-x.mean())/meta['age_sd'];w=x/(x@x)
        observed=w@np.array([r[2] for r in batch]);slopes=w@np.array([r[3] for r in batch])
        rows.append(dict(run=name,model=meta['model'],category='age_slope',label=p,measure='high_probability_per_sd_age',n_subjects=len(batch),observed=observed,**summarize(slopes)))
    pd.DataFrame(rows).to_csv(TABLE/f'predictive_{name}.csv',index=False)


def prior_predictive(model='H5',draws=500,age_terms=1,bounded=False,prior_scale=1):
    from scipy.stats import random_correlation
    data,meta,arrays,ratings,frames=inputs(model,age_terms=age_terms,bounded=bounded,prior_scale=prior_scale);rng=np.random.default_rng(stable_seed(20260924,'prior',model))
    names=SPECS[model][2];kinds=SPECS[model][3];rows=[]
    # Draw correlation matrices from the exact LKJ(2) prior using Stan's fixed-param RNG model.
    from cmdstanpy import CmdStanModel
    stan_model();file=WORK/'prior_rng.stan'
    source='''data { int K; int N; int A; matrix[N,A] age; vector[K] loc; vector[K] scale; real prior_scale; }
    generated quantities { vector[K] mu; vector[K] tau; matrix[K,A] beta; matrix[K,N] z; matrix[K,N] u; matrix[N,K] eta;
      matrix[K,K] L=lkj_corr_cholesky_rng(K,2);
      for(j in 1:K) { mu[j]=normal_rng(loc[j],scale[j]*prior_scale); tau[j]=abs(normal_rng(0,.8*prior_scale));
      for(a in 1:A) beta[j,a]=normal_rng(0,.5*prior_scale); for(i in 1:N) z[j,i]=normal_rng(0,1); }
      u=diag_pre_multiply(tau,L)*z; eta=rep_matrix(mu\',N)+age*beta\'+u\'; }'''
    if not file.exists() or file.read_text()!=source:file.write_text(source)
    sm=CmdStanModel(stan_file=str(file.resolve()))
    fit=sm.sample(data=dict(K=data['K'],N=data['N'],A=data['A'],age=data['age'],loc=data['mu_location'],scale=data['mu_scale'],prior_scale=prior_scale),
                  fixed_param=True,chains=1,iter_sampling=draws,seed=stable_seed(20260924,'prior',model),
                  output_dir=str((WORK/f'prior_{model}').resolve()),show_progress=False)
    natural=transform(fit.stan_variable('eta'),kinds,bounded)
    for d in range(draws):
        stats=[]
        for i,(sub,a) in enumerate(arrays.items()):
            sim,tr=simulate(a,SPECS[model][0],dict(zip(names,natural[d,i])),ratings.get(sub,np.zeros(3)),rng)
            valid=a[:,3]>=0;p=tr[valid,1];stats.append([sim[valid,3].mean(),((p<.01)|(p>.99)).mean(),float(((p<.01)|(p>.99)).mean()>.95)])
        v=np.mean(stats,axis=0);rows.append(dict(model=model,draw=d,high_probability=v[0],extreme_trial_fraction=v[1],nearly_deterministic_participant_fraction=v[2]))
    tag=model+('_quadratic' if age_terms==2 else '')+('_bounded' if bounded else '')+(f'_prior{prior_scale:g}' if prior_scale!=1 else '')
    pd.DataFrame(rows).to_csv(TABLE/f'prior_predictive_{tag}.csv',index=False)
    return pd.DataFrame(rows)


def collect():
    for pattern,dest in [('diagnostics_H*.csv','hierarchical_diagnostics.csv'),('parameters_H*.csv','hierarchical_parameter_summary.csv'),
                         ('age_H*.csv','hierarchical_age_effects.csv'),('curves_H*.csv','hierarchical_age_curves.csv'),
                         ('predictive_H*.csv','hierarchical_predictive_summary.csv')]:
        paths=sorted(TABLE.glob(pattern))
        if paths:pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(TABLE/dest,index=False)
    paths=sorted(TABLE.glob('heldout_H*.csv'))
    if paths:
        h=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);h.to_csv(TABLE/'hierarchical_heldout_participant.csv',index=False)
        h.groupby(['run','model']).agg(n_subjects=('participant_id','size'),n_trials=('n','sum'),log_loss=('log_loss','mean'),median_log_loss=('log_loss','median'),worst_log_loss=('log_loss','max'),brier=('brier','mean'),accuracy=('accuracy','mean')).to_csv(TABLE/'hierarchical_heldout_summary.csv')
        preds=pd.concat([pd.read_csv(p) for p in sorted(TABLE.glob('heldout_predictions_H*.csv'))]);preds['bin']=pd.cut(preds.probability,np.linspace(0,1,11),include_lowest=True)
        preds.groupby(['run','bin'],observed=True).agg(n=('choice','size'),mean_prediction=('probability','mean'),observed=('choice','mean')).to_csv(TABLE/'hierarchical_calibration.csv')
        comparisons=[]
        for model in SPECS:
            a=h[h.run.eq(model+'_train_age')].set_index('participant_id');b=h[h.run.eq(model+'_train_noage')].set_index('participant_id')
            if not len(a) or not len(b):continue
            diff=a.log_loss-b.log_loss;rng=np.random.default_rng(stable_seed(20260924,model,'comparison'));v=diff.to_numpy();boot=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
            comparisons.append(dict(model=model,contrast='age minus no age',n=len(v),mean_difference=v.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),fraction_improved=np.mean(v<0)))
        pd.DataFrame(comparisons).to_csv(TABLE/'hierarchical_age_prediction.csv',index=False)


def run_one(model,training=False,age_terms=1,bounded=False,prior_scale=1):
    name=run_name(model,training,age_terms,bounded,prior_scale=prior_scale)
    fit,meta,arrays,ratings,frames=sample(model,training,age_terms,bounded,prior_scale=prior_scale)
    if training:heldout(fit,meta,arrays,ratings,name)
    else:summaries(fit,meta,frames,name);predictive(fit,meta,arrays,ratings,frames,name)
    collect()


def sample(model,training=False,age_terms=1,bounded=False,independent=False,override=None,label=None,prior_scale=1,settings=None):
    import os,time
    name=label or run_name(model,training,age_terms,bounded,independent,prior_scale)
    WORK.mkdir(parents=True,exist_ok=True);lock=WORK/(name+'.lock');owned=False
    for attempt in range(480):
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
            owned=True;break
        except FileExistsError:
            try:
                owner=int(lock.read_text());os.kill(owner,0)
            except ProcessLookupError:
                lock.unlink(missing_ok=True);continue
            except (ValueError,FileNotFoundError):
                time.sleep(1);continue
            if owner==os.getpid():break
            time.sleep(15)
    else:raise RuntimeError(f'Another process still owns fit {name}')
    try:return _sample(model,training,age_terms,bounded,independent,override,label,prior_scale,settings)
    finally:
        if owned:lock.unlink(missing_ok=True)
