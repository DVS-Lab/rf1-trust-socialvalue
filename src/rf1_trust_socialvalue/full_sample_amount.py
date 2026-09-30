"""Exploratory, paired run-1 fits of general bias versus amount-sensitive bias."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import logsumexp
from . import full_sample_sampling as s
from . import full_sample_batch as batch
from . import full_sample_ppc as ppc
from .full_sample_residuals import groups as detailed_groups

SOURCES = ['stan/full_amount_fast.stan','stan/full_amount_reference.stan','stan/full_amount_fast.hpp']
BASELINES = {'H7':'Train_H7_zero_noage','HPreference':'Train_HPreference_zero_noage'}


def configuration():
    _, phase, c = ppc.configuration()
    root = Path(c['_root']); path = root/'config/full_sample_amount.json'; cfg = json.loads(path.read_text())
    if (cfg['models'] != ['H7','HPreference'] or cfg['extensions'] != ['bias','amount'] or cfg['subset'] != 'train'
            or cfg['automatic_retries'] or cfg['chains'] != 8 or cfg['parallel_fits'] != 4 or cfg['cpu_budget'] != 32
            or cfg['amount_feature'] != '(high^2-low^2)/64'):
        raise ValueError('Unreviewed amount-comparison scope')
    return cfg, phase, c


def entries(cfg):
    return [dict(model=m, extension=e, subset='train', name=f'Train_{m}_zero_{e}_v1') for m in cfg['models'] for e in cfg['extensions']]


def paths(c):
    out, work = s.paths(c)
    return out/'amount_comparison', work/'amount_comparison'


def sources(c):
    root=Path(c['_root']); files=SOURCES+['src/rf1_trust_socialvalue/full_sample_amount.py','src/rf1_trust_socialvalue/full_sample_residuals.py','config/full_sample_amount.json']
    return {**ppc.implementation(root), **{f:s.fs.sha(root/f) for f in files}}


def validate_live_audit(audit, c):
    """New phase's narrow reviewed exception; original fitting gate is unchanged."""
    if audit['read_errors'] or audit['status']=='audit_incomplete': raise ValueError('Live input audit incomplete')
    allowed={str(Path(c['bids_root'])/f'sub-10668/ses-01/func/sub-10668_ses-01_task-trust_run-2_part-{part}_events.tsv') for part in ['mag','phase']}
    records={row['filename']:row for receipt in audit['repair_10668']
             if Path(receipt['path']).name=='receipt.json' and receipt.get('repair_id')=='10668-sharedreward-v1' and receipt.get('status')=='complete'
             for row in receipt.get('implicated_originals',[])}
    for row in audit['changes']:
        original=records.get(Path(row['path']).name,{})
        if (row['path'] not in allowed or row['change']!='missing'
                or row['frozen_sha256']!='f49141d568d2bfebb7456958a33a9f3d67f923ec542a560f3c7cb6b2de599423'
                or not original.get('archive_matches_freeze') or original.get('archive_sha256')!=row['frozen_sha256']
                or original.get('receipt_original_sha256')!=row['frozen_sha256']):
            raise ValueError('Unreviewed live input change; amount fits blocked: '+row['path'])


def snapshot(cfg, phase, c):
    # saved_trials authenticates the old artifact, then the NEW phase explicitly
    # checks all live inputs with only the documented archived-template exception.
    t, prov = ppc.geometry.saved_trials(phase, c)
    audit = ppc.geometry.source_drift_audit(c, prov); validate_live_audit(audit, c)
    train = batch.entry_trials(t, dict(subset='train'))
    test = t[t.participant_id.isin(batch.paired_ids(t)) & t.run.eq(2)]
    if (train.participant_id.nunique()!=cfg['expected_train_n'] or int(train.valid_choice.sum())!=cfg['expected_train_choices']
            or int(test.valid_choice.sum())!=cfg['expected_test_choices']):
        raise ValueError('Paired cohort/count changed')
    # Verify old training models really target this exact training cohort.
    reused=[dict(model=m,variant='zero',subset='train',name=n) for m,n in BASELINES.items()]
    batch.review_reused(dict(reuse=reused,gamma_population_mean_prior_sd=phase['gamma_population_mean_prior_sd'],cmdstan_version=phase['cmdstan_version']),c,t)
    return t, audit


@njit(cache=True)
def engine(a, par, code):
    belief=np.full(3,.5); result=np.zeros((len(a),2))
    for t in range(len(a)):
        c=int(a[t,0]); low=a[t,1]; high=a[t,2]; p=belief[c]
        bonus=par[c+2] if code==7 and c<2 else 0.
        pref=par[c+2] if code==9 and c<2 else 0.
        z=par[1]*(high-low)*(-1+1.5*p+p*bonus+pref)+par[4]*(low==0)+par[5]
        if len(par)==7: z+=par[6]*(high*high-low*low)/64.
        prob=1/(1+np.exp(-z)) if z>=0 else np.exp(z)/(1+np.exp(z))
        result[t,0]=prob; result[t,1]=np.logaddexp(0.,z)-a[t,3]*z if a[t,3]>=0 else 0.
        if a[t,4]>0: belief[c]+=par[0]*(a[t,5]-p)
    return result


def model_data(t, entry, phase):
    data, meta, arrays=s.model_data(t,entry['model'],True,phase['gamma_population_mean_prior_sd'])
    blocks=np.concatenate([a[a[:,3]>=0] for a in arrays.values()])
    data.update(E=1 if entry['extension']=='bias' else 2,amount_feature=((blocks[:,2]**2-blocks[:,1]**2)/64).tolist())
    meta=dict(meta,extension=entry['extension'],parameter_names=meta['parameter_names']+['choice_bias']+(['amount_curvature'] if data['E']==2 else []))
    return data,meta,arrays


def models(phase,c,reference=False):
    from cmdstanpy import CmdStanModel
    s.cmdstan(phase,c); _,work=paths(c); build=work/'build'; build.mkdir(parents=True,exist_ok=True)
    root=Path(c['_root']); h=s.digest({f:s.fs.sha(root/f) for f in SOURCES}); result=[]
    for variant in (['fast','reference'] if reference else ['fast']):
        text=(root/f'stan/full_amount_{variant}.stan').read_text()+'\n// implementation '+h+'\n'
        target=build/f'amount_{variant}.stan'
        if not target.exists() or target.read_text()!=text: target.write_text(text)
        options=dict(user_header=str(root/'stan/full_amount_fast.hpp'),stanc_options={'allow-undefined':True}) if variant=='fast' else {}
        result.append(CmdStanModel(stan_file=str(target),**options))
    return result


def check_implementation(cfg,phase,c,t):
    fast,reference=models(phase,c,True); original=s.models(phase,c,True)[1]
    out,_=paths(c); out.mkdir(parents=True,exist_ok=True); rows=[]; prior_rows=[]; rng=np.random.default_rng(cfg['seed'])
    small=batch.entry_trials(t,dict(subset='train')); small=small[small.participant_id.isin(sorted(small.participant_id.unique())[:4])]
    for entry in entries(cfg):
        data,meta,arrays=model_data(small,entry,phase); k=data['K']; e=data['E']; n=data['N']; mu=np.array(data['mu_location'])
        for effect in [-1.,0.,1.]:
            pars=dict(mu=mu.tolist(),tau=[.6]*k,L=np.linalg.cholesky(.9*np.eye(k)+.1*np.ones((k,k))).tolist(),
                beta=[[] for _ in range(k)],z=rng.normal(0,.4,(k,n)).tolist(),mu_ext=[effect]*e,tau_ext=[.6]*e,z_ext=np.zeros((e,n)).tolist())
            if effect: pars['z_ext']=rng.normal(0,.4,(e,n)).tolist()
            ref=reference.log_prob(params=pars,data=data,jacobian=True,sig_figs=16).iloc[0]
            actual=fast.log_prob(params=pars,data=data,jacobian=True,sig_figs=16).iloc[0]
            if list(ref.index)!=list(actual.index):raise ValueError('Gradient coordinate mismatch')
            error=float(np.max(np.abs(ref.to_numpy()-actual.to_numpy())/(1+np.abs(ref.to_numpy()))))
            prior=float(reference.log_prob(params=pars,data=dict(data,prior_only=1),jacobian=True,sig_figs=16).iloc[0,0])
            base=s.transform(mu+(np.diag(pars['tau'])@np.array(pars['L'])@np.array(pars['z'])).T,data['kind'])
            ext=np.array(pars['mu_ext'])+np.array(pars['tau_ext'])*np.array(pars['z_ext']).T
            natural=np.concatenate([base,ext],axis=1)
            pyll=sum(-engine(a,natural[i],s.SPECS[entry['model']][1])[:,1].sum() for i,a in enumerate(arrays.values()))
            llerr=abs(float(ref.iloc[0])-prior-pyll); nesting=0.
            if effect==0:
                old_data,_,_=s.model_data(small,entry['model'],True,phase['gamma_population_mean_prior_sd'])
                old_pars={key:pars[key] for key in ['mu','tau','L','beta','z']}
                lp=float(original.log_prob(params=old_pars,data=old_data,jacobian=True,sig_figs=16).iloc[0,0])
                lp0=float(original.log_prob(params=old_pars,data=dict(old_data,prior_only=1),jacobian=True,sig_figs=16).iloc[0,0])
                nesting=abs(lp-lp0-pyll)
            rows.append(dict(model=entry['model'],extension=entry['extension'],effect=effect,max_scaled_gradient_error=error,
                python_likelihood_error=llerr,nesting_error=nesting,passed=bool(error<1e-9 and llerr<1e-8 and nesting<1e-8)))
        # Prior predictive probabilities on authentic run-1 schedules, no choice
        # fitting. Population draws are shared across participants in each replicate.
        for rep in range(200):
            # This lightweight screen concerns extension priors only.
            # Baseline is held at its prior median so this is explicitly a conditional screen.
            base0=s.transform(np.tile(data['mu_location'],(n,1)),data['kind'])
            ext_mu=rng.normal(0,1,e); ext_tau=abs(rng.normal(0,.8,e)); ext0=ext_mu+ext_tau*rng.normal(size=(n,e))
            probs=np.concatenate([engine(a,np.r_[base0[i],ext0[i]],s.SPECS[entry['model']][1])[:,0] for i,a in enumerate(arrays.values())])
            prior_rows.append(dict(model=entry['model'],extension=entry['extension'],replicate=rep,mean_probability=probs.mean(),fraction_below_01=np.mean(probs<.01),fraction_above_99=np.mean(probs>.99)))
    pd.DataFrame(rows).to_csv(out/'implementation_checks.tsv',sep='\t',index=False)
    screen=pd.DataFrame(prior_rows)
    if not np.isfinite(screen[['mean_probability','fraction_below_01','fraction_above_99']]).all().all():raise ValueError('Nonfinite extension prior screen')
    screen.to_csv(out/'extension_prior_screen.tsv',sep='\t',index=False)
    state=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',source_hashes=sources(c),
        check_sha256=s.fs.sha(out/'implementation_checks.tsv'),executable_sha256=s.fs.sha(fast.exe_file),
        prior_screen_sha256=s.fs.sha(out/'extension_prior_screen.tsv'),generated_at=s.now())
    s.save(out/'implementation_status.json',state)
    if state['status']!='passed':raise RuntimeError('Amount likelihood/gradient checks failed; no sampling launched')
    print('Passed 12 fast/reference target-gradient, Python likelihood, and zero-extension nesting checks.',flush=True)


def implementation_gate(c):
    out,work=paths(c); state=json.loads((out/'implementation_status.json').read_text())
    if (state['status']!='passed' or state['source_hashes']!=sources(c)
            or state['check_sha256']!=s.fs.sha(out/'implementation_checks.tsv')
            or state['executable_sha256']!=s.fs.sha(work/'build/amount_fast')):
        raise ValueError('Missing/stale amount implementation gate')


def diagnostics(fit,cfg,c):
    d=fit.summary(); d=d[d.index.str.startswith(('mu[','tau[','mu_ext[','tau_ext[','Omega[','natural['))]
    d=d[~d.index.str.match(r'Omega\[(\d+),\1\]')].rename_axis('parameter').reset_index()
    methods=fit.method_variables(); energy=methods['energy__']; bfmi=np.mean(np.diff(energy,axis=0)**2,axis=0)/np.var(energy,axis=0)
    limits=c['hierarchical']['acceptance']; div=int(methods['divergent__'].sum()); depth=int((methods['treedepth__']>=cfg['max_treedepth']).sum())
    passed=bool(len(d) and np.isfinite(d[['R_hat','ESS_bulk','ESS_tail']]).all().all()
        and (d.R_hat<limits['rhat_less_than']).all() and (d.ESS_bulk>=limits['minimum_bulk_ess']).all()
        and (d.ESS_tail>=limits['minimum_tail_ess']).all() and div==0 and depth==0 and np.isfinite(bfmi).all() and bfmi.min()>limits['bfmi_greater_than'])
    value=lambda x:float(x) if np.isfinite(x) else None
    return d,dict(passed=passed,max_rhat=value(d.R_hat.max()),min_bulk_ess=value(d.ESS_bulk.min()),min_tail_ess=value(d.ESS_tail.min()),divergences=div,max_depth_hits=depth,min_bfmi=value(bfmi.min()))


def heldout(t,entry,natural,meta):
    ids=batch.paired_ids(t)
    if meta['ids']!=ids or natural.shape[1:]!=(len(ids),len(meta['parameter_names'])) or not np.isfinite(natural).all():
        raise ValueError('Heldout posterior identity/shape mismatch')
    _,_,arrays=s.model_data(t[t.participant_id.isin(ids)],entry['model'],True)
    rows=[]; cells=[]
    for i,(sub,a) in enumerate(arrays.items()):
        valid=(a[:,7]==2)&(a[:,3]>=0); y=a[valid,3]
        traj=np.array([engine(a,par,s.SPECS[entry['model']][1]) for par in natural[:,i]])
        mean=traj[:,valid,0].mean(axis=0); loss=-logsumexp(-traj[:,valid,1],axis=0)+np.log(len(natural))
        rows.append(dict(name=entry['name'],model=entry['model'],variant=entry['extension'],participant_id=sub,n=len(y),posterior_draws=len(natural),log_loss=loss.mean(),brier=np.mean((mean-y)**2),accuracy=np.mean((mean>=.5)==y)))
        for strat,group,mask in detailed_groups(a):
            if strat not in ['partner','partner_zero','partner_offer']:continue
            mask=mask & (a[:,7]==2)
            if mask.any():cells.append(dict(participant_id=sub,stratification=strat,group=group,n=int(mask.sum()),observed=a[mask,3].mean(),predicted=traj[:,mask,0].mean()))
    return pd.DataFrame(rows),pd.DataFrame(cells)


def fit(cfg,phase,c,entry):
    s.require_linux();out,work=paths(c);name=entry['name'];folder=work/'fits'/name;dest=out/'fits'/name
    folder.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    state=dict(name=name,entry=entry,status='preparing',started_at=s.now());s.save(dest/'status.json',state);start=time.monotonic();manifest=folder/'manifest.json'
    try:
        t,audit=snapshot(cfg,phase,c);s.save(dest/'live_source_audit_before.json',audit);implementation_gate(c)
        data,meta,_=model_data(batch.entry_trials(t,entry),entry,phase);fingerprint=s.digest(dict(data=data,config=cfg,sources=sources(c)))
        sampler_seed=s.stable_seed(cfg['seed'],name);state.update(status='running',fingerprint=fingerprint,sampler_seed=sampler_seed,settings=cfg);s.save(dest/'status.json',state)
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved['fingerprint']!=fingerprint:raise ValueError('Existing amount target differs; cache untouched')
            for f,h in saved['posterior_sha256'].items():
                if s.fs.sha(f)!=h:raise ValueError('Cached amount posterior changed')
            posterior=s.load_chains(saved['csv_files']);state['reused_posterior']=True
        else:
            if list(folder.glob('*.csv')):raise RuntimeError('Partial draws exist; inspect before restart')
            from cmdstanpy import CmdStanModel
            s.cmdstan(phase,c);model=CmdStanModel(exe_file=str(work/'build/amount_fast'))
            posterior=model.sample(data=data,chains=cfg['chains'],parallel_chains=cfg['chains'],iter_warmup=cfg['warmup'],iter_sampling=cfg['draws'],seed=sampler_seed,
                adapt_delta=cfg['adapt_delta'],max_treedepth=cfg['max_treedepth'],metric=cfg['metric'],inits=cfg['inits'],sig_figs=cfg['sig_figs'],
                output_dir=str(folder),refresh=200,show_progress=False,show_console=True)
            saved=dict(fingerprint=fingerprint,meta=meta,settings=cfg,sampler_seed=sampler_seed,csv_files=posterior.runset.csv_files,
                posterior_sha256={f:s.fs.sha(f) for f in posterior.runset.csv_files},source_hashes=sources(c),sampling_seconds=time.monotonic()-start)
            s.save(manifest,saved);state['reused_posterior']=False
        _,audit=snapshot(cfg,phase,c);s.save(dest/'live_source_audit_after.json',audit);implementation_gate(c)
        d,info=diagnostics(posterior,cfg,c);d.to_csv(dest/'diagnostics.tsv',sep='\t',index=False);state.update(info,sampling_seconds=saved['sampling_seconds'])
        (dest/'diagnose.txt').write_text(posterior.diagnose())
        if not info['passed']:state['status']='diagnostic_failed';return 2
        natural=posterior.stan_variable('natural');scores,cells=heldout(t,entry,natural,meta)
        scores.to_csv(dest/'heldout_participants.tsv',sep='\t',index=False);cells.to_csv(dest/'heldout_cells.tsv',sep='\t',index=False)
        rows=[]
        for i,sub in enumerate(meta['ids']):
            for j,param in enumerate(meta['parameter_names']):rows.append(dict(participant_id=sub,parameter=param,**s.summarize(natural[:,i,j])))
        pd.DataFrame(rows).to_csv(dest/'participant_parameters.tsv',sep='\t',index=False)
        population=[]
        for field in ['mu','tau','mu_ext','tau_ext']:
            values=posterior.stan_variable(field)
            for j in range(values.shape[1]):population.append(dict(parameter=field+f'[{j+1}]',**s.summarize(values[:,j])))
        pd.DataFrame(population).to_csv(dest/'population_parameters.tsv',sep='\t',index=False)
        implementation_gate(c);state['status']='complete';return 0
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        state.update(finished_at=s.now(),elapsed_seconds=time.monotonic()-start,output_sha256={f.name:s.fs.sha(f) for f in dest.glob('*.tsv')})
        if manifest.exists():
            saved=json.loads(manifest.read_text());s.save(dest/'manifest_summary.json',dict(saved,csv_files=[Path(f).name for f in saved['csv_files']],posterior_sha256={Path(f).name:h for f,h in saved['posterior_sha256'].items()}))
        for f in folder.glob('*.txt'):s.write_tail(f,dest/f.name)
        s.save(dest/'status.json',state);print(name+': '+state['status'],flush=True)


def compare(cfg,c):
    out,_=paths(c);base,_=s.paths(c);frames=[]
    for model,name in BASELINES.items():
        folder=base/'fits'/name;st=json.loads((folder/'status.json').read_text());f=folder/'heldout_participants.tsv'
        if st['status']!='complete' or s.fs.sha(f)!=st['heldout_sha256']:raise ValueError('Baseline changed')
        frames.append(pd.read_csv(f,sep='\t').assign(extension='original'))
    for entry in entries(cfg):
        folder=out/'fits'/entry['name'];state=json.loads((folder/'status.json').read_text())
        if state['status']!='complete' or not state['passed']:continue
        f=folder/'heldout_participants.tsv'
        if s.fs.sha(f)!=state['output_sha256'][f.name]:raise ValueError('Amount scores changed')
        frames.append(pd.read_csv(f,sep='\t').assign(extension=entry['extension']))
    a=pd.concat(frames,ignore_index=True);wide=a.pivot(index='participant_id',columns='name',values='log_loss').sort_index()
    counts=a.pivot(index='participant_id',columns='name',values='n').reindex(wide.index)
    if wide.isna().any().any() or not counts.eq(counts.iloc[:,0],axis=0).all().all():raise ValueError('Unmatched score cohort')
    rng=np.random.default_rng(cfg['seed']);indices=rng.integers(0,len(wide),(2000,len(wide)));rows=[]
    for frame in frames:
        name=frame.name.iloc[0];model=frame.model.iloc[0];delta=(wide[name]-wide[BASELINES[model]]).to_numpy();lo,hi=np.quantile(delta[indices].mean(axis=1),[.025,.975])
        rows.append(dict(name=name,model=model,extension=frame.extension.iloc[0],participants=len(wide),choices=int(frame.n.sum()),log_loss=wide[name].mean(),delta_vs_original=delta.mean(),ci_low=lo,ci_high=hi))
    summary=pd.DataFrame(rows);summary.to_csv(out/'available_model_comparison.tsv',sep='\t',index=False)
    contrasts=[]
    for model in cfg['models']:
        bias=f'Train_{model}_zero_bias_v1';amount=f'Train_{model}_zero_amount_v1'
        if {bias,amount}.issubset(wide):
            delta=(wide[amount]-wide[bias]).to_numpy();lo,hi=np.quantile(delta[indices].mean(axis=1),[.025,.975]);contrasts.append(dict(model=model,amount_minus_bias=delta.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(contrasts,columns=['model','amount_minus_bias','ci_low','ci_high']).to_csv(out/'amount_vs_bias.tsv',sep='\t',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=summary.sort_values(['model','extension']);y=np.arange(len(summary))
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    ax.hlines(y,summary.ci_low,summary.ci_high,color='#176b87',lw=2);ax.scatter(summary.delta_vs_original,y,color='#176b87')
    ax.axvline(0,color='.5',ls='--');ax.set(yticks=y,yticklabels=[m+' / '+e for m,e in zip(summary.model,summary.extension)],
        xlabel='Run-2 log loss minus matching original (lower is better)',
        title=f'Exploratory amount comparison: {len(summary)} accepted models of 6\n304 paired participants · 95% participant bootstrap intervals')
    ax.invert_yaxis()
    for ext in ['png','pdf']:fig.savefig(out/f'available_model_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)



def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fit');a=parser.parse_args();cfg,phase,c=configuration();s.require_linux()
    if a.fit:
        entry=next(e for e in entries(cfg) if e['name']==a.fit);sys.exit(fit(cfg,phase,c,entry))
    import fcntl
    out,work=paths(c);out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    with (work/'batch.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Amount batch already running')
        state=dict(status='preparing',started_at=s.now(),runs={});s.save(out/'status.json',state)
        try:
            t,audit=snapshot(cfg,phase,c);s.save(out/'live_source_audit.json',audit);check_implementation(cfg,phase,c,t)
            available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
            workers=min(4,min(32,available)//cfg['chains'])
            if workers<1:raise RuntimeError('Eight CPUs required')
            state.update(status='running',workers=workers,active_chains=workers*cfg['chains']);s.save(out/'status.json',state)
            print(f'Amount comparison: four new training fits; {workers} concurrent x 8 chains = {workers*8} active cores.',flush=True)
            def launch(entry):
                folder=out/'fits'/entry['name'];folder.mkdir(parents=True,exist_ok=True)
                with (folder/'console.txt').open('a',buffering=1) as log:
                    log.write('\nAMOUNT WORKER '+s.now()+'\n')
                    return subprocess.run([sys.executable,'-m','rf1_trust_socialvalue.full_sample_amount','--fit',entry['name']],cwd=c['_root'],stdout=log,stderr=subprocess.STDOUT).returncode
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending={pool.submit(launch,e):e for e in entries(cfg)}
                for future in as_completed(pending):
                    name=pending[future]['name'];state['runs'][name]=future.result();s.save(out/'status.json',state);print(name+': exit '+str(state['runs'][name]),flush=True)
            compare(cfg,c)
            state['comparison_sha256']={f.name:s.fs.sha(f) for f in out.iterdir() if f.name.startswith(('available_model_comparison.','amount_vs_bias.'))}
            if any(state['runs'].values()):raise RuntimeError('Some fits failed diagnostics or execution; available comparisons saved, inspect logs; no automatic retry')
            state.update(status='complete',finished_at=s.now());s.save(out/'status.json',state)
        except BaseException as exc:
            state.update(status='error',finished_at=s.now(),error=repr(exc));s.save(out/'status.json',state);raise


if __name__=='__main__':main()
