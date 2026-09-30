"""Posterior predictive checks of accepted full-data fits using the frozen snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from . import full_sample_sampling as s
from . import full_sample_geometry as geometry
from . import full_sample_review as review

PARTNERS=['friend','stranger','computer']


def configuration():
    phase,c,_=s.configuration('config/full_sample_batch_retry.json')
    root=Path(c['_root']);path=root/'config/full_sample_ppc.json';p=json.loads(path.read_text())
    if p['simulations']!=c['hierarchical']['ppc_simulations'] or p['simulations']<2 or not 1<=p['workers']<=4:
        raise ValueError('Invalid reviewed predictive-check configuration')
    diagnostics,_,_=review.accepted_scores(root)
    names=set(diagnostics.loc[diagnostics.accepted & diagnostics.subset.eq('full'),'name'])
    if names!=set(p['accepted_full_fits']) or len(p['accepted_full_fits'])!=len(names):
        raise ValueError('Accepted full-fit inventory changed; review the predictive-check scope')
    for file,h in p['evidence_sha256'].items():
        if s.fs.sha(root/file)!=h:raise ValueError('Reviewed fit evidence changed: '+file)
    return p,phase,c


def implementation(root):
    hashes=s.implementation(root)
    for file in ['full_sample_ppc.py','full_sample_geometry.py','full_sample_review.py']:
        relative='src/rf1_trust_socialvalue/'+file;hashes[relative]=s.fs.sha(Path(root)/relative)
    return hashes


def groups(a):
    valid=a[:,3]>=0
    yield 'all','all',valid
    for i,partner in enumerate(PARTNERS):
        yield 'partner',partner,valid & (a[:,0]==i)
        for zero,label in [(True,'zero'),(False,'positive')]:
            yield 'partner_zero',partner+'_'+label,valid & (a[:,0]==i) & ((a[:,1]==0)==zero)
    for low,high in sorted(set(map(tuple,a[:,1:3]))):
        yield 'offer',f'{low:g}/{high:g}',valid & (a[:,1]==low) & (a[:,2]==high)
    for run in sorted(set(a[:,7])):
        yield 'run',str(int(run)),valid & (a[:,7]==run)
        for label,indices in zip(['early','middle','late'],np.array_split(np.flatnonzero(a[:,7]==run),3)):
            mask=np.zeros(len(a),dtype=bool);mask[indices]=True
            yield 'trial_block',f'run{int(run)}_{label}',valid & mask


def trajectories(a,parameters,model,zero,uniforms):
    conditional=[];replicated=[]
    for par,u in zip(parameters,uniforms):
        base=par[:-1] if zero else par;gamma=par[-1] if zero else 0.
        conditional.append(s.zero_engine(a,base,s.SPECS[model][1],gamma)[:,1])
        # The simulation sees a scheduled outcome only if its OWN choice invests
        # positively. Actual zero choices never update the conditional history.
        replicated.append(s.zero_engine(a,base,s.SPECS[model][1],gamma,True,u)[:,2])
    return np.asarray(conditional),np.asarray(replicated)


def summarize_cell(key,observed,predicted,nchoices):
    history,stratification,group,metric=key
    rows=[]
    statistics={'mean':lambda x:np.mean(x,axis=0)}
    if history=='generative' and stratification in ['all','partner'] and len(observed)>1:
        statistics['between_participant_sd']=lambda x:np.std(x,axis=0,ddof=1)
        if metric=='high_choice':
            statistics['all_low_fraction']=lambda x:np.mean(x==0,axis=0)
            statistics['all_high_fraction']=lambda x:np.mean(x==1,axis=0)
    for statistic,fn in statistics.items():
        actual=float(fn(observed));rep=fn(predicted);lo,hi=np.quantile(rep,[.025,.975])
        rows.append(dict(history=history,stratification=stratification,group=group,metric=metric,statistic=statistic,
            participants=len(observed),choices=nchoices,observed=actual,predicted_mean=float(np.mean(rep)),
            ci_low=float(lo),ci_high=float(hi),residual_observed_minus_predicted=actual-float(np.mean(rep)),
            replicated_fraction_ge_observed=float(np.mean(rep>=actual)) if history=='generative' else np.nan,
            interval_type='replicated_choices' if history=='generative' else 'conditional_expected_behavior'))
    return rows


def ppc_tables(arrays,natural,entry,simulations,seed):
    if natural.ndim!=3 or natural.shape[1]!=len(arrays) or not np.isfinite(natural).all():
        raise ValueError('Invalid participant posterior array')
    if len(natural)<simulations:raise ValueError('Not enough posterior draws for requested simulations')
    selected=np.linspace(0,len(natural)-1,simulations).astype(int)
    cells={};participant_rows=[]
    for i,(sub,a) in enumerate(arrays.items()):
        rng=np.random.default_rng(s.stable_seed(seed,entry['model'],sub,'full_cohort_ppc'))
        conditional,replicated=trajectories(a,natural[selected,i],entry['model'],entry['variant']=='zero',rng.random((simulations,len(a))))
        observed_amount=np.where(a[:,3]==1,a[:,2],a[:,1])
        for strat,group,mask in groups(a):
            if not mask.any():continue
            for history,pred in [('conditional',conditional),('generative',replicated)]:
                for metric,observed,values in [
                    ('high_choice',a[mask,3],pred[:,mask]),
                    ('investment',observed_amount[mask],a[mask,1]+pred[:,mask]*(a[mask,2]-a[mask,1]))]:
                    key=(history,strat,group,metric);obs=float(observed.mean());expected=values.mean(axis=1)
                    cell=cells.setdefault(key,dict(observed=[],predicted=[],choices=0,ids=[],counts=[]))
                    cell['observed'].append(obs);cell['predicted'].append(expected);cell['choices']+=int(mask.sum());cell['ids'].append(sub);cell['counts'].append(int(mask.sum()))
                    if strat=='partner':
                        lo,hi=np.quantile(expected,[.025,.975])
                        participant_rows.append(dict(participant_id=sub,partner=group,history=history,metric=metric,
                            choices=int(mask.sum()),observed=obs,predicted_mean=float(expected.mean()),ci_low=float(lo),ci_high=float(hi)))
    rows=[]
    for key,cell in cells.items():
        rows+=summarize_cell(key,np.asarray(cell['observed']),np.asarray(cell['predicted']),cell['choices'])
    rows+=partner_contrasts(cells)
    result=pd.DataFrame(rows)
    if not np.isfinite(result[['observed','predicted_mean','ci_low','ci_high']]).all().all():raise ValueError('Nonfinite predictive summary')
    return result,pd.DataFrame(participant_rows),selected



def partner_contrasts(cells):
    rows=[]
    for history in ['conditional','generative']:
        for metric in ['high_choice','investment']:
            for left,right in [('friend','computer'),('friend','stranger'),('stranger','computer')]:
                a=cells.get((history,'partner',left,metric));b=cells.get((history,'partner',right,metric))
                if a is None or b is None:continue
                ia={sub:i for i,sub in enumerate(a['ids'])};ib={sub:i for i,sub in enumerate(b['ids'])}
                common=sorted(set(ia)&set(ib))
                if not common:continue
                ai=[ia[sub] for sub in common];bi=[ib[sub] for sub in common]
                observed=np.asarray(a['observed'])[ai]-np.asarray(b['observed'])[bi]
                predicted=np.asarray(a['predicted'])[ai]-np.asarray(b['predicted'])[bi]
                choices=sum(a['counts'][i] for i in ai)+sum(b['counts'][i] for i in bi)
                rows+=summarize_cell((history,'partner_contrast',left+'_minus_'+right,metric),observed,predicted,choices)
    return rows

def fit_entry(p,phase,c,name):
    s.require_linux();root=Path(c['_root']);out,work=s.paths(c);dest=out/'ppc/fits'/name;dest.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',name=name,started_at=s.now(),new_sampling=False);s.save(dest/'status.json',state)
    try:
        t,_=geometry.saved_trials(phase,c)
        entry=next(e for e in review.resolved_entries(root) if e['name']==name and e['subset']=='full')
        evidence=out/'fits'/name;cache=work/'fits'/name
        status=json.loads((evidence/'status.json').read_text());manifest=json.loads((cache/'manifest.json').read_text())
        portable=json.loads((evidence/'manifest_summary.json').read_text())
        if status['status']!='complete' or not status['passed']:raise ValueError('PPC requires an accepted fit')
        data,meta,arrays=s.model_data(t,entry['model'],entry['variant']=='zero',phase['gamma_population_mean_prior_sd'])
        payload=dict(data=data,settings=status['settings'],sampler_seed=status['sampler_seed'],
            stan={file:s.fs.sha(root/file) for file in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
        if (s.digest(payload)!=status['fingerprint'] or manifest['fingerprint']!=status['fingerprint']
                or portable['fingerprint']!=status['fingerprint'] or manifest['meta']!=meta):
            raise ValueError('PPC target differs from fitted posterior')
        if {Path(file).name:h for file,h in manifest['posterior_sha256'].items()}!=portable['posterior_sha256']:
            raise ValueError('Cached posterior hashes differ from published manifest')
        for file in manifest['csv_files']:
            if s.fs.sha(file)!=manifest['posterior_sha256'][file]:raise ValueError('Raw posterior changed: '+file)
        sources=implementation(root);config_hash=s.fs.sha(root/'config/full_sample_ppc.json')
        fit=s.load_chains(manifest['csv_files']);natural=fit.stan_variable('natural')
        if list(arrays)!=meta['ids']:raise ValueError('Posterior participant order mismatch')
        table,participants,selected=ppc_tables(arrays,natural,entry,p['simulations'],c['seed'])
        table.to_csv(dest/'predictive_checks.tsv',sep='\t',index=False)
        participants.to_csv(dest/'participant_partner_checks.tsv',sep='\t',index=False)
        geometry.saved_trials(phase,c)
        if sources!=implementation(root) or config_hash!=s.fs.sha(root/'config/full_sample_ppc.json'):
            raise ValueError('Predictive-check implementation/configuration changed during execution')
        state.update(status='complete',finished_at=s.now(),model=entry['model'],variant=entry['variant'],participants=data['N'],
            simulations=p['simulations'],selected_posterior_draw_indices_zero_based=selected.tolist(),source_hashes=sources,
            config_sha256=config_hash,posterior_fingerprint=status['fingerprint'],posterior_sha256=portable['posterior_sha256'],
            accepted_status_sha256=s.fs.sha(evidence/'status.json'),manifest_sha256=s.fs.sha(evidence/'manifest_summary.json'),
            output_sha256={f.name:s.fs.sha(f) for f in dest.glob('*.tsv')},
            interpretation='Full-data posterior predictive checks conditional on offers, scheduled outcomes and observed missingness; not out-of-sample prediction or model recovery.')
        s.save(dest/'status.json',state)
    except BaseException as exc:
        state.update(status='error',finished_at=s.now(),error=repr(exc));s.save(dest/'status.json',state);raise


def render(p,c):
    out,_=s.paths(c);dest=out/'ppc';frames=[]
    for name in p['accepted_full_fits']:
        folder=dest/'fits'/name;status=json.loads((folder/'status.json').read_text())
        if status['status']!='complete':raise ValueError('Incomplete predictive checks: '+name)
        for file,h in status['output_sha256'].items():
            if s.fs.sha(folder/file)!=h:raise ValueError('Predictive table changed: '+name)
        frame=pd.read_csv(folder/'predictive_checks.tsv',sep='\t');frame['name']=name;frame['model']=status['model'];frame['variant']=status['variant'];frames.append(frame)
    all_rows=pd.concat(frames,ignore_index=True);all_rows.to_csv(dest/'predictive_checks_all_models.tsv',sep='\t',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharey=True,layout='constrained')
    labels=['friend_zero','friend_positive','stranger_zero','stranger_positive','computer_zero','computer_positive']
    for ax,name in zip(axes.flat,p['accepted_full_fits']):
        f=all_rows[(all_rows.name==name)&all_rows.history.eq('generative')&all_rows.stratification.eq('partner_zero')&all_rows.metric.eq('high_choice')&all_rows.statistic.eq('mean')].set_index('group').reindex(labels)
        if f.observed.isna().any():raise ValueError('Missing partner/zero cell')
        x=np.arange(6);ax.vlines(x,f.ci_low,f.ci_high,color='#176b87',linewidth=2)
        ax.plot(x,f.predicted_mean,'o',color='#176b87',label='Replicated mean + 95% interval')
        ax.plot(x,f.observed,'x',color='#bd3d32',markersize=7,label='Observed')
        ax.set(xticks=x,xticklabels=['F / 0','F / +','S / 0','S / +','C / 0','C / +'],ylim=(-.03,1.03),
            title=f"{f.model.iloc[0]} {'+ zero option' if f.variant.iloc[0]=='zero' else 'base'}")
    for ax in axes[:,0]:ax.set_ylabel('High-choice rate')
    for ax in axes[-1]:ax.set_xlabel('Partner / zero or positive lower option')
    axes[0,0].legend(fontsize=8,loc='best')
    fig.suptitle('Generative posterior predictive checks · 343 participants\nF = friend; S = stranger; C = computer; equal-participant averages')
    fig.savefig(dest/'partner_zero_predictive_checks.png',dpi=180,bbox_inches='tight');fig.savefig(dest/'partner_zero_predictive_checks.pdf',bbox_inches='tight');plt.close(fig)
    return {file:s.fs.sha(dest/file) for file in ['predictive_checks_all_models.tsv','partner_zero_predictive_checks.png','partner_zero_predictive_checks.pdf']}


def run(p,phase,c):
    import fcntl
    s.require_linux();out,work=s.paths(c);dest=out/'ppc';dest.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    with (work/'ppc.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Predictive-check batch already running')
        state=dict(status='preparing',started_at=s.now(),new_sampling=False,runs={});s.save(dest/'status.json',state)
        try:
            _,prov=geometry.saved_trials(phase,c);s.save(dest/'live_source_audit.json',geometry.source_drift_audit(c,prov))
            workers=min(p['workers'],os.cpu_count() or 1);state.update(status='running',workers=workers);s.save(dest/'status.json',state)
            print(f"Posterior predictive checks: {len(p['accepted_full_fits'])} accepted full fits, {p['simulations']} replicates each, {workers} workers; no MCMC.",flush=True)
            def launch(name):
                folder=dest/'fits'/name;folder.mkdir(parents=True,exist_ok=True)
                with (folder/'console.txt').open('a',buffering=1) as log:
                    log.write('\nPPC WORKER '+s.now()+'\n')
                    return subprocess.run([sys.executable,'-m','rf1_trust_socialvalue.full_sample_ppc','fit','--name',name],cwd=c['_root'],stdout=log,stderr=subprocess.STDOUT).returncode
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(launch,name):name for name in p['accepted_full_fits']}
                for future in as_completed(futures):
                    name=futures[future]
                    try:code=future.result()
                    except Exception as exc:code=1;state.setdefault('errors',{})[name]=repr(exc)
                    state['runs'][name]=code;s.save(dest/'status.json',state);print(name+': PPC exit '+str(code),flush=True)
            if any(state['runs'].values()):raise RuntimeError('Some predictive checks failed; see per-fit logs')
            state['output_sha256']=render(p,c);geometry.saved_trials(phase,c)
            state.update(status='complete',finished_at=s.now());s.save(dest/'status.json',state)
        except BaseException as exc:
            state.update(status='error',finished_at=s.now(),error=repr(exc));s.save(dest/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','fit']);parser.add_argument('--name');a=parser.parse_args()
    p,phase,c=configuration()
    if a.command=='fit':
        if a.name not in p['accepted_full_fits']:parser.error('Only configured accepted full fits are allowed')
        fit_entry(p,phase,c,a.name)
    else:run(p,phase,c)


if __name__=='__main__':main()
