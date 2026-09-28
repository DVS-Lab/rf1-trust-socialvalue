"""Matched full/run-1 no-age fits; reuse accepted pilots and score actual run 2."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from . import full_sample_sampling as s

REUSED = {
    ('H2','base'): 'Full_H2_base_noage',
    ('H5','zero'): 'Full_H5_zero_noage',
    ('H7','zero'): 'Full_H7_zero_noage',
    ('HPreference','zero'): 'Full_HPreference_zero_noage_draws4000',
}


def validate_configuration(phase,c,cfg):
    expected=[dict(model=m,variant=v,name=name) for (m,v),name in REUSED.items()]
    if phase['reuse']!=expected or phase['parallel_fits']!=16 or phase['cpu_budget']!=80:
        raise ValueError('Batch must retain the reviewed four fits and 16-fit/80-core limits')
    if phase['draws_by_model']!={'HPreference':4000} or cfg['draws']!=2000:
        raise ValueError('Only the reviewed preference-family draw allocation is allowed')
    original=json.loads((Path(c['_root'])/'config/full_sample_sampling.json').read_text())
    for key in ['gamma_population_mean_prior_sd','cmdstan_version','metric','cohort_config','integration_status_sha256']:
        if phase[key]!=original[key]:raise ValueError('Batch changed the accepted target: '+key)
    required={f'results/full_sample/hierarchical/fits/{name}/{file}' for name in REUSED.values()
              for file in ['status.json','manifest_summary.json']}
    if set(phase['accepted_evidence_sha256'])!=required:raise ValueError('Incomplete accepted-fit evidence')
    for name,h in phase['accepted_evidence_sha256'].items():
        if s.fs.sha(Path(c['_root'])/name)!=h:raise ValueError('Accepted-fit evidence changed: '+name)


def new_entries(phase):
    return [dict(model=m,variant=v,subset=subset,name=f"{'Full' if subset=='full' else 'Train'}_{m}_{v}_noage")
            for subset in ['full','train'] for m in s.MODELS for v in ['base','zero']
            if subset=='train' or (m,v) not in REUSED]


def entry_settings(phase,cfg,entry):
    return dict(cfg,draws=phase['draws_by_model'].get(entry['model'],cfg['draws']))


def paired_ids(t):
    if not set(t.run.unique()).issubset({1,2}):raise ValueError('Unreviewed run labels in canonical cohort')
    counts=t.groupby(['participant_id','run']).valid_choice.sum().unstack(fill_value=0)
    if not {1,2}.issubset(counts.columns):raise ValueError('No participants with valid choices in both runs')
    ids=sorted(counts.index[(counts[1]>0)&(counts[2]>0)].tolist())
    if not ids:raise ValueError('No participants with valid choices in both runs')
    return ids


def entry_trials(t,entry):
    if entry['subset']=='full':return t
    if entry['subset']!='train':raise ValueError('Unknown fit subset')
    return t[t.participant_id.isin(paired_ids(t)) & (t.run==1)].copy()


def heldout_scores(t,entry,natural,meta):
    """Run-1 posterior, observed history online, no run-2 parameter updating.

    Integrate each trial probability over the unchanged training posterior. This
    is an online conditional score, not a joint marginal likelihood of run 2.
    """
    ids=paired_ids(t)
    if meta['ids']!=ids or natural.ndim!=3 or natural.shape[1]!=len(ids):
        raise ValueError('Training posterior participant order differs from paired cohort')
    if not np.isfinite(natural).all():raise ValueError('Nonfinite predictive posterior')
    _,_,arrays=s.model_data(t[t.participant_id.isin(ids)],entry['model'],entry['variant']=='zero')
    rows=[];zero=entry['variant']=='zero'
    for i,(sub,a) in enumerate(arrays.items()):
        valid=(a[:,7]==2)&(a[:,3]>=0);y=a[valid,3]
        nll=[];prob=[]
        for par in natural[:,i,:]:
            trajectory=s.zero_engine(a,par[:-1] if zero else par,s.SPECS[entry['model']][1],par[-1] if zero else 0.)
            nll.append(trajectory[valid,3]);prob.append(trajectory[valid,1])
        loss=-logsumexp(-np.asarray(nll),axis=0)+np.log(len(natural));mean=np.mean(prob,axis=0)
        rows.append(dict(name=entry['name'],model=entry['model'],variant=entry['variant'],participant_id=sub,
            n=len(y),posterior_draws=len(natural),log_loss=float(loss.mean()),brier=float(np.mean((mean-y)**2)),
            accuracy=float(np.mean((mean>=.5)==y))))
    result=pd.DataFrame(rows)
    if not np.isfinite(result[['log_loss','brier','accuracy']]).all().all():raise ValueError('Nonfinite heldout scores')
    return result


def review_reused(phase,c,t):
    """Bind accepted summaries to the same current full-data Stan target."""
    out,_=s.paths(c);rows=[]
    for entry in phase['reuse']:
        folder=out/'fits'/entry['name'];status=json.loads((folder/'status.json').read_text())
        manifest=json.loads((folder/'manifest_summary.json').read_text())
        if status['status']!='complete' or not status['passed']:raise ValueError('Unaccepted reuse: '+entry['name'])
        for file,h in {**status['summary_hashes'],'diagnostics.tsv':status['diagnostics_sha256']}.items():
            if s.fs.sha(folder/file)!=h:raise ValueError('Accepted summary changed: '+str(folder/file))
        d=pd.read_csv(folder/'diagnostics.tsv',sep='\t');a=c['hierarchical']['acceptance']
        if (not np.isfinite(d[['R_hat','ESS_bulk','ESS_tail']]).all().all()
                or (d.R_hat>=a['rhat_less_than']).any() or (d.ESS_bulk<a['minimum_bulk_ess']).any()
                or (d.ESS_tail<a['minimum_tail_ess']).any() or status['divergences']!=a['divergences']
                or status['max_depth_hits']!=a['treedepth_hits'] or status['min_bfmi']<=a['bfmi_greater_than']):
            raise ValueError('Reused fit no longer passes diagnostics')
        data,meta,_=s.model_data(t,entry['model'],entry['variant']=='zero',phase['gamma_population_mean_prior_sd'])
        payload=dict(data=data,settings=status['settings'],sampler_seed=status['sampler_seed'],
            stan={p:s.fs.sha(Path(c['_root'])/p) for p in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
        if s.digest(payload)!=status['fingerprint'] or manifest['fingerprint']!=status['fingerprint'] or manifest['meta']!=meta:
            raise ValueError('Accepted fit differs from current full-data target')
        rows.append(dict(entry,subset='full',status='complete',reused=True,fingerprint=status['fingerprint']))
    return rows


def worker_count(phase,cfg,count):
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    workers=min(count,phase['parallel_fits'],min(phase['cpu_budget'],available)//cfg['chains'])
    if workers<1:raise RuntimeError('Fewer than four CPUs available')
    return workers


def compare_scores(frames,seed,iterations):
    scores=pd.concat(frames,ignore_index=True)
    wide=scores.pivot(index='participant_id',columns='name',values='log_loss').sort_index()
    if wide.isna().any().any():raise ValueError('Heldout models do not use identical participants')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(wide),(iterations,len(wide)));rows=[]
    baseline=wide['Train_H2_base_noage'].to_numpy()
    for name in sorted(wide):
        x=wide[name].to_numpy();delta=x-baseline
        lower,upper=np.quantile(delta[indices].mean(axis=1),[.025,.975])
        frame=scores[scores.name==name]
        rows.append(dict(name=name,participants=len(x),heldout_choices=int(frame.n.sum()),
            mean_log_loss=float(x.mean()),mean_brier=float(frame.brier.mean()),
            delta_log_loss_vs_H2_base=float(delta.mean()),delta_ci_low=float(lower),delta_ci_high=float(upper)))
    paired=[]
    for model in s.MODELS:
        delta=(wide[f'Train_{model}_zero_noage']-wide[f'Train_{model}_base_noage']).to_numpy()
        lo,hi=np.quantile(delta[indices].mean(axis=1),[.025,.975])
        paired.append(dict(model=model,participants=len(delta),delta_zero_minus_base=float(delta.mean()),ci_low=float(lo),ci_high=float(hi)))
    return pd.DataFrame(rows),pd.DataFrame(paired)


def comparison_outputs(phase,c):
    out,_=s.paths(c);batch=out/'batch';frames=[];hashes={}
    for entry in new_entries(phase):
        if entry['subset']!='train':continue
        folder=out/'fits'/entry['name'];status=json.loads((folder/'status.json').read_text());file=folder/'heldout_participants.tsv'
        if status['status']!='complete' or not status['passed'] or s.fs.sha(file)!=status['heldout_sha256']:
            raise ValueError('Heldout comparison requires ten accepted training fits with verified scores')
        frame=pd.read_csv(file,sep='\t')
        if set(frame.name)!={entry['name']} or frame.participant_id.duplicated().any():raise ValueError('Invalid heldout table')
        frames.append(frame);hashes[str(file.relative_to(out))]=s.fs.sha(file)
    summary,paired=compare_scores(frames,c['seed'],c['bootstrap_iterations'])
    summary.to_csv(batch/'heldout_comparison.tsv',sep='\t',index=False)
    paired.to_csv(batch/'zero_option_comparison.tsv',sep='\t',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    x=paired.delta_zero_minus_base.to_numpy();y=np.arange(len(paired))
    # Draw endpoints directly: percentile intervals need not contain the point estimate.
    ax.hlines(y,paired.ci_low,paired.ci_high,color='#176b87',linewidth=2)
    ax.plot(x,y,'o',color='#176b87');ax.axvline(0,color='.5',linestyle='--')
    ax.set(yticks=y,yticklabels=paired.model,xlabel='Run-2 log loss: zero-option model minus base\nNegative values favor adding the zero-option term',
           title='Prediction from run-1 parameters\nEqual-participant means and paired 95% bootstrap intervals')
    fig.savefig(batch/'zero_option_heldout.png',dpi=180);fig.savefig(batch/'zero_option_heldout.pdf');plt.close(fig)
    s.save(batch/'comparison_provenance.json',dict(score_hashes=hashes,bootstrap_seed=c['seed'],bootstrap_iterations=c['bootstrap_iterations'],
        estimand='Equal-participant mean per-trial log loss; online observed history with fixed run-1 posterior; paired participant bootstrap',
        generated_at=s.now()))
    return {p:s.fs.sha(batch/p) for p in ['heldout_comparison.tsv','zero_option_comparison.tsv','zero_option_heldout.png','zero_option_heldout.pdf','comparison_provenance.json']}


def run(phase,c,cfg):
    import fcntl
    s.require_linux();out,work=s.paths(c);batch=out/'batch';work.mkdir(parents=True,exist_ok=True)
    # Same parent lock as the pilot: code upgrades must wait until a pilot/retry finishes.
    with (work/'pilot.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A pilot, retry, or batch is already running')
        record=dict(status='preparing',started_at=s.now(),analysis_git_sha=s.fs.git_sha(c['_root']),runs={})
        s.save(batch/'status.json',record)
        try:
            t=s.integrated_trials(phase,c);record['reused']=review_reused(phase,c,t)
            paired=t[t.participant_id.isin(paired_ids(t))]
            counts=paired.groupby(['participant_id','run']).valid_choice.sum().unstack()
            counts.rename(columns={1:'train_choices',2:'test_choices'}).to_csv(batch/'heldout_cohort.tsv',sep='\t')
            s.cmdstan(phase,c,install=False)
            for subset in ['full','train']:
                s.check_implementation(dict(phase,_subset=subset),c,entry_trials(t,{'subset':subset}))
            entries=new_entries(phase);workers=worker_count(phase,cfg,len(entries))
            record.update(status='running',n_full=int(t.participant_id.nunique()),n_train=len(counts),parallel_fits=workers,
                parallel_chains=cfg['chains'],maximum_active_chains=workers*cfg['chains'],
                phase_config_sha256=s.fs.sha(phase['_path']),source_hashes=s.implementation(c['_root']))
            s.save(batch/'status.json',record)
            print(f"Batch: {len(entries)} new fits; {workers} concurrent fits x {cfg['chains']} chains = {workers*cfg['chains']} active cores; four full fits reused.",flush=True)
            def launch(entry):
                dest=out/'fits'/entry['name'];dest.mkdir(parents=True,exist_ok=True)
                # Preserve accepted statuses while a cached fit is verified/reloaded.
                if not (dest/'status.json').exists():s.save(dest/'status.json',dict(name=entry['name'],status='starting'))
                with (dest/'console.txt').open('a',buffering=1) as log:
                    log.write('\nBATCH WORKER '+s.now()+'\n')
                    code=subprocess.run([sys.executable,'-m','rf1_trust_socialvalue.full_sample_sampling','fit','--config',phase['_path'],'--name',entry['name']],
                        cwd=c['_root'],stdout=log,stderr=subprocess.STDOUT).returncode
                state=json.loads((dest/'status.json').read_text())
                if code!=0 and state.get('status') in {'starting','complete','running'}:
                    state.update(status='error',exit_code=code,error='Worker failed; see console.txt',finished_at=s.now());s.save(dest/'status.json',state)
                return code if code else (0 if state.get('status')=='complete' and state.get('passed') else 2)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(launch,e):e for e in entries}
                for future in as_completed(futures):
                    entry=futures[future]
                    try:code=future.result();result=dict(exit_code=code)
                    except Exception as exc:result=dict(exit_code=1,error=repr(exc))
                    record['runs'][entry['name']]=result;s.save(batch/'status.json',record)
                    print(entry['name']+': exit '+str(result['exit_code']),flush=True)
            if record['source_hashes']!=s.implementation(c['_root']) or record['phase_config_sha256']!=s.fs.sha(phase['_path']):
                raise ValueError('Batch implementation/configuration changed during execution')
            s.integrated_trials(phase,c);review_reused(phase,c,t)
            passed=all(r['exit_code']==0 for r in record['runs'].values())
            if passed:record['comparison_hashes']=comparison_outputs(phase,c)
            record.update(status='complete' if passed else 'blocked_diagnostics_or_error',finished_at=s.now(),
                next_stage='Review heldout comparisons and full-fit parameters, then posterior predictive checks, age/sensitivity and targeted recovery')
            s.save(batch/'status.json',record)
            return 0 if passed else 2
        except BaseException as exc:
            record.update(status='error',error=repr(exc),finished_at=s.now());s.save(batch/'status.json',record);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','status'])
    parser.add_argument('--config',default='config/full_sample_batch.json');a=parser.parse_args()
    phase,c,cfg=s.configuration(a.config)
    if phase['stage']!='full_cohort_noage_batch':parser.error('A batch configuration is required')
    if a.command=='status':
        out,_=s.paths(c)
        for e in new_entries(phase):
            p=out/'fits'/e['name']/'status.json';state=json.loads(p.read_text()) if p.exists() else {}
            print(e['name'],state.get('status','not_started'))
        return
    raise SystemExit(run(phase,c,cfg))


if __name__=='__main__':main()
