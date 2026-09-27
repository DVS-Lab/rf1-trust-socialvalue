"""Linux2 production-length no-age pilot; isolated from frozen N111 fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from . import full_sample as fs
from .fitting import stable_seed
from .hierarchical import SPECS, load_chains, summarize, transform
from .models import pack
from .n111_zero import zero_engine
from .run_logging import write_tail

MODELS = ['H2', 'H5', 'H7', 'HPreference', 'H8']
SOURCES = ['stan/n111_zero_fast.stan', 'stan/n111_zero_reference.stan', 'stan/n111_zero_fast.hpp']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');fs.save_json(temp,value);temp.replace(path)


def configuration(path='config/full_sample_sampling.json'):
    path=Path(path).resolve();phase=json.loads(path.read_text());root=path.parent.parent
    c=fs.resolve_config(root/phase['cohort_config'])
    if phase['stage'] not in {'full_cohort_noage_pilot','full_cohort_noage_retry'} or not phase['launch_authorized'] or phase['automatic_retries']:
        raise ValueError('Only the authorized pilot or reviewed retry, without automatic retries, is available')
    h=c['hierarchical']
    if h['chains']!=4 or h['parallel_chains']!=4 or phase['parallel_fits']<1:
        raise ValueError('Pilot requires four parallel chains per fit')
    settings={k:h[k] for k in ['chains','warmup','draws','adapt_delta','max_treedepth']}
    settings.update(seed=c['seed'],metric=phase['metric'],inits=.15,sig_figs=10)
    phase['_path']=str(path);phase['_root']=str(root)
    if phase['stage']=='full_cohort_noage_retry':validate_retry(phase,c,settings)
    return phase,c,settings


def validate_retry(phase,c,settings):
    """One explicit ESS-only retry, bound to the reviewed failed attempt."""
    r=phase['retry'];source='Full_HPreference_zero_noage'
    if (r['source_run']!=source or r['replacement_run']!=source+'_draws4000' or r['max_attempts']!=1
            or r['draws']!=4000 or phase['parallel_fits']!=1
            or phase['pilot']!=[dict(model='HPreference',variant='zero')]):
        raise ValueError('Only the reviewed HPreference draws4000 retry is allowed')
    folder=Path('results/full_sample/hierarchical/fits')/source
    required={str(folder/p) for p in ['status.json','diagnostics.tsv','manifest_summary.json']}|{'config/full_sample_sampling.json'}
    if set(r['evidence_sha256'])!=required:raise ValueError('Incomplete retry evidence')
    for name,h in r['evidence_sha256'].items():
        if fs.sha(Path(c['_root'])/name)!=h:raise ValueError('Reviewed retry evidence changed: '+name)
    old=json.loads((Path(c['_root'])/folder/'status.json').read_text())
    if old['status']!='diagnostic_failed' or old['passed'] or old['settings']!=settings:
        raise ValueError('Retry must retain the original sampler settings except retained draws')
    if old['integration_status_sha256']!=phase['integration_status_sha256']:
        raise ValueError('Retry cohort differs from failed attempt')
    a=c['hierarchical']['acceptance']
    if (old['max_rhat']>=a['rhat_less_than'] or old['min_tail_ess']<a['minimum_tail_ess']
            or old['divergences']!=0 or old['max_depth_hits']!=0 or old['min_bfmi']<=a['bfmi_greater_than']
            or not 0<old['min_bulk_ess']<a['minimum_bulk_ess']):
        raise ValueError('Retry rationale applies only to a bulk-ESS failure')
    original=json.loads((Path(c['_root'])/'config/full_sample_sampling.json').read_text())
    for key in ['gamma_population_mean_prior_sd','cmdstan_version','metric','cohort_config','integration_status_sha256']:
        if phase[key]!=original[key]:raise ValueError('Retry must preserve the reviewed target: '+key)
    settings['draws']=r['draws']


def verify_retry_target(phase,c,payload):
    if phase['stage']!='full_cohort_noage_retry':return
    source=Path(c['output'])/'hierarchical/fits'/phase['retry']['source_run']/'status.json'
    old=json.loads(source.read_text())
    original=dict(payload,settings=old['settings'])
    if digest(original)!=old['fingerprint']:
        raise ValueError('Retry data, priors, Stan implementation or seed differ from original target')


def phase_output(phase,c):
    out,_=paths(c)
    return out/'retries'/phase['retry']['replacement_run'] if phase['stage']=='full_cohort_noage_retry' else out


def paths(c):
    return Path(c['output'])/'hierarchical',Path(c['work'])/'hierarchical'


def entries(phase):
    rows=[dict(e,name=f"Full_{e['model']}_{e['variant']}_noage") for e in phase['pilot']]
    if phase['stage']=='full_cohort_noage_retry':
        rows=[dict(rows[0],name=phase['retry']['replacement_run'],seed_name=phase['retry']['source_run'])]
    if len({e['name'] for e in rows})!=len(rows) or any(e['model'] not in MODELS or e['variant'] not in ['base','zero'] for e in rows):
        raise ValueError('Invalid or duplicate pilot entry')
    return rows


def require_linux():
    if platform.system()!='Linux':raise RuntimeError('Compile and sampling run on Linux2 only; never on the laptop')


def integrated_trials(phase,c):
    status_path=Path(c['output'])/'milestone_status.json'
    if fs.sha(status_path)!=phase['integration_status_sha256']:
        raise ValueError('Integration gate changed since pilot review')
    s=json.loads(status_path.read_text())
    if s['status']!='ready_for_review' or s['cohort_summary']['unexplained_parity_fields']:
        raise ValueError('Integration/parity gate has not passed')
    for name,value in s['output_hashes'].items():
        if fs.sha(Path(c['output'])/name)!=value:raise ValueError('Reviewed integration output changed: '+name)
    fs.verify(c)
    t=pd.read_csv(Path(c['work'])/'canonical_trials.tsv',sep='\t',dtype={'trial_id':str,'session':str})
    if t.participant_id.nunique()!=s['cohort_summary']['primary_n'] or int(t.valid_choice.sum())!=s['cohort_summary']['total_valid_choices']:
        raise ValueError('Canonical model sample differs from reviewed cohort')
    return t


def model_data(t,model,zero,gamma_sd=1.):
    """Use canonical primary trials only. Misses and zero choices never update."""
    if model not in MODELS:raise ValueError('Unsupported model')
    if not set(t.session.astype(str).str.zfill(2)).issubset({'01'}):raise ValueError('Unreviewed model session')
    if t.duplicated(['participant_id','session','run','trial_id']).any():raise ValueError('Duplicate canonical decision')
    frames={sub:f.sort_values(['run','trial_in_run']) for sub,f in t.groupby('participant_id',sort=True)}
    arrays={sub:pack(f) for sub,f in frames.items()}
    first=[];last=[];blocks=[];cursor=1
    for sub,a in arrays.items():
        f=frames[sub]
        if not f.valid_choice.to_numpy().astype(bool).tolist()==(a[:,3]>=0).tolist():
            raise ValueError('Valid-choice flag disagrees with decisions')
        positive=(a[:,3]>=0)&(np.where(a[:,3]==1,a[:,2],a[:,1])>0)
        if not np.array_equal(a[:,4]>0,positive):raise ValueError('Feedback inconsistent with zero/missing choices')
        seen=f.observed_reciprocation.to_numpy(float)
        if not np.all(np.isnan(seen[~positive])) or not np.array_equal(seen[positive],a[positive,5]):
            raise ValueError('Observed outcomes disagree with schedule/feedback')
        b=a[a[:,3]>=0]
        if not len(b):raise ValueError('Participant without valid choices in model sample')
        first.append(cursor);last.append(cursor+len(b)-1);cursor+=len(b);blocks.append(b)
    if not blocks:raise ValueError('Empty canonical model sample')
    a=np.concatenate(blocks);_,code,names,kinds=SPECS[model];kinds=list(kinds);names=list(names)
    locations=[-1. if k==1 else -1.2 if k==2 else 1. if k==3 else 0. for k in kinds]
    scales=[1.25 if k==1 else .8 if k==2 else 1.5 if k==3 else 1. for k in kinds]
    if zero:kinds.append(4);names.append('gamma0');locations.append(0.);scales.append(gamma_sd)
    data=dict(N=len(arrays),T=len(a),K=len(kinds),A=0,model_code=code,bounded_theta=0,independent=0,
        kind=kinds,first=first,last=last,partner=(a[:,0]+1).astype(int).tolist(),gap=(a[:,2]-a[:,1]).tolist(),
        y=a[:,3].astype(int).tolist(),feedback=a[:,4].astype(int).tolist(),
        outcome=np.where(a[:,4]>0,a[:,5],0.).tolist(),ratings=np.zeros((len(arrays),3)).tolist(),
        age=[[] for _ in arrays],mu_location=locations,mu_scale=scales,prior_scale=1.,prior_only=0,
        zero_term=int(zero),zero_option=(a[:,1]==0).astype(int).tolist())
    meta=dict(ids=list(arrays),model=model,variant='zero' if zero else 'base',age_terms=0,
        n_trials=len(a),n_presented=len(t),parameter_names=['preference_friend' if model=='HPreference' and n=='theta' else n for n in names],
        run_transition='beliefs carry across observed runs; no within-run heldout fallback')
    return data,meta,arrays


def implementation(root):
    files=SOURCES+['src/rf1_trust_socialvalue/full_sample_sampling.py',
        'src/rf1_trust_socialvalue/n111_zero.py','src/rf1_trust_socialvalue/models.py',
        'src/rf1_trust_socialvalue/hierarchical.py']
    return {p:fs.sha(Path(root)/p) for p in files}


def cmdstan(phase,c,install=False):
    from cmdstanpy import set_cmdstan_path,install_cmdstan,cmdstan_version
    require_linux();root=Path(c['_root']);_,work=paths(c)
    version=phase['cmdstan_version'];old=root/'work/cmdstan'/('cmdstan-'+version)
    dest=work/'cmdstan';target=old if old.is_dir() else dest/('cmdstan-'+version)
    if not target.is_dir():
        if not install:raise RuntimeError('CmdStan missing; run pilot preparation first')
        if not install_cmdstan(version=version,dir=str(dest),cores=phase['compile_cores']):
            raise RuntimeError('CmdStan installation failed; see saved console log')
    set_cmdstan_path(str(target.resolve()))
    if cmdstan_version()!=tuple(map(int,version.split('.')[:2])):raise RuntimeError('Unexpected CmdStan version')
    return str(target)


def models(phase,c,reference=False):
    from cmdstanpy import CmdStanModel
    cmdstan(phase,c);root=Path(c['_root']);_,work=paths(c);build=work/'build';build.mkdir(parents=True,exist_ok=True)
    h=digest({p:fs.sha(root/p) for p in SOURCES});result=[]
    for variant in (['fast','reference'] if reference else ['fast']):
        text=(root/f'stan/n111_zero_{variant}.stan').read_text()+'\n// Full-cohort implementation '+h+'\n'
        target=build/f'full_zero_{variant}.stan'
        if not target.exists() or target.read_text()!=text:target.write_text(text)
        options=dict(user_header=str(root/'stan/n111_zero_fast.hpp'),stanc_options={'allow-undefined':True}) if variant=='fast' else {}
        result.append(CmdStanModel(stan_file=str(target),**options))
    return result


def check_implementation(phase,c,t):
    fast,reference=models(phase,c,reference=True);out=phase_output(phase,c);rng=np.random.default_rng(c['seed']);rows=[]
    small=t[t.participant_id.isin(sorted(t.participant_id.unique())[:4])]
    for model in MODELS:
        for zero,gamma in [(False,0.),(True,-2.),(True,0.),(True,2.)]:
            data,meta,arrays=model_data(small,model,zero,phase['gamma_population_mean_prior_sd']);k=data['K']
            corr=np.eye(k)*.9+np.ones((k,k))*.1;mu=np.array(data['mu_location'])
            if zero:mu[-1]=gamma
            pars=dict(mu=mu.tolist(),tau=[.6]*k,L=np.linalg.cholesky(corr).tolist(),
                beta=[[] for _ in range(k)],z=rng.normal(0,.4,(k,data['N'])).tolist())
            a=reference.log_prob(params=pars,data=data,jacobian=True,sig_figs=16).iloc[0]
            b=fast.log_prob(params=pars,data=data,jacobian=True,sig_figs=16).iloc[0]
            if list(a.index)!=list(b.index):raise ValueError('Stan gradient coordinate order mismatch')
            error=np.abs(a.to_numpy()-b.to_numpy());relative=float(np.max(error/(1+np.abs(a.to_numpy()))))
            prior=reference.log_prob(params=pars,data=dict(data,prior_only=1),jacobian=True,sig_figs=16).iloc[0,0]
            natural=transform(mu+(np.diag(pars['tau'])@np.array(pars['L'])@np.array(pars['z'])).T,data['kind'])
            pyll=sum(-zero_engine(trials,natural[i,:-1] if zero else natural[i],SPECS[model][1],natural[i,-1] if zero else 0.)[:,3].sum() for i,trials in enumerate(arrays.values()))
            likelihood_error=abs(float(a.iloc[0])-prior-pyll)
            passed=bool(np.isfinite(relative) and relative<1e-9 and np.isfinite(likelihood_error) and likelihood_error<1e-8)
            rows.append(dict(model=model,zero=zero,gamma=gamma,max_scaled_gradient_error=relative,python_likelihood_error=float(likelihood_error),passed=passed))
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'implementation_checks.tsv',sep='\t',index=False)
    record=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',source_hashes=implementation(c['_root']),
        integration_status_sha256=phase['integration_status_sha256'],checks_sha256=fs.sha(out/'implementation_checks.tsv'),cmdstan_version=phase['cmdstan_version'],generated_at=now())
    save(out/'implementation_status.json',record)
    if record['status']!='passed':raise RuntimeError('Implementation check failed; sampling blocked')
    print('Passed 20 Stan target/gradient and Python likelihood checks.',flush=True)


def implementation_gate(phase,c):
    out=phase_output(phase,c);s=json.loads((out/'implementation_status.json').read_text())
    if s['status']!='passed' or s['source_hashes']!=implementation(c['_root']) or s['integration_status_sha256']!=phase['integration_status_sha256'] or s['cmdstan_version']!=phase['cmdstan_version'] or s['checks_sha256']!=fs.sha(out/'implementation_checks.tsv'):
        raise ValueError('Implementation check missing or stale')


def diagnostics(fit,cfg,thresholds):
    d=fit.summary();d=d[d.index.str.startswith(('mu[','tau[','beta[','Omega[','natural['))].copy()
    d=d[~d.index.str.match(r'Omega\[(\d+),\1\]')].rename_axis('parameter').reset_index()
    method=fit.method_variables();e=method['energy__'];bfmi=np.mean(np.diff(e,axis=0)**2,axis=0)/np.var(e,axis=0)
    divergence=int(method['divergent__'].sum());depth=int((method['treedepth__']>=cfg['max_treedepth']).sum())
    good=(np.isfinite(d[['R_hat','ESS_bulk','ESS_tail']]).all(axis=1) & (d.R_hat<thresholds['rhat_less_than']) & (d.ESS_bulk>=thresholds['minimum_bulk_ess']) & (d.ESS_tail>=thresholds['minimum_tail_ess']))
    passed=bool(len(d)>0 and good.all() and divergence==thresholds['divergences'] and depth==thresholds['treedepth_hits'] and np.isfinite(bfmi).all() and bfmi.min()>thresholds['bfmi_greater_than'])
    finite=lambda x:float(x) if np.isfinite(x) else None
    info=dict(passed=passed,max_rhat=finite(d.R_hat.max()),min_bulk_ess=finite(d.ESS_bulk.min()),min_tail_ess=finite(d.ESS_tail.min()),divergences=divergence,max_depth_hits=depth,min_bfmi=finite(bfmi.min()))
    return d,info


def fit_entry(phase,c,cfg,entry):
    require_linux();t=integrated_trials(phase,c);implementation_gate(phase,c);cmdstan(phase,c)
    data,meta,arrays=model_data(t,entry['model'],entry['variant']=='zero',phase['gamma_population_mean_prior_sd'])
    out,work=paths(c);name=entry['name'];folder=work/'fits'/name;evidence=out/'fits'/name
    folder.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    sampler_seed=stable_seed(cfg['seed'],entry.get('seed_name',name))
    payload=dict(data=data,settings=cfg,sampler_seed=sampler_seed,stan={p:fs.sha(Path(c['_root'])/p) for p in SOURCES},cmdstan_version=phase['cmdstan_version'])
    verify_retry_target(phase,c,payload)
    target=digest(payload)
    source_hashes=implementation(c['_root'])
    state=dict(name=name,entry=entry,status='running',started_at=now(),n_participants=data['N'],n_choices=data['T'],fingerprint=target,
        settings=cfg,sampler_seed=sampler_seed,parallel_chains=cfg['chains'],analysis_git_sha=fs.git_sha(c['_root']),integration_status_sha256=phase['integration_status_sha256'])
    save(evidence/'status.json',state);manifest=folder/'manifest.json';start=time.monotonic()
    try:
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved['fingerprint']!=target:raise ValueError('Existing fit target/settings differ; cache left untouched')
            for p,h in saved['posterior_sha256'].items():
                if fs.sha(p)!=h:raise ValueError('Cached posterior changed: '+p)
            fit=load_chains(saved['csv_files']);state['reused_posterior']=True
        else:
            if list(folder.glob('*.csv')):raise RuntimeError('Partial posterior CSVs exist; inspect before an explicit restart')
            fast=models(phase,c)[0]
            fit=fast.sample(data=data,chains=cfg['chains'],parallel_chains=cfg['chains'],iter_warmup=cfg['warmup'],iter_sampling=cfg['draws'],
                seed=sampler_seed,adapt_delta=cfg['adapt_delta'],max_treedepth=cfg['max_treedepth'],metric=cfg['metric'],
                inits=cfg['inits'],sig_figs=cfg['sig_figs'],output_dir=str(folder),refresh=200,show_progress=False,show_console=True)
            saved=dict(fingerprint=target,csv_files=fit.runset.csv_files,posterior_sha256={p:fs.sha(p) for p in fit.runset.csv_files},
                sampling_seconds=time.monotonic()-start,meta=meta,settings=cfg,sampler_seed=sampler_seed,source_hashes=source_hashes,
                analysis_git_sha=state['analysis_git_sha'],integration_status_sha256=phase['integration_status_sha256'],
                package_versions={p:importlib.metadata.version(p) for p in ['cmdstanpy','numpy','pandas','scipy']})
            save(manifest,saved);state['reused_posterior']=False
        if source_hashes!=implementation(c['_root']):raise ValueError('Sampling implementation changed during the fit')
        integrated_trials(phase,c)
        d,info=diagnostics(fit,cfg,c['hierarchical']['acceptance']);state.update(info)
        state['sampling_seconds']=saved['sampling_seconds'];d.to_csv(evidence/'diagnostics.tsv',sep='\t',index=False)
        (evidence/'diagnose.txt').write_text(fit.diagnose())
        state['diagnostics_sha256']=fs.sha(evidence/'diagnostics.tsv')
        if not info['passed']:
            state['status']='diagnostic_failed';return 2
        natural=fit.stan_variable('natural');rows=[]
        for i,sub in enumerate(meta['ids']):
            for j,parameter in enumerate(meta['parameter_names']):
                rows.append(dict(participant_id=sub,parameter=parameter,**summarize(natural[:,i,j])))
        pd.DataFrame(rows).to_csv(evidence/'participant_parameters.tsv',sep='\t',index=False)
        population=[];mu=fit.stan_variable('mu');tau=fit.stan_variable('tau')
        for j,parameter in enumerate(meta['parameter_names']):
            for quantity,x in [('latent_population_location',mu[:,j]),('latent_population_sd',tau[:,j])]:
                population.append(dict(parameter=parameter,quantity=quantity,**summarize(x)))
        pd.DataFrame(population).to_csv(evidence/'population_parameters.tsv',sep='\t',index=False)
        state['summary_hashes']={p.name:fs.sha(p) for p in evidence.glob('*_parameters.tsv')}
        state['status']='complete';return 0
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        import resource
        divisor=1024 if platform.system()=='Linux' else 1024**2
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start,
            peak_python_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor,
            peak_child_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/divisor)
        if manifest.exists():
            saved=json.loads(manifest.read_text());portable=dict(saved,csv_files=[Path(p).name for p in saved['csv_files']],posterior_sha256={Path(p).name:h for p,h in saved['posterior_sha256'].items()})
            save(evidence/'manifest_summary.json',portable)
        for p in folder.glob('*.txt'):write_tail(p,evidence/p.name)
        save(evidence/'status.json',state)
        print(json.dumps(state,indent=2),flush=True)


def pilot(phase,c,cfg):
    import fcntl
    require_linux();out,work=paths(c);out=phase_output(phase,c);work.mkdir(parents=True,exist_ok=True)
    with (work/'pilot.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A full-cohort pilot is already running')
        save(out/'pilot_status.json',dict(status='preparing',started_at=now()))
        try:return run_pilot(phase,c,cfg)
        except BaseException as exc:
            record=json.loads((out/'pilot_status.json').read_text())
            record.update(status='error',finished_at=now(),error=repr(exc))
            save(out/'pilot_status.json',record)
            raise


def run_pilot(phase,c,cfg):
    out,work=paths(c);batch=phase_output(phase,c)
    t=integrated_trials(phase,c);cmdstan(phase,c,install=True);check_implementation(phase,c,t)
    available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    workers=min(len(entries(phase)),phase['parallel_fits'],min(phase['cpu_budget'],available)//cfg['chains'])
    if workers<1:raise RuntimeError('Fewer than four CPUs available for four-chain sampling')
    record=dict(status='running',started_at=now(),n_participants=t.participant_id.nunique(),parallel_fits=workers,
        parallel_chains=cfg['chains'],maximum_active_chains=workers*cfg['chains'],runs={},next_stage='review runtime and diagnostics before broad full/train batch')
    save(batch/'pilot_status.json',record)
    print(f"Pilot: {record['n_participants']} participants, {workers} concurrent fits × {cfg['chains']} chains = {workers*cfg['chains']} active cores.",flush=True)
    def launch(entry):
        dest=out/'fits'/entry['name'];dest.mkdir(parents=True,exist_ok=True)
        save(dest/'status.json',dict(name=entry['name'],status='starting',started_at=now()))
        with (dest/'console.txt').open('a',buffering=1) as log:
            log.write('\nPILOT WORKER '+now()+'\n')
            code=subprocess.run([sys.executable,'-m','rf1_trust_socialvalue.full_sample_sampling','fit','--config',phase['_path'],'--name',entry['name']],cwd=c['_root'],stdout=log,stderr=subprocess.STDOUT).returncode
        state=json.loads((dest/'status.json').read_text())
        if state['status']=='starting':
            state.update(status='error',exit_code=code,finished_at=now(),error='Worker exited before fit initialization; see console.txt')
            save(dest/'status.json',state)
        return code if code else (0 if state.get('status')=='complete' and state.get('passed') else 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(launch,e):e for e in entries(phase)}
        for future in as_completed(pending):
            entry=pending[future];code=future.result();record['runs'][entry['name']]=dict(exit_code=code)
            save(batch/'pilot_status.json',record);print(f"{entry['name']}: exit {code}",flush=True)
    integrated_trials(phase,c)
    record.update(status='ready_for_runtime_review' if all(r['exit_code']==0 for r in record['runs'].values()) else 'blocked_diagnostics_or_error',finished_at=now())
    save(batch/'pilot_status.json',record)
    print(json.dumps(record,indent=2),flush=True)
    return 0 if record['status']=='ready_for_runtime_review' else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['pilot','fit','status'])
    parser.add_argument('--config',default='config/full_sample_sampling.json');parser.add_argument('--name')
    a=parser.parse_args();phase,c,cfg=configuration(a.config)
    if a.command=='status':
        out,_=paths(c)
        for e in entries(phase):
            p=out/'fits'/e['name']/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
            print(e['name'],s.get('status','not_started'),s.get('sampling_seconds',''))
        return
    if a.command=='fit':
        selected=[e for e in entries(phase) if e['name']==a.name]
        if len(selected)!=1:parser.error('--name must identify a configured pilot fit')
        raise SystemExit(fit_entry(phase,c,cfg,selected[0]))
    raise SystemExit(pilot(phase,c,cfg))


if __name__=='__main__':main()
