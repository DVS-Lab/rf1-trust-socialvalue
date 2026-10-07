"""Bounded Linux2 closeout: three unchanged-target retries and hierarchical age."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from . import full_sample_amount as a
from . import full_sample_sampling as s
from . import full_sample_residuals as residuals

RETRIES = {'Train_H7_zero_bias_v1': 1, 'Train_H7_zero_amount_v1': 3,
           'Train_HPreference_zero_amount_v1': 4}
ACCEPTED = 'Train_HPreference_zero_bias_v1'


def configuration():
    old, phase, c = a.configuration()
    cfg = json.loads((Path(c['_root'])/'config/full_sample_closeout.json').read_text())
    if (cfg['stage'] != 'full_cohort_closeout_v1' or cfg['automatic_retries']
            or cfg['chains'] != 8 or cfg['parallel_fits'] != 8 or cfg['cpu_budget'] != 64
            or cfg['age_center'] != 50 or cfg['age_scale'] != 20
            or cfg['recovery_replicates'] != 8 or cfg['age_prior_sd'] != .5 or cfg['wide_age_prior_sd'] != 1.
            or cfg['warmup'] != 6000 or cfg['draws'] != 4000 or cfg['adapt_delta'] != .9995
            or cfg['max_treedepth'] != 16):
        raise ValueError('Unreviewed closeout scope')
    for file, h in cfg['evidence_sha256'].items():
        if s.fs.sha(Path(c['_root'])/file) != h:
            raise ValueError('Reviewed amount evidence changed: '+file)
    return cfg, old, phase, c


def entries():
    rows = []
    for name in RETRIES:
        _, model, _, extension, _ = name.split('_')
        rows.append(dict(name=name+'_retry1', source=name, model=model, extension=extension,
                         subset='train', age=False, prior_sd=.5))
    for subset, age, suffix, prior in [('train', True, 'age', .5), ('full', False, 'noage', .5),
                                     ('full', True, 'age', .5), ('sensitivity', True, 'age', .5),
                                     ('full', True, 'age_wideprior', 1.)]:
        rows.append(dict(name=f'{subset.title()}_HPreference_zero_bias_{suffix}_closeout1',
                         model='HPreference', extension='bias', subset=subset, age=age, prior_sd=prior))
    return rows


def paths(c):
    out, work = s.paths(c)
    return out/'closeout', work/'closeout'


def sources(c):
    root = Path(c['_root'])
    files = ['config/full_sample_closeout.json', 'stan/full_closeout_fast.stan',
             'stan/full_closeout_reference.stan', 'src/rf1_trust_socialvalue/full_sample_closeout.py',
             'src/rf1_trust_socialvalue/full_sample_closeout_checks.py']
    return {**a.sources(c), **{f: s.fs.sha(root/f) for f in files}}


def subset(t, entry):
    if entry['subset'] == 'train':
        return a.batch.entry_trials(t, dict(subset='train'))
    if entry['subset'] == 'sensitivity':
        if not set(t.include_sensitivity.unique()).issubset({True, False}):
            raise ValueError('Invalid frozen sensitivity flags')
        return t[t.include_sensitivity.astype(bool)].copy()
    if entry['subset'] != 'full':
        raise ValueError('Unknown cohort')
    return t.copy()


def model_data(t, entry, phase, cfg):
    data, meta, arrays = a.model_data(t, entry, phase)
    if 'source' in entry:
        return data, meta, arrays
    ages = residuals.ages_from_trials(t, meta['ids'])
    if np.any((ages < 20) | (ages > 89)):
        raise ValueError('Age outside frozen cohort range')
    data.update(A=int(entry['age']), age_prior_sd=entry['prior_sd'],
                age=(((ages-cfg['age_center'])/cfg['age_scale'])[:, None].tolist()
                     if entry['age'] else [[] for _ in ages]))
    meta.update(age_terms=data['A'], age_years=ages.tolist(), age_center=cfg['age_center'],
                age_scale=cfg['age_scale'], age_prior_sd=entry['prior_sd'])
    return data, meta, arrays


def diagnostics(fit, cfg, c):
    # Include ALL new population slopes in the unchanged acceptance gate.
    frame = fit.summary()
    selected = frame[frame.index.str.startswith(('mu[', 'tau[', 'mu_ext[', 'tau_ext[',
                                                  'beta[', 'beta_ext[', 'Omega[', 'natural['))]
    renamed = selected.copy()
    renamed.index = ['mu['+n+']' if n.startswith(('beta[', 'beta_ext[')) else n for n in selected.index]
    _, info = a.diagnostics(SimpleNamespace(summary=lambda: renamed, method_variables=fit.method_variables), cfg, c)
    selected = selected[~selected.index.str.match(r'Omega\[(\d+),\1\]')]
    return selected.rename_axis('parameter').reset_index(), info


def old_posterior(entry, old, phase, c, t):
    out, work = a.paths(c); name = entry.get('source', entry['name'])
    published = out/'fits'/name; cache = work/'fits'/name
    status = json.loads((published/'status.json').read_text())
    manifest = json.loads((cache/'manifest.json').read_text())
    portable = json.loads((published/'manifest_summary.json').read_text())
    original = next(e for e in a.entries(old) if e['name'] == name)
    data, meta, _ = a.model_data(subset(t, original), original, phase)
    fingerprint = s.digest(dict(data=data, config=old, sources=a.sources(c)))
    if any(x['fingerprint'] != fingerprint for x in [status, manifest, portable]) or manifest['meta'] != meta:
        raise ValueError('Original target/participant identity changed: '+name)
    if {Path(f).name:h for f,h in manifest['posterior_sha256'].items()} != portable['posterior_sha256']:
        raise ValueError('Published and cached posterior hashes differ')
    if set(manifest['csv_files']) != set(manifest['posterior_sha256']) or len(manifest['csv_files']) != old['chains']:
        raise ValueError('Incomplete original chain inventory')
    for file, h in manifest['posterior_sha256'].items():
        if s.fs.sha(file) != h:
            raise ValueError('Original posterior changed: '+file)
    fit = s.load_chains(manifest['csv_files'])
    _, info = a.diagnostics(fit, old, c)
    for key, value in info.items():
        if status[key] != value:
            raise ValueError('Recomputed diagnostic disagrees: '+key)
    if name in RETRIES:
        limits = c['hierarchical']['acceptance']
        if (status['status'] != 'diagnostic_failed' or info['passed'] or info['divergences'] != RETRIES[name]
                or info['max_depth_hits'] or info['max_rhat'] >= limits['rhat_less_than']
                or info['min_bulk_ess'] < limits['minimum_bulk_ess'] or info['min_tail_ess'] < limits['minimum_tail_ess']
                or info['min_bfmi'] <= limits['bfmi_greater_than']):
            raise ValueError('Retry rationale requires only the reviewed rare-divergence failure')
    elif not info['passed'] or status['status'] != 'complete':
        raise ValueError('Accepted reference no longer passes')
    return fit, data, meta, manifest


def export_geometry(fit, meta, name, c):
    dest = paths(c)[0]/'geometry'/name; dest.mkdir(parents=True, exist_ok=True)
    draws = fit.draws(inc_warmup=False, concat_chains=False); names = list(fit.column_names)
    # Reuse the original mapper for base parameters; explicitly include extension latents/hypers.
    context = a.ppc.geometry.parameter_context(draws, names, meta)
    extension_names = [i for i, n in enumerate(names) if n.startswith(('mu_ext[', 'tau_ext[', 'z_ext['))]
    divergences = np.argwhere(draws[:, :, names.index('divergent__')] > 0); rows = []
    for col in extension_names:
        name_col = names[col]; x = draws[:, :, col].ravel(); sd = x.std()
        if not np.isfinite(x).all(): raise ValueError('Nonfinite extension posterior')
        if not sd: continue
        participant = ''; parameter = name_col; match = re.fullmatch(r'z_ext\[(\d+),(\d+)\]', name_col)
        if match:
            j, i = map(int, match.groups()); participant = meta['ids'][i-1]; parameter = 'latent_z_'+meta['parameter_names'][5+j-1]
        lo, med, hi = np.quantile(x, [.05, .5, .95])
        for draw, chain in divergences:
            value = draws[draw, chain, col]
            rows.append(dict(chain=int(chain+1), draw=int(draw+1), stan_parameter=name_col,
                             participant_id=participant, parameter=parameter, value=value,
                             posterior_mean=x.mean(), posterior_sd=sd, q05=lo, median=med, q95=hi,
                             standardized_value=(value-x.mean())/sd, empirical_percentile=np.mean(x<=value)))
    pd.concat([context, pd.DataFrame(rows)], ignore_index=True).to_csv(dest/'parameter_context.tsv', sep='\t', index=False)
    a.ppc.geometry.draw_neighborhood(draws, names).to_csv(dest/'iteration_neighborhood.tsv', sep='\t', index=False)
    s.save(dest/'status.json', dict(status='complete', divergences=len(divergences),
        note='Saved iteration states, not internal leapfrog failure locations. Extreme ranks are descriptive, not causal.',
        output_sha256={p.name:s.fs.sha(p) for p in dest.glob('*.tsv')}))


def models(phase, c):
    from cmdstanpy import CmdStanModel
    s.cmdstan(phase, c); build = paths(c)[1]/'build'; build.mkdir(parents=True, exist_ok=True)
    root = Path(c['_root']); result = []
    for variant in ['fast', 'reference']:
        target = build/f'closeout_{variant}.stan'
        content = (root/f'stan/full_closeout_{variant}.stan').read_text()+'\n// header '+s.fs.sha(root/'stan/full_amount_fast.hpp')+'\n'
        if not target.exists() or target.read_text() != content: target.write_text(content)
        options = dict(user_header=str(root/'stan/full_amount_fast.hpp'), stanc_options={'allow-undefined':True}) if variant == 'fast' else {}
        result.append(CmdStanModel(stan_file=str(target), **options))
    return result


def preflight(cfg, old, phase, c, t):
    from .full_sample_closeout_checks import prior_screen
    out, _ = paths(c); out.mkdir(parents=True, exist_ok=True)
    a.implementation_gate(c)
    for entry in entries():
        if 'source' in entry:
            fit, _, meta, _ = old_posterior(entry, old, phase, c, t)
            export_geometry(fit, meta, entry['source'], c); del fit
    fit, _, _, _ = old_posterior(dict(name=ACCEPTED), old, phase, c, t); del fit
    fast, ref = models(phase, c); original = a.models(phase, c, True)[1]
    small = subset(t, dict(subset='train')); small = small[small.participant_id.isin(sorted(small.participant_id.unique())[:4])]
    rng = np.random.default_rng(cfg['seed']); rows = []
    for age in [False, True]:
        entry = dict(model='HPreference', extension='bias', age=age, prior_sd=.5)
        data, meta, arrays = model_data(small, entry, phase, cfg); n=data['N']; k=data['K']; e=data['E']; A=data['A']
        for effect in [-.4, 0., .4]:
            pars = dict(mu=data['mu_location'], tau=[.6]*k, L=np.linalg.cholesky(.9*np.eye(k)+.1*np.ones((k,k))).tolist(),
                        z=rng.normal(0,.4,(k,n)).tolist(), beta=np.full((k,A),effect).tolist(),
                        mu_ext=[.2]*e, tau_ext=[.6]*e, z_ext=rng.normal(0,.4,(e,n)).tolist(), beta_ext=np.full((e,A),-effect).tolist())
            lp = ref.log_prob(params=pars, data=data, jacobian=True, sig_figs=16).iloc[0]
            actual = fast.log_prob(params=pars, data=data, jacobian=True, sig_figs=16).iloc[0]
            if list(lp.index) != list(actual.index): raise ValueError('Gradient coordinate mismatch')
            error = float(np.max(np.abs(lp-actual)/(1+np.abs(lp))))
            prior = float(ref.log_prob(params=pars, data=dict(data, prior_only=1), jacobian=True, sig_figs=16).iloc[0,0])
            x = np.asarray(data['age']).reshape(n,A)
            eta = np.asarray(pars['mu']) + x@np.asarray(pars['beta']).T + (np.diag(pars['tau'])@pars['L']@np.asarray(pars['z'])).T
            ext = np.asarray(pars['mu_ext']) + x@np.asarray(pars['beta_ext']).T + np.asarray(pars['tau_ext'])*np.asarray(pars['z_ext']).T
            natural = np.c_[s.transform(eta, data['kind']), ext]
            pyll = sum(-a.engine(arr, natural[i],9)[:,1].sum() for i,arr in enumerate(arrays.values()))
            llerror = abs(lp.iloc[0]-prior-pyll); nesting = 0.
            if not age:
                original_data, _, _ = a.model_data(small, entry, phase)
                original_pars = {key:val for key,val in pars.items() if key != 'beta_ext'}
                previous = original.log_prob(params=original_pars, data=original_data, jacobian=True, sig_figs=16).iloc[0,0]
                nesting = abs(lp.iloc[0]-previous)
            rows.append(dict(age=age, effect=effect, gradient_error=error, likelihood_error=llerror,
                             noage_target_nesting_error=nesting, passed=bool(error<1e-9 and llerror<1e-8 and nesting<1e-8)))
    table=pd.DataFrame(rows); table.to_csv(out/'implementation_checks.tsv', sep='\t', index=False)
    if not table.passed.all(): raise ValueError('Age likelihood/gradient/nesting check failed')
    prior_screen(t, cfg, phase, c)
    s.save(out/'implementation_status.json', dict(status='passed', sources=sources(c),
        executable_sha256=s.fs.sha(fast.exe_file), checks_sha256=s.fs.sha(out/'implementation_checks.tsv'),
        prior_screen_sha256=s.fs.sha(out/'joint_prior_screen.tsv')))
    print('Age fast/reference, Python likelihood and full no-age target nesting passed; joint prior screen saved.', flush=True)


def gate(c):
    out, work = paths(c); state=json.loads((out/'implementation_status.json').read_text())
    if (state['status']!='passed' or state['sources']!=sources(c)
            or state['executable_sha256']!=s.fs.sha(work/'build/closeout_fast')
            or state['checks_sha256']!=s.fs.sha(out/'implementation_checks.tsv')
            or state['prior_screen_sha256']!=s.fs.sha(out/'joint_prior_screen.tsv')):
        raise ValueError('Stale closeout implementation gate')
    a.implementation_gate(c)


def authenticated_cache(folder, fingerprint):
    manifest = folder/'manifest.json'
    if not manifest.exists():
        if list(folder.glob('*.csv')): raise RuntimeError('Partial chains exist; inspect before restarting: '+str(folder))
        return None
    saved=json.loads(manifest.read_text())
    if saved['fingerprint']!=fingerprint: raise ValueError('Existing target differs; cache preserved')
    if set(saved['csv_files'])!=set(saved['posterior_sha256']) or len(saved['csv_files'])!=saved['settings']['chains']:
        raise ValueError('Incomplete chain inventory')
    for file,h in saved['posterior_sha256'].items():
        if s.fs.sha(file)!=h: raise ValueError('Cached posterior changed: '+file)
    return saved


def fit_entry(cfg, old, phase, c, entry, recovery=None):
    from cmdstanpy import CmdStanModel
    from . import full_sample_closeout_checks as checks
    s.require_linux(); out,work=paths(c); name=entry['name']; folder=work/'fits'/name; dest=out/'fits'/name
    folder.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); state=dict(name=name,entry=entry,status='preparing',started_at=s.now());s.save(dest/'status.json',state)
    try:
        t,audit=a.snapshot(old,phase,c);s.save(dest/'live_source_audit_before.json',audit);gate(c)
        selected=subset(t,entry)
        if recovery is not None:
            selected, truth, anchor = checks.synthetic_trials(selected, recovery, cfg, old, phase, c)
            s.save(dest/'recovery_truth.json', truth)
        else: truth=None;anchor=None
        data,meta,arrays=model_data(selected,entry,phase,cfg)
        expected=304 if entry['subset']=='train' else 338 if entry['subset']=='sensitivity' else 343
        if data['N']!=expected: raise ValueError('Closeout cohort size differs')
        if 'source' in entry:
            original,old_data,_,_=old_posterior(entry,old,phase,c,t);del original
            if data!=old_data: raise ValueError('Retry likelihood data/priors differ')
        fingerprint=s.digest(dict(data=data,meta=meta,entry=entry,config=cfg,sources=sources(c),recovery_anchor=anchor))
        seed=s.stable_seed(cfg['seed'],name)
        if 'source' in entry:
            seed=s.stable_seed(old['seed'],entry['source']) # same seed; adaptation is the sole target-independent change
        saved=authenticated_cache(folder,fingerprint)
        state.update(status='running',fingerprint=fingerprint,sampler_seed=seed,settings=cfg);s.save(dest/'status.json',state)
        if saved:
            posterior=s.load_chains(saved['csv_files']); state['reused_posterior']=True
        else:
            s.cmdstan(phase,c)
            exe=a.paths(c)[1]/'build/amount_fast' if 'source' in entry else work/'build/closeout_fast'
            posterior=CmdStanModel(exe_file=str(exe)).sample(data=data,chains=cfg['chains'],parallel_chains=cfg['chains'],
                iter_warmup=cfg['warmup'],iter_sampling=cfg['draws'],seed=seed,adapt_delta=cfg['adapt_delta'],
                max_treedepth=cfg['max_treedepth'],metric=cfg['metric'],inits=cfg['inits'],sig_figs=cfg['sig_figs'],
                output_dir=str(folder),refresh=200,show_progress=False,show_console=True)
            saved=dict(fingerprint=fingerprint,meta=meta,settings=cfg,entry=entry,sampler_seed=seed,
                       csv_files=posterior.runset.csv_files,posterior_sha256={f:s.fs.sha(f) for f in posterior.runset.csv_files},
                       source_hashes=sources(c),sampling_seconds=time.monotonic()-start,recovery_anchor=anchor)
            s.save(folder/'manifest.json',saved);state['reused_posterior']=False
        _,audit=a.snapshot(old,phase,c);s.save(dest/'live_source_audit_after.json',audit);gate(c)
        diagnostic,info=diagnostics(posterior,cfg,c);diagnostic.to_csv(dest/'diagnostics.tsv',sep='\t',index=False)
        state.update(info,sampling_seconds=saved['sampling_seconds']);(dest/'diagnose.txt').write_text(posterior.diagnose())
        if not info['passed']:
            state['status']='diagnostic_failed';return 2
        natural=posterior.stan_variable('natural')
        if recovery is not None:
            checks.recovery_tables(posterior,natural,truth,meta,dest)
        else:
            if entry['subset']=='train':
                scores,cells=a.heldout(t,entry,natural,meta)
                scores.to_csv(dest/'heldout_participants.tsv',sep='\t',index=False);cells.to_csv(dest/'heldout_cells.tsv',sep='\t',index=False)
            else:
                checks.predictive_tables(arrays,meta,natural,entry,cfg,dest)
            checks.parameter_tables(posterior,natural,data,meta,entry,cfg,dest)
        gate(c)
        _,audit=a.snapshot(old,phase,c);s.save(dest/'live_source_audit_after.json',audit)
        state['status']='complete';return 0
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        state.update(finished_at=s.now(),elapsed_seconds=time.monotonic()-start,output_sha256={p.name:s.fs.sha(p) for p in dest.glob('*.tsv')})
        if (dest/'recovery_truth.json').exists():state['truth_sha256']=s.fs.sha(dest/'recovery_truth.json')
        if (folder/'manifest.json').exists():
            m=json.loads((folder/'manifest.json').read_text())
            s.save(dest/'manifest_summary.json',dict(m,csv_files=[Path(p).name for p in m['csv_files']],posterior_sha256={Path(p).name:h for p,h in m['posterior_sha256'].items()}))
        for p in folder.glob('*.txt'):s.write_tail(p,dest/p.name)
        s.save(dest/'status.json',state);print(name+': '+state['status'],flush=True)


def recovery_entries():
    return [dict(name=f'Recovery_{"null" if i<4 else "effect"}_{i%4+1}_closeout1',model='HPreference',
                 extension='bias',subset='full',age=True,prior_sd=.5,recovery=i) for i in range(8)]


def run_batch(stage,cfg,c):
    out,_=paths(c);state=dict(status='running',started_at=s.now(),runs={});available=len(os.sched_getaffinity(0))
    workers=min(cfg['parallel_fits'],min(cfg['cpu_budget'],available)//cfg['chains'])
    if workers<1:raise RuntimeError('Eight CPUs required')
    state.update(workers=workers,active_chains=workers*cfg['chains']);record=out/f'{stage}_status.json';s.save(record,state)
    tasks=entries() if stage=='analysis' else recovery_entries()
    print(f'{stage}: {len(tasks)} fits; {workers} concurrent x 8 chains = {workers*8} active cores.',flush=True)
    def launch(entry):
        folder=out/'fits'/entry['name'];folder.mkdir(parents=True,exist_ok=True)
        with (folder/'console.txt').open('a',buffering=1) as log:
            log.write('\nCLOSEOUT WORKER '+s.now()+'\n')
            return subprocess.run([sys.executable,'-m','rf1_trust_socialvalue.full_sample_closeout','fit','--name',entry['name']],cwd=c['_root'],stdout=log,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(launch,e):e['name'] for e in tasks}
        for future in as_completed(pending):
            name=pending[future];state['runs'][name]=future.result();s.save(record,state);print(name+': exit '+str(state['runs'][name]),flush=True)
    state.update(status='complete' if not any(state['runs'].values()) else 'incomplete',finished_at=s.now());s.save(record,state)
    return any(state['runs'].values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['all','analysis','recovery','report','fit']);parser.add_argument('--name');args=parser.parse_args()
    cfg,old,phase,c=configuration();s.require_linux()
    if args.command=='fit':
        entry=next(e for e in entries()+recovery_entries() if e['name']==args.name)
        sys.exit(fit_entry(cfg,old,phase,c,entry,entry.get('recovery')))
    from . import full_sample_closeout_checks as checks
    import fcntl
    out,work=paths(c);out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    with (work/'batch.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Closeout already running')
        state=dict(status='running',started_at=s.now(),command=args.command);s.save(out/'status.json',state)
        try:
            t,audit=a.snapshot(old,phase,c);s.save(out/'live_source_audit.json',audit)
            failed=False
            if args.command in ['all','analysis']:
                preflight(cfg,old,phase,c,t);failed=run_batch('analysis',cfg,c)
            if args.command in ['all','recovery']:
                anchor=out/'fits/Full_HPreference_zero_bias_age_closeout1/status.json'
                if anchor.exists() and json.loads(anchor.read_text())['status']=='complete':
                    gate(c);failed=run_batch('recovery',cfg,c) or failed
                else:
                    failed=True;s.save(out/'recovery_status.json',dict(status='blocked',reason='Accepted full age fit required; no simulated fits launched'))
            checks.report(cfg,c)
            state.update(status='needs_review' if failed else 'complete',finished_at=s.now(),
                         source_hashes=sources(c),output_sha256={p.name:s.fs.sha(p) for p in out.iterdir() if p.suffix in ['tsv','png','pdf','md']})
            s.save(out/'status.json',state)
            if failed:sys.exit(2)
        except Exception as exc:
            state.update(status='error',error=repr(exc),finished_at=s.now());s.save(out/'status.json',state);raise


if __name__=='__main__':main()
