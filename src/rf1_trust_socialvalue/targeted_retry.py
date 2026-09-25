"""Explicit, fingerprint-bound retries without repeatedly replacing the same cache."""
import json
from pathlib import Path
import re
from .sampling_retry import archive_fit


def prepare_retries(root,checkpoint,plan):
    import pandas as pd
    if not re.fullmatch(r'[a-zA-Z0-9_]+',plan['id']):raise ValueError('Invalid retry plan ID')
    by_name={e['run']:dict(e) for e in checkpoint['runs']}
    prepared=[]
    for name,spec in plan['runs'].items():
        if name not in by_name or by_name[name]['kind']=='recovery':raise ValueError(f'Unknown real-data run: {name}')
        changes=spec['overrides']
        if set(changes)!={'adapt_delta','warmup','draws'}:raise ValueError('Retry may change only integrator target and iteration budgets')
        folder=root/'work/hierarchical'/name
        archive=folder.with_name(name+'_retry_'+plan['id'])
        source=archive if archive.exists() else folder
        saved=json.loads((source/'manifest.json').read_text())
        if saved['fingerprint']!=spec['expected_fingerprint']:raise RuntimeError(f'Retry baseline fingerprint differs: {name}')
        cfg=saved['settings'];settings=dict(cfg,**changes)
        if not cfg['adapt_delta']<settings['adapt_delta']<1 or settings['warmup']<cfg['warmup'] or settings['draws']<cfg['draws']:
            raise ValueError('Retry must tighten adapt_delta without reducing warmup or draws')
        if not archive.exists():
            diag=pd.read_csv(root/'results/tables'/f'diagnostics_{name}.csv')
            mixing=(diag.R_hat.lt(1.01)&diag.ESS_bulk.ge(400)&diag.ESS_tail.ge(400)).all()
            if diag.passed.all() or not mixing or not diag.divergences.eq(spec['expected_divergences']).all() or spec['expected_divergences']<=0 or not diag.max_depth_hits.eq(0).all() or not diag.min_bfmi.gt(.3).all():
                raise RuntimeError(f'Retry diagnostics no longer match the reviewed divergence-only failure: {name}')
        else:
            diag=None
            current=folder/'manifest.json'
            if current.exists():
                latest=json.loads(current.read_text())
                actual=latest['settings']
                if latest['fingerprint']==spec['expected_fingerprint'] or any(actual.get(k)!=v for k,v in settings.items() if k not in ('draws','warmup')) or actual['draws']<settings['draws'] or actual['warmup']<settings['warmup']:
                    raise RuntimeError(f'Existing retry does not match this plan: {name}')
        prepared.append((name,folder,archive,saved,diag,settings))
    # Preflight every target before moving any cache.
    for name,folder,archive,saved,diag,settings in prepared:
        if not archive.exists():
            actual=archive_fit(folder,saved,'_retry_'+plan['id'])
            if actual!=archive.resolve():raise RuntimeError('Unexpected retry archive path')
            diag['run']=archive.name
            diag.to_csv(root/'results/tables'/f'diagnostics_{archive.name}.csv',index=False)
            print(f'{name}: preserved reviewed attempt at {archive}; next settings={settings}',flush=True)
        else:print(f'{name}: retry already prepared; reusing its cache if present',flush=True)
        by_name[name]['settings']=settings
    return [by_name[e['run']] for e in checkpoint['runs']]



def audit_divergent_draws(root,plan):
    """Export population-parameter locations without copying raw posterior chains."""
    import numpy as np
    import pandas as pd
    table=root/'results/tables'
    for name,spec in plan['runs'].items():
        archive=root/'work/hierarchical'/(name+'_retry_'+plan['id'])
        saved=json.loads((archive/'manifest.json').read_text());cfg=saved['settings']
        frames=[]
        for chain,path in enumerate(saved['csv_files'],1):
            frame=pd.read_csv(path,comment='#',usecols=lambda c:c=='divergent__' or c.startswith(('mu.','tau.','beta.','Omega.','mu[','tau[','beta[','Omega[')))
            if len(frame)!=cfg['draws'] or frame.isna().any().any():raise RuntimeError(f'Incomplete retained-draw audit input: {path}')
            frame['chain']=chain;frame['retained_iteration']=np.arange(1,len(frame)+1);frames.append(frame)
        if len(frames)!=cfg['chains']:raise RuntimeError(f'Chain count differs in divergence audit: {name}')
        samples=pd.concat(frames,ignore_index=True)
        events=samples[samples.divergent__.eq(1)]
        if len(events)!=spec['expected_divergences']:raise RuntimeError(f'Divergence count differs from reviewed evidence: {name}')
        names=[c for c in samples if c not in ('chain','retained_iteration','divergent__')]
        rows=[];reference=[]
        for parameter in names:
            values=samples[parameter].to_numpy()
            reference.append(dict(run=name,parameter=parameter,mean=values.mean(),sd=values.std(ddof=1),
                ci_low=np.quantile(values,.025),ci_high=np.quantile(values,.975),divergences=len(events),accepted=False))
            for _,event in events.iterrows():
                rows.append(dict(run=name,chain=int(event.chain),retained_iteration=int(event.retained_iteration),parameter=parameter,
                    recorded_value=event[parameter],percentile=np.mean(values<=event[parameter]),
                    note='Recorded state associated with a divergent transition; not the numerical location along the failing trajectory'))
        pd.DataFrame(rows).to_csv(table/f'divergence_locations_{plan["id"]}_{name}.csv',index=False)
        pd.DataFrame(reference).to_csv(table/f'divergence_reference_{plan["id"]}_{name}.csv',index=False)
        print(f'{name}: exported {len(events)} divergent recorded state(s) across {len(names)} population parameters',flush=True)
