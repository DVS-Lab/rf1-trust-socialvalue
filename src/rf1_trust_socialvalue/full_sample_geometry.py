"""Export existing divergent-draw context on Linux2; never compile or sample."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from . import full_sample_sampling as s
from . import full_sample_batch as batch

TARGETS=['Train_H5_base_noage_retry1','Full_HPreference_base_noage_retry1']


def parameter_context(draws,names,meta):
    """Locations are saved iteration states, not the internal leapfrog failure point."""
    diverged=np.argwhere(draws[:,:,names.index('divergent__')]>0)
    rows=[]
    for column,name in enumerate(names):
        if not name.startswith(('mu[','tau[','L[','Omega[','z[','natural[')):continue
        x=draws[:,:,column].reshape(-1)
        if not np.isfinite(x).all():raise ValueError('Nonfinite retained parameter: '+name)
        mean=float(x.mean());sd=float(x.std());lo,median,hi=np.quantile(x,[.05,.5,.95])
        if sd==0:continue
        participant='';parameter=name
        natural=re.fullmatch(r'natural\[(\d+),(\d+)\]',name)
        latent=re.fullmatch(r'z\[(\d+),(\d+)\]',name)
        if natural:
            i,j=map(int,natural.groups());participant=meta['ids'][i-1];parameter=meta['parameter_names'][j-1]
        elif latent:
            j,i=map(int,latent.groups());participant=meta['ids'][i-1];parameter='latent_z_'+meta['parameter_names'][j-1]
        for draw,chain in diverged:
            value=float(draws[draw,chain,column])
            rows.append(dict(chain=int(chain+1),draw=int(draw+1),stan_parameter=name,participant_id=participant,parameter=parameter,
                value=value,posterior_mean=mean,posterior_sd=sd,q05=float(lo),median=float(median),q95=float(hi),
                standardized_value=(value-mean)/sd,empirical_percentile=float(np.mean(x<=value))))
    return pd.DataFrame(rows)


def draw_neighborhood(draws,names,radius=5):
    selected=[i for i,n in enumerate(names) if n in ['lp__','energy__','divergent__','treedepth__','accept_stat__','stepsize__','n_leapfrog__']]
    rows=[]
    for draw,chain in np.argwhere(draws[:,:,names.index('divergent__')]>0):
        for index in range(max(0,draw-radius),min(len(draws),draw+radius+1)):
            rows.append(dict(chain=int(chain+1),draw=int(index+1),offset=int(index-draw),**{names[i]:float(draws[index,chain,i]) for i in selected}))
    return pd.DataFrame(rows)



def saved_trials(phase,c):
    """Verify the historical analysis snapshot for export only, not a new fit.

    The pinned milestone authenticates provenance, which authenticates the saved
    canonical table. Existing posterior target fingerprints are checked by main.
    Live source drift is audited separately and never authorizes new sampling.
    """
    output=Path(c['output']);milestone=output/'milestone_status.json'
    if s.fs.sha(milestone)!=phase['integration_status_sha256']:
        raise ValueError('Historical integration gate changed')
    state=json.loads(milestone.read_text())
    if state['status']!='ready_for_review' or state['cohort_summary']['unexplained_parity_fields']:
        raise ValueError('Historical integration/parity gate has not passed')
    if 'provenance.json' not in state['output_hashes']:raise ValueError('Snapshot provenance is not authenticated')
    for file,h in state['output_hashes'].items():
        if s.fs.sha(output/file)!=h:raise ValueError('Historical integration output changed: '+file)
    prov=json.loads((output/'provenance.json').read_text())
    canonical=Path(c['work'])/'canonical_trials.tsv'
    if str(canonical) not in prov['artifact_hashes']:raise ValueError('Saved canonical table is not authenticated')
    config=str(Path(c['_config_path']))
    if config not in prov['input_hashes'] or s.fs.sha(config)!=prov['input_hashes'][config]:
        raise ValueError('Frozen cohort configuration changed')
    for file,h in {**prov['implementation_hashes'],**prov['artifact_hashes']}.items():
        if s.fs.sha(file)!=h:raise ValueError('Frozen analysis artifact changed: '+file)
    trials=pd.read_csv(canonical,sep='\t',dtype={'trial_id':str,'session':str})
    if (trials.participant_id.nunique()!=state['cohort_summary']['primary_n']
            or int(trials.valid_choice.sum())!=state['cohort_summary']['total_valid_choices']):
        raise ValueError('Saved model cohort differs from historical review')
    print('Historical analysis snapshot verified for existing-posterior export only.',flush=True)
    return trials,prov



def repair_10668_evidence(c,prov):
    """Read only the known repair marker and the two implicated archived files."""
    marker=Path(c['bids_root'])/'sub-10668/ses-01/.rf1-10668-repair.json'
    locations=[marker]
    if 'upstream_root' in c:
        locations.append(Path(c['upstream_root'])/'derivatives/source_repairs/10668-sharedreward-v1/receipt.json')
    records=[]
    for path in locations:
        row=dict(path=str(path),exists=path.is_file())
        if path.is_file():
            try:
                data=json.loads(path.read_text());row.update(sha256=s.fs.sha(path),repair_id=data.get('id'),status=data.get('status'))
                originals=data.get('original_files',data.get('original_session_sha256',{}));matches=[]
                for source,h in prov['input_hashes'].items():
                    name=Path(source).name
                    if not re.fullmatch(r'sub-10668_ses-01_task-trust_run-2_part-(?:mag|phase)_events\.tsv',name):continue
                    rel='func/'+name;item=dict(filename=name,frozen_sha256=h,receipt_original_sha256=originals.get(rel))
                    if path.name=='receipt.json':
                        archived=path.parent/'original_session'/rel
                        item['archive_exists']=archived.is_file()
                        item['archive_sha256']=s.fs.sha(archived) if archived.is_file() else None
                        item['archive_matches_freeze']=item['archive_sha256']==h
                    matches.append(item)
                row['implicated_originals']=matches
            except (OSError,ValueError,TypeError) as exc:row['read_error']=repr(exc)
        records.append(row)
    return records

def source_drift_audit(c,prov):
    """Record changed/missing/new live files without using them as model inputs."""
    before=prov['input_hashes'];paths=set(before)
    paths.update(str(p) for p in Path(c['bids_root']).glob('sub-*/ses-*/func/*task-trust*_events.tsv'))
    changed=[];errors=[]
    for name in sorted(paths):
        path=Path(name)
        try:current=s.fs.sha(path) if path.is_file() else None
        except OSError as exc:
            errors.append(dict(path=name,error=repr(exc)));continue
        if current==before.get(name):continue
        row=dict(path=name,frozen_sha256=before.get(name),current_sha256=current,
            change='added' if name not in before else 'missing' if current is None else 'modified',
            imaging_event_file=bool(s.fs.IMAGING_TEMPLATE_RE.fullmatch(path.name)))
        if row['imaging_event_file'] and current is not None:
            try:
                frame=s.fs.tsv(path);row.update(current_rows=len(frame),current_columns=list(frame.columns))
            except Exception as exc:row['inspection_error']=repr(exc)
        changed.append(row)
    return dict(status='audit_incomplete' if errors else 'drift_detected' if changed else 'matches_snapshot',
        changes=changed,read_errors=errors,repair_10668=repair_10668_evidence(c,prov),generated_at=s.now(),
        scope='Live files are audited only. This export uses the authenticated saved canonical table and existing posterior hashes. New fits still require the original strict live-input verification.')

def main():
    s.require_linux()
    phase,c,_=s.configuration('config/full_sample_batch_retry.json')
    # Loading existing CSVs only. No models(), sample(), or install/compile calls.
    out,work=s.paths(c);dest=out/'geometry_review';dest.mkdir(parents=True,exist_ok=True)
    record=dict(status='preparing',started_at=s.now(),analysis_git_sha=s.fs.git_sha(c['_root']),fits={},new_sampling=False,verification_mode='historical_snapshot')
    s.save(dest/'status.json',record)
    try:
        t,prov=saved_trials(phase,c)
        audit=source_drift_audit(c,prov);s.save(dest/'live_source_audit_before.json',audit)
        record.update(status='running',snapshot_provenance_sha256=s.fs.sha(Path(c['output'])/'provenance.json'),
            live_source_audit_before_sha256=s.fs.sha(dest/'live_source_audit_before.json'),live_source_status=audit['status'])
        s.save(dest/'status.json',record)
        print(f"Live-source audit: {audit['status']}; {len(audit['changes'])} changed files. Existing posteriors only.",flush=True)
        for name in TARGETS:
            entry=next(e for e in phase['retries'] if e['name']==name)
            evidence=out/'fits'/name;cache=work/'fits'/name
            status=json.loads((evidence/'status.json').read_text());portable=json.loads((evidence/'manifest_summary.json').read_text())
            manifest=json.loads((cache/'manifest.json').read_text())
            if status['status']!='diagnostic_failed' or status['divergences']!=1:raise ValueError('Expected reviewed one-divergence failure: '+name)
            if manifest['fingerprint']!=status['fingerprint'] or portable['fingerprint']!=status['fingerprint']:
                raise ValueError('Posterior/diagnostic fingerprint mismatch')
            data,meta,_=s.model_data(batch.entry_trials(t,entry),entry['model'],entry['variant']=='zero',phase['gamma_population_mean_prior_sd'])
            payload=dict(data=data,settings=status['settings'],sampler_seed=status['sampler_seed'],
                stan={p:s.fs.sha(Path(c['_root'])/p) for p in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
            if s.digest(payload)!=status['fingerprint'] or meta!=manifest['meta']:raise ValueError('Current target differs from saved posterior')
            if {Path(p).name:h for p,h in manifest['posterior_sha256'].items()}!=portable['posterior_sha256']:
                raise ValueError('Published posterior hashes differ from cache')
            for file in manifest['csv_files']:
                if s.fs.sha(file)!=manifest['posterior_sha256'][file]:raise ValueError('Raw posterior changed: '+file)
            fit=s.load_chains(manifest['csv_files']);draws=fit.draws(inc_warmup=False,concat_chains=False);names=list(fit.column_names)
            if int(draws[:,:,names.index('divergent__')].sum())!=1:raise ValueError('Divergence count differs from published result')
            folder=dest/name;folder.mkdir(parents=True,exist_ok=True)
            context=parameter_context(draws,names,meta);context.to_csv(folder/'divergent_parameter_context.tsv',sep='\t',index=False)
            draw_neighborhood(draws,names).to_csv(folder/'divergent_iteration_neighborhood.tsv',sep='\t',index=False)
            chain_rows=[]
            for chain in range(draws.shape[1]):
                energy=draws[:,chain,names.index('energy__')]
                chain_rows.append(dict(chain=chain+1,retained_draws=len(draws),divergences=int(draws[:,chain,names.index('divergent__')].sum()),
                    max_depth_hits=int((draws[:,chain,names.index('treedepth__')]>=status['settings']['max_treedepth']).sum()),
                    bfmi=float(np.mean(np.diff(energy)**2)/np.var(energy)),mean_acceptance=float(draws[:,chain,names.index('accept_stat__')].mean())))
            pd.DataFrame(chain_rows).to_csv(folder/'chain_diagnostics.tsv',sep='\t',index=False)
            info=dict(fingerprint=status['fingerprint'],status_sha256=s.fs.sha(evidence/'status.json'),manifest_sha256=s.fs.sha(evidence/'manifest_summary.json'),
                posterior_sha256=portable['posterior_sha256'],output_sha256={p.name:s.fs.sha(p) for p in folder.glob('*.tsv')},
                note='Descriptive context at saved divergent iteration state, not the internal leapfrog location. Extreme ranks among many parameters do not identify a causal geometry problem.')
            s.save(folder/'provenance.json',info);record['fits'][name]=info;s.save(dest/'status.json',record)
            print(name+': exported existing posterior context; no sampling',flush=True)
            del fit,draws
        saved_trials(phase,c)
        s.save(dest/'live_source_audit_after.json',source_drift_audit(c,prov))
        record.update(status='complete',finished_at=s.now(),live_source_audit_after_sha256=s.fs.sha(dest/'live_source_audit_after.json'));s.save(dest/'status.json',record)
    except BaseException as exc:
        record.update(status='error',error=repr(exc),finished_at=s.now());s.save(dest/'status.json',record);raise


if __name__=='__main__':main()
