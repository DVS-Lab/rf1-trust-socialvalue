"""One evidence-bound retry of nine failed fits, without changing model targets."""
import json
from pathlib import Path

from . import full_sample_sampling as s

# (retained draws per chain, adapt_delta); every retry has eight chains and
# 4,000 warmup. More draws address slow mixing; adaptation addresses divergences.
PLAN = {
    'Full_H2_zero_noage': (8000, .99),
    'Full_H5_base_noage': (4000, .99),
    'Full_HPreference_base_noage': (4000, .995),
    'Train_H2_base_noage': (4000, .99),
    'Train_H2_zero_noage': (8000, .99),
    'Train_H5_base_noage': (4000, .995),
    'Train_H5_zero_noage': (4000, .99),
    'Train_H8_zero_noage': (4000, .99),
    'Train_HPreference_base_noage': (4000, .995),
}


def validate_configuration(phase,c,cfg):
    from . import full_sample_batch as batch
    root=Path(c['_root']);original=json.loads((root/'config/full_sample_batch.json').read_text())
    batch.validate_configuration(original,c,cfg)
    if phase['max_attempts']!=1 or phase['parallel_fits']!=9 or phase['cpu_budget']!=80:
        raise ValueError('Only one reviewed nine-fit retry batch is authorized')
    for key in ['gamma_population_mean_prior_sd','cmdstan_version','metric','cohort_config','integration_status_sha256']:
        if phase[key]!=original[key]:raise ValueError('Retry changed the accepted model target: '+key)
    all_entries=original['reuse']+batch.new_entries(original)
    expected_evidence={'config/full_sample_batch.json','results/full_sample/hierarchical/batch/status.json'}
    expected_evidence|={f"results/full_sample/hierarchical/fits/{e['name']}/{file}" for e in all_entries
                       for file in ['status.json','manifest_summary.json','diagnostics.tsv']}
    if set(phase['evidence_sha256'])!=expected_evidence:raise ValueError('Incomplete original-batch evidence')
    for file,h in phase['evidence_sha256'].items():
        if s.fs.sha(root/file)!=h:raise ValueError('Reviewed batch evidence changed: '+file)
    reused=[];failed={}
    for entry in all_entries:
        old=json.loads((root/'results/full_sample/hierarchical/fits'/entry['name']/'status.json').read_text())
        if old['status']=='complete' and old['passed']:reused.append(entry)
        elif old['status']=='diagnostic_failed' and not old['passed']:failed[entry['name']]=entry
        else:raise ValueError('Retry review does not cover interrupted/error fits')
    if set(failed)!=set(PLAN) or phase['reuse']!=reused:raise ValueError('Accepted/failed fit inventory changed')
    if len(phase['retries'])!=len(PLAN):raise ValueError('Retry set must contain exactly nine fits')
    sources=[]
    for entry in phase['retries']:
        source=entry['source_name'];sources.append(source)
        if source not in failed:raise ValueError('Only failed fits can be retried')
        draws,adapt=PLAN[source]
        expected=dict(failed[source],name=source+'_retry1',source_name=source,seed_name=source,
                      settings=dict(chains=8,warmup=4000,draws=draws,adapt_delta=adapt))
        if entry!=expected:raise ValueError('Unreviewed retry entry/settings: '+source)
    if set(sources)!=set(PLAN):raise ValueError('Duplicate or missing retry')


def verify_target(phase,c,payload):
    """Reconstruct the original fingerprint, allowing only reviewed sampler edits."""
    entry=phase['_entry'];old=json.loads((Path(c['output'])/'hierarchical/fits'/entry['source_name']/'status.json').read_text())
    reconstructed=dict(payload,settings=old['settings'])
    if s.digest(reconstructed)!=old['fingerprint']:
        raise ValueError('Retry data, priors, Stan source, version or seed differs from the original target')


def trace_evidence(fit,diagnostics,evidence):
    """Portable chain traces for diagnosing a failed retry without raw CSV transfer.

    Keep at most 200 regularly spaced draws per chain, plus every divergent draw;
    report chain means/SDs over ALL retained draws. These are diagnostic exports,
    never replacements for the full posterior or the convergence calculations.
    """
    import re
    import numpy as np
    import pandas as pd
    names=list(fit.column_names);draws=fit.draws(inc_warmup=False,concat_chains=False)
    selected=[]
    bad=diagnostics[(diagnostics.R_hat>=1.01)|(diagnostics.ESS_bulk<400)|(diagnostics.ESS_tail<400)]
    natural=set(bad[bad.parameter.str.startswith('natural[')].sort_values('R_hat',ascending=False).parameter.head(20))
    for i,name in enumerate(names):
        corr=re.fullmatch(r'Omega\[(\d+),(\d+)\]',name)
        if (name in {'lp__','energy__','divergent__','treedepth__','accept_stat__'} or name.startswith(('mu[','tau['))
                or (corr and int(corr[1])<int(corr[2])) or name in natural):selected.append(i)
    rows=[];summary=[];divergent=names.index('divergent__')
    for chain in range(draws.shape[1]):
        keep=np.unique(np.r_[np.linspace(0,len(draws)-1,min(200,len(draws))).astype(int),np.flatnonzero(draws[:,chain,divergent])])
        frame=pd.DataFrame(draws[keep,chain][:,selected],columns=[names[i] for i in selected])
        frame.insert(0,'draw',keep+1);frame.insert(0,'chain',chain+1);rows.append(frame)
        for i in selected:
            summary.append(dict(chain=chain+1,parameter=names[i],mean=float(draws[:,chain,i].mean()),sd=float(draws[:,chain,i].std())))
    pd.concat(rows,ignore_index=True).to_csv(evidence/'chain_trace_excerpt.tsv',sep='\t',index=False)
    pd.DataFrame(summary).to_csv(evidence/'chain_summary.tsv',sep='\t',index=False)
    return {file:s.fs.sha(evidence/file) for file in ['chain_trace_excerpt.tsv','chain_summary.tsv']}
