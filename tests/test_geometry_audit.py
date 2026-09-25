import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue.geometry_audit import scan_chain,parameter_state,compare_attempts


def make_draws(path):
    rows=[]
    for i in range(3):
        row=dict(divergent__=int(i==2),energy__=i+1,lp__=-i-2,treedepth__=3,stepsize__=.1,accept_stat__=.99)
        for j in [1,2]:
            row[f'mu.{j}']=i+j;row[f'tau.{j}']=.5*j;row[f'beta.{j}.1']=.1*j
            for k in [1,2]:
                row[f'L.{j}.{k}']=float(j==k);row[f'z.{j}.{k}']=j*10+k
                row[f'Omega.{j}.{k}']=float(j==k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path,index=False)


def test_scan_keeps_previous_draw_across_chunk_boundary(tmp_path):
    path=tmp_path/'draws.csv';make_draws(path)
    frame,states=scan_chain(path,3,4,chunk_size=2)
    assert len(frame)==3 and [s['kind'] for s in states]==['flagged','previous']
    assert [s['iteration'] for s in states]==[3,2]
    assert all(s['chain']==4 for s in states)
    parameters=parameter_state(states[0]['row'],dict(K=2,N=2,A=1))
    assert parameters['mu']==[3,4] and parameters['z']==[[11,12],[21,22]]
    assert parameters['L']==[[1,0],[0,1]]


def test_scan_refuses_warmup_or_partial_chains(tmp_path):
    path=tmp_path/'draws.csv';make_draws(path)
    with pytest.raises(RuntimeError,match='Expected 4'):scan_chain(path,4,1)
    path.write_text('# save_warmup = 1\n'+path.read_text())
    with pytest.raises(RuntimeError,match='retained-only'):scan_chain(path,3,1)


def test_attempt_comparison_does_not_treat_constant_parameters_as_shifts():
    a=pd.DataFrame([dict(parameter='tau.1',mean=2.,sd=1.),dict(parameter='Omega.1.1',mean=1.,sd=0.)])
    b=pd.DataFrame([dict(parameter='tau.1',mean=1.,sd=2.),dict(parameter='Omega.1.1',mean=1.,sd=0.)])
    compared=compare_attempts(a,b).set_index('parameter')
    assert compared.loc['tau.1','mean_shift_baseline_sd']==.5
    assert np.isnan(compared.loc['Omega.1.1','mean_shift_baseline_sd'])


def test_full_audit_reads_cache_without_sampling_or_modifying_draws(monkeypatch,tmp_path):
    import platform
    import cmdstanpy
    from rf1_trust_socialvalue import hierarchical as h,linux_handoff as handoff,geometry_audit as audit
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(platform,'system',lambda:'Linux')
    folder=tmp_path/'work/hierarchical/H4_full_age';folder.mkdir(parents=True)
    tables=tmp_path/'results/tables';tables.mkdir(parents=True)
    stan=tmp_path/'stan';stan.mkdir();config=tmp_path/'config';config.mkdir()
    source='reference model';(stan/'hierarchical_shared.stan').write_text(source)
    (stan/'rl_fast.hpp').write_text('header');(stan/'hierarchical_fast.stan').write_text('fast model')
    data=dict(K=2,N=2,A=1);cfg=dict(chains=1,draws=3,max_treedepth=12)
    path=folder/'draws.csv';make_draws(path);original=path.read_bytes()
    fingerprint=hashlib.sha256((json.dumps(data,sort_keys=True)+json.dumps(cfg,sort_keys=True)+source).encode()).hexdigest()
    implementation=hashlib.sha256(b'headerfast model').hexdigest()
    (folder/'manifest.json').write_text(json.dumps(dict(fingerprint=fingerprint,settings=cfg,csv_files=[str(path)],implementation_sha256=implementation)))
    manifest=(folder/'manifest.json').read_bytes()
    entry=dict(run='H4_full_age',kind='real_data',model='H4',training=False,age_terms=1,bounded=False,prior_scale=1)
    (tmp_path/'results/hierarchical_checkpoint.json').write_text(json.dumps(dict(input_hashes={},runs=[entry])))
    (tmp_path/'results/linux_run_status.json').write_text(json.dumps({'runs':{'H4_full_age':{'status':'diagnostic_failed'}}}))
    pd.DataFrame([{'divergences':1}]).to_csv(tables/'diagnostics_H4_full_age.csv',index=False)
    calls=[]
    class FakeModel:
        def log_prob(self,**kwargs):
            calls.append(kwargs['params']);return pd.DataFrame([{'lp__':-1.,'g_mu1':.5}])
        def sample(self,**kwargs):pytest.fail('Audit must never start MCMC')
    monkeypatch.setattr(h,'stan_model',lambda:FakeModel())
    monkeypatch.setattr(h,'inputs',lambda *a,**k:(data,))
    monkeypatch.setattr(h,'WORK',folder.parent)
    monkeypatch.setattr(cmdstanpy,'CmdStanModel',lambda **k:FakeModel())
    monkeypatch.setattr(handoff,'existing_workers',lambda root:[])
    monkeypatch.setattr(handoff,'assert_no_live_fit_locks',lambda root:None)
    audit.main()
    assert len(calls)==4  # reference + fast at current + previous states
    assert path.read_bytes()==original and (folder/'manifest.json').read_bytes()==manifest
    output=tmp_path/'results/diagnostic_review'
    assert pd.read_csv(output/'reference_gradient_checks.csv').passed.all()
    assert (output/'H4_full_age.png').exists() and (output/'H4_full_age.pdf').exists()
