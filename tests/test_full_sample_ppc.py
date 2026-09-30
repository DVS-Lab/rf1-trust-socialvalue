import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample_ppc as ppc
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials


def test_ppc_scope_is_accepted_full_fits_only():
    p,phase,c=ppc.configuration()
    assert len(p['accepted_full_fits'])==9 and p['simulations']==200 and p['workers']==4
    assert 'Full_HPreference_base_noage_retry1' not in p['accepted_full_fits']
    assert all(name.startswith('Full_') for name in p['accepted_full_fits'])


def test_generated_choices_control_feedback_and_hidden_outcomes_do_not_affect_conditional_history():
    # First observed choice is zero; second has the same partner so feedback can
    # affect it. In a generative replicate, choose the positive option instead.
    a=np.array([[0,0,2,0,0,1,0,1],[0,0,2,1,1,0,0,1]],dtype=float)
    parameters=np.array([[.5,1.],[.5,1.]])
    u=np.zeros((2,2));conditional,replicated=ppc.trajectories(a,parameters,'H2',False,u)
    b=a.copy();b[0,5]=0
    changed,_=ppc.trajectories(b,parameters,'H2',False,u)
    np.testing.assert_array_equal(conditional,changed)
    assert np.all(replicated==1)
    high=s.zero_engine(a,parameters[0],2,0.,True,u[0])
    low=s.zero_engine(b,parameters[0],2,0.,True,u[0])
    assert high[0,4]==1 and high[1,1]>low[1,1]
    # If the simulated choice stays zero, the schedule is not revealed.
    zero=s.zero_engine(a,parameters[0],2,0.,True,np.ones(2))
    assert zero[0,2]==0 and zero[0,4]==0


def test_ppc_uses_equal_participant_weights_and_fixed_missingness():
    _,_,arrays=s.model_data(trials(2),'H2',False)
    natural=np.ones((20,2,2))*.5
    table,participants,indices=ppc.ppc_tables(arrays,natural,dict(model='H2',variant='base'),10,42)
    assert len(indices)==10 and table.choices.min()>0
    all_rows=table[(table.stratification=='all')&(table.metric=='high_choice')&(table.statistic=='mean')]
    assert set(all_rows.choices)=={6} and set(all_rows.participants)=={2}
    np.testing.assert_allclose(all_rows.observed,2/3)
    assert {'between_participant_sd','all_low_fraction','all_high_fraction'}.issubset(set(table.statistic))
    assert set(table[table.statistic!='mean'].history)=={'generative'}
    assert set(participants.partner)=={'friend','stranger','computer'}
    again,_,_=ppc.ppc_tables(arrays,natural,dict(model='H2',variant='base'),10,42)
    pd.testing.assert_frame_equal(table,again)
    assert 'trial_block' in set(table.stratification) and 'partner_contrast' in set(table.stratification)


def test_mean_checks_weight_people_not_choices():
    rows=ppc.summarize_cell(('generative','all','all','high_choice'),np.array([0.,1.]),np.array([[0.,0.],[1.,1.]]),101)
    mean=next(r for r in rows if r['statistic']=='mean')
    assert mean['observed']==.5 and mean['predicted_mean']==.5


def test_partner_contrasts_align_people_and_exclude_unmatched_subjects():
    a=dict(ids=['b','a','c'],observed=[.8,.6,1.],predicted=[[.8,.7],[.6,.5],[1.,1.]],counts=[2,2,2])
    b=dict(ids=['a','b'],observed=[.2,.3],predicted=[[.1,.1],[.2,.2]],counts=[3,4])
    rows=ppc.partner_contrasts({('generative','partner','friend','high_choice'):a,('generative','partner','computer','high_choice'):b})
    assert len(rows)==1 and rows[0]['participants']==2 and rows[0]['choices']==11
    assert rows[0]['observed']==pytest.approx(.45)
    assert rows[0]['predicted_mean']==pytest.approx(.5)


def test_ppc_parent_keeps_other_workers_running_after_failure(tmp_path,monkeypatch):
    p,phase,c=ppc.configuration();c=dict(c,output=str(tmp_path/'out'),work=str(tmp_path/'work'),_root=str(tmp_path))
    monkeypatch.setattr(s,'require_linux',lambda:None)
    monkeypatch.setattr(ppc.geometry,'saved_trials',lambda *a:(None,{}))
    monkeypatch.setattr(ppc.geometry,'source_drift_audit',lambda *a:{'status':'drift_detected'})
    commands=[]
    def worker(command,**kwargs):
        commands.append(command);return SimpleNamespace(returncode=1 if command[-1]==p['accepted_full_fits'][0] else 0)
    monkeypatch.setattr(ppc.subprocess,'run',worker)
    monkeypatch.setattr(ppc,'render',lambda *a:pytest.fail('Failed batch must not produce success plots'))
    with pytest.raises(RuntimeError,match='failed'):ppc.run(p,phase,c)
    assert len(commands)==9 and all('full_sample_ppc' in cmd[2] for cmd in commands)
    status=json.loads((Path(c['output'])/'hierarchical/ppc/status.json').read_text())
    assert status['status']=='error' and status['new_sampling'] is False


def test_ppc_worker_only_loads_hash_verified_existing_draws(tmp_path,monkeypatch):
    p,phase,c=ppc.configuration();p=dict(p,simulations=10)
    c=dict(c,_root=str(tmp_path),output=str(tmp_path/'out'),work=str(tmp_path/'work'))
    t=trials(2);data,meta,_=s.model_data(t,'H2',False)
    name='Full_H2_base_noage';entry=dict(name=name,model='H2',variant='base',subset='full')
    for file in s.SOURCES:
        f=tmp_path/file;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('synthetic implementation')
    s.save(tmp_path/'config/full_sample_ppc.json',p)
    out,work=s.paths(c);evidence=out/'fits'/name;cache=work/'fits'/name;cache.mkdir(parents=True)
    csv=cache/'posterior.csv';csv.write_text('synthetic posterior')
    settings=dict(draws=20,chains=4)
    payload=dict(data=data,settings=settings,sampler_seed=42,stan={file:s.fs.sha(tmp_path/file) for file in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
    fingerprint=s.digest(payload);s.save(evidence/'status.json',dict(status='complete',passed=True,fingerprint=fingerprint,settings=settings,sampler_seed=42))
    manifest=dict(fingerprint=fingerprint,meta=meta,csv_files=[str(csv)],posterior_sha256={str(csv):s.fs.sha(csv)})
    s.save(cache/'manifest.json',manifest)
    s.save(evidence/'manifest_summary.json',dict(manifest,posterior_sha256={csv.name:s.fs.sha(csv)}))
    monkeypatch.setattr(s,'require_linux',lambda:None)
    monkeypatch.setattr(ppc.geometry,'saved_trials',lambda *a:(t,{}))
    monkeypatch.setattr(ppc.review,'resolved_entries',lambda *a:[entry])
    monkeypatch.setattr(ppc,'implementation',lambda *a:{'synthetic':'hash'})
    monkeypatch.setattr(s,'models',lambda *a:pytest.fail('Post-fit checks must never compile or sample'))
    calls=[]
    def load(files):
        calls.append(files);return SimpleNamespace(stan_variable=lambda name:np.ones((20,2,2))*.5)
    monkeypatch.setattr(s,'load_chains',load)
    ppc.fit_entry(p,phase,c,name)
    state=json.loads((out/'ppc/fits'/name/'status.json').read_text())
    assert state['status']=='complete' and state['participants']==2 and state['new_sampling'] is False
    assert len(calls)==1 and state['posterior_fingerprint']==fingerprint
    csv.write_text('tampered posterior')
    with pytest.raises(ValueError,match='Raw posterior changed'):ppc.fit_entry(p,phase,c,name)
    assert len(calls)==1
