import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample as fs
from rf1_trust_socialvalue import full_sample_sampling as sampling
from test_full_sample import events


def trials(n=3):
    return pd.concat([fs.load_run(events(),f'sub-{i}','01',1) for i in range(n)],ignore_index=True)


@pytest.mark.parametrize('model',sampling.MODELS)
@pytest.mark.parametrize('zero',[False,True])
def test_full_cohort_adapter_has_no_n111_dependency(model,zero):
    data,meta,arrays=sampling.model_data(trials(),model,zero)
    assert data['N']==3 and data['T']==9 and meta['n_presented']==12
    assert data['first']==[1,4,7] and data['last']==[3,6,9]
    assert data['feedback']==[0,1,1]*3
    assert data['outcome']==[0.,0.,1.]*3
    assert data['zero_option']==[1,0,0]*3
    assert data['A']==0 and data['age']==[[],[],[]]
    assert data['K']==len(sampling.SPECS[model][2])+int(zero)
    assert data['zero_term']==int(zero)
    if zero:assert meta['parameter_names'][-1]=='gamma0' and data['kind'][-1]==4
    assert all(len(a)==4 for a in arrays.values())


def test_hidden_scheduled_outcomes_do_not_enter_empirical_likelihood():
    t=trials();before,_,_=sampling.model_data(t,'H5',True)
    mask=~t.feedback_observed;t.loc[mask,'scheduled_reciprocation']=1-t.loc[mask,'scheduled_reciprocation']
    after,_,_=sampling.model_data(t,'H5',True)
    assert before==after


def test_full_fit_retains_both_runs_in_order_without_65_percent_split():
    t=trials(1);r=t.copy();r['run']=2
    data,meta,a=sampling.model_data(pd.concat([r,t],ignore_index=True),'H5',False)
    assert data['T']==6 and meta['n_presented']==8
    assert a['sub-0'][:,7].tolist()==[1]*4+[2]*4
    assert data['first']==[1] and data['last']==[6]


@pytest.mark.parametrize('problem',['duplicate','no_choices','invented_feedback','invented_outcome','wrong_outcome','session'])
def test_model_adapter_rejects_invalid_canonical_inputs(problem):
    t=trials(1)
    if problem=='duplicate':t=pd.concat([t,t.iloc[:1]],ignore_index=True)
    if problem=='no_choices':
        t['valid_choice']=False;t['feedback_observed']=False
        t['observed_reciprocation']=np.nan;t['chose_high']=np.nan
    if problem=='invented_feedback':t.loc[0,'feedback_observed']=True
    if problem=='invented_outcome':t.loc[0,'observed_reciprocation']=1
    if problem=='wrong_outcome':t.loc[1,'observed_reciprocation']=1
    if problem=='session':t['session']='02'
    with pytest.raises(ValueError):sampling.model_data(t,'H5',True)


def fake_fit(rhat=1.001,bulk=900,tail=800,divergent=False,depth=False,constant_energy=False):
    summary=pd.DataFrame({'R_hat':[rhat,np.nan],'ESS_bulk':[bulk,np.nan],'ESS_tail':[tail,np.nan]},index=['mu[1]','Omega[1,1]'])
    energy=np.tile([0.,1.,0.,1.,0.,1.],(4,1)).T
    if constant_energy:energy[:]=0.
    method={'energy__':energy,'divergent__':np.full((6,4),int(divergent)),'treedepth__':np.full((6,4),14 if depth else 4)}
    return SimpleNamespace(summary=lambda:summary,method_variables=lambda:method)


def test_diagnostics_enforce_all_prespecified_thresholds():
    c=json.loads(Path('config/full_sample_linux2.json').read_text());cfg=c['hierarchical'];a=cfg['acceptance']
    d,s=sampling.diagnostics(fake_fit(),cfg,a)
    assert s['passed'] and len(d)==1
    for kwargs in [{'rhat':1.01},{'rhat':np.nan},{'bulk':399},{'tail':399},{'divergent':True},{'depth':True},{'constant_energy':True}]:
        with np.errstate(divide='ignore',invalid='ignore'):
            _,s=sampling.diagnostics(fake_fit(**kwargs),cfg,a)
        assert not s['passed'],kwargs
        json.dumps(s,allow_nan=False)


def test_reviewed_gate_rejects_output_drift_before_loading_trials(tmp_path,monkeypatch):
    output=tmp_path/'out';output.mkdir();(output/'table.tsv').write_text('original')
    state={'status':'ready_for_review','cohort_summary':{'unexplained_parity_fields':0},'output_hashes':{'table.tsv':fs.sha(output/'table.tsv')}}
    fs.save_json(output/'milestone_status.json',state)
    phase={'integration_status_sha256':fs.sha(output/'milestone_status.json')}
    (output/'table.tsv').write_text('changed')
    monkeypatch.setattr(fs,'verify',lambda c:pytest.fail('Drift must be caught before loading the live cohort'))
    with pytest.raises(ValueError,match='Reviewed integration output changed'):
        sampling.integrated_trials(phase,{'output':str(output)})


def test_sampling_refuses_laptop(monkeypatch):
    monkeypatch.setattr(sampling.platform,'system',lambda:'Darwin')
    with pytest.raises(RuntimeError,match='Linux2 only'):sampling.require_linux()


def test_pilot_configuration_is_separate_from_frozen_cohort():
    phase,c,cfg=sampling.configuration()
    assert c['hierarchical']['launch_authorized'] is False
    assert phase['launch_authorized'] and phase['parallel_fits']==4
    assert cfg['chains']==4 and cfg['warmup']==2000 and cfg['draws']==2000
    assert len(sampling.entries(phase))==4
    assert phase['integration_status_sha256']==fs.sha(Path(c['_root'])/'results/full_sample/milestone_status.json')


def test_pilot_launches_correct_worker_module_and_saves_failures(tmp_path,monkeypatch):
    phase,_,cfg=sampling.configuration();phase=dict(phase)
    c={'output':str(tmp_path/'out'),'work':str(tmp_path/'work'),'_root':str(tmp_path)}
    monkeypatch.setattr(sampling,'require_linux',lambda:None)
    monkeypatch.setattr(sampling,'integrated_trials',lambda p,c:trials())
    monkeypatch.setattr(sampling,'cmdstan',lambda *a,**kw:None)
    monkeypatch.setattr(sampling,'check_implementation',lambda *a:None)
    monkeypatch.setattr(sampling.os,'sched_getaffinity',lambda _:set(range(96)),raising=False)
    commands=[]
    def worker(command,**kwargs):
        commands.append(command)
        name=command[-1];dest=Path(c['output'])/'hierarchical/fits'/name/'status.json'
        if 'H2_' in name:return SimpleNamespace(returncode=1)
        sampling.save(dest,dict(status='complete',passed=True))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(sampling.subprocess,'run',worker)
    assert sampling.pilot(phase,c,cfg)==2
    state=json.loads((Path(c['output'])/'hierarchical/pilot_status.json').read_text())
    assert state['maximum_active_chains']==16 and len(commands)==4
    assert all(cmd[2]=='rf1_trust_socialvalue.full_sample_sampling' for cmd in commands)
    bad=json.loads((Path(c['output'])/'hierarchical/fits/Full_H2_base_noage/status.json').read_text())
    assert bad['status']=='error'
    assert state['status']=='blocked_diagnostics_or_error'


def test_pilot_preparation_failure_cannot_leave_success_status(tmp_path,monkeypatch):
    phase,_,cfg=sampling.configuration()
    c={'output':str(tmp_path/'out'),'work':str(tmp_path/'work')}
    monkeypatch.setattr(sampling,'require_linux',lambda:None)
    def fail(*a):raise ValueError('changed cohort')
    monkeypatch.setattr(sampling,'integrated_trials',fail)
    out=Path(c['output'])/'hierarchical/pilot_status.json'
    sampling.save(out,{'status':'ready_for_runtime_review'})
    with pytest.raises(ValueError,match='changed cohort'):sampling.pilot(phase,c,cfg)
    assert json.loads(out.read_text())['status']=='error'


def test_completed_fit_reuses_draws_and_rejects_cache_drift(tmp_path,monkeypatch):
    phase,base,cfg=sampling.configuration();c=dict(base,output=str(tmp_path/'out'),work=str(tmp_path/'work'),_root=str(tmp_path))
    for file in sampling.SOURCES:
        p=tmp_path/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic implementation')
    monkeypatch.setattr(sampling,'require_linux',lambda:None)
    monkeypatch.setattr(sampling,'integrated_trials',lambda *a:trials())
    monkeypatch.setattr(sampling,'implementation_gate',lambda *a:None)
    monkeypatch.setattr(sampling,'implementation',lambda *a:{'synthetic':'hash'})
    monkeypatch.setattr(sampling,'cmdstan',lambda *a:None)
    monkeypatch.setattr(fs,'git_sha',lambda *a:'test')
    fit=fake_fit();fit.diagnose=lambda:'ok'
    fit.stan_variable=lambda name:np.ones((6,3,2))*.5 if name=='natural' else np.ones((6,2))*.5
    calls=[]
    def sample(**kwargs):
        calls.append(kwargs)
        files=[]
        for i in range(4):
            p=Path(kwargs['output_dir'])/f'chain-{i}.csv';p.write_text('synthetic posterior');files.append(str(p))
        fit.runset=SimpleNamespace(csv_files=files)
        return fit
    monkeypatch.setattr(sampling,'models',lambda *a:[SimpleNamespace(sample=sample)])
    monkeypatch.setattr(sampling,'load_chains',lambda *a:fit)
    entry=sampling.entries(phase)[0]
    assert sampling.fit_entry(phase,c,cfg,entry)==0
    assert sampling.fit_entry(phase,c,cfg,entry)==0
    assert len(calls)==1
    assert calls[0]['chains']==4 and calls[0]['parallel_chains']==4
    with pytest.raises(ValueError,match='target/settings differ'):
        sampling.fit_entry(phase,c,dict(cfg,draws=cfg['draws']+1),entry)
    Path(fit.runset.csv_files[0]).write_text('tampered')
    with pytest.raises(ValueError,match='Cached posterior changed'):
        sampling.fit_entry(phase,c,cfg,entry)
    assert len(calls)==1


def test_preference_retry_is_evidence_bound_and_preserves_seed_and_settings():
    phase,c,cfg=sampling.configuration('config/full_sample_preference_retry.json')
    _,_,original=sampling.configuration()
    assert cfg==dict(original,draws=4000)
    entry=sampling.entries(phase)[0]
    assert entry['name']=='Full_HPreference_zero_noage_draws4000'
    source=Path(c['output'])/'hierarchical/fits'/entry['seed_name']/'status.json'
    old=json.loads(source.read_text())
    assert sampling.stable_seed(cfg['seed'],entry['seed_name'])==old['sampler_seed']
    assert sampling.phase_output(phase,c)==Path(c['output'])/'hierarchical/retries'/entry['name']


@pytest.mark.parametrize('problem',['hash','attempts','draws','metric','prior','replacement'])
def test_unreviewed_retry_changes_are_rejected(problem):
    phase,c,_=sampling.configuration('config/full_sample_preference_retry.json')
    _,_,settings=sampling.configuration()
    phase=copy.deepcopy(phase)
    if problem=='hash':
        key=next(iter(phase['retry']['evidence_sha256']));phase['retry']['evidence_sha256'][key]='changed'
    if problem=='attempts':phase['retry']['max_attempts']=2
    if problem=='draws':phase['retry']['draws']=8000
    if problem=='metric':phase['metric']='dense_e'
    if problem=='prior':phase['gamma_population_mean_prior_sd']=2.
    if problem=='replacement':phase['retry']['replacement_run']='another_fit'
    with pytest.raises(ValueError):sampling.validate_retry(phase,c,settings)


def test_retry_preparation_preserves_original_pilot_status(tmp_path,monkeypatch):
    phase,_,cfg=sampling.configuration('config/full_sample_preference_retry.json')
    c={'output':str(tmp_path/'out'),'work':str(tmp_path/'work')}
    original=Path(c['output'])/'hierarchical/pilot_status.json'
    sampling.save(original,{'status':'blocked_diagnostics_or_error','original':True})
    before=original.read_bytes()
    monkeypatch.setattr(sampling,'require_linux',lambda:None)
    def fail(*a):raise RuntimeError('test preparation error')
    monkeypatch.setattr(sampling,'integrated_trials',fail)
    with pytest.raises(RuntimeError,match='test preparation error'):sampling.pilot(phase,c,cfg)
    assert original.read_bytes()==before
    assert json.loads((sampling.phase_output(phase,c)/'pilot_status.json').read_text())['status']=='error'


def test_retry_reconstructs_original_target_fingerprint(tmp_path):
    phase={'stage':'full_cohort_noage_retry','retry':{'source_run':'source'}}
    c={'output':str(tmp_path)}
    old_settings={'draws':2000,'chains':4}
    payload={'data':{'y':[1,0],'mu_scale':[1.]},'settings':dict(old_settings,draws=4000),'sampler_seed':123,'stan':{'model':'hash'},'cmdstan_version':'2.40.0'}
    source=tmp_path/'hierarchical/fits/source/status.json'
    sampling.save(source,{'settings':old_settings,'fingerprint':sampling.digest(dict(payload,settings=old_settings))})
    sampling.verify_retry_target(phase,c,payload)
    for changed in [dict(payload,data={'y':[0,0],'mu_scale':[1.]}),dict(payload,data={'y':[1,0],'mu_scale':[2.]}),dict(payload,sampler_seed=456),dict(payload,stan={'model':'changed'})]:
        with pytest.raises(ValueError,match='differ from original target'):
            sampling.verify_retry_target(phase,c,changed)
