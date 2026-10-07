import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample_closeout as q
from rf1_trust_socialvalue import full_sample_closeout_checks as ch
from rf1_trust_socialvalue import full_sample_amount as a
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials, fake_fit


def age_trials(n=3):
    t=trials(n);t['age']=t.participant_id.map({f'sub-{i}':20+20*i for i in range(n)})
    t['include_sensitivity']=t.participant_id.ne('sub-0')
    return t


def test_closeout_scope_and_immutable_old_sources():
    cfg,old,phase,c=q.configuration()
    assert len(q.entries())==8 and len(q.recovery_entries())==8
    assert cfg['parallel_fits']*cfg['chains']==64
    assert len([e for e in q.entries() if 'source' in e])==3
    manifest=json.loads(Path('results/full_sample/hierarchical/amount_comparison/fits/'+q.ACCEPTED+'/manifest_summary.json').read_text())
    assert a.sources(c)==manifest['source_hashes']
    assert all(e['model']=='HPreference' and e['age'] for e in q.recovery_entries())


@pytest.mark.parametrize('age',[False,True])
def test_age_adapter_aligns_subjects_and_adds_no_choice_information(age):
    cfg,_,phase,_=q.configuration();t=age_trials().sample(frac=1,random_state=3)
    entry=dict(model='HPreference',extension='bias',age=age,prior_sd=.5)
    data,meta,_=q.model_data(t,entry,phase,cfg);original,_,_=a.model_data(t,entry,phase)
    assert {k:v for k,v in data.items() if k not in ['A','age','age_prior_sd']}=={k:v for k,v in original.items() if k not in ['A','age']}
    assert meta['age_years']==[20.,40.,60.]
    assert data['age']==([[-1.5],[-.5],[.5]] if age else [[],[],[]])
    assert data['A']==int(age) and meta['age_terms']==int(age)
    bad=t.copy();bad.loc[bad.index[0],'age']=np.nan
    with pytest.raises(ValueError):q.model_data(bad,entry,phase,cfg)


def test_retry_data_is_bitwise_original_and_sensitivity_uses_frozen_flag():
    cfg,_,phase,_=q.configuration();entry=q.entries()[0];t=age_trials()
    assert q.model_data(t,entry,phase,cfg)[0]==a.model_data(t,entry,phase)[0]
    assert set(q.subset(t,dict(subset='sensitivity')).participant_id)=={'sub-1','sub-2'}
    t['include_sensitivity']='False'
    with pytest.raises(ValueError):q.subset(t,dict(subset='sensitivity'))


def test_age_and_extension_slopes_cannot_escape_diagnostic_gate():
    cfg,_,_,c=q.configuration()
    for name in ['beta[1,1]','beta_ext[1,1]']:
        base=fake_fit();summary=base.summary().copy()
        summary.loc[name]=[1.1,1000,1000]
        fit=SimpleNamespace(summary=lambda:summary,method_variables=base.method_variables)
        table,info=q.diagnostics(fit,cfg,c)
        assert name in set(table.parameter) and not info['passed']
    _,info=q.diagnostics(fake_fit(),cfg,c);assert info['passed']
    _,info=q.diagnostics(fake_fit(divergent=True),cfg,c);assert not info['passed']


def test_simulation_matches_independent_stepwise_likelihood_and_own_feedback():
    arr=np.array([[0,0,2,0,0,1,0,1],[0,0,4,-1,0,1,0,1],[0,0,4,0,0,0,0,2],[0,2,4,1,1,1,0,2]],float)
    par=np.array([.5,1.,.8,.4,.3,.1,-.2]);u=np.array([0.,0.,.999,0.])
    for code in [7,9]:
        generated=ch.simulate(arr,par,code,u)
        probabilities=a.engine(generated,par,code)[:,0]
        valid=arr[:,3]>=0
        np.testing.assert_array_equal(generated[valid,3],u[valid]<probabilities[valid])
        assert generated[0,4]==1 and generated[2,4]==0
        assert generated[1,3]==-1 and generated[1,4]==0
        np.testing.assert_array_equal(arr[:,[0,1,2,5,6,7]],generated[:,[0,1,2,5,6,7]])
        hidden=arr.copy();hidden[1:3,5]=1-arr[1:3,5]
        np.testing.assert_array_equal(ch.simulate(hidden,par,code,u)[:,3],generated[:,3])


def test_joint_prior_correlation_sampler_has_lkj2_moments():
    rng=np.random.default_rng(11);Ls=np.array([ch.lkj_cholesky(rng,5) for _ in range(12000)])
    corr=Ls@Ls.transpose(0,2,1)
    np.testing.assert_allclose(np.diagonal(corr,axis1=1,axis2=2),1,atol=1e-14)
    # LKJ(2), dimension 5: marginal variance 1/(2*eta+K-1) = 1/8.
    off=corr[:,np.tril_indices(5,-1)[0],np.tril_indices(5,-1)[1]]
    np.testing.assert_allclose(off.mean(axis=0),0,atol=.012)
    np.testing.assert_allclose(off.var(axis=0),1/8,atol=.008)


def test_latent_variance_fraction_is_not_choice_r_squared(tmp_path):
    cfg,_,phase,_=q.configuration();entry=dict(model='HPreference',extension='bias',age=True,prior_sd=.5)
    data,meta,_=q.model_data(age_trials(),entry,phase,cfg);D=100
    variables={'mu':np.zeros((D,5)),'mu_ext':np.zeros((D,1)),
               'tau':np.ones((D,5)),'tau_ext':np.ones((D,1)),
               'beta':np.full((D,5,1),.5),'beta_ext':np.full((D,1,1),.5)}
    fit=SimpleNamespace(stan_variable=lambda name:variables[name]);natural=np.ones((D,3,6))
    ch.parameter_tables(fit,natural,data,meta,entry,cfg,tmp_path)
    f=pd.read_csv(tmp_path/'age_effects.tsv',sep='\t');v=np.var([-1.5,-.5,.5])
    np.testing.assert_allclose(f.loc[f.quantity.eq('latent_age_variance_fraction'),'mean'],.25*v/(.25*v+1))
    curves=pd.read_csv(tmp_path/'age_curves.tsv',sep='\t');alpha=curves[curves.parameter.eq(meta['parameter_names'][0])]
    assert len(alpha)==70 and ((alpha['mean']>0)&(alpha['mean']<1)).all()


def test_cache_refuses_partial_changed_and_incomplete_chains(tmp_path):
    assert q.authenticated_cache(tmp_path,'target') is None
    (tmp_path/'partial.csv').write_text('partial')
    with pytest.raises(RuntimeError,match='Partial'):q.authenticated_cache(tmp_path,'target')
    (tmp_path/'partial.csv').unlink()
    f=tmp_path/'one.csv';f.write_text('posterior')
    manifest=dict(fingerprint='target',csv_files=[str(f)],posterior_sha256={str(f):s.fs.sha(f)},settings={'chains':1})
    s.save(tmp_path/'manifest.json',manifest)
    assert q.authenticated_cache(tmp_path,'target')==manifest
    with pytest.raises(ValueError,match='target differs'):q.authenticated_cache(tmp_path,'other')
    f.write_text('changed')
    with pytest.raises(ValueError,match='posterior changed'):q.authenticated_cache(tmp_path,'target')


def test_paired_comparison_rejects_participant_or_trial_mismatch():
    x=pd.DataFrame(dict(participant_id=['a','b','c'],n=[5,6,7],log_loss=[.5,.6,.7]))
    y=x.assign(log_loss=x.log_loss+.1)
    assert ch.paired_contrast(x,y,1,100)['delta_log_loss']==pytest.approx(-.1)
    for change in ['id','count']:
        bad=y.copy()
        if change=='id':bad.loc[0,'participant_id']='d'
        else:bad.loc[0,'n']=4
        with pytest.raises(ValueError,match='Unmatched'):ch.paired_contrast(x,bad,1,100)


def test_recovery_exports_coverage_and_rank_not_only_correlation(tmp_path):
    meta={'ids':['a','b','c'],'parameter_names':['alpha','kappa','friend','stranger','gamma','bias']}
    truth={'ids':meta['ids'],'beta':[0.]*6,'natural':np.arange(18).reshape(3,6).tolist()}
    natural=np.tile(np.asarray(truth['natural']),(100,1,1))
    fit=SimpleNamespace(stan_variable=lambda name:np.zeros((100,5 if name=='beta' else 1,1)))
    ch.recovery_tables(fit,natural,truth,meta,tmp_path)
    age=pd.read_csv(tmp_path/'recovery_age.tsv',sep='\t');people=pd.read_csv(tmp_path/'recovery_participants.tsv',sep='\t')
    assert age.covered.all() and not age.excludes_zero.any()
    assert (people.rmse==0).all() and (people.interval_coverage==1).all()
    np.testing.assert_allclose(people.rank_correlation,1)


def test_failed_fit_tables_are_excluded_even_if_left_on_disk(tmp_path):
    s.save(tmp_path/'status.json',dict(status='diagnostic_failed',passed=False))
    (tmp_path/'age_effects.tsv').write_text('stale')
    status,frames=ch.accepted_tables(tmp_path)
    assert not frames


def test_recovery_regenerates_consistent_feedback_and_known_age_truth(monkeypatch):
    cfg,old,phase,c=q.configuration();t=age_trials();entry=dict(model='HPreference',extension='bias',age=True,prior_sd=.5)
    data,meta,_=q.model_data(t,entry,phase,cfg);D=10
    variables={'mu':np.tile([-1.,-1.,.2,.1,0.],(D,1)),'tau':np.full((D,5),.4),
               'Omega':np.tile(np.eye(5),(D,1,1)),'mu_ext':np.full((D,1),.3),'tau_ext':np.full((D,1),.7)}
    fake=SimpleNamespace(stan_variable=lambda field:variables[field])
    monkeypatch.setattr(ch,'age_anchor',lambda *args:(fake,{'fingerprint':'anchor'},data,meta))
    for rep in [0,4,5]:
        generated,truth,anchor=ch.synthetic_trials(t,rep,cfg,old,phase,c)
        adapted,generated_meta,_=q.model_data(generated,entry,phase,cfg)
        assert generated_meta['ids']==meta['ids'] and adapted['T']==data['T']
        assert generated.valid_choice.tolist()==t.valid_choice.tolist()
        if rep==0:assert truth['beta']==[0.]*6
        else:np.testing.assert_allclose(np.abs(truth['beta']),.3)
        feedback=generated.feedback_observed
        assert generated.loc[~feedback,'observed_reciprocation'].isna().all()
        np.testing.assert_array_equal(generated.loc[feedback,'observed_reciprocation'],generated.loc[feedback,'scheduled_reciprocation'])
        again,truth2,_=ch.synthetic_trials(t,rep,cfg,old,phase,c)
        pd.testing.assert_frame_equal(generated,again);assert truth==truth2


def test_accepted_worker_and_postfit_failure_keep_raw_draws(monkeypatch,tmp_path):
    import cmdstanpy
    cfg,old,phase,c=q.configuration();c=dict(c,output=str(tmp_path/'out'),work=str(tmp_path/'work'))
    cfg=dict(cfg,chains=1)
    t=trials(343);t['age']=50.;t['include_sensitivity']=True
    entry=next(e for e in q.entries() if e['name']=='Full_HPreference_zero_bias_age_closeout1')
    monkeypatch.setattr(s,'require_linux',lambda:None);monkeypatch.setattr(s,'cmdstan',lambda *args:None)
    monkeypatch.setattr(q,'snapshot',lambda *args:(t,{'status':'matches_snapshot'}));monkeypatch.setattr(q,'gate',lambda c:None)
    monkeypatch.setattr(q,'sources',lambda c:{'test':'source'})
    diagnostic=pd.DataFrame({'parameter':['beta_ext[1,1]'],'R_hat':[1.001],'ESS_bulk':[1000],'ESS_tail':[1000]})
    info=dict(passed=True,max_rhat=1.001,min_bulk_ess=1000,min_tail_ess=1000,divergences=0,max_depth_hits=0,min_bfmi=.7)
    monkeypatch.setattr(q,'diagnostics',lambda *args:(diagnostic,info))
    class FakeModel:
        def __init__(self,**kwargs):pass
        def sample(self,**kwargs):
            path=Path(kwargs['output_dir'])/'one.csv';path.write_text('authenticated fake draws')
            return SimpleNamespace(runset=SimpleNamespace(csv_files=[str(path)]),diagnose=lambda:'passed',stan_variable=lambda field:np.zeros((2,343,6)))
    monkeypatch.setattr(cmdstanpy,'CmdStanModel',FakeModel)
    def export(*args):
        pd.DataFrame({'test':[1]}).to_csv(args[-1]/'test.tsv',sep='\t',index=False)
    monkeypatch.setattr(ch,'predictive_tables',export);monkeypatch.setattr(ch,'parameter_tables',export)
    assert q.fit_entry(cfg,old,phase,c,entry)==0
    out,work=q.paths(c);status=json.loads((out/'fits'/entry['name']/'status.json').read_text())
    assert status['passed'] and status['status']=='complete' and 'test.tsv' in status['output_sha256']
    # A postprocessing failure must leave a usable manifest and captured error.
    manifest=json.loads((work/'fits'/entry['name']/'manifest.json').read_text())
    cached=FakeModel().sample(output_dir=work/'fits'/entry['name'])
    monkeypatch.setattr(s,'load_chains',lambda files:cached)
    def fail(*args):raise RuntimeError('postfit failed')
    monkeypatch.setattr(ch,'predictive_tables',fail)
    with pytest.raises(RuntimeError,match='postfit failed'):q.fit_entry(cfg,old,phase,c,entry)
    assert json.loads((work/'fits'/entry['name']/'manifest.json').read_text())==manifest
    status=json.loads((out/'fits'/entry['name']/'status.json').read_text())
    assert status['status']=='error' and status['reused_posterior']


def test_saved_chain_loader_initializes_project_cmdstan_without_install(monkeypatch):
    calls=[];phase={'cmdstan_version':'2.40.0'};c={'_root':'project'};files=['chain.csv'];posterior=object()
    def configure(actual_phase,actual_c,install=False):
        assert actual_phase is phase and actual_c is c and install is False
        calls.append('configure')
    def load(actual_files):
        assert actual_files is files and calls==['configure']
        calls.append('load');return posterior
    monkeypatch.setattr(s,'cmdstan',configure);monkeypatch.setattr(s,'load_chains',load)
    assert q.load_posterior(files,phase,c) is posterior
    assert calls==['configure','load']


def test_missing_project_cmdstan_stops_before_csv_loading(monkeypatch):
    def missing(*args):raise RuntimeError('CmdStan missing')
    monkeypatch.setattr(s,'cmdstan',missing)
    monkeypatch.setattr(s,'load_chains',lambda *args:pytest.fail('Must initialize before loading'))
    with pytest.raises(RuntimeError,match='CmdStan missing'):q.load_posterior(['chain.csv'],{}, {})


def test_old_posterior_configures_cmdstan_before_recomputing_diagnostics(tmp_path,monkeypatch):
    cfg,old,phase,c=q.configuration();c=dict(c,output=str(tmp_path/'out'),work=str(tmp_path/'work'))
    entry=q.entries()[0];name=entry['source'];t=trials(2);t=pd.concat([t,t.assign(run=2)],ignore_index=True)
    original=next(e for e in a.entries(old) if e['name']==name)
    data,meta,_=a.model_data(q.subset(t,original),original,phase)
    fingerprint=s.digest(dict(data=data,config=old,sources=a.sources(c)))
    out,work=a.paths(c);cache=work/'fits'/name;cache.mkdir(parents=True)
    files=[]
    for i in range(old['chains']):
        f=cache/f'chain{i}.csv';f.write_text(f'fake retained chain {i}\n');files.append(str(f))
    hashes={f:s.fs.sha(f) for f in files}
    manifest=dict(fingerprint=fingerprint,meta=meta,csv_files=files,posterior_sha256=hashes)
    published=out/'fits'/name;s.save(cache/'manifest.json',manifest)
    s.save(published/'manifest_summary.json',dict(manifest,posterior_sha256={Path(f).name:h for f,h in hashes.items()}))
    info=dict(passed=False,max_rhat=1.004,min_bulk_ess=1000.,min_tail_ess=1000.,divergences=1,max_depth_hits=0,min_bfmi=.7)
    s.save(published/'status.json',dict(status='diagnostic_failed',fingerprint=fingerprint,**info))
    events=[];fit=object()
    def configure(*args,**kwargs):events.append('configured')
    def load(files):
        assert events==['configured'];events.append('loaded');return fit
    def diagnostics(actual_fit,*args):
        assert actual_fit is fit and events==['configured','loaded']
        events.append('summarized');return None,info
    monkeypatch.setattr(s,'cmdstan',configure);monkeypatch.setattr(s,'load_chains',load);monkeypatch.setattr(a,'diagnostics',diagnostics)
    result=q.old_posterior(entry,old,phase,c,t)
    assert result[0] is fit and result[1]==data and events==['configured','loaded','summarized']
