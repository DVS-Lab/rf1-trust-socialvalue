import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from rf1_trust_socialvalue import full_sample_batch as batch
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials, fake_fit


def paired_trials():
    one=trials(3);two=one[one.participant_id!='sub-2'].copy();two['run']=2
    return pd.concat([two,one],ignore_index=True)


def test_batch_config_reuses_four_and_schedules_six_full_ten_train():
    phase,c,cfg=s.configuration('config/full_sample_batch.json');entries=s.entries(phase)
    assert len(entries)==16
    assert sum(e['subset']=='full' for e in entries)==6
    assert sum(e['subset']=='train' for e in entries)==10
    assert not ({e['name'] for e in entries}&set(batch.REUSED.values()))
    assert c['hierarchical']['launch_authorized'] is False
    for e in entries:
        settings=batch.entry_settings(phase,cfg,e)
        assert settings==dict(cfg,draws=4000 if e['model']=='HPreference' else 2000)
    assert s.phase_output(dict(phase,_subset='train'),c)!=s.phase_output(phase,c)


def test_batch_cap_respects_cpu_affinity(monkeypatch):
    phase,_,cfg=s.configuration('config/full_sample_batch.json')
    monkeypatch.setattr(batch.os,'sched_getaffinity',lambda _:set(range(96)),raising=False)
    assert batch.worker_count(phase,cfg,16)==16
    monkeypatch.setattr(batch.os,'sched_getaffinity',lambda _:set(range(12)))
    assert batch.worker_count(phase,cfg,16)==3
    monkeypatch.setattr(batch.os,'sched_getaffinity',lambda _:set(range(3)))
    with pytest.raises(RuntimeError):batch.worker_count(phase,cfg,16)


def test_train_contains_only_real_run_one_and_no_single_run_fallback():
    t=paired_trials();entry=dict(subset='train')
    selected=batch.entry_trials(t,entry)
    assert set(selected.run)=={1} and set(selected.participant_id)=={'sub-0','sub-1'}
    data,_,_=s.model_data(selected,'H2',False)
    assert data['N']==2 and data['T']==6
    # Changing every heldout offer/outcome/choice does not alter the fitted target.
    mutated=t.copy();test=mutated.run==2
    mutated.loc[test,['chose_high','scheduled_reciprocation','observed_reciprocation']]=0
    mutated.loc[test,['c_low','c_high']]=99
    after,_,_=s.model_data(batch.entry_trials(mutated,entry),'H2',False)
    assert after==data
    with pytest.raises(ValueError):batch.paired_ids(t[t.run==1])
    # Run 2 present but no scorable decisions is ineligible, with no fallback.
    t.loc[(t.run==2)&(t.participant_id=='sub-1'),'valid_choice']=False
    assert batch.paired_ids(t)==['sub-0']


def test_heldout_integrates_probabilities_and_masks_unseen_outcomes():
    t=paired_trials();entry=dict(subset='train',model='H2',variant='base',name='Train_H2_base_noage')
    _,meta,_=s.model_data(batch.entry_trials(t,entry),'H2',False)
    natural=np.array([[[.1,.2],[.1,.2]],[[.8,3.],[.8,3.]]])
    result=batch.heldout_scores(t,entry,natural,meta)
    assert result.n.tolist()==[3,3] and result.participant_id.tolist()==meta['ids']
    _,_,arrays=s.model_data(t[t.participant_id.isin(meta['ids'])],'H2',False)
    a=arrays['sub-0'];mask=(a[:,7]==2)&(a[:,3]>=0)
    nll=np.array([s.zero_engine(a,p,2,0.)[mask,3] for p in natural[:,0]])
    expected=(-logsumexp(-nll,axis=0)+np.log(2)).mean()
    assert result.log_loss.iloc[0]==pytest.approx(expected)
    assert abs(result.log_loss.iloc[0]-nll.mean())>1e-3
    hidden=~t.feedback_observed;t.loc[hidden,'scheduled_reciprocation']=1-t.loc[hidden,'scheduled_reciprocation']
    pd.testing.assert_frame_equal(result,batch.heldout_scores(t,entry,natural,meta))
    with pytest.raises(ValueError,match='order'):
        batch.heldout_scores(t,entry,natural,dict(meta,ids=list(reversed(meta['ids']))))


def test_comparison_is_paired_and_equal_participant_weighted():
    frames=[]
    for m in s.MODELS:
        for v in ['base','zero']:
            frames.append(pd.DataFrame(dict(name=[f'Train_{m}_{v}_noage']*2,model=[m]*2,variant=[v]*2,
                participant_id=['a','b'],n=[2,40],log_loss=np.array([.2,.8])-(.1 if v=='zero' else 0),brier=[.1,.2])))
    summary,paired=batch.compare_scores(frames,3,200)
    assert summary[summary.name=='Train_H2_base_noage'].mean_log_loss.iloc[0]==pytest.approx(.5)
    np.testing.assert_allclose(paired.delta_zero_minus_base,-.1)
    frames[0]=frames[0].iloc[:1]
    with pytest.raises(ValueError,match='identical participants'):batch.compare_scores(frames,3,200)


def test_batch_dispatches_sixteen_fits_and_retains_other_results_on_failure(tmp_path,monkeypatch):
    phase,_,cfg=s.configuration('config/full_sample_batch.json')
    c=dict(output=str(tmp_path/'out'),work=str(tmp_path/'work'),_root=str(tmp_path))
    monkeypatch.setattr(s,'require_linux',lambda:None)
    monkeypatch.setattr(s,'integrated_trials',lambda *a:paired_trials())
    monkeypatch.setattr(s.fs,'git_sha',lambda *a:'test')
    monkeypatch.setattr(s,'implementation',lambda *a:{'test':'hash'})
    monkeypatch.setattr(batch,'review_reused',lambda *a:[dict(name=x) for x in batch.REUSED.values()])
    monkeypatch.setattr(s,'cmdstan',lambda *a,**kw:None)
    checks=[]
    monkeypatch.setattr(s,'check_implementation',lambda p,c,t:checks.append((p['_subset'],set(t.run))))
    monkeypatch.setattr(batch.os,'sched_getaffinity',lambda _:set(range(96)),raising=False)
    commands=[]
    def worker(command,**kwargs):
        commands.append(command);name=command[-1]
        if name=='Train_H2_base_noage':return SimpleNamespace(returncode=1)
        s.save(Path(c['output'])/'hierarchical/fits'/name/'status.json',dict(status='complete',passed=True))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess,'run',worker)
    monkeypatch.setattr(batch,'comparison_outputs',lambda *a:pytest.fail('Do not compare failed posteriors'))
    assert batch.run(phase,c,cfg)==2
    assert checks==[('full',{1,2}),('train',{1})] and len(commands)==16
    assert all(x[2]=='rf1_trust_socialvalue.full_sample_sampling' for x in commands)
    status=json.loads((Path(c['output'])/'hierarchical/batch/status.json').read_text())
    assert status['maximum_active_chains']==64 and status['n_train']==2
    assert sum(x['exit_code']==0 for x in status['runs'].values())==15
    assert status['status']=='blocked_diagnostics_or_error'


def test_batch_preparation_failure_is_logged(tmp_path,monkeypatch):
    phase,_,cfg=s.configuration('config/full_sample_batch.json')
    c=dict(output=str(tmp_path/'out'),work=str(tmp_path/'work'),_root=str(tmp_path))
    monkeypatch.setattr(s,'require_linux',lambda:None);monkeypatch.setattr(s.fs,'git_sha',lambda *a:'test')
    def fail(*a):raise ValueError('changed inputs')
    monkeypatch.setattr(s,'integrated_trials',fail)
    with pytest.raises(ValueError,match='changed inputs'):batch.run(phase,c,cfg)
    status=json.loads((Path(c['output'])/'hierarchical/batch/status.json').read_text())
    assert status['status']=='error'


def test_train_worker_samples_only_run_one_and_saves_scores(tmp_path,monkeypatch):
    phase,base,cfg=s.configuration('config/full_sample_batch.json');c=dict(base,output=str(tmp_path/'out'),work=str(tmp_path/'work'),_root=str(tmp_path))
    for file in s.SOURCES:
        p=tmp_path/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
    monkeypatch.setattr(s,'require_linux',lambda:None);monkeypatch.setattr(s,'integrated_trials',lambda *a:paired_trials())
    monkeypatch.setattr(s,'implementation_gate',lambda *a:None);monkeypatch.setattr(s,'implementation',lambda *a:{'test':'hash'})
    monkeypatch.setattr(s,'cmdstan',lambda *a:None);monkeypatch.setattr(s.fs,'git_sha',lambda *a:'test')
    fit=fake_fit();fit.diagnose=lambda:'ok'
    fit.stan_variable=lambda name:np.ones((6,2,2))*.5 if name=='natural' else np.ones((6,2))*.5
    calls=[]
    def sample(**kw):
        calls.append(kw);p=Path(kw['output_dir'])/'chain.csv';p.write_text('synthetic posterior')
        fit.runset=SimpleNamespace(csv_files=[str(p)]);return fit
    monkeypatch.setattr(s,'models',lambda *a:[SimpleNamespace(sample=sample)])
    entry=next(e for e in batch.new_entries(phase) if e['name']=='Train_H2_base_noage')
    assert s.fit_entry(phase,c,cfg,entry)==0
    assert calls[0]['data']['N']==2 and calls[0]['data']['T']==6
    out=Path(c['output'])/'hierarchical/fits'/entry['name']
    status=json.loads((out/'status.json').read_text())
    assert status['heldout_n']==2 and status['heldout_choices']==6
    assert status['heldout_sha256']==s.fs.sha(out/'heldout_participants.tsv')


def test_reuse_verifies_target_and_summary_hashes(tmp_path):
    phase,base,_=s.configuration('config/full_sample_batch.json');phase=copy.deepcopy(phase)
    phase['reuse']=phase['reuse'][:1]
    c=dict(base,output=str(tmp_path/'out'),_root=str(tmp_path))
    for file in s.SOURCES:
        p=tmp_path/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
    entry=phase['reuse'][0];folder=Path(c['output'])/'hierarchical/fits'/entry['name'];folder.mkdir(parents=True)
    data,meta,_=s.model_data(paired_trials(),'H2',False)
    payload=dict(data=data,settings={'draws':2000},sampler_seed=42,stan={p:s.fs.sha(tmp_path/p) for p in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
    fake_fit().summary().iloc[:1].to_csv(folder/'diagnostics.tsv',sep='\t')
    (folder/'participant_parameters.tsv').write_text('synthetic parameters')
    status=dict(status='complete',passed=True,fingerprint=s.digest(payload),settings=payload['settings'],sampler_seed=42,
        summary_hashes={'participant_parameters.tsv':s.fs.sha(folder/'participant_parameters.tsv')},diagnostics_sha256=s.fs.sha(folder/'diagnostics.tsv'),
        divergences=0,max_depth_hits=0,min_bfmi=.7)
    s.save(folder/'status.json',status);s.save(folder/'manifest_summary.json',dict(fingerprint=status['fingerprint'],meta=meta))
    assert len(batch.review_reused(phase,c,paired_trials()))==1
    changed=paired_trials();changed.loc[0,'c_high']=10
    with pytest.raises(ValueError,match='current full-data target'):batch.review_reused(phase,c,changed)
    (folder/'participant_parameters.tsv').write_text('tampered')
    with pytest.raises(ValueError,match='summary changed'):batch.review_reused(phase,c,paired_trials())


def test_comparison_artifacts_are_hash_bound_and_failed_fits_block_them(tmp_path):
    phase,c,_=s.configuration('config/full_sample_batch.json');c=dict(c,output=str(tmp_path),bootstrap_iterations=200)
    out=tmp_path/'hierarchical';(out/'batch').mkdir(parents=True)
    for e in batch.new_entries(phase):
        if e['subset']!='train':continue
        folder=out/'fits'/e['name'];folder.mkdir(parents=True)
        file=folder/'heldout_participants.tsv'
        pd.DataFrame(dict(name=[e['name']]*2,participant_id=['a','b'],n=[3,4],log_loss=[.4,.5],brier=[.1,.2])).to_csv(file,sep='\t',index=False)
        s.save(folder/'status.json',dict(status='complete',passed=True,heldout_sha256=s.fs.sha(file)))
    hashes=batch.comparison_outputs(phase,c)
    assert len(hashes)==5 and all(s.fs.sha(out/'batch'/file)==h for file,h in hashes.items())
    s.save(folder/'status.json',dict(status='diagnostic_failed',passed=False))
    with pytest.raises(ValueError,match='accepted training fits'):batch.comparison_outputs(phase,c)
