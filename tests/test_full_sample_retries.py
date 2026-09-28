import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rf1_trust_socialvalue import full_sample_sampling as s
from rf1_trust_socialvalue import full_sample_batch as batch
from rf1_trust_socialvalue import full_sample_retries as retry
from test_full_sample_batch import paired_trials


def test_retry_inventory_settings_and_core_budget(monkeypatch):
    p,c,cfg=s.configuration('config/full_sample_batch_retry.json')
    assert len(p['reuse'])==11 and len(s.entries(p))==9
    assert not ({e['name'] for e in p['reuse']}&{e['name'] for e in p['retries']})
    for e in p['retries']:
        old=json.loads((Path(c['output'])/'hierarchical/fits'/e['source_name']/'status.json').read_text())
        assert old['status']=='diagnostic_failed'
        effective=batch.entry_settings(p,cfg,e)
        assert effective['chains']==8 and effective['warmup']==4000
        assert effective['adapt_delta']==(.995 if old['divergences'] else .99)
        assert effective['draws']==(8000 if e['source_name'] in ['Full_H2_zero_noage','Train_H2_zero_noage'] else 4000)
        assert s.stable_seed(effective['seed'],e['seed_name'])==old['sampler_seed']
    monkeypatch.setattr(batch.os,'sched_getaffinity',lambda _:set(range(96)),raising=False)
    assert batch.worker_count(p,dict(cfg,chains=8),9)*8==72
    assert 'batch_retry' in str(s.phase_output(p,c))


@pytest.mark.parametrize('change',['draws','accepted','duplicate','attempts','prior','evidence','chains'])
def test_retry_rejects_unreviewed_changes(change):
    p,c,cfg=s.configuration('config/full_sample_batch_retry.json');p=copy.deepcopy(p)
    if change=='draws':p['retries'][0]['settings']['draws']=123
    if change=='accepted':p['reuse']=p['reuse'][:-1]
    if change=='duplicate':p['retries'][0]=copy.deepcopy(p['retries'][1])
    if change=='attempts':p['max_attempts']=2
    if change=='prior':p['gamma_population_mean_prior_sd']=2
    if change=='evidence':p['evidence_sha256'][next(iter(p['evidence_sha256']))]='changed'
    if change=='chains':p['retries'][0]['settings']['chains']=16
    with pytest.raises(ValueError):retry.validate_configuration(p,c,cfg)


def test_retry_target_allows_sampler_only_changes(tmp_path):
    old=dict(data={'y':[1,0],'prior':[1]},settings={'chains':4,'draws':2000},sampler_seed=42,stan={'source':'hash'},cmdstan_version='2.40.0')
    file=tmp_path/'hierarchical/fits/source/status.json';s.save(file,dict(fingerprint=s.digest(old),settings=old['settings']))
    p=dict(stage='full_cohort_noage_batch_retry',_entry={'source_name':'source'})
    payload=dict(old,settings={'chains':8,'draws':8000})
    s.verify_retry_target(p,{'output':str(tmp_path)},payload)
    for changed in [dict(payload,sampler_seed=43),dict(payload,data={'y':[1,1],'prior':[1]}),dict(payload,stan={'source':'new'})]:
        with pytest.raises(ValueError,match='original target'):s.verify_retry_target(p,{'output':str(tmp_path)},changed)


def test_retry_parent_dispatches_only_failed_fits_and_preserves_old_status(tmp_path,monkeypatch):
    p,_,cfg=s.configuration('config/full_sample_batch_retry.json')
    c=dict(output=str(tmp_path/'out'),work=str(tmp_path/'work'),_root=str(tmp_path))
    old=Path(c['output'])/'hierarchical/batch/status.json';s.save(old,{'historical':'keep'});before=old.read_bytes()
    monkeypatch.setattr(s,'require_linux',lambda:None);monkeypatch.setattr(s,'integrated_trials',lambda *a:paired_trials())
    monkeypatch.setattr(s.fs,'git_sha',lambda *a:'test');monkeypatch.setattr(s,'implementation',lambda *a:{'source':'hash'})
    monkeypatch.setattr(batch,'review_reused',lambda *a:p['reuse']);monkeypatch.setattr(s,'cmdstan',lambda *a,**kw:None)
    monkeypatch.setattr(s,'check_implementation',lambda *a:None)
    monkeypatch.setattr(batch.os,'sched_getaffinity',lambda _:set(range(96)),raising=False)
    commands=[]
    def worker(command,**kw):
        commands.append(command)
        dest=Path(c['output'])/'hierarchical/fits'/command[-1]/'status.json'
        s.save(dest,dict(status='diagnostic_failed',passed=False));return SimpleNamespace(returncode=2)
    monkeypatch.setattr(batch.subprocess,'run',worker)
    assert batch.run(p,c,cfg)==2 and len(commands)==9
    assert {cmd[-1] for cmd in commands}=={e['name'] for e in p['retries']}
    state=json.loads((Path(c['output'])/'hierarchical/batch_retry/status.json').read_text())
    assert state['maximum_active_chains']==72 and old.read_bytes()==before


def test_comparison_resolves_replacement_and_original_training_scores(tmp_path):
    p,c,_=s.configuration('config/full_sample_batch_retry.json');original=json.loads(Path('config/full_sample_batch.json').read_text())
    c=dict(c,output=str(tmp_path/'out'),_root=str(tmp_path),bootstrap_iterations=100)
    s.save(tmp_path/'config/full_sample_batch.json',original)
    root=Path(c['output'])/'hierarchical';(root/'batch_retry').mkdir(parents=True)
    mapping={e['source_name']:e['name'] for e in p['retries']}
    for e in batch.new_entries(original):
        if e['subset']!='train':continue
        actual=mapping.get(e['name'],e['name']);folder=root/'fits'/actual;folder.mkdir(parents=True)
        file=folder/'heldout_participants.tsv'
        pd.DataFrame(dict(name=[actual]*2,participant_id=['a','b'],n=[3,4],log_loss=[.4,.5],brier=[.1,.2])).to_csv(file,sep='\t',index=False)
        s.save(folder/'status.json',dict(status='complete',passed=True,heldout_sha256=s.fs.sha(file)))
    hashes=batch.comparison_outputs(p,c)
    assert len(hashes)==5
    scores=pd.read_csv(root/'batch_retry/heldout_comparison.tsv',sep='\t')
    assert len(scores)==10 and not scores.name.str.contains('retry').any()
    provenance=json.loads((root/'batch_retry/comparison_provenance.json').read_text())
    assert sum('retry1' in f for f in provenance['score_hashes'])==6


def test_trace_excerpt_retains_divergent_draw_and_all_draw_chain_summary(tmp_path):
    names=['lp__','divergent__','mu[1]','Omega[1,2]','natural[1,1]']
    draws=np.zeros((800,8,5));draws[:,0,2]=np.arange(800);draws[345,3,1]=1
    fit=SimpleNamespace(column_names=names,draws=lambda **kw:draws)
    d=pd.DataFrame(dict(parameter=['natural[1,1]'],R_hat=[1.02],ESS_bulk=[300],ESS_tail=[500]))
    hashes=retry.trace_evidence(fit,d,tmp_path)
    trace=pd.read_csv(tmp_path/'chain_trace_excerpt.tsv',sep='\t');summary=pd.read_csv(tmp_path/'chain_summary.tsv',sep='\t')
    assert len(hashes)==2 and trace['divergent__'].sum()==1
    assert ((trace.chain==4)&(trace.draw==346)).any()
    row=summary[(summary.chain==1)&(summary.parameter=='mu[1]')]
    assert row['mean'].iloc[0]==pytest.approx(399.5)


def test_reused_training_target_uses_run_one_and_checks_score_hash(tmp_path):
    from test_full_sample_sampling import fake_fit
    phase,base,_=s.configuration('config/full_sample_batch_retry.json')
    entry=dict(name='Train_H7_base_noage',model='H7',variant='base',subset='train')
    phase=dict(phase,reuse=[entry]);c=dict(base,output=str(tmp_path/'out'),_root=str(tmp_path))
    for file in s.SOURCES:
        p=tmp_path/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
    folder=Path(c['output'])/'hierarchical/fits'/entry['name'];folder.mkdir(parents=True)
    data,meta,_=s.model_data(batch.entry_trials(paired_trials(),entry),'H7',False)
    payload=dict(data=data,settings={'draws':2000},sampler_seed=42,stan={p:s.fs.sha(tmp_path/p) for p in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
    fake_fit().summary().iloc[:1].to_csv(folder/'diagnostics.tsv',sep='\t')
    (folder/'participant_parameters.tsv').write_text('synthetic parameters')
    (folder/'heldout_participants.tsv').write_text('synthetic heldout')
    status=dict(status='complete',passed=True,fingerprint=s.digest(payload),settings=payload['settings'],sampler_seed=42,
        summary_hashes={'participant_parameters.tsv':s.fs.sha(folder/'participant_parameters.tsv')},diagnostics_sha256=s.fs.sha(folder/'diagnostics.tsv'),
        divergences=0,max_depth_hits=0,min_bfmi=.7,heldout_sha256=s.fs.sha(folder/'heldout_participants.tsv'))
    s.save(folder/'status.json',status);s.save(folder/'manifest_summary.json',dict(fingerprint=status['fingerprint'],meta=meta))
    reused=batch.review_reused(phase,c,paired_trials())
    assert len(reused)==1 and reused[0]['subset']=='train'
    (folder/'heldout_participants.tsv').write_text('tampered')
    with pytest.raises(ValueError,match='heldout scores changed'):batch.review_reused(phase,c,paired_trials())
