import json
import threading
from pathlib import Path
import pytest
from rf1_trust_socialvalue.parallel_checkpoint import execution_chains,dispatch
from rf1_trust_socialvalue import linux_handoff as handoff


def test_execution_parallelism_does_not_change_sampling_settings(monkeypatch):
    cfg={'parallel_chains':2,'chains':4,'seed':24,'draws':8000}
    original=dict(cfg)
    monkeypatch.setenv('RF1_PARALLEL_CHAINS','4')
    assert execution_chains(cfg)==4 and cfg==original
    monkeypatch.setenv('RF1_PARALLEL_CHAINS','48')
    with pytest.raises(ValueError):execution_chains(cfg)


def test_dispatch_parallelizes_independent_fits_and_records_failure():
    barrier=threading.Barrier(2,timeout=5);active=0;peak=0;lock=threading.Lock();events={}
    def launch(entry):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        barrier.wait()
        with lock:active-=1
        return {'status':'diagnostic_failed' if entry['run']=='a' else 'complete'}
    def update(name,value):
        with lock:events[name]=value
    dispatch([{'run':'a'},{'run':'b'}],2,launch,update)
    assert peak==2
    assert events['a']['status']=='diagnostic_failed' and events['b']['status']=='complete'


def test_dispatch_operational_failure_is_recorded_without_losing_other_results():
    results={}
    def launch(entry):
        if entry['run']=='a':raise OSError('disk full')
        return {'status':'complete'}
    dispatch([{'run':'a'},{'run':'b'}],2,launch,lambda n,v:results.update({n:v}))
    assert results['a']['status']=='error' and results['b']['status']=='complete'


def test_worker_selection_requires_exact_code_project_and_run_flag(tmp_path):
    info={'cwd':str(tmp_path),'command':['python','-c','from rf1_trust_socialvalue.checkpoint_handoff import main; main()','--run']}
    assert handoff.is_sequential_worker(info,tmp_path)
    assert not handoff.is_sequential_worker(info,tmp_path/'another')
    assert not handoff.is_sequential_worker(dict(info,command=['python','unrelated.py']),tmp_path)


def test_process_identity_change_never_receives_signals(monkeypatch):
    saved={'pid':1,'started':'100','cwd':'/project','command':['python']}
    monkeypatch.setattr(handoff,'process',lambda pid:dict(saved,started='101'))
    signals=[];monkeypatch.setattr(handoff.os,'kill',lambda *args:signals.append(args))
    with pytest.raises(RuntimeError,match='identity changed'):handoff.stop_verified_tree(saved)
    assert not signals


def test_takeover_waits_for_complete_fit_before_stopping(monkeypatch,tmp_path):
    path=tmp_path/'results/linux_run_status.json';path.parent.mkdir()
    path.write_text(json.dumps({'runs':{'current':{'status':'running'}}}))
    saved={'pid':111,'ppid':99,'started':'100','state':'S','cwd':str(tmp_path),'command':['python','-c','worker']}
    monkeypatch.setattr(handoff,'process',lambda pid:saved if pid==111 else None)
    sleeps=[]
    def sleep(seconds):
        sleeps.append(seconds)
        path.write_text(json.dumps({'runs':{'current':{'status':'complete'},'next':{'status':'running'}}}))
    monkeypatch.setattr(handoff.time,'sleep',sleep)
    stopped=[]
    def stop(info):
        assert json.loads(path.read_text())['runs']['current']['status']=='complete'
        stopped.append(info['pid']);return [111]
    monkeypatch.setattr(handoff,'stop_verified_tree',stop)
    handoff.take_over(tmp_path,[saved])
    assert sleeps==[2] and stopped==[111]
    record=json.loads(next((tmp_path/'results/run_logs').glob('handoff-*.json')).read_text())
    assert record['waited_for']=='current'


def test_coordinator_lock_prevents_duplicate_coordinators(tmp_path):
    with handoff.coordinator_lock(tmp_path):
        with pytest.raises(RuntimeError,match='already owns'):
            with handoff.coordinator_lock(tmp_path):pass


def test_worker_produces_only_its_own_outputs(monkeypatch,tmp_path):
    import pandas as pd
    from rf1_trust_socialvalue import hierarchical as h
    from rf1_trust_socialvalue.parallel_checkpoint import worker
    monkeypatch.chdir(tmp_path);table=tmp_path/'results/tables';table.mkdir(parents=True)
    pd.DataFrame([{'passed':True}]).to_csv(table/'diagnostics_H2_train_age.csv',index=False)
    monkeypatch.setattr(h,'TABLE',table)
    calls=[]
    monkeypatch.setattr(h,'sample',lambda *a,**k:('fit','meta','arrays','ratings','frames'))
    monkeypatch.setattr(h,'heldout',lambda *a:calls.append('heldout'))
    monkeypatch.setattr(h,'collect',lambda:pytest.fail('Workers must not write aggregate tables'))
    worker(dict(run='H2_train_age',kind='real_data',model='H2',training=True,age_terms=1,bounded=False,prior_scale=1,settings={}))
    assert calls==['heldout']
