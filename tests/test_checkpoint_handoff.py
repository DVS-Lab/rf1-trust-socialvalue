import hashlib
import json
import pytest
from rf1_trust_socialvalue.checkpoint_handoff import prepare_caches,verify_inputs


def test_relocate_cache_preserves_posterior_and_fingerprint(tmp_path):
    folder=tmp_path/'work/hierarchical/HPreference_full_age/chain1';folder.mkdir(parents=True)
    draws=folder/'draws.csv';draws.write_text('unchanged posterior bytes\n')
    manifest=folder.parent/'manifest.json'
    manifest.write_text(json.dumps(dict(csv_files=['/old/mac/project/work/hierarchical/HPreference_full_age/chain1/draws.csv'],fingerprint='scientific-target',settings={'draws':2000},sampling_segments={'adaptation_source':[{'csv':'/old/mac/project/work/hierarchical/pilot/chain1.csv'}]})))
    prepare_caches(tmp_path,{'runs':[{'run':'HPreference_full_age'}]})
    data=json.loads(manifest.read_text())
    assert data['csv_files']==[str(draws)]
    assert data['fingerprint']=='scientific-target' and data['settings']=={'draws':2000}
    assert draws.read_text()=='unchanged posterior bytes\n'
    assert data['sampling_segments']['adaptation_source'][0]['csv']==str(tmp_path/'work/hierarchical/pilot/chain1.csv')


def test_incomplete_transfer_does_not_rewrite_manifest(tmp_path):
    folder=tmp_path/'work/hierarchical/H5_full_age';folder.mkdir(parents=True)
    manifest=folder/'manifest.json';original=json.dumps({'csv_files':['/old/work/hierarchical/H5_full_age/missing.csv']});manifest.write_text(original)
    with pytest.raises(RuntimeError,match='Incomplete copied'):prepare_caches(tmp_path,{'runs':[{'run':'H5_full_age'}]})
    assert manifest.read_text()==original


def test_changed_input_prevents_resume(tmp_path):
    path=tmp_path/'choices.csv';path.write_text('original')
    checkpoint={'input_hashes':{'choices.csv':hashlib.sha256(path.read_bytes()).hexdigest()}}
    verify_inputs(tmp_path,checkpoint);path.write_text('modified')
    with pytest.raises(RuntimeError,match='input differs'):verify_inputs(tmp_path,checkpoint)



def test_failed_diagnostics_allow_independent_jobs_but_operational_errors_stop():
    from rf1_trust_socialvalue.checkpoint_handoff import execute_jobs
    from rf1_trust_socialvalue.sampling_retry import DiagnosticFailure
    seen=[];events=[]
    def execute(entry):
        seen.append(entry['run'])
        if entry['run']=='bad':raise DiagnosticFailure('depth hits')
        if entry['run']=='broken':raise OSError('disk full')
    update=lambda *event:events.append(event)
    assert execute_jobs([{'run':'bad'},{'run':'good'}],execute,update)==['bad']
    assert seen==['bad','good'] and events[-1]==('good','complete',None)
    with pytest.raises(OSError):execute_jobs([{'run':'broken'},{'run':'never'}],execute,update)
    assert 'never' not in seen and events[-1]==('broken','error','disk full')


def test_explicit_retry_uses_clean_attempt_and_preserves_failed_cache(tmp_path):
    import pandas as pd
    from rf1_trust_socialvalue.checkpoint_handoff import prepare_retry
    work=tmp_path/'work/hierarchical';table=tmp_path/'results/tables';table.mkdir(parents=True)
    name='HPreference_full_age'
    for suffix,delta,depth in [('',.99,3635),('_attempt95',.95,0)]:
        folder=work/(name+suffix);folder.mkdir(parents=True)
        csv=folder/'draws.csv';csv.write_text('unchanged')
        cfg=dict(warmup=2000,draws=2000,adapt_delta=delta,seed=24,parallel_chains=2)
        (folder/'manifest.json').write_text(json.dumps(dict(csv_files=[str(csv)],settings=cfg,fingerprint='target')))
        pd.DataFrame([dict(passed=False,divergences=0,max_depth_hits=depth,min_bfmi=.63)]).to_csv(table/f'diagnostics_{name+suffix}.csv',index=False)
    settings=prepare_retry(tmp_path,name,2)
    assert settings['adapt_delta']==.95 and settings['warmup']==3000 and settings['draws']==8000
    assert not (work/name).exists()
    assert (work/(name+'_before_manual_retry')/'draws.csv').read_text()=='unchanged'
    assert (work/(name+'_attempt95')/'draws.csv').read_text()=='unchanged'


def test_explicit_retry_refuses_passing_cache(tmp_path):
    import pandas as pd
    from rf1_trust_socialvalue.checkpoint_handoff import prepare_retry
    folder=tmp_path/'work/hierarchical/good';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text('{}')
    table=tmp_path/'results/tables';table.mkdir(parents=True)
    pd.DataFrame([dict(passed=True)]).to_csv(table/'diagnostics_good.csv',index=False)
    with pytest.raises(RuntimeError,match='passing posterior'):prepare_retry(tmp_path,'good',2)
    assert folder.exists()
