import json
from pathlib import Path
import pandas as pd
import pytest
from rf1_trust_socialvalue.targeted_retry import prepare_retries,audit_divergent_draws


def setup_case(root,name='H4_full_age',fingerprint='reviewed'):
    folder=root/'work/hierarchical'/name;folder.mkdir(parents=True)
    table=root/'results/tables';table.mkdir(parents=True,exist_ok=True)
    csv=folder/'draws.csv';csv.write_text('divergent__,mu.1,tau.1\n0,1,0.5\n1,3,0.1\n0,2,0.3\n')
    cfg=dict(seed=24,chains=1,parallel_chains=1,draws=3,warmup=2,adapt_delta=.99,metric='diag_e',max_treedepth=12)
    (folder/'manifest.json').write_text(json.dumps(dict(fingerprint=fingerprint,settings=cfg,csv_files=[str(csv)])))
    pd.DataFrame([dict(passed=False,R_hat=1.001,ESS_bulk=800,ESS_tail=900,divergences=1,max_depth_hits=0,min_bfmi=.7)]).to_csv(table/f'diagnostics_{name}.csv',index=False)
    entry=dict(run=name,kind='real_data',settings=cfg)
    spec=dict(expected_fingerprint='reviewed',expected_divergences=1,overrides=dict(adapt_delta=.995,warmup=4,draws=4))
    return entry,spec


def test_retry_preserves_original_and_is_idempotent(tmp_path):
    entry,spec=setup_case(tmp_path);plan=dict(id='test',runs={entry['run']:spec});checkpoint={'runs':[entry]}
    updated=prepare_retries(tmp_path,checkpoint,plan)
    folder=tmp_path/'work/hierarchical'/entry['run'];archive=folder.with_name(folder.name+'_retry_test')
    assert not folder.exists() and (archive/'draws.csv').read_text().startswith('divergent__')
    assert updated[0]['settings']['adapt_delta']==.995 and entry['settings']['adapt_delta']==.99
    assert prepare_retries(tmp_path,checkpoint,plan)==updated  # safe after interruption before resampling
    folder.mkdir();current=dict(fingerprint='new-target-settings',settings=updated[0]['settings'],csv_files=[])
    (folder/'manifest.json').write_text(json.dumps(current))
    assert prepare_retries(tmp_path,checkpoint,plan)==updated
    assert json.loads((folder/'manifest.json').read_text())==current
    assert len(list(folder.parent.glob('*retry*')))==1


def test_all_targets_preflight_before_any_archive(tmp_path):
    first,spec=setup_case(tmp_path,'one');second,bad=setup_case(tmp_path,'two',fingerprint='unreviewed')
    plan=dict(id='test',runs={'one':spec,'two':bad})
    with pytest.raises(RuntimeError,match='fingerprint differs'):prepare_retries(tmp_path,{'runs':[first,second]},plan)
    assert (tmp_path/'work/hierarchical/one/manifest.json').exists()


def test_passing_fit_is_never_replaced(tmp_path):
    entry,spec=setup_case(tmp_path)
    path=tmp_path/'results/tables'/f'diagnostics_{entry["run"]}.csv';diag=pd.read_csv(path);diag['passed']=True;diag['divergences']=0;diag.to_csv(path,index=False)
    with pytest.raises(RuntimeError,match='no longer match'):prepare_retries(tmp_path,{'runs':[entry]},dict(id='test',runs={entry['run']:spec}))
    assert (tmp_path/'work/hierarchical'/entry['run']/'manifest.json').exists()


def test_retry_cannot_change_seed_or_reduce_draws(tmp_path):
    entry,spec=setup_case(tmp_path);spec['overrides']['seed']=25
    with pytest.raises(ValueError,match='only integrator'):prepare_retries(tmp_path,{'runs':[entry]},dict(id='test',runs={entry['run']:spec}))
    del spec['overrides']['seed'];spec['overrides']['draws']=1
    with pytest.raises(ValueError,match='without reducing'):prepare_retries(tmp_path,{'runs':[entry]},dict(id='test',runs={entry['run']:spec}))


def test_divergence_audit_uses_archived_draws_without_modifying_them(tmp_path):
    entry,spec=setup_case(tmp_path);plan=dict(id='test',runs={entry['run']:spec})
    prepare_retries(tmp_path,{'runs':[entry]},plan)
    original=tmp_path/'work/hierarchical'/f'{entry["run"]}_retry_test/draws.csv';before=original.read_bytes()
    audit_divergent_draws(tmp_path,plan)
    rows=pd.read_csv(tmp_path/'results/tables'/f'divergence_locations_test_{entry["run"]}.csv')
    assert set(rows.parameter)=={'mu.1','tau.1'}
    assert rows.retained_iteration.eq(2).all() and rows.chain.eq(1).all()
    assert rows.set_index('parameter').loc['mu.1','percentile']==1
    assert original.read_bytes()==before


def test_divergence_audit_rejects_partial_or_different_draws(tmp_path):
    entry,spec=setup_case(tmp_path);plan=dict(id='test',runs={entry['run']:spec})
    prepare_retries(tmp_path,{'runs':[entry]},plan)
    path=tmp_path/'work/hierarchical'/f'{entry["run"]}_retry_test/draws.csv'
    path.write_text('divergent__,mu.1,tau.1\n0,1,0.5\n')
    with pytest.raises(RuntimeError,match='Incomplete retained'):audit_divergent_draws(tmp_path,plan)
