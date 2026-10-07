import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample_closeout as q
from rf1_trust_socialvalue import full_sample_closeout_snapshot as snap
from rf1_trust_socialvalue import full_sample_amount as a
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials

ROOT=Path(__file__).resolve().parents[1]


def setup_qc(tmp_path,monkeypatch):
    root=tmp_path/'upstream';qc=root/'qc/events/results';qc.mkdir(parents=True)
    old=b'subject\tsession\ttask\trun\r\n10668\t01\ttrust\t1\r\n'
    added=b'10668\t01\tsharedreward\t2\r\n';new=old+added
    table=qc/'run_response_qc.tsv';table.write_bytes(new)
    prov=qc/'provenance.json';before=b'{"events_run_count": 1}';after=b'{"events_run_count": 2}'
    prov.write_bytes(after);h=lambda b:hashlib.sha256(b).hexdigest()
    proof=dict(reviewed_upstream_commit='reviewed',trust_rows_unchanged=1,reason='One non-Trust row added',
               added_qc_line=added.decode(),approved_qc_provenance=json.loads(after),
               approved_transitions={
                   'qc/events/results/run_response_qc.tsv':dict(frozen_sha256=h(old),current_sha256=h(new)),
                   'qc/events/results/provenance.json':dict(frozen_sha256=h(before),current_sha256=h(after))})
    monkeypatch.setattr(snap,'review',lambda c:proof)
    c=dict(_root=str(ROOT),upstream_root=str(root),response_qc=str(table),qc_provenance=str(prov),
           bids_root='/ZPOOL/data/projects/rf1-sra-linux2/bids')
    audit=dict(status='drift_detected',read_errors=[],repair_10668=[],changes=[
        dict(path=str(root/rel),change='modified',**values) for rel,values in proof['approved_transitions'].items()])
    return c,audit,proof


def test_production_review_is_pinned_and_specific_to_sharedreward():
    proof=snap.review({'_root':str(ROOT)})
    assert proof['added_qc_row']['task']=='sharedreward' and proof['added_qc_row']['subject']=='10668'
    assert proof['added_qc_row']['run']=='2' and proof['trust_rows_unchanged']==647
    before=proof['frozen_qc_provenance'];after=proof['approved_qc_provenance']
    assert {k for k in before if before[k]!=after[k]}=={'generated_at','events_run_count','events_manifest_sha256'}
    assert after['events_run_count']==before['events_run_count']+1
    frozen=json.loads((ROOT/'results/full_sample/provenance.json').read_text())
    for rel,hashes in proof['approved_transitions'].items():
        assert frozen['input_hashes']['/ZPOOL/data/projects/rf1-sra-linux2/'+rel]==hashes['frozen_sha256']


def test_review_artifact_cannot_be_changed(tmp_path):
    path=tmp_path/snap.REVIEW_FILE;path.parent.mkdir(parents=True);path.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):snap.review({'_root':str(tmp_path)})


def test_exact_qc_pair_passes_and_does_not_modify_audit_or_files(tmp_path,monkeypatch):
    c,audit,_=setup_qc(tmp_path,monkeypatch);original=copy.deepcopy(audit)
    content=Path(c['response_qc']).read_bytes();metadata=Path(c['qc_provenance']).read_bytes()
    result=snap.validate_live_audit(audit,c)
    assert result['qc_transition_applied'] and result['trust_qc_rows_unchanged']==1
    assert audit==original and Path(c['response_qc']).read_bytes()==content
    assert Path(c['qc_provenance']).read_bytes()==metadata
    # Original strict amount gate remains strict, preserving old target provenance.
    with pytest.raises(ValueError,match='Unreviewed live input'):a.validate_live_audit(audit,c)


@pytest.mark.parametrize('problem',['current_hash','frozen_hash','deleted','half_pair','read_error','trust_events','qc_policy','eligibility','changed_after_audit'])
def test_other_drift_still_blocks(tmp_path,monkeypatch,problem):
    c,audit,proof=setup_qc(tmp_path,monkeypatch)
    if problem=='current_hash':audit['changes'][0]['current_sha256']='other'
    if problem=='frozen_hash':audit['changes'][0]['frozen_sha256']='other'
    if problem=='deleted':audit['changes'][0]['change']='missing'
    if problem=='half_pair':audit['changes'].pop()
    if problem=='read_error':audit['read_errors']=[dict(error='permission denied')]
    if problem in ['trust_events','qc_policy','eligibility']:
        path={'trust_events':c['bids_root']+'/sub-10668/ses-01/func/sub-10668_ses-01_task-trust_run-1_events.tsv',
              'qc_policy':str(Path(c['upstream_root'])/'qc/events/policy.json'),
              'eligibility':str(Path(c['upstream_root'])/'qc/trust_analysis/run_eligibility.tsv')}[problem]
        audit['changes'].append(dict(path=path,change='modified',frozen_sha256='old',current_sha256='new'))
    if problem=='changed_after_audit':Path(c['response_qc']).write_bytes(b'changed')
    with pytest.raises(ValueError):snap.validate_live_audit(audit,c)


def test_exact_reconstruction_required_even_if_hash_pair_were_edited(tmp_path,monkeypatch):
    c,audit,proof=setup_qc(tmp_path,monkeypatch)
    altered=Path(c['response_qc']).read_bytes().replace(b'trust\t1',b'trust\t2')
    Path(c['response_qc']).write_bytes(altered)
    h=hashlib.sha256(altered).hexdigest()
    proof['approved_transitions']['qc/events/results/run_response_qc.tsv']['current_sha256']=h
    audit['changes'][0]['current_sha256']=h
    with pytest.raises(ValueError,match='exact frozen QC'):snap.validate_live_audit(audit,c)


def test_prior_empty_template_receipt_check_is_preserved(tmp_path,monkeypatch):
    c,audit,_=setup_qc(tmp_path,monkeypatch)
    previous=json.loads((ROOT/'results/full_sample/hierarchical/residual_review/live_source_audit.json').read_text())
    audit['changes']+=previous['changes'];audit['repair_10668']=previous['repair_10668']
    assert snap.validate_live_audit(audit,c)['qc_transition_applied']
    audit['repair_10668']=[]
    with pytest.raises(ValueError,match='Unreviewed live input'):snap.validate_live_audit(audit,c)


def test_snapshot_saves_complete_blocked_audit_before_raising(tmp_path,monkeypatch):
    c,audit,_=setup_qc(tmp_path,monkeypatch);audit['changes'][0]['current_sha256']='unreviewed'
    monkeypatch.setattr(a.ppc.geometry,'saved_trials',lambda *args:(None,{}))
    monkeypatch.setattr(a.ppc.geometry,'source_drift_audit',lambda *args:audit)
    record=tmp_path/'out/audit.json'
    with pytest.raises(ValueError,match='Unreviewed global QC'):snap.snapshot({}, {}, c, record)
    saved=json.loads(record.read_text())
    assert saved['closeout_validation']['status']=='blocked' and len(saved['changes'])==2
    assert saved['changes'][0]['current_sha256']=='unreviewed'


def test_snapshot_keeps_paired_counts_and_baseline_review(tmp_path,monkeypatch):
    c,audit,_=setup_qc(tmp_path,monkeypatch);t=trials(2);t=pd.concat([t,t.assign(run=2)],ignore_index=True)
    monkeypatch.setattr(a.ppc.geometry,'saved_trials',lambda *args:(t,{}))
    monkeypatch.setattr(a.ppc.geometry,'source_drift_audit',lambda *args:copy.deepcopy(audit))
    calls=[];monkeypatch.setattr(a.batch,'review_reused',lambda *args:calls.append(args))
    old=dict(expected_train_n=2,expected_train_choices=6,expected_test_choices=6)
    phase=dict(gamma_population_mean_prior_sd=1.,cmdstan_version='2.40.0')
    record=tmp_path/'out/audit.json';returned,saved=snap.snapshot(old,phase,c,record)
    pd.testing.assert_frame_equal(returned,t)
    assert saved['closeout_validation']['status']=='passed' and len(calls)==1
    assert len(saved['changes'])==2 # Accepted exceptions remain visible.
    with pytest.raises(ValueError,match='cohort/count'):snap.snapshot(dict(old,expected_train_n=3),phase,c,record)
    assert len(calls)==1 and json.loads(record.read_text())['closeout_validation']['status']=='blocked'
