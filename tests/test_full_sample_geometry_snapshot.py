import json
from pathlib import Path
import pytest
from rf1_trust_socialvalue import full_sample_geometry as geometry
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials


def snapshot(tmp_path):
    out=tmp_path/'results';work=tmp_path/'work';bids=tmp_path/'bids'
    out.mkdir();work.mkdir();bids.mkdir()
    config=tmp_path/'config.json';config.write_text('{}')
    code=tmp_path/'frozen_parser.py';code.write_text('# frozen implementation')
    table=work/'canonical_trials.tsv';t=trials(2);t.to_csv(table,sep='\t',index=False)
    template=bids/'sub-1/ses-01/func/sub-1_ses-01_task-trust_run-2_part-mag_events.tsv'
    template.parent.mkdir(parents=True);template.write_text('onset\tduration\n')
    prov=dict(input_hashes={str(config):s.fs.sha(config),str(template):s.fs.sha(template)},
        implementation_hashes={str(code):s.fs.sha(code)},artifact_hashes={str(table):s.fs.sha(table)})
    s.save(out/'provenance.json',prov)
    state=dict(status='ready_for_review',cohort_summary=dict(primary_n=2,total_valid_choices=6,unexplained_parity_fields=0),
        output_hashes={'provenance.json':s.fs.sha(out/'provenance.json')})
    s.save(out/'milestone_status.json',state)
    phase=dict(integration_status_sha256=s.fs.sha(out/'milestone_status.json'))
    c=dict(output=str(out),work=str(work),bids_root=str(bids),_config_path=str(config),_root=str(tmp_path))
    return phase,c,prov,template


def test_modified_imaging_template_is_audited_without_changing_saved_trials(tmp_path,monkeypatch):
    phase,c,prov,template=snapshot(tmp_path)
    before=Path(c['work'],'canonical_trials.tsv').read_bytes()
    template.write_text('onset\tduration\tnote\n1\t2\tnew-content\n')
    monkeypatch.setattr(s,'integrated_trials',lambda *a:pytest.fail('Export must not use the live fitting gate'))
    t,p=geometry.saved_trials(phase,c)
    audit=geometry.source_drift_audit(c,p)
    assert len(t)==8 and audit['status']=='drift_detected'
    assert len(audit['changes'])==1
    row=audit['changes'][0]
    assert row['change']=='modified' and row['imaging_event_file'] and row['current_rows']==1
    assert row['frozen_sha256']==prov['input_hashes'][str(template)]
    assert row['current_sha256']==s.fs.sha(template)
    assert Path(c['work'],'canonical_trials.tsv').read_bytes()==before


def test_deleted_and_added_live_files_are_recorded(tmp_path):
    phase,c,prov,template=snapshot(tmp_path);template.unlink()
    added=template.with_name('sub-1_ses-01_task-trust_run-1_events.tsv');added.write_text('onset\n1\n')
    geometry.saved_trials(phase,c)
    audit=geometry.source_drift_audit(c,prov)
    assert {r['change'] for r in audit['changes']}=={'missing','added'}


@pytest.mark.parametrize('changed',['canonical','provenance','parser','config','milestone'])
def test_historical_snapshot_tampering_still_blocks_export(tmp_path,changed):
    phase,c,_,_=snapshot(tmp_path)
    paths=dict(canonical=Path(c['work'])/'canonical_trials.tsv',provenance=Path(c['output'])/'provenance.json',
        parser=tmp_path/'frozen_parser.py',config=Path(c['_config_path']),milestone=Path(c['output'])/'milestone_status.json')
    with paths[changed].open('a') as f:f.write('\nchanged\n')
    with pytest.raises(ValueError):geometry.saved_trials(phase,c)


def test_missing_canonical_hash_cannot_bypass_snapshot_authentication(tmp_path):
    phase,c,prov,_=snapshot(tmp_path)
    prov['artifact_hashes']={};s.save(Path(c['output'])/'provenance.json',prov)
    state=json.loads(Path(c['output'],'milestone_status.json').read_text())
    state['output_hashes']['provenance.json']=s.fs.sha(Path(c['output'])/'provenance.json')
    s.save(Path(c['output'])/'milestone_status.json',state)
    phase['integration_status_sha256']=s.fs.sha(Path(c['output'])/'milestone_status.json')
    with pytest.raises(ValueError,match='not authenticated'):geometry.saved_trials(phase,c)


def test_geometry_preparation_failure_writes_status(tmp_path,monkeypatch):
    phase,c,_,_=snapshot(tmp_path)
    monkeypatch.setattr(s,'require_linux',lambda:None)
    monkeypatch.setattr(s,'configuration',lambda *a:(phase,c,{}))
    monkeypatch.setattr(s.fs,'git_sha',lambda *a:'test')
    def fail(*a):raise ValueError('saved canonical table changed')
    monkeypatch.setattr(geometry,'saved_trials',fail)
    with pytest.raises(ValueError,match='canonical'):geometry.main()
    state=json.loads(Path(c['output'],'hierarchical/geometry_review/status.json').read_text())
    assert state['status']=='error' and state['new_sampling'] is False


def test_export_completes_with_live_drift_without_using_live_fitting_gate(tmp_path,monkeypatch):
    phase,c,_,template=snapshot(tmp_path);template.unlink()
    monkeypatch.setattr(s,'require_linux',lambda:None)
    monkeypatch.setattr(s,'configuration',lambda *a:(phase,c,{}))
    monkeypatch.setattr(s.fs,'git_sha',lambda *a:'test')
    monkeypatch.setattr(s,'integrated_trials',lambda *a:pytest.fail('Must retain historical snapshot path'))
    monkeypatch.setattr(geometry,'TARGETS',[])
    geometry.main()
    dest=Path(c['output'])/'hierarchical/geometry_review'
    state=json.loads((dest/'status.json').read_text())
    assert state['status']=='complete' and state['live_source_status']=='drift_detected'
    assert state['verification_mode']=='historical_snapshot'
    assert json.loads((dest/'live_source_audit_after.json').read_text())['changes'][0]['change']=='missing'


def test_repair_receipt_and_archived_templates_are_checked(tmp_path):
    phase,c,prov,template=snapshot(tmp_path)
    archive=tmp_path/'upstream/derivatives/source_repairs/10668-sharedreward-v1'
    archive.mkdir(parents=True);c['upstream_root']=str(tmp_path/'upstream')
    name='sub-10668_ses-01_task-trust_run-2_part-mag_events.tsv'
    preserved=archive/'original_session/func'/name;preserved.parent.mkdir(parents=True)
    preserved.write_text('onset\tduration\n')
    h=s.fs.sha(preserved);source=Path(c['bids_root'])/'sub-10668/ses-01/func'/name
    prov['input_hashes'][str(source)]=h
    s.save(archive/'receipt.json',dict(id='10668-sharedreward-v1',status='complete',original_files={'func/'+name:h}))
    audit=geometry.source_drift_audit(c,prov)
    receipt=next(r for r in audit['repair_10668'] if r['path'].endswith('/receipt.json'))
    assert receipt['status']=='complete' and receipt['implicated_originals'][0]['archive_matches_freeze']
    assert receipt['implicated_originals'][0]['receipt_original_sha256']==h
    preserved.write_text('altered')
    assert not geometry.repair_10668_evidence(c,prov)[-1]['implicated_originals'][0]['archive_matches_freeze']
