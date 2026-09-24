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
