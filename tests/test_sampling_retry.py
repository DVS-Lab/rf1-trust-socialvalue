import json
from pathlib import Path
import pytest
from rf1_trust_socialvalue.sampling_retry import retry_settings,archive_fit


def test_precision_retry_keeps_integrator_then_stops_at_budget():
    cfg=dict(warmup=2000,draws=2000,adapt_delta=.95,seed=24,metric='diag_e')
    diagnostics=dict(passed=False,divergences=0,max_depth_hits=0,min_bfmi=.6)
    reason,retry=retry_settings(cfg,**diagnostics)
    assert reason=='short' and retry==dict(cfg,warmup=3000,draws=8000)
    reason,retry=retry_settings(retry,**diagnostics)
    assert reason=='8000' and retry['draws']==16000
    reason,retry=retry_settings(retry,**diagnostics)
    assert reason=='delta' and retry['adapt_delta']==.99
    assert retry_settings(retry,**diagnostics) is None
    assert retry_settings(cfg,**dict(diagnostics,passed=True)) is None


def test_depth_failure_does_not_trigger_longer_chains():
    cfg=dict(warmup=2000,draws=2000,adapt_delta=.99)
    assert retry_settings(cfg,passed=False,divergences=0,max_depth_hits=3635,min_bfmi=.63) is None


def test_archive_preserves_nested_draws_and_existing_attempt(tmp_path):
    folder=tmp_path/'fit';(folder/'chain1').mkdir(parents=True)
    csv=folder/'chain1/draws.csv';csv.write_text('posterior bytes')
    saved={'csv_files':[str(csv)],'fingerprint':'unchanged','settings':{'seed':1}}
    old=tmp_path/'fit_attempt95';old.mkdir();(old/'keep.txt').write_text('keep')
    archive=archive_fit(folder,saved,'_attempt95')
    assert archive.name=='fit_attempt95_2'
    restored=json.loads((archive/'manifest.json').read_text())
    assert restored['fingerprint']=='unchanged'
    assert Path(restored['csv_files'][0]).read_text()=='posterior bytes'
    assert (old/'keep.txt').read_text()=='keep'


def test_archive_rejects_chain_outside_fit_before_rename(tmp_path):
    folder=tmp_path/'fit';folder.mkdir()
    with pytest.raises(ValueError):archive_fit(folder,{'csv_files':[str(tmp_path/'outside.csv')]},'_attempt')
    assert folder.exists()
