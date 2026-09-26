from contextlib import nullcontext
import json
from pathlib import Path

import pandas as pd
import pytest

from rf1_trust_socialvalue import n111_zero as z
from rf1_trust_socialvalue import n111_zero_retry as r
from rf1_trust_socialvalue import linux_handoff as h


def test_reviewed_plan_matches_uploaded_batch():
    plan, states = r.validate_plan()
    assert sum(s['status'] == 'complete' for s in states.values()) == 11
    assert plan['settings'] == dict(z.settings(), max_treedepth=14)


def test_changed_reviewed_evidence_blocks_retry(tmp_path, monkeypatch):
    plan = json.loads(r.PLAN.read_text())
    plan['evidence_sha256']['config/n111_zero.json'] = 'changed'
    p = tmp_path/'plan.json'; p.write_text(json.dumps(plan))
    monkeypatch.setattr(r, 'PLAN', p)
    with pytest.raises(ValueError, match='evidence changed'):
        r.validate_plan()


def test_retry_cannot_sample_on_laptop(monkeypatch):
    monkeypatch.setattr(r.platform, 'system', lambda: 'Darwin')
    with pytest.raises(RuntimeError, match='linux1'):
        r.run()


@pytest.mark.parametrize('code', [0, 2])
def test_only_selected_retry_runs_and_failed_attempt_is_never_substituted(tmp_path, monkeypatch, code):
    cfg = dict(z.settings(), max_treedepth=14)
    states = {e['name']: {'status': 'complete'} for e in z.entries()}
    states[r.ORIGINAL] = {'status': 'diagnostic_failed'}
    monkeypatch.setattr(r.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(h, 'coordinator_lock', lambda root: nullcontext())
    monkeypatch.setattr(h, 'existing_workers', lambda root: [])
    monkeypatch.setattr(h, 'assert_no_live_fit_locks', lambda root: None)
    monkeypatch.setattr(z, 'gate_stage_a', lambda: None)
    monkeypatch.setattr(r, 'validate_plan', lambda: ({'settings': cfg}, states))
    monkeypatch.setattr(r, 'STATUS', tmp_path/'status.json')
    calls = []; collected = []
    def fit(entry, parallel_chains, cfg):
        calls.append((entry, parallel_chains, cfg)); return code
    monkeypatch.setattr(z, 'fit_entry', fit)
    monkeypatch.setattr(z, 'collect_results', lambda s, mapping: collected.append((s, mapping)) or True)
    assert r.run() == code
    assert len(calls) == 1
    assert calls[0][0] == dict(name=r.RETRY, model='HPreference', zero=True, training=False, seed_name=r.ORIGINAL)
    assert calls[0][1:] == (4, cfg)
    assert len(collected) == int(code == 0)
    if code == 0:
        assert collected[0][1] == {r.ORIGINAL: r.RETRY}
    status = json.loads(r.STATUS.read_text())
    assert len(status['reused_runs']) == 11
    assert status['exit_code'] == code


def test_collector_rejects_unreviewed_source_replacement():
    with pytest.raises(ValueError, match='reviewed'):
        z.collect_results({}, {'N111_H5_zero_full': 'another_run'})


def test_collector_reads_retry_source_and_never_failed_original(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    z.TABLE.mkdir(parents=True)
    # Use accepted fixture tables with a new source label; no artificial posterior fitting.
    for kind in ['ppc', 'parameters']:
        frame = pd.read_csv(repo/z.TABLE/f'zero_{kind}_N111_HPreference_base_full.csv')
        frame['run'] = r.RETRY; frame['variant'] = 'zero'
        frame.to_csv(z.TABLE/f'zero_{kind}_{r.RETRY}.csv', index=False)
        (z.TABLE/f'zero_{kind}_{r.ORIGINAL}.csv').write_text('excluded stale file')
    d = pd.read_csv(repo/z.TABLE/'zero_diagnostics_N111_HPreference_base_full.csv')
    d['run'] = r.RETRY
    d.to_csv(z.TABLE/f'zero_diagnostics_{r.RETRY}.csv', index=False)
    (z.TABLE/f'zero_diagnostics_{r.ORIGINAL}.csv').write_text('excluded stale file')
    monkeypatch.setattr(z, 'entries', lambda: [dict(name=r.ORIGINAL)])
    monkeypatch.setattr(z, 'FOCAL', ['HPreference'])
    monkeypatch.setattr(z, 'plot_comparison', lambda *a, **k: None)
    assert z.collect_results({r.ORIGINAL: dict(status='complete')}, {r.ORIGINAL: r.RETRY})
    assert set(pd.read_csv(z.TABLE/'zero_option_parameter_summary.csv').run) == {r.RETRY}
    assert r.RETRY in (z.OUT/'zero_option_report.md').read_text()
