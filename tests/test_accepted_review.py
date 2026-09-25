import json

import pandas as pd
import pytest

from rf1_trust_socialvalue.accepted_review import Inputs, diagnostic_pass, paired_difference


def diagnostics(run='good', **overrides):
    values = dict(run=run, R_hat=1.001, ESS_bulk=1000., ESS_tail=1000.,
                  divergences=0, max_depth_hits=0, min_bfmi=.7, passed=True,
                  chains=4, draws_per_chain=2000)
    values.update(overrides)
    return pd.DataFrame([values])


@pytest.mark.parametrize('overrides', [dict(divergences=1), dict(R_hat=1.01),
    dict(ESS_bulk=399), dict(ESS_tail=399), dict(max_depth_hits=1),
    dict(min_bfmi=.3), dict(R_hat=float('nan')), dict(passed=False)])
def test_numerical_gate_cannot_be_overridden_by_passed_flag(overrides):
    assert diagnostic_pass(diagnostics())
    assert not diagnostic_pass(diagnostics(**overrides))


def test_stale_fit_summaries_cannot_enter_review(tmp_path):
    results = tmp_path / 'results'
    tables = results / 'tables'
    tables.mkdir(parents=True)
    statuses = {'good': 'complete', 'status_failed': 'diagnostic_failed', 'numeric_failed': 'complete'}
    (results / 'linux_run_status.json').write_text(json.dumps({'runs': {k: {'status': v} for k, v in statuses.items()}}))
    for run in statuses:
        diagnostics(run, divergences=int(run == 'numeric_failed')).to_csv(tables / f'diagnostics_{run}.csv', index=False)
        pd.DataFrame({'run': [run], 'mean': [42]}).to_csv(tables / f'age_{run}.csv', index=False)
    src = Inputs(tmp_path)
    assert src.accepted == {'good'}
    assert src.fit('age', 'good')['mean'].iloc[0] == 42
    for run in ['status_failed', 'numeric_failed', 'unlisted']:
        with pytest.raises(ValueError, match='Excluded or unknown'):
            src.fit('age', run)
    assert all('age_status_failed' not in path for path in src.hashes)


def test_paired_scores_align_by_id_and_reject_mismatches():
    a = pd.DataFrame({'participant_id': ['b', 'a'], 'n': [10, 12], 'log_loss': [.6, .5]})
    b = pd.DataFrame({'participant_id': ['a', 'b'], 'n': [12, 10], 'log_loss': [.4, .3]})
    assert paired_difference(a, b) == pytest.approx([.1, .3])
    for invalid in [b.iloc[:1], pd.concat([b, b.iloc[:1]]), b.assign(n=[11, 10])]:
        with pytest.raises(ValueError):
            paired_difference(a, invalid)
