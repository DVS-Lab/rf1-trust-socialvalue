import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rf1_trust_socialvalue import full_sample_residuals as r
from rf1_trust_socialvalue import full_sample_ppc as ppc
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials


def test_age_slopes_recover_known_decade_trend_and_bootstrap_people():
    age = np.linspace(20, 80, 40)
    observed = .2 + .004*(age-20)
    predicted = .2 + .001*(age[:, None]-20) + np.linspace(-.03, .03, 20)[None, :]
    result = r.cell_summary(observed, predicted, age, 19, 100)
    assert result['age_status'] == 'estimable'
    assert result['observed_age_slope'] == pytest.approx(.04)
    assert result['predicted_age_slope'] == pytest.approx(.01)
    for key in ['residual_age_slope', 'residual_age_predictive_low', 'residual_age_predictive_high', 'residual_age_bootstrap_low', 'residual_age_bootstrap_high']:
        assert result[key] == pytest.approx(.03)
    # Each participant is one observation even if their underlying cell had
    # many choices. Predictive uncertainty remains separate from resampling.
    assert result['participants'] == 40
    assert result['residual_predictive_high'] > result['residual_predictive_low']
    assert r.cell_summary(observed, predicted, np.ones(40)*30, 19, 100)['age_status'] == 'not_estimable'


def test_age_alignment_uses_id_order_and_refuses_missing_or_inconsistent_age():
    t = trials(3); t['age'] = t.participant_id.map({'sub-0':20, 'sub-1':40, 'sub-2':60})
    np.testing.assert_array_equal(r.ages_from_trials(t, ['sub-2','sub-0','sub-1']), [60,20,40])
    bad = t.copy(); bad.loc[0, 'age'] = 21
    with pytest.raises(ValueError, match='varies'): r.ages_from_trials(bad, ['sub-0','sub-1','sub-2'])
    t.loc[t.participant_id.eq('sub-1'), 'age'] = np.nan
    with pytest.raises(ValueError, match='Missing'): r.ages_from_trials(t, ['sub-0','sub-1','sub-2'])


def test_residual_tables_match_original_ppc_and_preserve_misses():
    _, _, arrays = s.model_data(trials(12), 'H7', True)
    natural = np.ones((20,12,5))*.5
    result, selected = r.tables(arrays, np.linspace(20,80,12), natural, 'H7', 10, 42, 50)
    original, _, original_selected = ppc.ppc_tables(arrays, natural, dict(model='H7',variant='zero'), 10, 42)
    np.testing.assert_array_equal(selected, original_selected)
    for history in ['conditional','generative']:
        row = result[result.history.eq(history) & result.stratification.eq('all')].iloc[0]
        reference = original[original.history.eq(history) & original.stratification.eq('all') & original.metric.eq('high_choice') & original.statistic.eq('mean')].iloc[0]
        assert row['choices'] == 36 and row['participants'] == 12
        assert row['observed'] == pytest.approx(reference.observed)
        assert row['predicted_mean'] == pytest.approx(reference.predicted_mean)
        assert row['predicted_ci_low'] == pytest.approx(reference.ci_low)
    offers = result[result.stratification.eq('partner_offer')]
    assert set(offers.group) == {'friend_0/2','stranger_2/4','computer_4/8'}
    assert set(offers.choices) == {12}  # Missing computer 0/8 never enters.


def test_verified_posterior_blocks_corrupt_raw_csv_before_loading(tmp_path, monkeypatch):
    p, phase, c = ppc.configuration()
    c = dict(c, _root=str(tmp_path), output=str(tmp_path/'out'), work=str(tmp_path/'work'))
    name = r.TARGETS[0]; entry = dict(name=name,model='H7',variant='zero',subset='full')
    t = trials(2); data, meta, _ = s.model_data(t,'H7',True,phase['gamma_population_mean_prior_sd'])
    for file in s.SOURCES:
        f = tmp_path/file; f.parent.mkdir(parents=True,exist_ok=True); f.write_text('synthetic')
    out, work = s.paths(c); evidence = out/'fits'/name; cache = work/'fits'/name; cache.mkdir(parents=True)
    csv = cache/'draw.csv'; csv.write_text('synthetic posterior')
    settings = dict(draws=20,chains=4)
    payload = dict(data=data,settings=settings,sampler_seed=42,stan={file:s.fs.sha(tmp_path/file) for file in s.SOURCES},cmdstan_version=phase['cmdstan_version'])
    fingerprint = s.digest(payload)
    s.save(evidence/'status.json',dict(status='complete',passed=True,fingerprint=fingerprint,settings=settings,sampler_seed=42))
    manifest = dict(fingerprint=fingerprint,meta=meta,csv_files=[str(csv)],posterior_sha256={str(csv):s.fs.sha(csv)})
    s.save(cache/'manifest.json',manifest)
    s.save(evidence/'manifest_summary.json',dict(manifest,posterior_sha256={csv.name:s.fs.sha(csv)}))
    monkeypatch.setattr(ppc.review,'resolved_entries',lambda *a:[entry])
    monkeypatch.setattr(s,'models',lambda *a:pytest.fail('No Stan compilation or sampling'))
    calls = []
    def load(files):
        calls.append(files); return SimpleNamespace(stan_variable=lambda _:np.ones((20,2,5))*.5)
    monkeypatch.setattr(s,'load_chains',load)
    _, arrays, natural, provenance = r.verified_posterior(phase,c,name,t)
    assert len(arrays) == 2 and natural.shape == (20,2,5)
    assert provenance['posterior_fingerprint'] == fingerprint and len(calls) == 1
    csv.write_text('corrupted')
    with pytest.raises(ValueError,match='Raw posterior changed'): r.verified_posterior(phase,c,name,t)
    assert len(calls) == 1
