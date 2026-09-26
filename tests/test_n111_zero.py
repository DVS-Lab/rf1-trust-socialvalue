import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from rf1_trust_socialvalue import n111_zero as z
from rf1_trust_socialvalue.models import trajectory, simulate


def trials():
    return np.array([[0, 0, 2, 0, 0, 1, 0, 1], [1, 2, 4, 1, 1, 0, 1, 1],
                     [0, 0, 4, 1, 1, 1, 0, 1], [2, 0, 8, -1, 0, 1, 1, 1],
                     [0, 2, 8, 0, 1, 0, 0, 2], [1, 0, 2, 0, 0, 1, 0, 2],
                     [2, 4, 8, 1, 1, 1, 1, 2]], dtype=float)


def params(model):
    p = dict(alpha=.25, kappa=.6, alpha_negative=.12, theta=1.4, theta_stranger=.7, preference_stranger=-.2)
    return {name: p[name] for name in z.SPECS[model][2]}


@pytest.mark.parametrize('model', z.FOCAL)
def test_zero_gamma_exactly_reduces_to_existing_engine(model):
    a = trials(); par = params(model); ordered = np.array(list(par.values()))
    expected = trajectory(a, z.SPECS[model][0], par, np.zeros(3))
    actual = z.zero_engine(a, ordered, z.SPECS[model][1], 0.)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    rng = np.random.default_rng(13)
    sim, expected = simulate(a, z.SPECS[model][0], par, np.zeros(3), rng)
    actual = z.zero_engine(a, ordered, z.SPECS[model][1], 0., True, np.random.default_rng(13).random(len(a)))
    np.testing.assert_allclose(actual, expected, atol=1e-13)


@pytest.mark.parametrize('model', z.FOCAL)
def test_gamma_is_common_unscaled_logit_increment_only_for_zero_offers(model):
    a = trials(); par = np.array(list(params(model).values())); code = z.SPECS[model][1]
    base = z.zero_engine(a, par, code, 0.)
    extended = z.zero_engine(a, par, code, 1.3)
    np.testing.assert_allclose(logit(extended[:, 1])-logit(base[:, 1]), 1.3*(a[:, 1] == 0), atol=1e-12)
    np.testing.assert_allclose(base[:, 0], extended[:, 0])
    changed = a.copy(); changed[2:, 5] = 1-changed[2:, 5]
    other = z.zero_engine(changed, par, code, 1.3)
    np.testing.assert_allclose(extended[:3, 1], other[:3, 1])


def test_generative_zero_choices_hide_outcomes_and_missing_trials_stay_missing():
    a = trials(); par = np.array(list(params('H5').values()))
    low = z.zero_engine(a, par, 5, -100., True, np.ones(len(a)))
    np.testing.assert_array_equal(low[:, 4], [0, 1, 0, 0, 1, 0, 1])
    assert low[3, 2] == -1 and low[2, 0] == .5
    high = z.zero_engine(a, par, 5, 100., True, np.zeros(len(a)))
    np.testing.assert_array_equal(high[:, 4], [1, 1, 1, 0, 1, 1, 1])
    assert high[2, 0] > .5


def test_data_extensions_do_not_mutate_existing_specs_or_age_models():
    original = copy.deepcopy(z.SPECS)
    for model in z.FOCAL:
        base, _, arrays = z.model_data(model, False, False)
        extended, _, _ = z.model_data(model, True, False)
        again, _, _ = z.model_data(model, True, False)
        assert z.SPECS == original
        assert extended == again
        assert extended['K'] == base['K']+1 and len(extended['kind']) == extended['K']
        assert extended['kind'][:-1] == base['kind']
        assert extended['A'] == 0 and extended['N'] == 111
        assert len(extended['zero_option']) == extended['T'] == 8251
        train, _, _ = z.model_data(model, True, True)
        assert train['T'] == 4234
        assert train['mu_location'][-1] == 0 and train['mu_scale'][-1] == 1


def test_paired_metrics_align_participants_and_reject_mismatches():
    base = pd.DataFrame(dict(participant_id=[f's{i}' for i in range(111)], n=[36]*110+[57], log_loss=np.linspace(.3, .8, 111)))
    extended = base.assign(log_loss=base.log_loss-.02).iloc[::-1]
    result = z.paired_metric(extended, base, 'log_loss', 1)
    assert result['mean_delta'] == pytest.approx(-.02)
    assert result['ci_high'] < 0
    with pytest.raises(ValueError):
        z.paired_metric(extended.iloc[:-1], base, 'log_loss', 1)
    with pytest.raises(ValueError):
        z.paired_metric(extended.assign(n=35), base, 'log_loss', 1)


def test_macos_cannot_launch_new_sampler(monkeypatch):
    monkeypatch.setattr(z.platform, 'system', lambda: 'Darwin')
    with pytest.raises(RuntimeError, match='only on linux1'):
        z.fit_entry(z.entries()[0], 4)


def test_collector_excludes_failed_fit_even_with_stale_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    z.TABLE.mkdir(parents=True)
    states = {e['name']: {'status': 'diagnostic_failed'} for e in z.entries()}
    # Presence of stale numerical outputs must never be treated as acceptance.
    (z.TABLE/'zero_heldout_N111_H5_zero_train.csv').write_text('this stale file must not be read')
    done = z.collect_results(states)
    assert not done
    d = pd.read_csv(z.TABLE/'zero_option_model_comparison.csv')
    assert len(d) == 12 and d.status.eq('unavailable').all()
    assert 'Partial results' in (z.OUT/'zero_option_report.md').read_text()


def test_full_and_heldout_export_then_comparison_without_sampling(tmp_path, monkeypatch):
    # Synthetic draws on the fixed original schedule test export/aggregation, never inference.
    repo = Path(__file__).resolve().parents[1]
    data, meta, arrays = z.model_data('H5', False, False)
    monkeypatch.chdir(tmp_path)
    z.TABLE.mkdir(parents=True)
    (z.OUT/'figures').mkdir()
    monkeypatch.setattr(z, 'settings', lambda: dict(predictive_draws=4, seed=1))
    states = {e['name']: {'status': 'diagnostic_failed'} for e in z.entries()}
    names = ['N111_H5_base_full', 'N111_H5_zero_full', 'N111_H5_zero_train']
    for name in names:
        entry = next(e for e in z.entries() if e['name'] == name)
        par = list(params('H5').values())+([.8] if entry['zero'] else [])
        draws = np.tile(par, (4, 111, 1))
        if entry['training']:
            z.heldout_outputs(entry, draws, meta, arrays)
        else:
            z.full_outputs(entry, draws, meta, arrays)
        pd.DataFrame([dict(run=name, R_hat=1., ESS_bulk=1000, ESS_tail=1000, divergences=0,
                          max_depth_hits=0, min_bfmi=.7, passed=True)]).to_csv(z.TABLE/f'zero_diagnostics_{name}.csv', index=False)
        states[name] = {'status': 'complete'}
    baseline = Path('results/tables'); baseline.mkdir()
    import shutil
    shutil.copyfile(repo/'results/tables/heldout_H5_train_noage.csv', baseline/'heldout_H5_train_noage.csv')
    assert not z.collect_results(states)  # Other three models remain unavailable.
    d = pd.read_csv(z.TABLE/'zero_option_model_comparison.csv')
    assert d[d.model.eq('H5')].status.eq('complete').all()
    ppc = pd.read_csv(z.TABLE/'zero_option_ppc_comparison.csv')
    assert len(ppc) > 0 and set(ppc.model) == {'H5'}
    pars = pd.read_csv(z.TABLE/'zero_option_parameter_summary.csv')
    assert len(pars[pars.parameter.eq('gamma0')]) == 111
    assert set(pars.variant) == {'base', 'zero'}
