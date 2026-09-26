import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from rf1_trust_socialvalue import n111_recovery as r
from rf1_trust_socialvalue import n111_zero as z
from rf1_trust_socialvalue.models import trajectory
from rf1_trust_socialvalue.hierarchical import SPECS


def trials(n=60):
    rng = np.random.default_rng(54)
    a = np.zeros((n, 8)); a[:, 0] = rng.integers(0, 3, n)
    a[:, 1] = rng.choice([0., 2.], n); a[:, 2] = rng.choice([4., 8.], n)
    a[:, 3] = rng.integers(0, 2, n); a[5, 3] = -1
    a[:, 4] = (a[:, 3] >= 0)&((a[:, 1] > 0)|(a[:, 3] == 1))
    a[:, 5] = rng.integers(0, 2, n); a[:, 7] = np.where(np.arange(n) < n//2, 1, 2)
    return a


def parameters(model, zero):
    values = dict(alpha=.24, kappa=.42, theta=2.7, theta_stranger=.9, preference_stranger=-.3, alpha_negative=.12, gamma0=1.4)
    return np.array([values[p] for p in r.parameter_names(model, zero)])


@pytest.mark.parametrize('model', r.FOCAL)
@pytest.mark.parametrize('zero', [False, True])
def test_fast_gradient_matches_reference_likelihood_and_finite_difference(model, zero):
    a = trials(); x = parameters(model, zero); code = SPECS[model][1]
    val, grad = r.value_gradient(x, a, code, zero)
    def reference(x):
        return z.zero_engine(a, x[:-1] if zero else x, code, x[-1] if zero else 0.)[:, 3].sum()
    assert val == pytest.approx(reference(x), abs=1e-10)
    np.testing.assert_allclose(grad, approx_derivative(reference, x).ravel(), atol=2e-6, rtol=2e-6)
    if not zero:
        expected = trajectory(a, SPECS[model][0], dict(zip(SPECS[model][2], x)), np.zeros(3))[:, 3].sum()
        assert val == pytest.approx(expected, abs=1e-10)


def test_population_generation_preserves_correlations_and_stress_only_changes_friend_theta():
    kinds = [1, 2, 3, 3, 4]; k = len(kinds)
    corr = np.eye(k)*.8+np.ones((k, k))*.2
    L = np.linalg.cholesky(corr)
    draw = np.r_[[-1., -1., 5., 1., 1.5], [.5]*k, L.ravel()]
    a = r.population_parameters(draw, kinds, 111, np.random.default_rng(1))
    b = r.population_parameters(draw, kinds, 111, np.random.default_rng(1), stress=True)
    np.testing.assert_array_equal(a[:, [0, 1, 3, 4]], b[:, [0, 1, 3, 4]])
    assert ((b[:, 2] >= 5)&(b[:, 2] <= 10)).sum() == 55
    assert ((b[:, 2] > 10)&(b[:, 2] <= 20)).sum() == 56
    expected = r.transform(draw[:k]+np.random.default_rng(1).normal(size=(111, k))@(np.diag(draw[k:2*k])@L).T, kinds)
    np.testing.assert_array_equal(a, expected)


def test_wide_bounds_cover_all_generating_parameters_without_clipping():
    p = [dict(model='H7', variant='zero', parameters=np.array([[.2, 130., 37., 4., 14.]])),
         dict(model='HPreference', variant='base', parameters=np.array([[.3, 1., -18., 12.]]))]
    limits = r.make_bounds(p)
    assert limits['theta'] > 37 and limits['kappa'] > 130 and limits['preference'] > 18 and limits['gamma0'] > 14
    for item in p:
        b = r.bounds_for(item['model'], item['variant'] == 'zero', limits)
        assert all(lo <= x <= hi for x, (lo, hi) in zip(item['parameters'][0], b))
    assert p[0]['parameters'][0, 2] == 37


def test_population_csv_checks_acceptance_and_column_order(tmp_path):
    k = 2; columns = ['mu.1','mu.2','tau.1','tau.2','L.1.1','L.1.2','L.2.1','L.2.2']
    data = pd.DataFrame(np.tile([-.5, .7, .4, .6, 1., 0., 0., 1.], (8, 1)), columns=columns)
    data['divergent__'] = 0; data['treedepth__'] = 5; data['energy__'] = [0, 1]*4
    path = tmp_path/'draws.csv'; data[data.columns[::-1]].to_csv(path, index=False)
    out, bfmi = r.scan_population(path, k, 8, np.array([1, 5]), 12)
    np.testing.assert_array_equal(out[0], data[columns].iloc[1])
    assert bfmi > .3
    data.loc[7, 'treedepth__'] = 12; data.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Unaccepted'):
        r.scan_population(path, k, 8, np.array([1, 5]), 12)


def test_simulation_preserves_schedule_missingness_and_feedback():
    a = trials(); sim = r.simulate_case(a, 'H7', True, parameters('H7', True), 22)
    np.testing.assert_array_equal(a[:, [0, 1, 2, 5, 6, 7]], sim[:, [0, 1, 2, 5, 6, 7]])
    np.testing.assert_array_equal(a[:, 3] < 0, sim[:, 3] < 0)
    np.testing.assert_array_equal(sim[:, 4], (sim[:, 3] >= 0)&((sim[:, 1] > 0)|(sim[:, 3] == 1)))


def test_candidate_fit_scores_own_likelihood_on_toy_schedule():
    a = trials(); limits = dict(theta=20., kappa=100., preference=10., gamma0=10.)
    fitted = r.fit_candidate(a, 'H5', True, limits, 3, 18)
    x = np.array(list(fitted['parameters'].values()))
    assert fitted['nll'] == pytest.approx(z.zero_engine(a, x[:-1], 5, x[-1])[:, 3].sum())
    assert fitted['converged'] and fitted['n'] == 59 and fitted['k'] == 4


def test_recovery_cache_reuses_complete_and_error_records_and_rejects_stale(tmp_path, monkeypatch):
    case = dict(variant='zero', regime='empirical', generating='H5', replicate=0, participant_id='toy')
    path = tmp_path/'case.json'
    monkeypatch.setattr(r, 'simulate_case', lambda *a: pytest.fail('A cached case must not simulate or refit'))
    for status in ['complete', 'error']:
        saved = dict(fingerprint='known', case=case, rows=[], status=status)
        path.write_text(json.dumps(saved))
        assert r.case_worker(case, None, None, None, None, 'known', path) == saved
    with pytest.raises(ValueError, match='fingerprint'):
        r.case_worker(case, None, None, None, None, 'changed', path)


def test_selection_ties_and_aggregate_excludes_incomplete_datasets(tmp_path, monkeypatch):
    np.testing.assert_array_equal(r.selection_weights([2., 2., 4.]), [.5, .5, 0.])
    monkeypatch.setattr(r, 'TABLE', tmp_path)
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'plot_recovery', lambda _: None)
    records = []
    for rep, n in [(0, 111), (1, 110)]:
        for i in range(n):
            rows = []
            for model in r.FOCAL:
                value = 1. if model == 'HPreference' else 2.
                rows.append(dict(variant='zero', regime='empirical', generating='H7', replicate=rep, participant_id=f's{i}',
                    fitted=model, AICc=value, BIC=value, heldout_log_loss=value, heldout_brier=value,
                    boundary='', train_boundary='', near_best=2, train_near_best=3))
            records.append(dict(status='complete', rows=rows))
    assert not r.summarize_results(records, 222)
    c = pd.read_csv(tmp_path/'realistic_model_recovery_confusion.csv')
    assert c[c.unit=='dataset'].n_units.eq(1).all()
    assert c[c.unit=='participant'].n_units.eq(221).all()
    assert c[c.fitted=='HPreference'].selection_probability.eq(1).all()
    assert c[c.fitted=='H7'].selection_probability.eq(0).all()


def test_linux_only_and_current_published_evidence(monkeypatch):
    assert r.gate()['source_runs']['zero:HPreference'].endswith('_depth14')
    monkeypatch.setattr(r.platform, 'system', lambda: 'Darwin')
    with pytest.raises(RuntimeError, match='linux1'):
        r.run(40)


def test_complete_toy_case_can_be_saved_and_reloaded(tmp_path):
    a = trials(); limits = dict(theta=20., kappa=100., preference=10., gamma0=10.)
    case = dict(variant='zero', regime='empirical', generating='H7', replicate=0, participant_id='toy')
    path = tmp_path/'case.json'; cfg = dict(seed=32, starts=2)
    result = r.case_worker(case, a, parameters('H7', True), limits, cfg, 'fixture', path)
    assert result['status'] == 'complete', result
    assert len(result['rows']) == 4
    assert all(row['n'] == 59 and row['heldout_n'] == 30 for row in result['rows'])
    assert json.loads(path.read_text()) == result
    assert r.case_worker(case, a, parameters('H7', True), limits, cfg, 'fixture', path) == result


def test_population_loader_checks_full_cache_fingerprint_and_streams_hyperparameters(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(z, 'WORK', tmp_path/'work')
    monkeypatch.setattr(r, 'OUT', tmp_path/'results')
    model = 'H5'; variant = 'zero'; run = 'N111_H5_zero_full'; k = 4
    data = dict(K=k, kind=[1, 2, 3, 4]); meta = dict(ids=['fixture'])
    monkeypatch.setattr(z, 'model_data', lambda *a: (data, meta, {'fixture': trials()}))
    cfg = dict(seed=2, source_runs={'zero:H5': run})
    settings = dict(draws=8, max_treedepth=12)
    folder = z.WORK/'fits'/run; folder.mkdir(parents=True)
    files = []
    for i in range(4):
        values = {f'mu.{j}': [-1.]*8 for j in range(1, k+1)}
        values.update({f'tau.{j}': [.5]*8 for j in range(1, k+1)})
        values.update({f'L.{j}.{l}': [float(j == l)]*8 for j in range(1, k+1) for l in range(1, k+1)})
        values.update(divergent__=[0]*8, treedepth__=[4]*8, energy__=[0., 1.]*4)
        path = folder/f'chain{i}.csv'; pd.DataFrame(values).to_csv(path, index=False); files.append(str(path.resolve()))
    fingerprint = hashlib.sha256((json.dumps(data, sort_keys=True)+json.dumps(settings, sort_keys=True)+z.implementation_hash()).encode()).hexdigest()
    saved = dict(settings=settings, csv_files=files, fingerprint=fingerprint, meta=meta, implementation_sha256=z.implementation_hash())
    (folder/'manifest.json').write_text(json.dumps(saved))
    public_folder = r.OUT/'sampler_evidence'/run; public_folder.mkdir(parents=True)
    public = dict(saved, csv_files=[Path(f).name for f in files])
    (public_folder/'manifest_summary.json').write_text(json.dumps(public))
    draws, kinds, ids, arrays, audit = r.load_population(model, variant, cfg)
    assert draws.shape == (8, 2*k+k*k) and len(audit['chains']) == 4
    assert all(len(chain['retained_indices']) == 2 for chain in audit['chains'])
    saved['fingerprint'] = 'stale'; (folder/'manifest.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='manifest differs'):
        r.load_population(model, variant, cfg)


@pytest.mark.parametrize('model', r.FOCAL)
def test_gradient_in_high_parameter_regime(model):
    a = trials(); x = parameters(model, True); x[0] = .015; x[1] = 3.; x[2] = .8 if model == 'H8' else 18.; x[-1] = 5.
    code = SPECS[model][1]
    _, grad = r.value_gradient(x, a, code, True)
    numerical = approx_derivative(lambda y: z.zero_engine(a, y[:-1], code, y[-1])[:, 3].sum(), x).ravel()
    np.testing.assert_allclose(grad, numerical, atol=3e-5, rtol=3e-5)


def test_two_process_case_execution_serializes_results_and_caches(tmp_path):
    from joblib import Parallel, delayed, parallel_config
    a = trials(); limits = dict(theta=20., kappa=100., preference=20., gamma0=10.)
    cases = [dict(variant='base', regime='empirical', generating='H5', replicate=i, participant_id='toy') for i in range(2)]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        result = Parallel(n_jobs=2)(delayed(r.case_worker)(case, a, parameters('H5', False), limits,
                                  dict(seed=12, starts=2), 'parallel-fixture', tmp_path/f'{i}.json') for i, case in enumerate(cases))
    assert all(rec['status']=='complete' for rec in result), result
    assert all(len(rec['rows'])==4 for rec in result)
