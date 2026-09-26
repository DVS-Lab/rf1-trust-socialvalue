import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rf1_trust_socialvalue import n111_diagnostics as mod


def sequence():
    # partner, low, high, actual choice, observed feedback, programmed outcome, side, run
    return np.array([[0, 0, 2, 0, 0, 1, 0, 1],
                     [1, 2, 4, 1, 1, 0, 1, 1],
                     [0, 0, 4, 1, 1, 1, 1, 1],
                     [0, 2, 8, 0, 1, 0, 0, 1],
                     [2, 0, 8, -1, 0, 1, 0, 2],
                     [0, 4, 8, 1, 1, 1, 1, 2],
                     [1, 0, 2, 0, 0, 1, 0, 2],
                     [2, 2, 4, 1, 1, 0, 1, 2]], dtype=float)


def natural(model, count=12):
    params = dict(alpha=.3, kappa=.7, theta=1.5, theta_stranger=.8,
                  preference_stranger=-.2, alpha_negative=.1)
    return np.array([[params[p] for p in mod.SPECS[model][2]]]*count)


@pytest.mark.parametrize('model', mod.MODELS)
def test_conditional_predictions_use_only_actual_past_feedback(model):
    a = sequence()
    before = a.copy()
    pred = mod.prediction_arrays(a, model, natural(model), 13, 'subject')
    assert np.array_equal(a, before)
    changed = a.copy()
    changed[3, 5] = 1-changed[3, 5]
    altered = mod.prediction_arrays(changed, model, natural(model), 13, 'subject')
    # Outcome at decision 3 is unavailable until after that decision.
    np.testing.assert_array_equal(pred['conditional_history'][0][:, :4], altered['conditional_history'][0][:, :4])
    # A programmed outcome concealed by an actual zero choice has NO effect on conditional beliefs.
    changed = a.copy()
    changed[0, 5] = 1-changed[0, 5]
    hidden = mod.prediction_arrays(changed, model, natural(model), 13, 'subject')
    np.testing.assert_array_equal(pred['conditional_history'][0], hidden['conditional_history'][0])
    # Conditional Bernoulli replications cannot feed back: changing seed leaves probabilities unchanged.
    alternate_seed = mod.prediction_arrays(a, model, natural(model), 19, 'subject')
    np.testing.assert_array_equal(pred['conditional_history'][0], alternate_seed['conditional_history'][0])
    assert (pred['generative_history'][0][:, 4] == -1).all()


def test_generative_exposure_uses_simulated_amount_and_excludes_missing():
    a = sequence()
    pars = mod.BASE.copy()
    pars[mod.INDEX['alpha']] = .4
    # All simulated high decisions expose outcomes except on the missed trial.
    high = mod.engine(a, pars, 2, np.zeros(3), False, True, np.zeros(len(a)))
    np.testing.assert_array_equal(high[:, 4], [1, 1, 1, 1, 0, 1, 1, 1])
    # All simulated low decisions expose outcomes only when even the low option is positive.
    low = mod.engine(a, pars, 2, np.zeros(3), False, True, np.ones(len(a)))
    np.testing.assert_array_equal(low[:, 4], [0, 1, 0, 1, 0, 1, 0, 1])
    assert high[0, 0] == .5 and high[2, 0] == pytest.approx(.7)
    assert low[2, 0] == .5


def test_previous_feedback_is_same_partner_pretrial_and_grouping_is_not_cross_product():
    a = sequence()
    np.testing.assert_array_equal(mod.previous_feedback(a), [-1, -1, -1, 1, -1, 0, 0, -1])
    groups = list(mod.trial_groups(a))
    assert len({key for key, _ in groups}) == len(groups)
    for key, mask in groups:
        fields = dict(zip(mod.DIMENSIONS, key))
        assert not mask[4]  # Missing trial never contributes.
        if fields['stratification'] != 'previous_feedback':
            assert fields['previous_feedback'] == 'all'
    single = a.copy()
    single[:, 7] = 1
    runs = [dict(zip(mod.DIMENSIONS, k))['run_or_bin'] for k, _ in mod.trial_groups(single) if k[0] == 'run']
    assert set(runs) == {'single_run'}


def test_equal_participant_aggregation_and_paired_history_alignment():
    a = sequence()
    b = np.tile(a, (2, 1))
    # Same choice fraction per subject; b contributes twice as many trials, not twice the weight.
    b[:, 3] = np.where(b[:, 3] >= 0, 1, -1)
    b[:, 4] = (b[:, 3] >= 0).astype(float)
    pars = np.repeat(natural('H5')[:, None, :], 2, axis=1)
    residuals, participants = mod.summarize_predictions('H5', {'a': a, 'b': b}, ['a', 'b'], pars)
    row = residuals.query("prediction_type=='conditional_history' and stratification=='partner' and partner=='friend' and evaluation_period=='all'").iloc[0]
    assert row.observed_high == pytest.approx(.75)
    assert row.n_participants == 2 and row.n_trials == 12
    again, _ = mod.summarize_predictions('H5', {'a': a, 'b': b}, ['a', 'b'], pars)
    pd.testing.assert_frame_equal(residuals, again)
    paired = mod.paired_histories(residuals)
    assert len(paired)*2 == len(residuals)
    corrupt = residuals.copy()
    corrupt.loc[corrupt.prediction_type.eq('generative_history'), 'observed_high'] += .01
    with pytest.raises(ValueError, match='different observed cells'):
        mod.paired_histories(corrupt)
    contrast = mod.zero_contrasts(participants)
    assert not contrast.empty and contrast.n_paired_participants.eq(2).all()


def chain_file(path, n=2, k=2, draws=8, divergence=False):
    columns = {f'natural.{i}.{j}': np.arange(draws)+100*i+10*j for i in range(1, n+1) for j in range(1, k+1)}
    columns.update(divergent__=np.repeat(int(divergence), draws), treedepth__=np.repeat(4, draws), energy__=np.tile([-1., 1.], draws//2))
    pd.DataFrame(columns)[list(reversed(columns))].to_csv(path, index=False)


def test_stream_reader_preserves_parameter_order_and_joint_draws(tmp_path):
    path = tmp_path/'posterior.csv'
    chain_file(path)
    before = path.read_bytes()
    x, info = mod.scan_natural(path, 2, 2, 8, np.array([0, 3, 7]), 12)
    assert x.shape == (3, 2, 2)
    np.testing.assert_array_equal(x[1], [[113, 123], [213, 223]])
    assert info['selected_draws'] == 3 and path.read_bytes() == before
    with pytest.raises(RuntimeError, match='draw count'):
        mod.scan_natural(path, 2, 2, 9, np.array([0]), 12)
    chain_file(path, divergence=True)
    with pytest.raises(RuntimeError, match='Unaccepted sampler'):
        mod.scan_natural(path, 2, 2, 8, np.array([0]), 12)


def test_scope_refuses_changed_inputs_without_inspecting_other_data(tmp_path):
    for rel in mod.PINNED:
        p = tmp_path/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('changed sample')
    with pytest.raises(RuntimeError, match='input changed'):
        mod.verify_scope(tmp_path)


def test_posterior_loader_no_sampling_and_cache_bytes_unchanged(tmp_path, monkeypatch):
    run = 'H5_train_noage'
    ids = [f'sub-{i:03}' for i in range(111)]
    monkeypatch.setattr(mod, 'verify_scope', lambda root: ids)
    data = dict(N=111, K=3)
    meta = dict(ids=ids, model='H5', training=True, age_terms=0, bounded=False,
                independent=False, prior_scale=1, n_trials=111)
    monkeypatch.setattr(mod, 'inputs', lambda *a, **k: (data, meta, {}, {}, {}))
    tables = tmp_path/'results/tables'
    tables.mkdir(parents=True)
    folder = tmp_path/'work/hierarchical'/run
    folder.mkdir(parents=True)
    stan = tmp_path/'stan'
    stan.mkdir()
    for name in ['hierarchical_shared.stan', 'hierarchical_fast.stan', 'rl_fast.hpp']:
        (stan/name).write_text(name)
    cfg = dict(chains=4, draws=8, warmup=2000, adapt_delta=.95, max_treedepth=12)
    files = []
    for i in range(4):
        p = folder/f'chain{i}.csv'
        chain_file(p, n=111, k=3)
        files.append(str(p))
    fingerprint = hashlib.sha256((json.dumps(data, sort_keys=True)+json.dumps(cfg, sort_keys=True)+(stan/'hierarchical_shared.stan').read_text()).encode()).hexdigest()
    implementation = hashlib.sha256((stan/'rl_fast.hpp').read_bytes()+(stan/'hierarchical_fast.stan').read_bytes()).hexdigest()
    manifest = folder/'manifest.json'
    manifest.write_text(json.dumps(dict(meta=meta, settings=cfg, fingerprint=fingerprint, implementation_sha256=implementation, csv_files=files)))
    diag = pd.DataFrame([dict(run=run, R_hat=1.001, ESS_bulk=1000, ESS_tail=1000, divergences=0,
        max_depth_hits=0, min_bfmi=.7, passed=True, chains=4, draws_per_chain=8, warmup_per_chain=2000, adapt_delta=.95)])
    diag.to_csv(tables/f'diagnostics_{run}.csv', index=False)
    statepath = tmp_path/'results/linux_run_status.json'
    statepath.write_text(json.dumps({'runs': {run: {'status': 'complete'}}}))
    before = {str(p): p.read_bytes() for p in folder.iterdir()}
    # No CmdStan object or sample method is supplied: this path can only read CSVs.
    natural, _, _, evidence = mod.load_training_draws(tmp_path, 'H5', 8)
    assert natural.shape == (8, 111, 3) and len(evidence['chains']) == 4
    assert before == {str(p): p.read_bytes() for p in folder.iterdir()}
    statepath.write_text(json.dumps({'runs': {run: {'status': 'diagnostic_failed'}}}))
    with pytest.raises(RuntimeError, match='Unaccepted posterior'):
        mod.load_training_draws(tmp_path, 'H5', 8)


def test_complete_diagnostic_stage_with_synthetic_parameters_only(tmp_path, monkeypatch):
    """Exercise aggregation/plots/status on the fixed schedule without any chain read or fit."""
    from concurrent.futures import Future
    import shutil
    from rf1_trust_socialvalue.models import pack
    repo = Path(__file__).resolve().parents[1]
    for rel in mod.PINNED:
        target = tmp_path/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/rel, target)
    ids = mod.verify_scope(tmp_path)
    t = pd.read_csv(tmp_path/'results/tables/trial_table.csv')
    arrays = {sub: pack(frame) for sub, frame in t[t.participant_id.isin(ids)].groupby('participant_id', sort=True)}
    def fake_load(root, model, count):
        pars = np.repeat(natural(model, count)[:, None, :], 111, axis=1)
        return pars, {'ids': ids}, arrays, {'test_only': True, 'run': f'{model}_train_noage'}
    class InlinePool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, *args):
            f = Future()
            try:
                f.set_result(function(*args))
            except Exception as exc:
                f.set_exception(exc)
            return f
    monkeypatch.setattr(mod, 'load_training_draws', fake_load)
    monkeypatch.setattr(mod, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *args, **kwargs: 'test-source\n')
    mod.run_audit(tmp_path, draws=4, jobs=1)
    out = tmp_path/'results/n111_wrapup'
    status = json.loads((out/'audit_status.json').read_text())
    assert status['status'] == 'diagnostics_complete_closeout_pending'
    assert status['sampling_started'] is False
    assert status['excluded_runs'] == ['H4_full_age', 'H5_train_age']
    residuals = pd.read_csv(out/'tables/predictive_residuals.csv')
    mod.validate_residuals(residuals)
    assert set(residuals.model) == set(mod.MODELS)
    assert len(pd.read_csv(out/'tables/conditional_vs_generative.csv'))*2 == len(residuals)
    assert len(list((out/'figures').iterdir())) == 6
    assert 'closeout not yet complete' in (out/'README.md').read_text()
