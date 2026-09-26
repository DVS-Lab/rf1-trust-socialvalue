"""Finite N=111 structural recovery screen; population posterior generators, no MCMC."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from numba import njit
from scipy.optimize import minimize

from . import n111_zero as z
from .accepted_review import diagnostic_pass, table
from .hierarchical import SPECS, transform, split_index
from .fitting import stable_seed
from .n111_diagnostics import verify_scope, sha, FOCAL, save_figure, style
from .parallel_checkpoint import write_json

OUT = Path('results/n111_wrapup')
TABLE = OUT/'tables'
WORK = Path('work/n111_wrapup/realistic_recovery')
CONFIG = Path('config/n111_recovery.json')
STATUS = OUT/'realistic_recovery_status.json'


def configuration():
    cfg = json.loads(CONFIG.read_text())
    if cfg['models'] != FOCAL or cfg['variants'] != ['zero', 'base']:
        raise ValueError('Only the four focal models and their scoped variants are allowed')
    if cfg['empirical_replicates'] != 8 or cfg['stress_replicates'] != 2 or cfg['starts'] != 32:
        raise ValueError('The finite screen is fixed at 8 empirical, 2 stress and 32 starts')
    if cfg['automatic_hierarchical_confirmation']:
        raise ValueError('Hierarchical confirmation requires scientific review')
    return cfg


def gate():
    z.gate_stage_a()
    from .n111_zero_retry import validate_plan, PLAN
    validate_plan()
    retry = json.loads((OUT/'zero_option_retry_status.json').read_text())
    if retry['status'] != 'complete' or retry['comparison_status'] != 'complete_retention_review_pending' or retry['plan_sha256'] != sha(PLAN):
        raise ValueError('An accepted reviewed HPreference source is required')
    decision = json.loads(Path('config/n111_zero_decision.json').read_text())
    if decision['retention'] != 'retain_candidate_with_residual_misfit':
        raise ValueError('Record the stage-B feature decision before recovery')
    cfg = configuration()
    for path, digest in cfg['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Recovery evidence changed: {path}')
    for run in cfg['source_runs'].values():
        d = pd.read_csv(TABLE/f'zero_diagnostics_{run}.csv')
        if not d.run.eq(run).all() or not diagnostic_pass(d):
            raise ValueError(f'Excluded recovery generator: {run}')
    return cfg


def parameter_names(model, zero):
    return list(SPECS[model][2])+(['gamma0'] if zero else [])


@njit(cache=True)
def value_gradient(x, a, code, zero):
    """Analytic natural-scale gradient for the already-validated four choice models."""
    k = len(x); belief = np.full(3, .5); deriv = np.zeros((3, k))
    value = 0.; grad = np.zeros(k)
    for t in range(len(a)):
        c = int(a[t, 0]); p = belief[c]; gap = a[t, 2]-a[t, 1]
        bonus = 0.; preference = 0.; bonus_index = -1; pref_index = -1
        if code == 5 and c == 0:
            bonus_index = 2; bonus = x[2]
        elif code == 7 and c < 2:
            bonus_index = c+2; bonus = x[c+2]
        elif code == 9 and c < 2:
            pref_index = c+2; preference = x[c+2]
        dv = gap*(-1+1.5*p+p*bonus+preference)
        logit = x[1]*dv
        dz = x[1]*gap*(1.5+bonus)*deriv[c].copy()
        dz[1] += dv
        if bonus_index >= 0:
            dz[bonus_index] += x[1]*gap*p
        if pref_index >= 0:
            dz[pref_index] += x[1]*gap
        if zero and a[t, 1] == 0:
            logit += x[-1]; dz[-1] += 1
        if a[t, 3] >= 0:
            prob = 1/(1+np.exp(-logit)) if logit >= 0 else np.exp(logit)/(1+np.exp(logit))
            value += np.logaddexp(0., logit)-a[t, 3]*logit
            grad += (prob-a[t, 3])*dz
        if a[t, 4] > 0:
            err = a[t, 5]-p
            j = 2 if code == 8 and err < 0 else 0
            deriv[c] *= 1-x[j]; deriv[c, j] += err
            belief[c] += x[j]*err
    return value, grad


def scan_population(path, k, expected, keep, max_depth):
    columns = [f'mu.{j}' for j in range(1, k+1)]+[f'tau.{j}' for j in range(1, k+1)]
    columns += [f'L.{i}.{j}' for i in range(1, k+1) for j in range(1, k+1)]
    with path.open() as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            if re.search(r'save_warmup\s*=\s*1', line):
                raise ValueError('Only retained-only CSVs are allowed')
    offset = 0; selected = []; energy = []
    for chunk in pd.read_csv(path, comment='#', usecols=columns+['divergent__', 'treedepth__', 'energy__'], chunksize=1024):
        if not np.isfinite(chunk.to_numpy(float)).all() or chunk.divergent__.ne(0).any() or chunk.treedepth__.ge(max_depth).any():
            raise ValueError(f'Unaccepted or nonfinite posterior transitions: {path}')
        local = keep[(keep >= offset) & (keep < offset+len(chunk))]-offset
        if len(local):
            selected.append(chunk.iloc[local][columns].to_numpy())
        energy.extend(chunk.energy__.to_list()); offset += len(chunk)
    e = np.asarray(energy); bfmi = np.mean(np.diff(e)**2)/np.var(e)
    if offset != expected or not np.isfinite(bfmi) or bfmi <= .3:
        raise ValueError('Posterior count or BFMI mismatch')
    return np.concatenate(selected), float(bfmi)


def load_population(model, variant, cfg):
    """Stream only hyperparameters; verify all transitions and the cached data target."""
    run = cfg['source_runs'][variant+':'+model]
    folder = (z.WORK/'fits'/run).resolve(); manifest = folder/'manifest.json'
    saved = json.loads(manifest.read_text())
    public = json.loads((OUT/'sampler_evidence'/run/'manifest_summary.json').read_text())
    portable = dict(saved, csv_files=[Path(f).name for f in saved['csv_files']])
    if portable != public:
        raise ValueError(f'Cache manifest differs from reviewed evidence: {run}')
    data, meta, arrays = z.model_data(model, variant == 'zero', False)
    fingerprint = hashlib.sha256((json.dumps(data, sort_keys=True)+json.dumps(saved['settings'], sort_keys=True)+z.implementation_hash()).encode()).hexdigest()
    if saved['fingerprint'] != fingerprint or saved['meta'] != meta or saved['implementation_sha256'] != z.implementation_hash():
        raise ValueError(f'Posterior target mismatch: {run}')
    settings = saved['settings']; files = [Path(f).resolve() for f in saved['csv_files']]
    if len(files) != 4 or len(set(files)) != 4 or any(not f.is_relative_to(folder) or not f.is_file() for f in files):
        raise ValueError('Missing, external or duplicate posterior chains; no refit is automatic')
    draws = []; audit = []
    for chain, path in enumerate(files):
        rng = np.random.default_rng(stable_seed(cfg['seed'], run, chain, 'hyper_draws'))
        keep = np.sort(rng.choice(settings['draws'], 2, replace=False))
        digest = sha(path)
        values, bfmi = scan_population(path, data['K'], settings['draws'], keep, settings['max_treedepth'])
        if sha(path) != digest:
            raise ValueError('Posterior changed during read-only scan')
        draws.extend(values)
        audit.append(dict(file=path.name, sha256=digest, chain=chain+1, retained_indices=keep.tolist(), bfmi=bfmi))
    return np.asarray(draws), data['kind'], meta['ids'], arrays, dict(run=run, fingerprint=fingerprint, chains=audit)


def population_parameters(draw, kinds, n, rng, stress=False):
    k = len(kinds); mu = draw[:k]; tau = draw[k:2*k]; L = draw[2*k:].reshape(k, k)
    if (tau <= 0).any() or not np.allclose(L, np.tril(L)) or not np.allclose(np.diag(L@L.T), 1, atol=1e-7):
        raise ValueError('Invalid population scale or Cholesky factor')
    eta = mu+rng.normal(size=(n, k))@(np.diag(tau)@L).T
    natural = transform(eta, kinds)
    if stress:
        # Deliberate intervention on friend theta; not a draw from the posterior joint distribution.
        if kinds[2] != 3:
            raise ValueError('High-theta stress is only for social-value generators')
        order = rng.permutation(n); half = n//2
        natural[order[:half], 2] = rng.uniform(5., 10., half)
        natural[order[half:], 2] = rng.uniform(10., 20., n-half)
    if not np.isfinite(natural).all():
        raise ValueError('Nonfinite population generation; no clipping is permitted')
    return natural


def make_bounds(populations):
    # Shared bounds across every candidate, variant, replicate and regime. Generators are never clipped.
    limits = dict(theta=20., kappa=100., preference=10., gamma0=10.)
    for item in populations:
        names = parameter_names(item['model'], item['variant'] == 'zero')
        for j, name in enumerate(names):
            group = 'preference' if item['model'] == 'HPreference' and name in ['theta', 'preference_stranger'] else 'theta' if name.startswith('theta') else name
            if group in limits:
                limits[group] = max(limits[group], float(np.max(np.abs(item['parameters'][:, j])))*1.1+1e-6)
    # A preference can match p*theta at fixed p<=1 without a narrower arbitrary cap.
    limits['preference'] = max(limits['preference'], limits['theta'])
    return limits


def bounds_for(model, zero, limits):
    bounds = []
    for name in parameter_names(model, zero):
        if name.startswith('alpha'):
            bounds.append((0., 1.))
        elif name == 'kappa':
            bounds.append((0., limits['kappa']))
        elif name == 'gamma0':
            bounds.append((-limits['gamma0'], limits['gamma0']))
        elif model == 'HPreference':
            bounds.append((-limits['preference'], limits['preference']))
        else:
            bounds.append((0., limits['theta']))
    return bounds


def fit_candidate(a, model, zero, limits, starts, seed, initial=None):
    names = parameter_names(model, zero); bounds = bounds_for(model, zero, limits)
    code = SPECS[model][1]; rng = np.random.default_rng(seed); solutions = []
    n = int((a[:, 3] >= 0).sum()); k = len(names)
    if n <= k+1:
        raise ValueError('Too few valid choices for AICc')
    for start in range(starts):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        for j, name in enumerate(names):
            if name == 'kappa':
                x[j] = np.exp(rng.uniform(np.log(.01), np.log(min(5., limits['kappa']))))
            elif start % 2 == 0 and not name.startswith('alpha'):
                x[j] = np.clip(rng.normal(0., 2.) if bounds[j][0] < 0 else np.exp(rng.uniform(-2, 2)), *bounds[j])
        if start == 0:
            x = np.array([.2 if p.startswith('alpha') else .5 if p == 'kappa' else 1. for p in names])
        if start == 1 and initial is not None:
            # H5-derived nested H7 start, never an oracle/true-parameter initialization.
            x = np.array(initial)
        result = minimize(value_gradient, x, args=(a, code, zero), jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=600, ftol=1e-11, gtol=1e-6, maxls=40))
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            solutions.append(result)
    if not solutions:
        raise RuntimeError('No finite optimizer solution')
    best = min(solutions, key=lambda r: r.fun)
    successful = [r for r in solutions if r.success and r.fun <= best.fun+1e-4]
    if successful:
        best = min(successful, key=lambda r: r.fun)
    if not best.success:
        fallback = minimize(lambda x: value_gradient(x, a, code, zero)[0], best.x, method='Powell', bounds=bounds,
                            options=dict(maxiter=2000, ftol=1e-10, xtol=1e-8))
        if fallback.success and fallback.fun <= best.fun+1e-5:
            best = fallback
    if not best.success:
        raise RuntimeError(f'Unresolved optimizer failure: {model}: {best.message}')
    value, gradient = value_gradient(best.x, a, code, zero)
    boundary = [p for p, v, (lo, hi) in zip(names, best.x, bounds) if min(v-lo, hi-v) < 1e-4*(hi-lo)]
    return dict(n=n, k=k, nll=float(value), AICc=float(2*value+2*k+2*k*(k+1)/(n-k-1)), BIC=float(2*value+k*np.log(n)),
                parameters=dict(zip(names, map(float, best.x))), boundary=';'.join(boundary), starts=starts,
                near_best=int(sum(r.success and r.fun <= value+1e-4 for r in solutions)), converged=True)


def simulate_case(a, model, zero, parameters, seed):
    tr = z.zero_engine(a, parameters[:-1] if zero else parameters, SPECS[model][1], parameters[-1] if zero else 0.,
                       True, np.random.default_rng(seed).random(len(a)))
    sim = a.copy(); sim[:, 3] = tr[:, 2]; sim[:, 4] = tr[:, 4]
    return sim


def case_worker(case, a, truth, limits, cfg, fingerprint, cache):
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['fingerprint'] != fingerprint or saved['case'] != case:
            raise ValueError('Recovery case fingerprint mismatch')
        return saved
    started = time.monotonic()
    saved = dict(fingerprint=fingerprint, case=case, rows=[])
    try:
        zero = case['variant'] == 'zero'
        seed = stable_seed(cfg['seed'], *case.values())
        sim = simulate_case(a, case['generating'], zero, truth, seed)
        split = split_index(sim); valid = (np.arange(len(sim)) >= split) & (sim[:, 3] >= 0)
        full = {}; train = {}
        for model in FOCAL:
            initials = [None, None]
            if model == 'H7':
                for j, fits in enumerate([full, train]):
                    v = fits['H5']['parameters']
                    initials[j] = [v['alpha'], v['kappa'], v['theta'], 0.]+([v['gamma0']] if zero else [])
            full[model] = fit_candidate(sim, model, zero, limits, cfg['starts'], stable_seed(seed, model, 'full'), initials[0])
            train[model] = fit_candidate(sim[:split], model, zero, limits, cfg['starts'], stable_seed(seed, model, 'train'), initials[1])
            par = np.array(list(train[model]['parameters'].values()))
            tr = z.zero_engine(sim, par[:-1] if zero else par, SPECS[model][1], par[-1] if zero else 0.)
            y = sim[valid, 3]; prob = tr[valid, 1]
            row = dict(**case, fitted=model, n=full[model]['n'], k=full[model]['k'], AICc=full[model]['AICc'], BIC=full[model]['BIC'],
                       nll=full[model]['nll'], train_n=train[model]['n'], heldout_n=int(valid.sum()),
                       heldout_log_loss=float(tr[valid, 3].mean()), heldout_brier=float(np.mean((prob-y)**2)),
                       boundary=full[model]['boundary'], train_boundary=train[model]['boundary'],
                       near_best=full[model]['near_best'], train_near_best=train[model]['near_best'], starts=cfg['starts'])
            row.update({'fit_'+p: v for p, v in full[model]['parameters'].items()})
            row.update({'train_'+p: v for p, v in train[model]['parameters'].items()})
            saved['rows'].append(row)
        for fits in [full, train]:
            if fits['H7']['nll'] > fits['H5']['nll']+1e-4:
                raise RuntimeError('Nested H7 likelihood worse than H5; optimizer audit failed')
        saved['status'] = 'complete'
    except Exception as exc:
        saved.update(status='error', error=repr(exc), rows=[])
    saved['seconds'] = time.monotonic()-started
    write_json(cache, saved)
    return saved


def selection_weights(values):
    a = np.asarray(values, float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite recovery criterion')
    winners = a <= a.min()+1e-6
    return winners/winners.sum()


def summarize_results(records, expected_cases):
    good = [r for r in records if r['status'] == 'complete']
    d = pd.DataFrame([row for rec in good for row in rec['rows']])
    if d.empty:
        raise RuntimeError('No accepted optimizer cases to summarize')
    keys = ['variant', 'regime', 'generating', 'replicate']
    casekeys = keys+['participant_id']
    for _, group in d.groupby(casekeys):
        if set(group.fitted) != set(FOCAL) or len(group) != 4:
            raise ValueError('Incomplete or duplicate candidate comparison')
    d.to_csv(TABLE/'realistic_model_recovery.csv', index=False)
    selections = []
    for casekey, group in d.groupby(casekeys):
        for metric in ['AICc', 'BIC', 'heldout_log_loss', 'heldout_brier']:
            for model, weight in zip(group.fitted, selection_weights(group[metric])):
                selections.append(dict(zip(casekeys, casekey), fitted=model, metric=metric, unit='participant', selected_weight=weight))
    datasets = []
    for datasetkey, group in d.groupby(keys):
        if group.participant_id.nunique() != 111 or len(group) != 444:
            continue
        agg = group.groupby('fitted').agg(AICc=('AICc', 'sum'), BIC=('BIC', 'sum'),
            heldout_log_loss=('heldout_log_loss', 'mean'), heldout_brier=('heldout_brier', 'mean')).reset_index()
        for metric in ['AICc', 'BIC', 'heldout_log_loss', 'heldout_brier']:
            for model, score, weight in zip(agg.fitted, agg[metric], selection_weights(agg[metric])):
                row = dict(zip(keys, datasetkey), fitted=model, metric=metric, criterion=score, selected_weight=weight)
                datasets.append(row)
                selections.append(dict(row, participant_id='all_111', unit='dataset'))
    selected = pd.DataFrame(selections)
    selected.to_csv(TABLE/'realistic_recovery_selections.csv', index=False)
    confusion = selected.groupby(['variant', 'regime', 'unit', 'metric', 'generating', 'fitted']).agg(
        selection_probability=('selected_weight', 'mean'), n_units=('selected_weight', 'size'), n_replicates=('replicate', 'nunique')).reset_index()
    confusion.to_csv(TABLE/'realistic_model_recovery_confusion.csv', index=False)
    pd.DataFrame(datasets).to_csv(TABLE/'realistic_recovery_dataset_scores.csv', index=False)
    pair = d[d.fitted.eq('H7')].merge(d[d.fitted.eq('HPreference')], on=casekeys, suffixes=('_H7', '_HPreference'), validate='one_to_one')
    pair['H7_minus_HPreference_log_loss'] = pair.heldout_log_loss_H7-pair.heldout_log_loss_HPreference
    paired = pair.groupby(keys).agg(n_subjects=('participant_id', 'size'),
        H7_minus_HPreference_log_loss=('H7_minus_HPreference_log_loss', 'mean')).reset_index()
    paired.to_csv(TABLE/'realistic_recovery_H7_vs_HPreference.csv', index=False)
    for col in ['boundary', 'train_boundary']:
        d[col+'_any'] = d[col].fillna('').ne('')
    d.groupby(['variant', 'regime', 'generating', 'fitted']).agg(
        n_cases=('participant_id', 'size'), full_boundary_rate=('boundary_any', 'mean'), training_boundary_rate=('train_boundary_any', 'mean'),
        median_near_best=('near_best', 'median'), median_train_near_best=('train_near_best', 'median')).to_csv(TABLE/'realistic_recovery_optimizer_audit.csv')
    plot_recovery(confusion)
    complete = len(good) == expected_cases
    report = '# N=111 realistic mechanism recovery: fast structural screen\n\n'
    report += f'{len(good)}/{expected_cases} participant simulations have all four full/training candidate fits accepted by the optimizer checks. Complete screen: {complete}. No hierarchical fits were launched.\n\n'
    report += 'Population parameters come from joint posterior hyperparameter draws, with new correlated participant effects on the actual 111 schedules. Primary zero-adjusted and secondary base families are separate. High-theta stress deliberately intervenes on friend theta (half 5–10, half 10–20); it is not a joint posterior draw and is never pooled with empirical simulations. Generators are never clipped; fitting bounds expand above the realized upper tail.\n\n'
    report += 'Participant rows describe selection from one short task. Dataset rows sum individual AICc/BIC or average participant held-out losses across all 111 participants; these are nonhierarchical aggregate choices, not hierarchical model comparisons. Ties within 1e-6 receive fractional weights. Eight empirical population draws per generator limit population-level Monte Carlo precision; participant selections within a replicate are dependent through shared hyperparameters. No binomial confidence intervals treating them as independent are used. Dataset summaries exclude incomplete replicates.\n\n'
    central = confusion[(confusion.variant == 'zero') & (confusion.regime == 'empirical') & confusion.generating.isin(['H7', 'HPreference'])]
    report += table(central, ['unit', 'metric', 'generating', 'fitted', 'selection_probability', 'n_units'], 4)+'\n\n'
    report += '![Targeted recovery](figures/04_realistic_recovery.png)\n\nReview H7↔HPreference confusion, H8 selections, paired prediction differences, boundary rates and stress results before deciding whether a small hierarchical confirmation is useful. No arbitrary selection-rate cutoff establishes identifiability. Strong confusion is a result, not authorization for additional fits. Final scientific interpretation and N=111 synthesis remain pending.\n'
    (OUT/'realistic_recovery_report.md').write_text(report)
    return complete


def plot_recovery(confusion):
    plt = style(); fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for i, unit in enumerate(['participant', 'dataset']):
        for j, metric in enumerate(['AICc', 'BIC', 'heldout_log_loss']):
            sub = confusion[(confusion.variant == 'zero') & (confusion.regime == 'empirical') & (confusion.unit == unit) & (confusion.metric == metric)]
            matrix = sub.pivot(index='generating', columns='fitted', values='selection_probability').reindex(index=FOCAL, columns=FOCAL)
            ax = axes[i, j]; ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
            for r in range(4):
                for c in range(4):
                    v = matrix.iloc[r, c]
                    ax.text(c, r, f'{v:.2f}' if np.isfinite(v) else 'NA', ha='center', va='center', color='white' if v > .55 else '#222222')
            ax.set(xticks=range(4), xticklabels=FOCAL, yticks=range(4), yticklabels=FOCAL, title=f'{unit}: {metric}', xlabel='Selected', ylabel='Generating')
            ax.tick_params(axis='x', rotation=25)
    fig.suptitle('Realistic recovery · zero-adjusted models · empirical population draws\nNonhierarchical screen: participant and complete N=111 dataset selection', fontsize=14)
    save_figure(fig, OUT/'figures/04_realistic_recovery')


def run(jobs):
    if platform.system() != 'Linux':
        raise RuntimeError('Run the recovery screen on linux1, not the laptop')
    from .linux_handoff import coordinator_lock, existing_workers, assert_no_live_fit_locks
    root = Path.cwd().resolve()
    with coordinator_lock(root):
        if existing_workers(root):
            raise RuntimeError('Another fitting coordinator is active')
        assert_no_live_fit_locks(root)
        cfg = gate(); WORK.mkdir(parents=True, exist_ok=True); (WORK/'cases').mkdir(exist_ok=True)
        state = dict(status='preparing_population_draws', started_at=datetime.now(timezone.utc).isoformat(), jobs=jobs,
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), sampling_started=False,
                     hierarchical_confirmation='not_launched_review_required')
        write_json(STATUS, state)
        try:
            populations = []; sources = []; range_rows = []; schedule_sha = None
            for variant in cfg['variants']:
                for model in FOCAL:
                    draws, kinds, ids, arrays, evidence = load_population(model, variant, cfg)
                    current_schedule = hashlib.sha256(b''.join(arrays[sub].tobytes() for sub in ids)).hexdigest()
                    if ids != verify_scope(root) or (schedule_sha is not None and current_schedule != schedule_sha):
                        raise ValueError('Generator participant order or schedules differ')
                    schedule_sha = current_schedule
                    if sum(int((a[:, 3] >= 0).sum()) for a in arrays.values()) != 8251 or sum(len(a) for a in arrays.values()) != 8442:
                        raise ValueError('Generator sample or missing-trial counts differ')
                    if sum(int((a[split_index(a):, 3] >= 0).sum()) for a in arrays.values()) != 4017:
                        raise ValueError('Generator temporal split differs')
                    sources.append(evidence)
                    for regime, n in [('empirical', 8), ('high_theta_stress', 2 if model in ['H5', 'H7'] else 0)]:
                        for rep in range(n):
                            rng = np.random.default_rng(stable_seed(cfg['seed'], variant, model, regime, rep, 'population'))
                            pars = population_parameters(draws[rep % 8], kinds, 111, rng, regime == 'high_theta_stress')
                            populations.append(dict(model=model, variant=variant, regime=regime, replicate=rep, parameters=pars))
                            for j, param in enumerate(parameter_names(model, variant == 'zero')):
                                x = pars[:, j]; label = 'preference_friend' if model == 'HPreference' and param == 'theta' else param
                                range_rows.append(dict(model=model, variant=variant, regime=regime, replicate=rep, parameter=label,
                                    minimum=x.min(), q05=np.quantile(x, .05), median=np.median(x), q95=np.quantile(x, .95), maximum=x.max(),
                                    fraction_5_to_10=np.mean((x >= 5)&(x <= 10)), fraction_above_10=np.mean(x > 10), fraction_above_20=np.mean(x > 20)))
            limits = make_bounds(populations)
            # Content binds caches to exact populations, sources, scope, config and implementation.
            population_sha = hashlib.sha256(b''.join(p['parameters'].tobytes() for p in populations)).hexdigest()
            provenance = dict(config_sha256=sha(CONFIG), implementation_sha256={str(p): sha(p) for p in [Path(__file__), Path(z.__file__)]},
                              sources=sources, population_sha256=population_sha, schedule_sha256=schedule_sha, fitting_upper_bounds=limits, n_datasets=len(populations),
                              n_participants=111, chains_sampled=0, generator='posterior_population_hyperparameters_with_new_correlated_effects')
            fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
            manifest = WORK/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Recovery cache changed; no overwrite or restart is automatic')
            write_json(manifest, dict(fingerprint=fingerprint, provenance=provenance))
            write_json(OUT/'realistic_recovery_provenance.json', dict(fingerprint=fingerprint, **provenance))
            pd.DataFrame(range_rows).to_csv(TABLE/'realistic_recovery_generating_ranges.csv', index=False)
            tasks = []
            for pop in populations:
                for i, sub in enumerate(ids):
                    case = dict(variant=pop['variant'], regime=pop['regime'], generating=pop['model'], replicate=pop['replicate'], participant_id=sub)
                    key = '_'.join(map(str, case.values()))
                    tasks.append((case, arrays[sub], pop['parameters'][i], limits, cfg, fingerprint, WORK/'cases'/f'{key}.json'))
            state.update(status='running_fast_screen', expected_cases=len(tasks), completed_cases=0, failed_cases=0, n_datasets=len(populations), fingerprint=fingerprint)
            write_json(STATUS, state)
            print(f'Finite screen: {len(populations)} N=111 datasets, {len(tasks)} participant cases, four candidates each; full and temporal fits; {jobs} processes. No MCMC.', flush=True)
            results = []
            # Compile once in the parent so workers can load the Linux Numba cache.
            first = populations[0]
            value_gradient(first['parameters'][0], arrays[ids[0]], SPECS[first['model']][1], first['variant'] == 'zero')
            with parallel_config(backend='loky', inner_max_num_threads=1):
                stream = Parallel(n_jobs=jobs, return_as='generator_unordered', batch_size=1)(delayed(case_worker)(*task) for task in tasks)
                for result in stream:
                    results.append(result)
                    state['completed_cases'] += result['status'] == 'complete'
                    state['failed_cases'] += result['status'] != 'complete'
                    if len(results) % 20 == 0 or len(results) == len(tasks):
                        write_json(STATUS, state)
                        print(f'{len(results)}/{len(tasks)} cases finished; {state["failed_cases"]} optimizer errors', flush=True)
            # Deterministic output ordering independent of worker completion order.
            results.sort(key=lambda r: tuple(str(v) for v in r['case'].values()))
            errors = [dict(**r['case'], error=r['error']) for r in results if r['status'] != 'complete']
            write_json(OUT/'realistic_recovery_errors.json', errors)
            complete = summarize_results(results, len(tasks))
            state.update(status='screen_complete_scientific_review_pending' if complete else 'incomplete_optimizer_errors', exit_code=0 if complete else 2)
            return state['exit_code']
        except BaseException as exc:
            state.update(status='error_or_interrupted', error=repr(exc)); raise
        finally:
            state['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(STATUS, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--jobs', type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 40:
        parser.error('Use 1–40 single-threaded worker processes')
    if not args.run:
        gate()
        print('Ready: 64 empirical + 8 high-theta stress N=111 datasets; 7,992 participant cases; 63,936 full/training candidate fits.\n32 starts each, cache-resumable, Linux-only. No Stan sampling or automatic hierarchical confirmation.')
        return 0
    return run(args.jobs)


if __name__ == '__main__':
    raise SystemExit(main())
