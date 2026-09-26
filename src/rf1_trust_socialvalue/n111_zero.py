"""One matched no-age zero-option comparison; isolated caches, no automatic retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import logsumexp

from .hierarchical import SPECS, inputs, split_index, transform, load_chains, summarize
from .fitting import stable_seed
from .n111_diagnostics import verify_scope, sha, trial_groups, save_figure, style, FOCAL, DIMENSIONS

OUT = Path('results/n111_wrapup')
TABLE = OUT/'tables'
WORK = Path('work/n111_wrapup')
SOURCES = ['stan/n111_zero_fast.stan', 'stan/n111_zero_reference.stan', 'stan/n111_zero_fast.hpp']


@njit(cache=True)
def zero_engine(a, par, code, gamma, simulate=False, uniforms=np.zeros(1)):
    """Same base likelihood, plus gamma I(low=0) on the LOGIT scale."""
    belief = np.full(3, .5)
    out = np.zeros((len(a), 5))
    for t in range(len(a)):
        c = int(a[t, 0]); low = a[t, 1]; high = a[t, 2]; p = belief[c]
        bonus = 0.; preference = 0.
        if code == 5 and c == 0:
            bonus = par[2]
        if code == 7 and c < 2:
            bonus = par[c+2]
        if code == 9 and c < 2:
            preference = par[c+2]
        dv = (high-low)*(-1+1.5*p+p*bonus+preference)
        z = par[1]*dv+gamma*(low == 0)
        prob = 1/(1+np.exp(-z)) if z >= 0 else np.exp(z)/(1+np.exp(z))
        choice = a[t, 3]
        if simulate and choice >= 0:
            choice = 1. if uniforms[t] < prob else 0.
        feedback = a[t, 4] > 0
        if simulate:
            feedback = choice >= 0 and (high if choice == 1 else low) > 0
        nll = np.logaddexp(0., z)-choice*z if choice >= 0 else 0.
        out[t, 0] = p; out[t, 1] = prob; out[t, 2] = choice
        out[t, 3] = nll; out[t, 4] = 1. if feedback else 0.
        if feedback:
            err = a[t, 5]-p
            alpha = par[2] if code == 8 and err < 0 else par[0]
            belief[c] += alpha*err
    return out


def settings():
    cfg = json.loads(Path('config/n111_zero.json').read_text())
    if cfg['automatic_retries'] or cfg['chains'] != 4:
        raise ValueError('This closeout requires four chains and no automatic retries')
    return cfg


def entries():
    return [dict(name=f'N111_{m}_{variant}_{period}', model=m, zero=variant == 'zero', training=period == 'train')
            for m in FOCAL for variant, period in [('base', 'full'), ('zero', 'full'), ('zero', 'train')]]


def model_data(model, zero, training):
    verify_scope(Path.cwd())
    data, meta, arrays, ratings, frames = inputs(model, training=training, age_terms=0)
    if model not in FOCAL or data['N'] != 111:
        raise ValueError('Only the four scoped primary models at N=111 are allowed')
    flags = []
    for a in arrays.values():
        part = a[:split_index(a)] if training else a
        flags.extend((part[part[:, 3] >= 0, 1] == 0).astype(int).tolist())
    data['zero_option'] = flags
    data['zero_term'] = int(zero)
    if zero:
        data['K'] += 1
        data['kind'] = data['kind']+[4]
        data['mu_location'] += [0.]
        data['mu_scale'] += [settings()['gamma_population_mean_prior_sd']]
    return data, meta, arrays


def implementation_hash():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in SOURCES)).hexdigest()


def models(compile_reference=False):
    from cmdstanpy import CmdStanModel, set_cmdstan_path
    installed = sorted(Path('work/cmdstan').glob('cmdstan-*'))
    if not installed:
        raise RuntimeError('Use the existing CmdStan installation in work/cmdstan')
    set_cmdstan_path(str(installed[-1].resolve()))
    build = WORK/'build'; build.mkdir(parents=True, exist_ok=True)
    result = []
    for variant in (['fast', 'reference'] if compile_reference else ['fast']):
        source = Path(f'stan/n111_zero_{variant}.stan').read_text()+'\n// implementation '+implementation_hash()+'\n'
        target = build/f'n111_zero_{variant}.stan'
        if not target.exists() or target.read_text() != source:
            target.write_text(source)
        options = dict(user_header=str(Path('stan/n111_zero_fast.hpp').resolve()), stanc_options={'allow-undefined': True}) if variant == 'fast' else {}
        result.append(CmdStanModel(stan_file=str(target.resolve()), **options))
    return result


def check_implementation():
    """Full posterior/gradient parity and Python likelihood parity; NO SAMPLING."""
    fast, reference = models(compile_reference=True)
    rows = []; rng = np.random.default_rng(20260926)
    for model in FOCAL:
        for zero, gamma in [(False, 0.), (True, -2.), (True, 0.), (True, 2.)]:
            data, meta, arrays = model_data(model, zero, False)
            # Four actual schedules suffice for a target/gradient parity check; fits use all 111.
            n = 4; end = data['last'][n-1]
            for field in ['partner', 'gap', 'y', 'feedback', 'outcome', 'zero_option']:
                data[field] = data[field][:end]
            for field in ['first', 'last', 'ratings', 'age']:
                data[field] = data[field][:n]
            data['N'] = n; data['T'] = end; k = data['K']
            corr = np.eye(k)*.9+np.ones((k, k))*.1
            mu = np.array(data['mu_location'])
            if zero:
                mu[-1] = gamma
            pars = dict(mu=mu.tolist(), tau=[.6]*k, L=np.linalg.cholesky(corr).tolist(),
                        beta=[[] for _ in range(k)], z=rng.normal(0, .4, (k, n)).tolist())
            a = reference.log_prob(params=pars, data=data, jacobian=True, sig_figs=16).iloc[0]
            b = fast.log_prob(params=pars, data=data, jacobian=True, sig_figs=16).iloc[0]
            if list(a.index) != list(b.index):
                raise ValueError('Gradient coordinate order differs')
            error = np.abs(a.to_numpy()-b.to_numpy())
            relative = float(np.max(error/(1+np.abs(a.to_numpy()))))
            prior = reference.log_prob(params=pars, data=dict(data, prior_only=1), jacobian=True, sig_figs=16).iloc[0, 0]
            eta = mu+(np.diag(pars['tau'])@np.array(pars['L'])@np.array(pars['z'])).T
            natural = transform(eta, data['kind'])
            pyll = 0.
            for i, trials in enumerate(list(arrays.values())[:n]):
                par = natural[i, :-1] if zero else natural[i]
                pyll -= zero_engine(trials, par, SPECS[model][1], natural[i, -1] if zero else 0.)[:, 3].sum()
            likelihood_error = abs(float(a.iloc[0])-prior-pyll)
            passed = np.isfinite(relative) and relative < 1e-9 and likelihood_error < 1e-8
            rows.append(dict(model=model, zero=zero, gamma_location=gamma, max_scaled_gradient_error=relative,
                             max_absolute_gradient_error=float(error.max()), python_likelihood_error=likelihood_error, passed=passed))
    TABLE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE/'zero_option_implementation_checks.csv', index=False)
    if not all(r['passed'] for r in rows):
        raise RuntimeError('Zero-option implementation check failed; sampling is blocked')
    provenance = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={p: sha(Path(p)) for p in SOURCES}, generator_sha256=sha(Path(__file__)),
        config_sha256=sha(Path('config/n111_zero.json')), cases=len(rows), sampling_started=False)
    (OUT/'zero_option_implementation_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Passed {len(rows)} full-target/gradient and Python likelihood checks; no MCMC.', flush=True)


def diagnostic(fit, name, cfg):
    d = fit.summary(); d = d[d.index.str.startswith(('mu[', 'tau[', 'beta[', 'Omega[', 'natural['))].copy()
    d = d[~d.index.str.match(r'Omega\[(\d+),\1\]')].rename_axis('parameter').reset_index()
    method = fit.method_variables(); energy = method['energy__']
    bfmi = np.mean(np.diff(energy, axis=0)**2, axis=0)/np.var(energy, axis=0)
    d['run'] = name; d['n_subjects'] = 111; d['chains'] = cfg['chains']
    d['draws_per_chain'] = cfg['draws']; d['warmup_per_chain'] = cfg['warmup']
    d['adapt_delta'] = cfg['adapt_delta']; d['divergences'] = int(method['divergent__'].sum())
    d['max_depth_hits'] = int((method['treedepth__'] >= cfg['max_treedepth']).sum()); d['min_bfmi'] = bfmi.min()
    d['passed'] = (d.R_hat < 1.01) & (d.ESS_bulk >= 400) & (d.ESS_tail >= 400) & (d.divergences == 0) & (d.max_depth_hits == 0) & (d.min_bfmi > .3)
    d.to_csv(TABLE/f'zero_diagnostics_{name}.csv', index=False)
    from .accepted_review import diagnostic_pass
    return d, diagnostic_pass(d)


def fit_entry(entry, parallel_chains):
    if platform.system() != 'Linux':
        raise RuntimeError('New posterior sampling is permitted only on linux1')
    cfg = settings(); name = entry['name']; data, meta, arrays = model_data(entry['model'], entry['zero'], entry['training'])
    folder = WORK/'fits'/name; folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'manifest.json'
    fingerprint = hashlib.sha256((json.dumps(data, sort_keys=True)+json.dumps(cfg, sort_keys=True)+implementation_hash()).encode()).hexdigest()
    fast = models()[0]
    try:
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['fingerprint'] != fingerprint:
                raise RuntimeError(f'Changed target/settings in existing cache {name}; no overwrite allowed')
            fit = load_chains(saved['csv_files'])
        else:
            if list(folder.glob('*.csv')):
                raise RuntimeError(f'Incomplete posterior files in {folder}; inspect before an explicit restart')
            fit = fast.sample(data=data, chains=cfg['chains'], parallel_chains=parallel_chains,
                iter_warmup=cfg['warmup'], iter_sampling=cfg['draws'], seed=stable_seed(cfg['seed'], name),
                adapt_delta=cfg['adapt_delta'], max_treedepth=cfg['max_treedepth'], metric=cfg['metric'],
                output_dir=str(folder.resolve()), inits=.15, sig_figs=10, refresh=200, show_progress=False, show_console=False)
            saved = dict(fingerprint=fingerprint, settings=cfg, meta=meta, entry=entry, implementation_sha256=implementation_hash(),
                         csv_files=fit.runset.csv_files, execution=dict(parallel_chains=parallel_chains))
            manifest.write_text(json.dumps(saved, indent=2)+'\n')
        d, passed = diagnostic(fit, name, cfg)
        (folder/'diagnose.txt').write_text(fit.diagnose())
        print(f'{name}: Rhat={d.R_hat.max():.5f}, bulk ESS={d.ESS_bulk.min():.1f}, tail ESS={d.ESS_tail.min():.1f}, divergences={d.divergences.max()}, depth={d.max_depth_hits.max()}, BFMI={d.min_bfmi.min():.3f}', flush=True)
        if not passed:
            print('Diagnostic failure retained; no automatic retry and no inference exported.', flush=True)
            return 2
        natural = fit.stan_variable('natural')
        if entry['training']:
            heldout_outputs(entry, natural, meta, arrays)
        else:
            full_outputs(entry, natural, meta, arrays)
        return 0
    finally:
        # New-fit console and diagnostic evidence is tracked even when inference fails.
        from .run_logging import write_tail
        dest = OUT/'sampler_evidence'/name; dest.mkdir(parents=True, exist_ok=True)
        for path in folder.glob('*.txt'):
            write_tail(path, dest/path.name)
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            saved['csv_files'] = [Path(p).name for p in saved['csv_files']]
            (dest/'manifest_summary.json').write_text(json.dumps(saved, indent=2)+'\n')


def heldout_outputs(entry, natural, meta, arrays):
    rows = []; code = SPECS[entry['model']][1]
    for i, (sub, a) in enumerate(arrays.items()):
        nll = []; probs = []
        valid = (np.arange(len(a)) >= split_index(a)) & (a[:, 3] >= 0)
        y = a[valid, 3]
        for par in natural[:, i]:
            tr = zero_engine(a, par[:-1], code, par[-1])
            nll.append(tr[valid, 3]); probs.append(tr[valid, 1])
        mean = np.mean(probs, axis=0)
        loss = -logsumexp(-np.asarray(nll), axis=0)+np.log(len(natural))
        rows.append(dict(run=entry['name'], model=entry['model'], participant_id=sub, n=len(y),
            posterior_draws=len(natural), log_loss=loss.mean(), brier=np.mean((mean-y)**2), accuracy=np.mean((mean >= .5) == y)))
    pd.DataFrame(rows).to_csv(TABLE/f'zero_heldout_{entry["name"]}.csv', index=False)


def full_outputs(entry, natural, meta, arrays):
    model = entry['model']; name = entry['name']; code = SPECS[model][1]
    names = SPECS[model][2]+(['gamma0'] if entry['zero'] else [])
    rows = []
    for i, sub in enumerate(meta['ids']):
        for j, param in enumerate(names):
            label = 'preference_friend' if model == 'HPreference' and param == 'theta' else param
            rows.append(dict(run=name, model=model, variant='zero' if entry['zero'] else 'base',
                participant_id=sub, parameter=label, probability_positive=float((natural[:, i, j] > 0).mean()), **summarize(natural[:, i, j])))
    pd.DataFrame(rows).to_csv(TABLE/f'zero_parameters_{name}.csv', index=False)
    cfg = settings()
    count = min(cfg['predictive_draws'], len(natural))
    selected = np.linspace(0, len(natural)-1, count).astype(int)
    cells = {}
    for i, (sub, a) in enumerate(arrays.items()):
        observed_amount = np.where(a[:, 3] == 1, a[:, 2], a[:, 1])
        ph = []; gh = []
        for d, index in enumerate(selected):
            params = natural[index, i]; gamma = params[-1] if entry['zero'] else 0.
            base = params[:-1] if entry['zero'] else params
            ph.append(zero_engine(a, base, code, gamma)[:, 1])
            # Match randomness across base/extended counterparts for less noisy PPC differences.
            u = np.random.default_rng(stable_seed(cfg['seed'], model, sub, d, 'zero_ppc')).random(len(a))
            gh.append(zero_engine(a, base, code, gamma, True, u)[:, 2])
        for pred, values in [('conditional_history', np.asarray(ph)), ('generative_history', np.asarray(gh))]:
            for key, mask in trial_groups(a):
                fields = dict(zip(DIMENSIONS, key))
                if fields['evaluation_period'] != 'all' or fields['stratification'] not in ['partner', 'offer', 'zero_option']:
                    continue
                group = (pred,)+key
                if group not in cells:
                    cells[group] = dict(n=0, trials=0, oh=0., oi=0., h=np.zeros(count), inv=np.zeros(count))
                cell = cells[group]; cell['n'] += 1; cell['trials'] += int(mask.sum())
                cell['oh'] += a[mask, 3].mean(); cell['oi'] += observed_amount[mask].mean()
                cell['h'] += values[:, mask].mean(axis=1)
                cell['inv'] += (a[mask, 1]+values[:, mask]*(a[mask, 2]-a[mask, 1])).mean(axis=1)
    out = []
    for key, cell in cells.items():
        hs = summarize(cell['h']/cell['n']); inv = summarize(cell['inv']/cell['n'])
        out.append(dict(run=name, model=model, variant='zero' if entry['zero'] else 'base', prediction_type=key[0],
            **dict(zip(DIMENSIONS, key[1:])), n_subjects=cell['n'], n_trials=cell['trials'], observed_high=cell['oh']/cell['n'],
            predicted_high=hs['mean'], residual_high=cell['oh']/cell['n']-hs['mean'], ci_low=hs['ci_low'], ci_high=hs['ci_high'],
            observed_investment=cell['oi']/cell['n'], predicted_investment=inv['mean'], residual_investment=cell['oi']/cell['n']-inv['mean'],
            investment_ci_low=inv['ci_low'], investment_ci_high=inv['ci_high'],
            interval_type='posterior_expected_probability' if key[0] == 'conditional_history' else 'replicated_choice_predictive'))
    pd.DataFrame(out).to_csv(TABLE/f'zero_ppc_{name}.csv', index=False)


def paired_metric(extended, base, metric, seed):
    if extended.participant_id.duplicated().any() or base.participant_id.duplicated().any():
        raise ValueError('Duplicate held-out participant')
    a = extended.set_index('participant_id').sort_index(); b = base.set_index('participant_id').sort_index()
    if not a.index.equals(b.index) or not a.n.equals(b.n) or len(a) != 111 or a.n.sum() != 4017:
        raise ValueError('Held-out sample or trial counts differ')
    x = (a[metric]-b[metric]).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite held-out metric')
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(len(x), size=(5000, len(x)))].mean(axis=1)
    return dict(n_subjects=len(x), n_trials=int(a.n.sum()), base_mean=b[metric].mean(), zero_mean=a[metric].mean(),
                mean_delta=x.mean(), ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975))


def gate_stage_a():
    verify_scope(Path.cwd())
    from .n111_diagnostics import validate_residuals
    status = json.loads((OUT/'audit_status.json').read_text())
    if status['status'] != 'diagnostics_complete_closeout_pending' or any(v != 'complete' for v in status['models'].values()):
        raise RuntimeError('Stage A is not complete')
    validate_residuals(pd.read_csv(TABLE/'predictive_residuals.csv'))
    from .accepted_review import diagnostic_pass
    state = json.loads(Path('results/linux_run_status.json').read_text())['runs']
    for model in FOCAL:
        run = f'{model}_train_noage'; d = pd.read_csv(f'results/tables/diagnostics_{run}.csv')
        if state[run]['status'] != 'complete' or not diagnostic_pass(d):
            raise RuntimeError(f'Excluded baseline posterior: {run}')
        scores = pd.read_csv(f'results/tables/heldout_{run}.csv')
        if len(scores) != 111 or scores.n.sum() != 4017 or not scores.run.eq(run).all():
            raise RuntimeError(f'Invalid existing held-out baseline: {run}')
        old = json.loads((OUT/f'posterior_audit_{model}.json').read_text())
        if old['diagnostic_sha256'] != sha(Path(f'results/tables/diagnostics_{run}.csv')):
            raise RuntimeError('Baseline diagnostics changed since stage A')
    decision = json.loads(Path('config/n111_zero_decision.json').read_text())
    for filename, digest in decision['evidence'].items():
        if sha(Path(filename)) != digest:
            raise RuntimeError('Stage-A evidence changed; reassess the extension decision')
    if decision['decision'] != 'test_one_generic_zero_option_term':
        raise RuntimeError('Zero-option experiment is not enabled')


def collect_results(states):
    from .accepted_review import diagnostic_pass, table
    accepted = set()
    for entry in entries():
        name = entry['name']
        if states[name]['status'] == 'complete' and diagnostic_pass(pd.read_csv(TABLE/f'zero_diagnostics_{name}.csv')):
            accepted.add(name)
    comparisons = []; ppcs = []; pars = []
    for model in FOCAL:
        train = f'N111_{model}_zero_train'
        for metric in ['log_loss', 'brier', 'accuracy']:
            row = dict(model=model, metric=metric, status='unavailable', source_run=train,
                       n_subjects=np.nan, n_trials=np.nan, base_mean=np.nan, zero_mean=np.nan,
                       mean_delta=np.nan, ci_low=np.nan, ci_high=np.nan)
            if train in accepted:
                new = pd.read_csv(TABLE/f'zero_heldout_{train}.csv')
                old = pd.read_csv(f'results/tables/heldout_{model}_train_noage.csv')
                row.update(status='complete', **paired_metric(new, old, metric, stable_seed(20260926, model, metric, 'paired')))
            comparisons.append(row)
        for variant in ['base', 'zero']:
            run = f'N111_{model}_{variant}_full'
            if run in accepted:
                ppcs.append(pd.read_csv(TABLE/f'zero_ppc_{run}.csv'))
                pars.append(pd.read_csv(TABLE/f'zero_parameters_{run}.csv'))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(TABLE/'zero_option_model_comparison.csv', index=False)
    if ppcs:
        ppc = pd.concat(ppcs, ignore_index=True)
        ppc.to_csv(TABLE/'zero_option_predictive_summary.csv', index=False)
        keys = ['model', 'prediction_type']+DIMENSIONS
        ppc_pair = ppc[ppc.variant.eq('zero')].merge(ppc[ppc.variant.eq('base')], on=keys, suffixes=('_zero', '_base'), validate='one_to_one')
        ppc_pair['change_absolute_residual_high'] = ppc_pair.residual_high_zero.abs()-ppc_pair.residual_high_base.abs()
        ppc_pair.to_csv(TABLE/'zero_option_ppc_comparison.csv', index=False)
    else:
        ppc = pd.DataFrame()
    if pars:
        parameters = pd.concat(pars, ignore_index=True)
        parameters.to_csv(TABLE/'zero_option_parameter_summary.csv', index=False)
        joined = parameters[parameters.variant.eq('zero')].merge(parameters[parameters.variant.eq('base')],
            on=['model', 'participant_id', 'parameter'], suffixes=('_zero', '_base'), validate='one_to_one')
        joined['change_posterior_mean'] = joined.mean_zero-joined.mean_base
        joined.to_csv(TABLE/'zero_option_parameter_changes.csv', index=False)
    else:
        parameters = pd.DataFrame(columns=['run', 'model', 'variant', 'participant_id', 'parameter', 'probability_positive', 'mean', 'median', 'ci_low', 'ci_high'])
        parameters.to_csv(TABLE/'zero_option_parameter_summary.csv', index=False)
    done = len(accepted) == len(entries())
    message = '# N=111 zero-option comparison\n\n'
    message += ('All 12 new fits passed the unchanged diagnostic gate.\n\n' if done else 'Partial results: one or more new fits failed or are unavailable. Failed fits are excluded; no automatic retries were attempted.\n\n')
    message += 'This is one common zero-option logit term, partially pooled across participants and shared across partners. New full-data no-age base fits provide the matched PPC/parameter comparison; existing accepted no-age training fits provide the predictive baseline. Negative log-loss/Brier differences favor the extension; positive accuracy differences favor it. Intervals are paired participant bootstrap intervals, conditional on the fitted posteriors.\n\n'
    available = comparison[comparison.status.eq('complete')]
    if len(available):
        message += table(available, ['model', 'metric', 'mean_delta', 'ci_low', 'ci_high'], 5)+'\n\n'
    message += '## Parameter and model-fit checks\n\n[Parameter summaries](tables/zero_option_parameter_summary.csv), [comparison status](tables/zero_option_model_comparison.csv), and per-fit diagnostics are available in tables/. Full-data conditional intervals describe posterior expected probabilities; generative intervals include replicated choice variation. Comparisons remain exploratory because the zero feature was motivated by this dataset, including held-out diagnostics; these scores are not an untouched confirmatory test.\n\n'
    if done:
        plot_comparison(ppc, comparison)
        message += '![Zero-option comparison](figures/03_zero_option_comparison.png)\n\n'
    message += '## Pending scientific decision\n\nReview exact-offer and zero/positive-positive PPC changes alongside held-out log loss/Brier and parameter shifts before deciding retention. No automatic retention rule or additional model extension is applied. The targeted realistic recovery screen is the next stage after that decision; it has not run. Age and rating conclusions, N=111 scope, and exclusion of H4_full_age/H5_train_age remain unchanged.\n'
    (OUT/'zero_option_report.md').write_text(message)
    return done


def plot_comparison(ppc, comparison):
    plt = style()
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), layout='constrained')
    for i, model in enumerate(FOCAL):
        for j, group in enumerate(['zero', 'positive_positive']):
            ax = axes[i, j]
            data = ppc[ppc.model.eq(model) & ppc.stratification.eq('zero_option') & ppc.zero_option.eq(group) & ppc.prediction_type.eq('generative_history')]
            for v, offset, color in [('base', -.09, '#64748b'), ('zero', .09, '#007f86')]:
                frame = data[data.variant.eq(v)].set_index('partner').reindex(['friend', 'stranger', 'computer'])
                x = np.arange(3)+offset
                ax.vlines(x, frame.ci_low, frame.ci_high, color=color)
                ax.scatter(x, frame.predicted_high, color=color, s=25, label=v)
            ax.scatter(range(3), frame.observed_high, color='#222222', marker='x', label='observed')
            ax.set(xticks=range(3), xticklabels=['Friend', 'Stranger', 'Computer'], ylim=(0, 1), title=f'{model}: {group.replace("_", " ")}')
            if j == 0:
                ax.set_ylabel('High-choice probability')
        ax = axes[i, 2]
        d = comparison[comparison.model.eq(model) & comparison.metric.isin(['log_loss', 'brier'])].set_index('metric').loc[['log_loss', 'brier']]
        ax.hlines(range(2), d.ci_low, d.ci_high, color='#007f86', lw=2)
        ax.scatter(d.mean_delta, range(2), color='#007f86')
        ax.axvline(0, color='#999999', ls='--')
        ax.set(yticks=range(2), yticklabels=['Log loss', 'Brier'], xlabel='Zero extension − base (lower is better)', title='Paired temporal prediction')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('One shared zero-option term · matched no-age models · N=111\nLeft: generative 95% predictive intervals. Right: paired participant bootstrap 95% intervals.', fontsize=14)
    save_figure(fig, OUT/'figures/03_zero_option_comparison')


def launch(entry, logfolder, chains):
    print(f'Launching {entry["name"]}: {chains} parallel chains; console in {logfolder}', flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log = logfolder/(entry['name']+'.txt')
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-u', '-m', 'rf1_trust_socialvalue.n111_zero', '--worker', entry['name'], '--parallel-chains', str(chains)], env=env, stdout=stream, stderr=subprocess.STDOUT)
    status = 'complete' if result.returncode == 0 else 'diagnostic_failed' if result.returncode == 2 else 'error'
    return dict(status=status, exit_code=result.returncode, log=str(log))


def run(jobs, chains):
    gate_stage_a()
    check_implementation()  # Serial compilation and checks before concurrent workers.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logs = OUT/'logs'/('zero-'+stamp); logs.mkdir(parents=True, exist_ok=True)
    state = dict(stage='B_zero_option_comparison', status='running', settings=settings(),
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 excluded_original_runs=['H4_full_age', 'H5_train_age'], jobs=jobs, parallel_chains=chains,
                 runs={e['name']: {'status': 'queued_or_running'} for e in entries()})
    from .parallel_checkpoint import write_json
    path = OUT/'zero_option_status.json'
    write_json(path, state)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(launch, e, logs, chains): e for e in entries()}
        for future in as_completed(futures):
            e = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(status='error', error=str(exc))
            state['runs'][e['name']] = result
            write_json(path, state)
            print(f'{e["name"]}: {result["status"]}', flush=True)
    done = collect_results(state['runs'])
    state['status'] = 'comparison_complete_retention_review_pending' if done else 'incomplete_no_automatic_retry'
    write_json(path, state)
    print('Zero-option comparison finished; results in results/n111_wrapup/zero_option_report.md', flush=True)
    if not done:
        print('Some new fits are excluded. Exit 2 records incomplete diagnostics; no automatic retry will run.', flush=True)
    return 0 if done else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--check', action='store_true', help='Compile and validate target/gradients; no MCMC')
    parser.add_argument('--worker', choices=[e['name'] for e in entries()])
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--parallel-chains', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.parallel_chains <= 4 or not 1 <= args.jobs <= 8:
        parser.error('Use 1–8 jobs and 1–4 parallel chains')
    TABLE.mkdir(parents=True, exist_ok=True)
    if args.worker:
        entry = next(e for e in entries() if e['name'] == args.worker)
        return fit_entry(entry, args.parallel_chains)
    if args.check:
        gate_stage_a(); check_implementation(); return 0
    if not args.run:
        print('Plan: 4 matched full-data no-age baselines + 4 full-data and 4 training zero-option fits.\nExisting training baselines reused; fixed 4-chain settings; no automatic retries. --run is Linux-only.')
        return 0
    if platform.system() != 'Linux':
        raise RuntimeError('Run fitting on linux1, not the laptop')
    from .linux_handoff import coordinator_lock, existing_workers, assert_no_live_fit_locks
    root = Path.cwd().resolve()
    with coordinator_lock(root):
        if existing_workers(root):
            raise RuntimeError('Another fitting coordinator is active')
        assert_no_live_fit_locks(root)
        return run(args.jobs, args.parallel_chains)


if __name__ == '__main__':
    raise SystemExit(main())
