"""Explore offer and continuous-age misfit using authenticated existing draws."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from . import full_sample_ppc as ppc
from . import full_sample_sampling as s

TARGETS = ['Full_H7_zero_noage', 'Full_HPreference_zero_noage_draws4000']


def groups(a):
    for strat, group, mask in ppc.groups(a):
        if strat in ['all', 'partner', 'partner_zero']:
            yield strat, group, mask
    valid = a[:, 3] >= 0
    for i, partner in enumerate(ppc.PARTNERS):
        for low, high in sorted(set(map(tuple, a[:, 1:3]))):
            yield 'partner_offer', f'{partner}_{low:g}/{high:g}', valid & (a[:, 0] == i) & (a[:, 1] == low) & (a[:, 2] == high)


def ages_from_trials(t, ids):
    grouped = t.groupby('participant_id').age
    if (grouped.nunique(dropna=False) != 1).any():
        raise ValueError('Age varies within participant')
    ages = grouped.first().reindex(ids).to_numpy(float)
    if not np.isfinite(ages).all():
        raise ValueError('Missing/nonfinite frozen age; review scope before excluding anyone')
    return ages


def cell_summary(observed, predicted, ages, seed, boot):
    """One row per participant; slopes are probability units per decade."""
    observed = np.asarray(observed, float); predicted = np.asarray(predicted, float); ages = np.asarray(ages, float)
    if predicted.ndim != 2 or predicted.shape[0] != len(observed) or ages.shape != observed.shape:
        raise ValueError('Participant arrays are misaligned')
    if not all(np.isfinite(x).all() for x in [observed, predicted, ages]):
        raise ValueError('Nonfinite residual inputs')
    residual_draws = observed[:, None] - predicted
    rep = residual_draws.mean(axis=0)
    out = dict(participants=len(ages), age_min=ages.min(), age_max=ages.max(),
               observed=observed.mean(), predicted_mean=predicted.mean(),
               predicted_ci_low=np.quantile(predicted.mean(axis=0), .025),
               predicted_ci_high=np.quantile(predicted.mean(axis=0), .975),
               residual_mean=rep.mean(), residual_predictive_low=np.quantile(rep, .025),
               residual_predictive_high=np.quantile(rep, .975))
    x = (ages - ages.mean()) / 10
    if len(ages) < 10 or x @ x <= 0:
        out['age_status'] = 'not_estimable'; return out
    obs_slope = x @ observed / (x @ x)
    pred_slopes = x @ predicted / (x @ x)
    residual_slopes = obs_slope - pred_slopes
    # Resample people, preserving their entire cell summary. Each person has
    # equal weight regardless of trials or runs. This interval conditions on
    # posterior-mean predictions; it is separate from the predictive interval.
    rng = np.random.default_rng(seed); indices = rng.integers(0, len(ages), (boot, len(ages)))
    bx = x[indices]; bx -= bx.mean(axis=1, keepdims=True)
    denom = (bx * bx).sum(axis=1); good = denom > 0
    if good.sum() < .95 * boot:
        raise ValueError('Too many degenerate age bootstrap samples')
    br = observed - predicted.mean(axis=1)
    obs_boot = (bx[good] * observed[indices[good]]).sum(axis=1) / denom[good]
    res_boot = (bx[good] * br[indices[good]]).sum(axis=1) / denom[good]
    out.update(age_status='estimable', observed_age_slope=obs_slope,
               observed_age_bootstrap_low=np.quantile(obs_boot, .025), observed_age_bootstrap_high=np.quantile(obs_boot, .975),
               predicted_age_slope=pred_slopes.mean(), predicted_age_low=np.quantile(pred_slopes, .025), predicted_age_high=np.quantile(pred_slopes, .975),
               residual_age_slope=residual_slopes.mean(), residual_age_predictive_low=np.quantile(residual_slopes, .025),
               residual_age_predictive_high=np.quantile(residual_slopes, .975),
               residual_age_bootstrap_low=np.quantile(res_boot, .025), residual_age_bootstrap_high=np.quantile(res_boot, .975))
    return out


def tables(arrays, ages, natural, model, simulations, seed, boot):
    if natural.ndim != 3 or natural.shape[1] != len(arrays) or len(natural) < simulations or not np.isfinite(natural).all():
        raise ValueError('Invalid existing posterior array')
    if len(ages) != len(arrays): raise ValueError('Participant age order mismatch')
    selected = np.linspace(0, len(natural)-1, simulations).astype(int)
    cells = {}
    for i, (sub, a) in enumerate(arrays.items()):
        rng = np.random.default_rng(s.stable_seed(seed, model, sub, 'full_cohort_ppc'))
        conditional, generative = ppc.trajectories(a, natural[selected, i], model, True, rng.random((simulations, len(a))))
        for strat, group, mask in groups(a):
            if not mask.any(): continue
            for history, predictions in [('conditional', conditional), ('generative', generative)]:
                cell = cells.setdefault((strat, group, history), dict(observed=[], predicted=[], ages=[], choices=0))
                cell['observed'].append(a[mask, 3].mean()); cell['predicted'].append(predictions[:, mask].mean(axis=1))
                cell['ages'].append(ages[i]); cell['choices'] += int(mask.sum())
    rows = []
    for (strat, group, history), cell in cells.items():
        rows.append(dict(stratification=strat, group=group, history=history, choices=cell['choices'],
                         interval_type='replicated_choices' if history == 'generative' else 'conditional_expected_behavior',
                         **cell_summary(cell['observed'], cell['predicted'], cell['ages'],
                                        s.stable_seed(seed, strat, group, 'residual_age'), boot)))
    return pd.DataFrame(rows), selected


def verified_posterior(phase, c, name, t):
    root = Path(c['_root']); out, work = s.paths(c)
    entry = next(e for e in ppc.review.resolved_entries(root) if e['name'] == name)
    evidence = out/'fits'/name; cache = work/'fits'/name
    status = json.loads((evidence/'status.json').read_text())
    portable = json.loads((evidence/'manifest_summary.json').read_text())
    manifest = json.loads((cache/'manifest.json').read_text())
    if status['status'] != 'complete' or not status['passed'] or entry['subset'] != 'full' or entry['variant'] != 'zero':
        raise ValueError('Residual checks require accepted full zero-option fit')
    data, meta, arrays = s.model_data(t, entry['model'], True, phase['gamma_population_mean_prior_sd'])
    payload = dict(data=data, settings=status['settings'], sampler_seed=status['sampler_seed'],
                   stan={file:s.fs.sha(root/file) for file in s.SOURCES}, cmdstan_version=phase['cmdstan_version'])
    if (s.digest(payload) != status['fingerprint'] or manifest['fingerprint'] != status['fingerprint']
            or portable['fingerprint'] != status['fingerprint'] or manifest['meta'] != meta):
        raise ValueError('Residual target differs from existing posterior')
    if {Path(f).name:h for f,h in manifest['posterior_sha256'].items()} != portable['posterior_sha256']:
        raise ValueError('Cached posterior differs from published manifest')
    for file in manifest['csv_files']:
        if s.fs.sha(file) != manifest['posterior_sha256'][file]: raise ValueError('Raw posterior changed: '+file)
    fit = s.load_chains(manifest['csv_files'])
    return entry, arrays, fit.stan_variable('natural'), dict(posterior_fingerprint=status['fingerprint'],
        posterior_sha256=portable['posterior_sha256'], accepted_status_sha256=s.fs.sha(evidence/'status.json'),
        manifest_sha256=s.fs.sha(evidence/'manifest_summary.json'))


def source_hashes(root):
    sources = ppc.implementation(root)
    file = 'src/rf1_trust_socialvalue/full_sample_residuals.py'; sources[file] = s.fs.sha(Path(root)/file)
    return sources


def fit(p, phase, c, name):
    s.require_linux(); out, _ = s.paths(c); dest = out/'residual_review/fits'/name; dest.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', name=name, started_at=s.now(), new_sampling=False); s.save(dest/'status.json', state)
    try:
        sources = source_hashes(c['_root']); t, _ = ppc.geometry.saved_trials(phase, c)
        entry, arrays, natural, provenance = verified_posterior(phase, c, name, t)
        ages = ages_from_trials(t, list(arrays))
        table, indices = tables(arrays, ages, natural, entry['model'], p['simulations'], c['seed'], c['bootstrap_iterations'])
        table.to_csv(dest/'offer_age_residuals.tsv', sep='\t', index=False)
        ppc.geometry.saved_trials(phase, c)
        if sources != source_hashes(c['_root']): raise ValueError('Implementation changed during execution')
        state.update(status='complete', finished_at=s.now(), model=entry['model'], participants=len(arrays),
            simulations=p['simulations'], bootstrap_iterations=c['bootstrap_iterations'], seed=c['seed'], source_hashes=sources,
            slope_units='high-choice probability per ten years',
            interpretation='Exploratory linear residual age associations; not hierarchical age effects, causal inference, or new heldout prediction.',
            selected_posterior_draw_indices_zero_based=indices.tolist(), **provenance,
            output_sha256={'offer_age_residuals.tsv':s.fs.sha(dest/'offer_age_residuals.tsv')})
        s.save(dest/'status.json', state)
    except BaseException as exc:
        state.update(status='error', error=repr(exc), finished_at=s.now()); s.save(dest/'status.json', state); raise


def render(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, _ = s.paths(c); dest = out/'residual_review'; frames = []
    for name in TARGETS:
        folder = dest/'fits'/name; state = json.loads((folder/'status.json').read_text())
        if state['status'] != 'complete': raise ValueError('Incomplete residual worker')
        for f, h in state['output_sha256'].items():
            if s.fs.sha(folder/f) != h: raise ValueError('Residual table changed')
        f = pd.read_csv(folder/'offer_age_residuals.tsv', sep='\t'); f['model'] = state['model']; frames.append(f)
    combined = pd.concat(frames, ignore_index=True); combined.to_csv(dest/'offer_age_residuals.tsv', sep='\t', index=False)
    for age in [False, True]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.3), sharex=True, layout='constrained')
        for ax, partner in zip(axes, ppc.PARTNERS):
            for model, color, shift in [('H7', '#176b87', -.12), ('HPreference', '#c36e24', .12)]:
                f = combined[combined.model.eq(model) & combined.history.eq('generative') & combined.stratification.eq('partner_offer') & combined.group.str.startswith(partner+'_')].copy()
                f['offer'] = f.group.str.split('_').str[-1]
                offers = sorted(f.offer, key=lambda x:tuple(map(float, x.split('/')))); f = f.set_index('offer').loc[offers]
                if age and (f.age_status != 'estimable').any(): raise ValueError('Some age cells not estimable; inspect tables')
                prefix = 'residual_age' if age else 'residual'
                mean = f.residual_age_slope if age else f.residual_mean
                y = np.arange(len(f)) + shift
                ax.hlines(y, 100*f[prefix+'_predictive_low'], 100*f[prefix+'_predictive_high'], color=color, lw=2)
                ax.scatter(100*mean, y, color=color, label=model+' + zero')
            ax.axvline(0, color='.5', ls='--'); ax.set(yticks=range(len(offers)), yticklabels=offers, title=partner.title())
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.15)
            ax.set_xlabel('Residual age slope\n(percentage points / decade)' if age else 'Observed minus predicted\n(percentage points)')
        axes[0].set_ylabel('Lower / higher investment')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=9)
        fig.suptitle(('Age dependence of model misfit' if age else 'Model misfit by exact offer and partner')+'\n95% predictive intervals · fixed observed cohort · equal participant weighting')
        stem = 'age_residual_by_offer' if age else 'residual_by_offer'
        for ext in ['png', 'pdf']: fig.savefig(dest/f'{stem}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    return {f.name:s.fs.sha(f) for f in dest.iterdir() if f.suffix in ['.tsv', '.png', '.pdf']}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--fit', choices=TARGETS); a = parser.parse_args()
    p, phase, c = ppc.configuration(); s.require_linux()
    if a.fit: fit(p, phase, c, a.fit); return
    import fcntl
    out, work = s.paths(c); dest = out/'residual_review'; dest.mkdir(parents=True, exist_ok=True)
    with (work/'residual_review.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Residual review already running')
        state = dict(status='preparing', started_at=s.now(), new_sampling=False, runs={}, workers=2)
        s.save(dest/'status.json', state)
        try:
            _, prov = ppc.geometry.saved_trials(phase, c)
            s.save(dest/'live_source_audit.json', ppc.geometry.source_drift_audit(c, prov))
            def launch(name):
                folder = dest/'fits'/name; folder.mkdir(parents=True, exist_ok=True)
                with (folder/'console.txt').open('a', buffering=1) as log:
                    log.write('\nRESIDUAL WORKER '+s.now()+'\n')
                    return subprocess.run([sys.executable, '-m', 'rf1_trust_socialvalue.full_sample_residuals', '--fit', name], cwd=c['_root'], stdout=log, stderr=subprocess.STDOUT).returncode
            state['status'] = 'running'; s.save(dest/'status.json', state)
            print('Two accepted models; 200 predictive replicates; continuous age; no new MCMC.', flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {name:pool.submit(launch, name) for name in TARGETS}
                for name, future in futures.items():
                    state['runs'][name] = future.result(); s.save(dest/'status.json', state)
                    print(name+': residual exit '+str(state['runs'][name]), flush=True)
            if any(state['runs'].values()): raise RuntimeError('Residual checks failed; inspect per-fit logs')
            state['output_sha256'] = render(c); ppc.geometry.saved_trials(phase, c)
            state.update(status='complete', finished_at=s.now()); s.save(dest/'status.json', state)
        except BaseException as exc:
            state.update(status='error', finished_at=s.now(), error=repr(exc)); s.save(dest/'status.json', state); raise


if __name__ == '__main__': main()
