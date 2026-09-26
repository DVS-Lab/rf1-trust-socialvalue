"""N=111 closeout, stage A: read-only posterior prediction diagnostics.

This module has no sampling entry point. It reads five accepted training/no-age
posteriors; inference inputs and the earlier reports are never modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np
import pandas as pd

from .fitting import stable_seed
from .hierarchical import SPECS, inputs, split_index
from .models import engine, BASE, INDEX

MODELS = ['H2', 'H5', 'H7', 'HPreference', 'H8']
FOCAL = ['H5', 'H7', 'HPreference', 'H8']
PARTNERS = ['friend', 'stranger', 'computer']
OFFERS = ['0-2', '0-4', '0-8', '2-4', '2-8', '4-8']
DIMENSIONS = ['stratification', 'partner', 'offer_pair', 'zero_option', 'run_or_bin', 'previous_feedback', 'evaluation_period']
PINNED = {
    'results/tables/trial_table.csv': 'b819604280bf12d0325424b727cb6a4ee585966e62f4c1cc7cb968061d73f8d8',
    'results/tables/sample_audit.csv': '1226af5ad06437764d9238071c9b8c5cae7326ad2a4d0baa9461baf61a10a02a',
    'results/tables/ratings.csv': 'a0bb51734a2231151782fb70b1afddc51e33386796ad9b89737cf69aa8f5ad05',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_scope(root):
    for rel, expected in PINNED.items():
        if sha(root/rel) != expected:
            raise RuntimeError(f'N=111 input changed: {rel}; this workflow cannot admit an expanded sample')
    sample = pd.read_csv(root/'results/tables/sample_audit.csv')
    ids = sorted(sample.loc[sample.primary_include.eq(True), 'participant_id'])
    if len(ids) != 111 or len(set(ids)) != 111:
        raise RuntimeError('Closeout requires the original 111 primary participants')
    return ids


def scan_natural(path, n, k, expected_draws, keep, max_depth):
    """Stream retained CSVs; preserve joint participant/parameter draw identity."""
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            if re.search(r'save_warmup\s*=\s*1', line):
                raise RuntimeError('Retained-only posterior CSVs are required')
    columns = [f'natural.{i}.{j}' for i in range(1, n+1) for j in range(1, k+1)]
    method = ['divergent__', 'treedepth__', 'energy__']
    chosen, energies = [], []
    offset = 0
    for chunk in pd.read_csv(path, comment='#', usecols=columns+method, chunksize=1024):
        if not np.isfinite(chunk.to_numpy(float)).all():
            raise RuntimeError(f'Nonfinite posterior value: {path}')
        if chunk.divergent__.ne(0).any() or chunk.treedepth__.ge(max_depth).any():
            raise RuntimeError(f'Unaccepted sampler transitions: {path}')
        local = keep[(keep >= offset) & (keep < offset+len(chunk))]-offset
        if len(local):
            chosen.append(chunk.iloc[local][columns].to_numpy().reshape(len(local), n, k))
        energies.extend(chunk.energy__.to_list())
        offset += len(chunk)
    if offset != expected_draws:
        raise RuntimeError(f'Posterior draw count mismatch: {path}: {offset} != {expected_draws}')
    e = np.asarray(energies)
    bfmi = np.mean(np.diff(e)**2)/np.var(e)
    if not np.isfinite(bfmi) or bfmi <= .3:
        raise RuntimeError(f'Unaccepted BFMI in {path}: {bfmi}')
    return np.concatenate(chosen), dict(retained_draws=offset, selected_draws=len(keep), bfmi=float(bfmi))


def load_training_draws(root, model, count):
    from .accepted_review import diagnostic_pass
    run = f'{model}_train_noage'
    state = json.loads((root/'results/linux_run_status.json').read_text())
    diagpath = root/'results/tables'/f'diagnostics_{run}.csv'
    diag = pd.read_csv(diagpath)
    if state['runs'][run]['status'] != 'complete' or not diag.run.eq(run).all() or not diagnostic_pass(diag):
        raise RuntimeError(f'Unaccepted posterior: {run}')
    folder = root/'work/hierarchical'/run
    manifest = folder/'manifest.json'
    saved = json.loads(manifest.read_text())
    cfg = saved['settings']
    if cfg['chains'] < 4 or count % cfg['chains'] or count//cfg['chains'] > cfg['draws']:
        raise ValueError('Requested draw count must balance chains and not exceed the cache')
    data, meta, arrays, ratings, frames = inputs(model, training=True, age_terms=0)
    ids = verify_scope(root)
    if meta['ids'] != ids or saved['meta']['ids'] != ids:
        raise RuntimeError(f'Participant order/sample differs for {run}')
    for field in ['model', 'training', 'age_terms', 'bounded', 'independent', 'prior_scale', 'n_trials']:
        if saved['meta'][field] != meta[field]:
            raise RuntimeError(f'Posterior metadata differs: {run}: {field}')
    source = (root/'stan/hierarchical_shared.stan').read_text()
    fingerprint = hashlib.sha256((json.dumps(data, sort_keys=True)+json.dumps(cfg, sort_keys=True)+source).encode()).hexdigest()
    implementation = hashlib.sha256((root/'stan/rl_fast.hpp').read_bytes()+(root/'stan/hierarchical_fast.stan').read_bytes()).hexdigest()
    if saved['fingerprint'] != fingerprint or saved.get('implementation_sha256') != implementation:
        raise RuntimeError(f'Stale input or likelihood fingerprint: {run}')
    for field, setting in [('chains', 'chains'), ('draws_per_chain', 'draws'), ('warmup_per_chain', 'warmup'), ('adapt_delta', 'adapt_delta')]:
        if not diag[field].eq(cfg[setting]).all():
            raise RuntimeError(f'Diagnostic settings disagree with {run} manifest')
    files = []
    for value in saved['csv_files']:
        path = Path(value)
        if not path.is_absolute():
            path = root/path
        path = path.resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise RuntimeError(f'Missing or external posterior file: {path}; no refit or relocation is automatic')
        files.append(path)
    if len(files) != cfg['chains'] or len(set(files)) != len(files):
        raise RuntimeError(f'Incorrect/duplicate chain files: {run}')
    natural, evidence = [], []
    keep = np.linspace(0, cfg['draws']-1, count//cfg['chains']).astype(int)
    for path in files:
        before = sha(path)
        x, check = scan_natural(path, data['N'], data['K'], cfg['draws'], keep, cfg['max_treedepth'])
        if sha(path) != before:
            raise RuntimeError('Posterior changed during read-only audit')
        natural.append(x)
        evidence.append(dict(file=str(path.relative_to(root)), sha256=before, **check))
    return np.concatenate(natural), meta, arrays, dict(run=run, fingerprint=fingerprint,
        manifest_sha256=sha(manifest), diagnostic_sha256=sha(diagpath), chains=evidence)


def previous_feedback(a):
    """Label information available BEFORE each decision, never the current outcome."""
    last = np.full(3, -1, dtype=int)
    out = []
    for row in a:
        c = int(row[0])
        out.append(last[c])
        if row[3] >= 0 and row[4] > 0:
            last[c] = int(row[5])
    return np.asarray(out)


def trial_groups(a):
    """Marginal decompositions, not a sparse Cartesian product of every feature."""
    n = len(a)
    valid = a[:, 3] >= 0
    partner = np.asarray(PARTNERS)[a[:, 0].astype(int)]
    offers = np.asarray([f'{int(lo)}-{int(hi)}' for lo, hi in a[:, 1:3]])
    zero = np.where(a[:, 1] == 0, 'zero', 'positive_positive')
    runs = np.unique(a[:, 7])
    if len(runs) not in (1, 2):
        raise ValueError('Unexpected ordinary run structure')
    run = np.asarray(['single_run']*n) if len(runs) == 1 else np.asarray([f'run_{np.flatnonzero(runs == v)[0]+1}' for v in a[:, 7]])
    bins = np.asarray(['early', 'middle', 'late'])[np.minimum(2, (3*np.arange(n)//n).astype(int))]
    history = np.asarray(['no_previous', 'defection', 'reciprocation'])[previous_feedback(a)+1]
    period = np.where(np.arange(n) < split_index(a), 'training', 'heldout')
    features = [('partner', None, None), ('offer', offers, 'offer_pair'),
                ('zero_option', zero, 'zero_option'), ('run', run, 'run_or_bin'),
                ('time_bin', bins, 'run_or_bin'), ('previous_feedback', history, 'previous_feedback')]
    for window in ['all', 'training', 'heldout']:
        for p in PARTNERS:
            base = valid & (partner == p) & ((period == window) if window != 'all' else True)
            for kind, values, field in features:
                for label in (['all'] if values is None else np.unique(values)):
                    mask = base if values is None else base & (values == label)
                    if not mask.any():
                        continue
                    fields = dict(stratification=kind, partner=p, offer_pair='all', zero_option='all',
                                  run_or_bin='all', previous_feedback='all', evaluation_period=window)
                    if field:
                        fields[field] = label
                    if kind == 'offer':
                        fields['zero_option'] = 'zero' if label.startswith('0-') else 'positive_positive'
                    yield tuple(fields[x] for x in DIMENSIONS), mask


def prediction_arrays(a, model, draws, seed, participant):
    """Equal posterior draws for both histories; independent predictive randomness."""
    code = SPECS[model][1]
    names = SPECS[model][2]
    n = len(a)
    conditional, replicated, generative = [], [], []
    for d, pars in enumerate(draws):
        natural = BASE.copy()
        for p, v in zip(names, pars):
            natural[INDEX[p]] = v
        observed = engine(a, natural, code, np.zeros(3))
        u = np.random.default_rng(stable_seed(seed, model, participant, d, 'conditional_history')).random(n)
        y = (u < observed[:, 1]).astype(float)
        y[a[:, 3] < 0] = -1
        # Conditional replications never enter a belief update. engine sees the ORIGINAL a.
        conditional.append(observed[:, 1])
        replicated.append(y)
        u = np.random.default_rng(stable_seed(seed, model, participant, d, 'generative_history')).random(n)
        gen = engine(a, natural, code, np.zeros(3), False, True, u)
        if not np.array_equal(gen[:, 4] > 0, (gen[:, 2] >= 0) & (np.where(gen[:, 2] == 1, a[:, 2], a[:, 1]) > 0)):
            raise RuntimeError('Generative feedback exposure mismatch')
        generative.append(gen[:, 2])
    return dict(conditional_history=(np.asarray(conditional), np.asarray(replicated)),
                generative_history=(np.asarray(generative), np.asarray(generative)))


def summarize_predictions(model, arrays, ids, natural, seed=20260925):
    accum = {}
    participant_rows = []
    for i, participant in enumerate(ids):
        a = arrays[participant]
        observed_amount = np.where(a[:, 3] == 1, a[:, 2], a[:, 1])
        predictions = prediction_arrays(a, model, natural[:, i], seed, participant)
        groups = list(trial_groups(a))
        for prediction_type, (expected, replicated) in predictions.items():
            for key, mask in groups:
                yh = a[mask, 3].mean()
                yi = observed_amount[mask].mean()
                ph = expected[:, mask].mean(axis=1)
                pi = (a[mask, 1] + expected[:, mask]*(a[mask, 2]-a[mask, 1])).mean(axis=1)
                rh = replicated[:, mask].mean(axis=1)
                ri = (a[mask, 1] + replicated[:, mask]*(a[mask, 2]-a[mask, 1])).mean(axis=1)
                fullkey = (prediction_type,)+key
                if fullkey not in accum:
                    accum[fullkey] = dict(n_trials=0, n_participants=0, observed_high=0., observed_investment=0.,
                                         expected_high=np.zeros(len(natural)), expected_investment=np.zeros(len(natural)),
                                         replicated_high=np.zeros(len(natural)), replicated_investment=np.zeros(len(natural)))
                item = accum[fullkey]
                item['n_trials'] += int(mask.sum())
                item['n_participants'] += 1
                item['observed_high'] += yh
                item['observed_investment'] += yi
                for field, values in [('expected_high', ph), ('expected_investment', pi), ('replicated_high', rh), ('replicated_investment', ri)]:
                    item[field] += values
                participant_rows.append(dict(model=model, prediction_type=prediction_type, participant_id=participant,
                    **dict(zip(DIMENSIONS, key)), n_trials=int(mask.sum()), observed_high=yh, predicted_high=ph.mean(),
                    residual_high=yh-ph.mean(), observed_investment=yi, predicted_investment=pi.mean(), residual_investment=yi-pi.mean()))
    rows = []
    for key, item in accum.items():
        n = item['n_participants']
        ph = item['expected_high']/n
        pi = item['expected_investment']/n
        rh = item['replicated_high']/n
        ri = item['replicated_investment']/n
        oh, oi = item['observed_high']/n, item['observed_investment']/n
        rows.append(dict(model=model, source_run=f'{model}_train_noage', posterior_fit_window='training',
            prediction_type=key[0], **dict(zip(DIMENSIONS, key[1:])), n_trials=item['n_trials'], n_participants=n,
            observed_high=oh, predicted_high=ph.mean(), residual_high=oh-ph.mean(),
            observed_investment=oi, predicted_investment=pi.mean(), residual_investment=oi-pi.mean(),
            predictive_interval_low=np.quantile(rh, .025), predictive_interval_high=np.quantile(rh, .975),
            investment_predictive_interval_low=np.quantile(ri, .025), investment_predictive_interval_high=np.quantile(ri, .975),
            posterior_draws=len(natural), cell_weighting='equal_participant', sparse_cell=n < 20,
            history_strata='observed_pretrial_labels_for_both_predictions'))
    return pd.DataFrame(rows), pd.DataFrame(participant_rows)


def paired_histories(residuals):
    keys = ['model']+DIMENSIONS
    a = residuals[residuals.prediction_type.eq('conditional_history')]
    b = residuals[residuals.prediction_type.eq('generative_history')]
    joined = a.merge(b, on=keys, suffixes=('_conditional', '_generative'), validate='one_to_one')
    if len(joined) != len(a) or len(a) != len(b):
        raise ValueError('Prediction history cells differ')
    for field in ['n_trials', 'n_participants', 'observed_high', 'observed_investment']:
        if not np.allclose(joined[field+'_conditional'], joined[field+'_generative']):
            raise ValueError(f'Prediction histories use different observed cells: {field}')
    joined['generative_minus_conditional_residual_high'] = joined.residual_high_generative-joined.residual_high_conditional
    joined['generative_minus_conditional_residual_investment'] = joined.residual_investment_generative-joined.residual_investment_conditional
    return joined


def zero_contrasts(participants, seed=20260925):
    """Paired descriptive zero-minus-positive residuals; no causal interpretation."""
    d = participants[participants.stratification.eq('zero_option')]
    keys = ['model', 'prediction_type', 'partner', 'evaluation_period']
    rows = []
    for values, frame in d.groupby(keys, sort=True):
        wide = frame.pivot(index='participant_id', columns='zero_option', values='residual_high')
        if not {'zero', 'positive_positive'} <= set(wide.columns):
            continue
        paired = wide.dropna(subset=['zero', 'positive_positive'])
        x = (paired.zero-paired.positive_positive).to_numpy()
        if len(x) < 2:
            continue
        rng = np.random.default_rng(stable_seed(seed, *values, 'zero_contrast'))
        boots = x[rng.integers(len(x), size=(5000, len(x)))].mean(axis=1)
        rows.append(dict(zip(keys, values)) | dict(n_paired_participants=len(x),
            zero_residual_high=paired.zero.mean(), positive_positive_residual_high=paired.positive_positive.mean(),
            zero_minus_positive_residual=x.mean(), ci_low=np.quantile(boots, .025), ci_high=np.quantile(boots, .975),
            interval_type='paired_participant_bootstrap_95_percent', bootstrap_replicates=5000))
    return pd.DataFrame(rows)


def published_check(root):
    """Verify the brief using committed age-fit PPCs, explicitly distinct from the new audit."""
    from .accepted_review import diagnostic_pass
    verify_scope(root)
    state = json.loads((root/'results/linux_run_status.json').read_text())['runs']
    rows, hashes = [], {}
    for model in FOCAL:
        run = f'{model}_full_age'
        dp = root/'results/tables'/f'diagnostics_{run}.csv'
        path = root/'results/tables'/f'predictive_{run}.csv'
        if state[run]['status'] != 'complete' or not diagnostic_pass(pd.read_csv(dp)):
            raise RuntimeError(f'Unaccepted published input: {run}')
        hashes[str(path.relative_to(root))] = sha(path)
        hashes[str(dp.relative_to(root))] = sha(dp)
        d = pd.read_csv(path)
        if not d.run.eq(run).all():
            raise ValueError('Published PPC run mismatch')
        d = d[d.category.eq('offer') & d.measure.eq('high_probability')].copy()
        if len(d) != 18:
            raise ValueError('Expected 3 partners × 6 offers')
        labels = d.label.str.split(':', expand=True)
        d['partner'], d['offer_pair'] = labels[0], labels[1]
        d['zero_option'] = np.where(d.offer_pair.str.startswith('0-'), 'zero', 'positive_positive')
        d['residual_high'] = d.observed-d['mean']
        d['posterior_fit_window'] = 'full_age'
        rows.append(d)
    out = root/'results/n111_wrapup'
    (out/'tables').mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(out/'tables/published_offer_check.csv', index=False)
    summary = d.groupby(['model', 'partner', 'zero_option'], as_index=False).agg(mean_offer_residual=('residual_high', 'mean'))
    summary['weighting'] = 'equal_offer_cells_not_trial_weighted'
    summary.to_csv(out/'tables/published_zero_contrast.csv', index=False)
    (out/'published_check_provenance.json').write_text(json.dumps(dict(inputs=PINNED | hashes,
        source_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        generator_sha256=sha(Path(__file__))), indent=2)+'\n')
    plot_published(d, out/'figures/00_published_offer_check')
    if not (out/'audit_status.json').exists():
        write_preview_report(d, out)
    print('Verified published full-age PPCs; no no-age posterior draws or new fits were used.', flush=True)
    return d


def write_preview_report(d, out):
    from .accepted_review import table
    selected = d[d.model.eq('HPreference') & d.partner.eq('computer')].copy()
    text = """# N=111 closeout — diagnostic stage prepared

**Closeout incomplete: awaiting the no-age history diagnostic on linux1.** The [fixed brief](../../docs/n111_wrapup_scope.md) and [decision record](../../docs/n111_wrapup_decisions.md) define the finite scope. No additional participant data have been accessed, no new fits have been started, and H4_full_age / H5_train_age remain excluded. The [accepted-fit report](../review/README.md) and caches are preserved.

The motivating zero-option pattern is verified directly from the accepted, committed full-age PPC outputs. HPreference computer trials:

"""
    text += table(selected, ['offer_pair', 'observed', 'mean', 'ci_low', 'ci_high', 'residual_high'])
    text += """

H5 and H7 show similar computer zero-option underprediction. H8 shows a different pattern, including overprediction on large positive-positive offers. This existing full-age check does not establish whether the discrepancy persists under actual histories, or whether a generic zero-option term improves prospective performance.

![Published offer check](figures/00_published_offer_check.png)

Shading marks zero-containing offers. Black crosses show observations; teal points/intervals show existing generative means and 95% predictive intervals. The plot is a verification of prior outputs, not newly fitted no-age results. [Source values](tables/published_offer_check.csv) and [provenance](published_check_provenance.json) are available. PDF and SVG accompany the PNG.

## Next execution

Follow the [Linux commands](../../docs/n111_linux.md). Stage A reads the five accepted no-age training posterior caches, computes both history predictions, produces matched residual tables and figures, and stops without sampling. The main figures use held-out trials; all/training/held-out table rows are labeled separately. No automatic model-extension or recovery run follows this command.

Zero-option retention, realistic H7-versus-HPreference recovery, and the final synthesis remain pending. No placeholders are presented as estimates for these unrun analyses.

## What should carry forward to the full dataset

No final new modeling decision is available yet. Preserve the fixed sample/trial conventions, accepted diagnostics, unresolved rating timing, and age uncertainty while completing the scoped diagnostic. There are no instructions here to access or import additional participants.
"""
    (out/'README.md').write_text(text)


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    for ext in ['png', 'pdf', 'svg']:
        options = dict(dpi=200, bbox_inches='tight')
        if ext == 'pdf':
            options['metadata'] = {'CreationDate': None}
        if ext == 'svg':
            options['metadata'] = {'Date': None}
        fig.savefig(path.with_suffix('.'+ext), **options)
        if ext == 'svg':
            svg = path.with_suffix('.svg')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'n111-closeout', 'figure.facecolor': '#fafbf9', 'axes.facecolor': '#fafbf9'})
    return plt


def plot_published(d, path):
    plt = style()
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True, sharey=True, layout='constrained')
    for i, model in enumerate(FOCAL):
        for j, partner in enumerate(PARTNERS):
            ax = axes[i, j]
            v = d[d.model.eq(model) & d.partner.eq(partner)].set_index('offer_pair').loc[OFFERS]
            ax.axvspan(-.5, 2.5, color='#e5b767', alpha=.15)
            x = np.arange(6)
            ax.vlines(x, v.ci_low, v.ci_high, color='#007f86', lw=2)
            ax.scatter(x, v['mean'], color='#007f86', s=18, label='Generative mean / 95% PI')
            ax.scatter(x, v.observed, color='#222222', marker='x', s=28, label='Observed')
            ax.set(ylim=(0, 1), xticks=x, xticklabels=OFFERS, title=f'{model} · {partner}')
            if j == 0:
                ax.set_ylabel('Probability of higher investment')
            if i == 3:
                ax.set_xlabel('Offer pair ($); shaded = zero option')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Verified existing full-age PPCs · N=111\nStage A preview; not the forthcoming no-age history comparison', fontsize=14)
    save_figure(fig, path)


def plot_audit(d, out):
    plt = style()
    # Main forecast figure: same posteriors and held-out choices across both histories.
    v = d[d.stratification.eq('offer') & d.evaluation_period.eq('heldout')]
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True, sharey=True, layout='constrained')
    for i, model in enumerate(FOCAL):
        for j, partner in enumerate(PARTNERS):
            ax = axes[i, j]
            ax.axvspan(-.5, 2.5, color='#e5b767', alpha=.15)
            for pred, color, offset, label in [('conditional_history', '#007f86', -.08, 'Conditional'), ('generative_history', '#a55e9b', .08, 'Generative')]:
                frame = v[v.model.eq(model) & v.partner.eq(partner) & v.prediction_type.eq(pred)].set_index('offer_pair').reindex(OFFERS)
                x = np.arange(6)+offset
                ax.vlines(x, frame.predictive_interval_low, frame.predictive_interval_high, color=color, alpha=.65)
                ax.scatter(x, frame.predicted_high, color=color, s=16, label=label)
            ax.scatter(np.arange(6), frame.observed_high, marker='x', color='#222222', label='Observed')
            ax.set(xticks=np.arange(6), xticklabels=OFFERS, ylim=(0, 1), title=f'{model} · {partner}')
            if j == 0:
                ax.set_ylabel('Probability of higher investment')
            if i == 3:
                ax.set_xlabel('Offer pair ($); shaded = zero option')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Where the models miss · held-out trials · accepted no-age training posteriors\n95% replicated-choice intervals; equal participant weight within cells', fontsize=14)
    save_figure(fig, out/'figures/01_where_models_miss')
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), layout='constrained')
    extent = max(.05, np.nanmax(np.abs(v.residual_high)))
    for i, model in enumerate(FOCAL):
        for j, pred in enumerate(['conditional_history', 'generative_history']):
            ax = axes[i, j]
            x = v[v.model.eq(model) & v.prediction_type.eq(pred)]
            grid = x.pivot(index='partner', columns='offer_pair', values='residual_high').reindex(index=PARTNERS, columns=OFFERS)
            im = ax.imshow(grid, vmin=-extent, vmax=extent, cmap='RdBu_r', aspect='auto')
            for r in range(3):
                for c in range(6):
                    value = grid.iloc[r, c]
                    ax.text(c, r, f'{value:+.2f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(value) > extent*.6 else '#222222')
            ax.axvline(2.5, color='#222222', lw=2)
            ax.set(xticks=range(6), xticklabels=OFFERS, yticks=range(3), yticklabels=PARTNERS,
                   title=f'{model} · {pred.replace("_", " ")}')
    fig.colorbar(im, ax=axes, label='Observed − predicted high-choice probability', shrink=.8)
    fig.suptitle('Choice-history decomposition · held-out trials\nLeft three columns contain a $0 option; identical observed cells across histories', fontsize=14)
    save_figure(fig, out/'figures/02_conditional_vs_generative')


def audit_model(root, model, draws, seed):
    natural, meta, arrays, evidence = load_training_draws(root, model, draws)
    print(f'{model}: read {len(natural)} balanced posterior draws; generating history comparisons', flush=True)
    residuals, participants = summarize_predictions(model, arrays, meta['ids'], natural, seed)
    out = root/'results/n111_wrapup'
    residuals.to_csv(out/'tables'/f'predictive_residuals_{model}.csv', index=False)
    participants.to_csv(out/'tables'/f'predictive_residuals_participants_{model}.csv', index=False)
    (out/f'posterior_audit_{model}.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return model


def run_audit(root, draws=400, jobs=5, seed=20260925):
    verify_scope(root)
    out = root/'results/n111_wrapup'
    (out/'tables').mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    record = dict(stage='A_history_diagnostics', status='running', models={m: 'pending' for m in MODELS},
                  posterior_draws=draws, seed=seed, jobs=jobs, git_commit=revision,
                  generator_sha256=sha(Path(__file__)), inputs=PINNED,
                  excluded_runs=['H4_full_age', 'H5_train_age'], sampling_started=False)
    def save():
        temp = out/'audit_status.json.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(out/'audit_status.json')
    save()
    (out/'README.md').write_text('# N=111 closeout — history audit in progress\n\nSee [audit_status.json](audit_status.json) for current progress. Prior aggregate files, if present, must not be treated as outputs of this unfinished run. No sampling is performed; the closeout remains incomplete.\n')
    errors = []
    with ProcessPoolExecutor(max_workers=min(jobs, len(MODELS))) as pool:
        tasks = {pool.submit(audit_model, root, m, draws, seed): m for m in MODELS}
        for future in as_completed(tasks):
            m = tasks[future]
            try:
                future.result()
                record['models'][m] = 'complete'
            except Exception as exc:
                record['models'][m] = f'error: {exc}'
                errors.append(f'{m}: {exc}')
            save()
    if errors:
        record['status'] = 'error'
        save()
        (out/'README.md').write_text('# N=111 closeout — history audit stopped\n\nOne or more posterior checks or prediction jobs failed. See [audit_status.json](audit_status.json) and the tracked console logs. No aggregate result from this invocation is accepted. No sampling or automatic retry was performed.\n')
        raise RuntimeError('No aggregate published; inspect per-model audit logs: '+'; '.join(errors))
    d = pd.concat([pd.read_csv(out/'tables'/f'predictive_residuals_{m}.csv') for m in MODELS], ignore_index=True)
    p = pd.concat([pd.read_csv(out/'tables'/f'predictive_residuals_participants_{m}.csv') for m in MODELS], ignore_index=True)
    validate_residuals(d)
    d.to_csv(out/'tables/predictive_residuals.csv', index=False)
    paired_histories(d).to_csv(out/'tables/conditional_vs_generative.csv', index=False)
    contrasts = zero_contrasts(p, seed)
    contrasts.to_csv(out/'tables/zero_option_residual_contrasts.csv', index=False)
    plot_audit(d, out)
    record['status'] = 'diagnostics_complete_closeout_pending'
    record['next_stage'] = 'Interpret zero-option evidence before any model extension or recovery run'
    save()
    write_audit_report(d, contrasts, out)
    print('Stage A diagnostic outputs complete. No new fits; zero-option/model-recovery decisions remain pending.', flush=True)


def validate_residuals(d):
    numeric = ['observed_high', 'predicted_high', 'residual_high', 'observed_investment', 'predicted_investment',
               'residual_investment', 'predictive_interval_low', 'predictive_interval_high']
    if not np.isfinite(d[numeric].to_numpy()).all():
        raise ValueError('Nonfinite predictive residual')
    if d.duplicated(['model', 'prediction_type']+DIMENSIONS).any():
        raise ValueError('Duplicate diagnostic cells')
    if not d.n_participants.between(1, 111).all():
        raise ValueError('Unexpected participant count')
    for field in ['observed_high', 'predicted_high', 'predictive_interval_low', 'predictive_interval_high']:
        if not d[field].between(0, 1).all():
            raise ValueError(f'Invalid probability: {field}')
    if (d.predictive_interval_low > d.predictive_interval_high).any():
        raise ValueError('Reversed predictive intervals')
    if not d.observed_investment.between(0, 8).all() or not d.predicted_investment.between(0, 8).all():
        raise ValueError('Invalid investment prediction')
    if not np.allclose(d.observed_high-d.predicted_high, d.residual_high):
        raise ValueError('Residual arithmetic mismatch')
    if not np.allclose(d.observed_investment-d.predicted_investment, d.residual_investment):
        raise ValueError('Investment residual arithmetic mismatch')
    for _, part in d[d.stratification.eq('partner')].groupby(['model', 'prediction_type', 'evaluation_period']):
        period = part.evaluation_period.iloc[0]
        expected = {'all': 8251, 'training': 4234, 'heldout': 4017}[period]
        if part.n_trials.sum() != expected:
            raise ValueError(f'Incorrect {period} trial count')
    paired_histories(d)


def write_audit_report(d, contrasts, out):
    from .accepted_review import table
    selected = contrasts[contrasts.partner.eq('computer') & contrasts.evaluation_period.eq('heldout')]
    text = '''# N=111 closeout — stage A diagnostic results

**Diagnostic stage complete; closeout not yet complete.** These outputs use the five accepted no-age TRAINING posteriors. The two excluded fits remain excluded. No new sampling or additional participant data were used. The [accepted-fit review](../review/README.md) is preserved.

The primary figures use the 4,017 held-out choices. Tables also distinguish all 8,251 valid choices and the 4,234 training choices; all-trial summaries include in-sample decisions and must not be described as prospective performance. Parameters stay fixed at their training posterior. Conditional predictions replay actual pretrial feedback; generative predictions simulate feedback exposure from the beginning of each participant's sequence. Thus the comparison includes divergence of simulated training histories as well as held-out histories.

Conditional predictive means average choice probabilities over posterior draws. Their predictive intervals use independent Bernoulli replications that do not affect subsequent feedback exposure. Generative means and intervals use simulated choices whose feedback changes subsequent beliefs. Intervals are pointwise replicated-choice intervals, not confidence intervals on residuals; no multiplicity correction is applied. Both retain the same posterior draw indices and observed cell definitions.

History-stratified cells use the actual PRETRIAL same-partner feedback label for both prediction types so the compared observed subsets are identical. The actual label is used only for grouping, never to update a generative belief. These are fixed-observed-stratum diagnostics, not a replication of the distribution of simulated feedback-category membership. Run 1/2 cells include the 90 two-run participants; 21 single-run participants are identified separately. Early/middle/late bins use chronological thirds of each participant's full sequence including missed trials. Cells with fewer than 20 participants are retained and flagged, not given equal evidential weight.

## Zero-option residual contrast

Computer trials in the held-out period, paired within participants who contribute both zero and positive-positive offers. Positive values mean greater underprediction for zero-containing offers. The bootstrap resamples participants (5,000 draws), conditional on the posterior prediction estimates. It is descriptive, not a causal effect of presenting zero, and does not match exact offer amounts.

'''
    text += table(selected, ['model', 'prediction_type', 'n_paired_participants', 'zero_residual_high', 'positive_positive_residual_high', 'zero_minus_positive_residual', 'ci_low', 'ci_high'])
    text += '''

## Figures

![Where the models miss](figures/01_where_models_miss.png)

![Conditional versus generative residuals](figures/02_conditional_vs_generative.png)

PNG, PDF and SVG are available in [figures](figures/). All partners, exact offers, zero-option status, runs, chronological bins, and prior-feedback categories are in [predictive_residuals.csv](tables/predictive_residuals.csv). [Conditional versus generative](tables/conditional_vs_generative.csv) matches the same observed cells.

## Pending closeout decisions

Interpret the signed residuals, exact-offer plots, and paired zero contrasts together. If the common zero-option pattern warrants the extension, fit only the shared gamma0 term specified in the [closeout brief](../../docs/n111_wrapup_scope.md), then assess PPCs and paired held-out log loss/Brier. No code in this stage fits that extension, makes a retention decision, or runs realistic model recovery. Those results and the final synthesis remain pending.

Age conclusions remain unchanged: behavioral friend-minus-computer age-25-to-75 change +.025, 95% CI [-.150, .201]; friend-minus-stranger +.002 [-.141, .146]; accepted H5 canonical friend-value change −.050, 95% credible interval [-.185, .078]. These admit meaningful effects in either direction. Latent variance fractions are not behavioral variance explained. Ratings timing remains unresolved; no ratings investigation was performed.

## What should carry forward to the full dataset

Decisions about the zero-option term and mechanism distinguishability remain pending. Preserve the sample/trial conventions, separate conditional and generative checks, report age uncertainty, and retain the unresolved ratings limitation. No additional participant data have been accessed by this workflow.
'''
    (out/'README.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published-check', action='store_true', help='Verify existing committed full-age PPCs only; safe on laptop')
    parser.add_argument('--run', action='store_true', help='Read current Linux no-age training posteriors; never sample')
    parser.add_argument('--jobs', type=int, default=5)
    parser.add_argument('--draws', type=int, default=400)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.jobs < 1 or args.draws < 4:
        parser.error('jobs must be positive and draws at least 4')
    verify_scope(root)
    if args.published_check:
        published_check(root)
    if not args.run:
        if not args.published_check:
            print('Plan: read five accepted no-age training posteriors; 400 balanced draws each by default; no MCMC.\nRun --run on linux1. This stage stops after the history/zero-option diagnostic.')
        return
    if platform.system() != 'Linux':
        raise RuntimeError('Current posterior audit runs on linux1, not the laptop')
    from .linux_handoff import coordinator_lock, existing_workers, assert_no_live_fit_locks
    with coordinator_lock(root):
        if existing_workers(root):
            raise RuntimeError('Wait for current sampling to finish before this read-only audit')
        assert_no_live_fit_locks(root)
        run_audit(root, args.draws, args.jobs)


if __name__ == '__main__':
    main()
