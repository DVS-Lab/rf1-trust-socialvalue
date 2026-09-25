"""Rebuild an explicitly partial review from accepted, committed summary tables.

No sampling, chain access, or edits to finalization thresholds. All hierarchical
inputs must be named by the current Linux inventory and pass numerical checks.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ['H2', 'H5', 'H8', 'HPreference', 'H7']
LABELS = {'H2': 'Monetary RL', 'H5': 'Friend value', 'H8': 'Asymmetric learning',
          'HPreference': 'Partner preference', 'H7': 'Friend / stranger value'}
PARAMS = {'alpha': 'Learning rate α', 'kappa': 'Choice sensitivity κ', 'theta': 'Friend value θ',
          'friend_value_probability_effect': 'Canonical friend-value effect'}
COLORS = ['#64748b', '#007f86', '#bc6c25', '#7b58a4', '#c04969']


def diagnostic_pass(d):
    if d.empty:
        return False
    cols = ['R_hat', 'ESS_bulk', 'ESS_tail', 'divergences', 'max_depth_hits', 'min_bfmi']
    return bool(np.isfinite(d[cols].to_numpy(float)).all()
                and (d.R_hat < 1.01).all() and (d.ESS_bulk >= 400).all()
                and (d.ESS_tail >= 400).all() and (d.divergences == 0).all()
                and (d.max_depth_hits == 0).all() and (d.min_bfmi > .3).all()
                and d.passed.eq(True).all())


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.hashes = {}
        path = self.root / 'results/linux_run_status.json'
        self.track(path)
        status = json.loads(path.read_text())['runs']
        rows = []
        self.accepted = set()
        for run, entry in status.items():
            d = self.read('tables', f'diagnostics_{run}.csv')
            if not d.run.eq(run).all():
                raise ValueError(f'Diagnostic run mismatch: {run}')
            passed = diagnostic_pass(d)
            accepted = entry['status'] == 'complete' and passed
            if accepted:
                self.accepted.add(run)
            rows.append(dict(run=run, status=entry['status'], accepted=accepted,
                             max_rhat=d.R_hat.max(), min_bulk_ess=d.ESS_bulk.min(),
                             min_tail_ess=d.ESS_tail.min(), divergences=d.divergences.max(),
                             depth_hits=d.max_depth_hits.max(), min_bfmi=d.min_bfmi.min(),
                             retained_draws=int(d.chains.iloc[0]*d.draws_per_chain.iloc[0])))
        self.inventory = pd.DataFrame(rows)

    def track(self, path):
        self.hashes[str(path.relative_to(self.root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(self, folder, filename):
        path = self.root / 'results' / folder / filename
        self.track(path)
        return pd.read_csv(path)

    def fit(self, prefix, run):
        if run not in self.accepted:
            raise ValueError(f'Excluded or unknown fit: {run}')
        d = self.read('tables', f'{prefix}_{run}.csv')
        if 'run' in d and not d.run.eq(run).all():
            raise ValueError(f'Summary run mismatch: {run}')
        return d


def paired_difference(a, b):
    """One loss per identical participant and held-out trial count; no silent joins."""
    if a.participant_id.duplicated().any() or b.participant_id.duplicated().any():
        raise ValueError('Duplicate participant scores')
    a = a.set_index('participant_id').sort_index()
    b = b.set_index('participant_id').sort_index()
    if not a.index.equals(b.index) or not a.n.equals(b.n):
        raise ValueError('Held-out participants or trial counts differ')
    return a.log_loss.to_numpy() - b.log_loss.to_numpy()


def interval(values):
    x = np.asarray(values, float)
    if len(x) != 111 or not np.isfinite(x).all():
        raise ValueError('Expected 111 finite participant scores')
    rng = np.random.default_rng(20260925)
    draws = x[rng.integers(len(x), size=(5000, len(x)))].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(mean=x.mean(), ci_low=lo, ci_high=hi)


def table(d, cols, digits=3):
    def fmt(v):
        if pd.isna(v):
            return '—'
        return f'{v:.{digits}f}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |'] +
                     ['| ' + ' | '.join(fmt(row[c]) for c in cols) + ' |' for _, row in d.iterrows()])


def forest(ax, d, labels, color='#007f86'):
    y = np.arange(len(d))
    ax.hlines(y, d.ci_low, d.ci_high, color=color, linewidth=2)
    ax.scatter(d['mean'], y, color=color, s=30, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)


def build(root):
    root = Path(root)
    src = Inputs(root)
    if set(src.inventory.loc[~src.inventory.accepted, 'run']) != {'H4_full_age', 'H5_train_age'} or len(src.accepted) != 32:
        raise ValueError('Fit inventory changed; update the scoped review narrative before rebuilding')
    out = root / 'results/review'
    figs = out / 'figures'
    tables = out / 'tables'
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220,
                         'figure.facecolor': '#fafbf9', 'axes.facecolor': '#fafbf9'})
    def save(fig, name, title):
        fig.suptitle(title, fontsize=16, fontweight='bold')
        fig.savefig(figs / f'{name}.png', bbox_inches='tight')
        fig.savefig(figs / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})
        plt.close(fig)
    def export(d, name):
        d.to_csv(tables / f'{name}.csv', index=False)

    export(src.inventory, 'fit_inventory')
    # Hierarchical sources are explicitly whitelisted; historical aggregates are never read.
    full = {m: src.fit('parameters', f'{m}_full_age') for m in MODELS}
    age = src.fit('age', 'H5_full_age')
    curves = src.fit('curves', 'H5_full_age')
    variants = ['H5_full_age', 'H5_full_age_bounded', 'H5_full_quadratic', 'H5_full_age_prior1.5']
    sensitivity = pd.concat([src.fit('age', run) for run in variants], ignore_index=True)
    export(age, 'h5_age')
    export(sensitivity, 'h5_age_sensitivity')

    behavior = src.read('tables', 'age_partner_marginal_effects.csv')
    change = src.read('tables', 'age_partner_change_25_to_75.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for label, color in zip(['computer', 'stranger', 'friend'], [COLORS[0], COLORS[2], COLORS[1]]):
        d = behavior[behavior.contrast.eq(label)]
        axes[0].plot(d.age, d.estimate, label=label.title(), color=color)
        axes[0].fill_between(d.age, d.ci_low, d.ci_high, color=color, alpha=.13)
    axes[0].set(xlabel='Age (years)', ylabel='Probability of higher investment', ylim=(0, 1), title='Offer-standardized behavior')
    axes[0].legend(frameon=False)
    for label, color in zip(['friend - computer', 'friend - stranger'], [COLORS[1], COLORS[3]]):
        d = behavior[behavior.contrast.eq(label)]
        axes[1].plot(d.age, d.estimate, label=label.title(), color=color)
        axes[1].fill_between(d.age, d.ci_low, d.ci_high, color=color, alpha=.13)
    axes[1].axhline(0, color='#999999', ls='--')
    axes[1].set(xlabel='Age (years)', ylabel='Probability difference', title='Friend advantage across age')
    axes[1].legend(frameon=False)
    save(fig, '01_behavior_age', 'Behavioral age effects · N = 111 · GEE 95% confidence bands')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, p in zip(axes[0], ['alpha', 'kappa', 'theta']):
        d = curves[curves.parameter.eq(p)]
        ax.plot(d.age, d['mean'], color=COLORS[1])
        ax.fill_between(d.age, d.ci_low, d.ci_high, color=COLORS[1], alpha=.18)
        ax.set(xlabel='Age (years)', ylabel=PARAMS[p], title=PARAMS[p])
    fraction = age[age.quantity.eq('latent_age_variance_fraction')].copy()
    fraction[['mean', 'ci_low', 'ci_high']] *= 100
    forest(axes[1, 0], fraction, ['α', 'κ', 'θ'])
    axes[1, 0].set(xlabel='Age-associated latent variance (%)', title='Age accounts for a small, uncertain fraction', xlim=(0, 20))
    effect = sensitivity[sensitivity.quantity.eq('natural_change_25_to_75') & sensitivity.parameter.eq('friend_value_probability_effect')]
    forest(axes[1, 1], effect, ['Main', 'θ bounded at 10', 'Quadratic age', 'Wider age prior'])
    axes[1, 1].axvline(0, color='#999999', ls='--')
    axes[1, 1].set(xlabel='Effect at 75 − effect at 25', title='Age contrast across H5 specifications')
    d = curves[curves.parameter.eq('friend_value_probability_effect')]
    axes[1, 2].plot(d.age, d['mean'], color=COLORS[1])
    axes[1, 2].fill_between(d.age, d.ci_low, d.ci_high, color=COLORS[1], alpha=.18)
    axes[1, 2].set(xlabel='Age (years)', ylabel='Probability effect', title='Canonical friend-value effect')
    save(fig, '02_hierarchical_age', 'Hierarchical H5 · typical-participant curves · 95% credible intervals')

    scores = {run: src.fit('heldout', run) for run in sorted(src.accepted) if '_train_' in run}
    for d in scores.values():
        if len(d) != 111 or d.n.sum() != 4017:
            raise ValueError('Unexpected held-out sample')
    absolute, contrasts, gains = [], [], []
    for m in MODELS:
        run = f'{m}_train_noage'
        absolute.append(dict(model=m, run=run, **interval(scores[run].log_loss)))
        if m != 'H5':
            contrasts.append(dict(model=m, reference='H5', **interval(paired_difference(scores[run], scores['H5_train_noage']))))
        arun = f'{m}_train_age'
        if arun in scores:
            gains.append(dict(model=m, **interval(paired_difference(scores[arun], scores[run]))))
    absolute, contrasts, gains = map(pd.DataFrame, [absolute, contrasts, gains])
    pref7 = pd.DataFrame([dict(model='H7', reference='HPreference', **interval(paired_difference(scores['H7_train_noage'], scores['HPreference_train_noage'])))])
    export(absolute, 'heldout_noage')
    export(pd.concat([contrasts, pref7], ignore_index=True), 'heldout_paired_contrasts')
    export(gains, 'heldout_age_minus_noage')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
    forest(axes[0], absolute, [LABELS[m] for m in absolute.model])
    axes[0].set(xlabel='Mean participant log loss (lower is better)', title='All five models: no age covariate')
    forest(axes[1], contrasts, [LABELS[m] for m in contrasts.model], COLORS[3])
    axes[1].axvline(0, color='#999999', ls='--')
    axes[1].set(xlabel='Log loss − H5 log loss', title='Paired differences from friend value')
    gain_plot = gains.copy()
    gain_plot[['mean', 'ci_low', 'ci_high']] *= 1000
    forest(axes[2], gain_plot, [LABELS[m] for m in gains.model], COLORS[2])
    axes[2].axvline(0, color='#999999', ls='--')
    axes[2].set(xlabel='Age-model loss − no-age loss (×10⁻³)', title='Age prediction gain: H5 unavailable')
    save(fig, '03_heldout', 'Temporal prediction · 111 participants / 4,017 choices · paired bootstrap 95% intervals')

    ppc = pd.concat([src.fit('predictive', f'{m}_full_age') for m in MODELS], ignore_index=True)
    ppc = ppc[ppc.category.eq('partner') & ppc.measure.eq('high_probability')].copy()
    if len(ppc) != 15 or not ppc.n_subjects.eq(111).all():
        raise ValueError('Unexpected PPC sample')
    ppc['observed_outside_interval'] = (ppc.observed < ppc.ci_low) | (ppc.observed > ppc.ci_high)
    export(ppc, 'partner_posterior_predictive')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained', sharey=True)
    for ax, partner in zip(axes, ['computer', 'stranger', 'friend']):
        d = ppc[ppc.label.eq(partner)].set_index('model').loc[MODELS].reset_index()
        forest(ax, d, [LABELS[m] for m in MODELS])
        if np.ptp(d.observed) > 1e-10:
            raise ValueError('PPC observations differ between models')
        ax.axvline(d.observed.iloc[0], color='#222222', ls='--', label='Observed')
        ax.set(title=partner.title(), xlabel='Probability of higher investment')
        ax.legend(frameon=False, fontsize=9)
    # Shared axes must not be inverted once per panel.
    axes[0].set_ylim(len(MODELS)-.5, -.5)
    save(fig, '04_predictive_checks', 'Full-data posterior predictive checks · 95% simulated-data intervals')

    bounds = src.read('tables', 'theta_bound_sensitivity.csv')
    bounds = bounds[bounds.base_model.eq('M5')]
    ceiling = bounds.groupby('theta_upper', as_index=False).agg(at_ceiling=('theta_at_upper', 'sum'), n=('participant_id', 'size'))
    export(ceiling, 'm5_ceiling_counts')
    mle = src.read('tables', 'model_fits_theta10.csv')
    mle = mle[mle.model.eq('M5')].set_index('participant_id')
    estimates = full['H5'][full['H5'].parameter.eq('theta')].set_index('participant_id')
    if len(mle) != 111 or not mle.index.sort_values().equals(estimates.index.sort_values()):
        raise ValueError('Mismatched MLE / hierarchy sample')
    matched = estimates.join(mle[['theta']], validate='one_to_one')
    export(matched.reset_index(), 'h5_theta_mle_hierarchy')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    axes[0].bar(ceiling.theta_upper.astype(str), 100*ceiling.at_ceiling/ceiling.n, color=COLORS[1], width=.6)
    for i, row in ceiling.iterrows():
        axes[0].text(i, 100*row.at_ceiling/row.n+2, f'{int(row.at_ceiling)}/{int(row.n)}', ha='center')
    axes[0].set(ylim=(0, 70), xlabel='MLE θ upper bound', ylabel='Participants at ceiling (%)', title='Widening the bound does not resolve saturation')
    axes[1].scatter(matched.theta, matched['mean'], alpha=.6, color=COLORS[1], s=25)
    limit = max(10, matched['mean'].max())*1.06
    axes[1].plot([0, limit], [0, limit], color='#999999', ls='--')
    axes[1].axvline(10, color=COLORS[2], ls=':')
    axes[1].set(xlabel='Individual MLE θ (bound = 10)', ylabel='Hierarchical posterior mean θ (unbounded)', title='Regularization changes individual estimates', xlim=(0, limit), ylim=(0, limit))
    save(fig, '05_theta_identifiability', 'Raw social-value scale remains fragile · N = 111')

    recnames = [f'H5_recovery_{c}_{i}' for c in ['zero', 'positive', 'negative'] for i in range(5)]
    recovery = pd.concat([src.fit('recovery', r) for r in recnames], ignore_index=True)
    recovery['interval_coverage'] = pd.to_numeric(recovery.interval_coverage.replace({'True': 1., 'False': 0.}))
    individual = pd.concat([src.fit('recovery_individual', r) for r in recnames], ignore_index=True)
    participants = recovery[recovery.level.eq('participant')]
    summary = participants.groupby(['condition', 'method', 'parameter'], as_index=False).agg(datasets=('replicate', 'nunique'), mean_rmse=('rmse', 'mean'), min_rmse=('rmse', 'min'), max_rmse=('rmse', 'max'), mean_coverage=('interval_coverage', 'mean'))
    if not summary.datasets.eq(5).all():
        raise ValueError('Incomplete recovery grid')
    pop = recovery[recovery.level.eq('population') & recovery.parameter.eq('beta_theta')].copy()
    pop['excludes_zero'] = (pop.ci_low > 0) | (pop.ci_high < 0)
    popsum = pop.groupby('condition', as_index=False).agg(datasets=('replicate', 'nunique'), true_slope=('true', 'first'), mean_estimate=('mean', 'mean'), intervals_excluding_zero=('excludes_zero', 'sum'), truth_coverage=('interval_coverage', 'mean'))
    export(recovery, 'recovery_by_dataset')
    export(summary, 'recovery_participant_summary')
    export(popsum, 'recovery_age_summary')
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), layout='constrained')
    for ax, p in zip(axes, PARAMS):
        for method, offset, color, label in [('MLE_theta10', -.12, COLORS[2], 'Individual MLE'), ('hierarchical', .12, COLORS[1], 'Hierarchical')]:
            for i, condition in enumerate(['zero', 'positive', 'negative']):
                d = participants[participants.parameter.eq(p) & participants.method.eq(method) & participants.condition.eq(condition)].sort_values('replicate')
                ax.scatter(i+offset+np.linspace(-.04, .04, len(d)), d.rmse, color=color, alpha=.55, s=23, label=label if i == 0 else None)
                ax.plot([i+offset-.065, i+offset+.065], [d.rmse.mean()]*2, color=color, lw=3)
        ax.set(xticks=range(3), xticklabels=['Zero', 'Positive', 'Negative'], xlabel='Generating θ age slope', ylabel='Participant RMSE', title=PARAMS[p])
    axes[0].legend(frameon=False)
    save(fig, '06_recovery', 'Matched recovery · identical synthetic data · five datasets per condition')

    grad = src.read('diagnostic_review', 'reference_gradient_checks.csv')
    shifts = {r: src.read('diagnostic_review', f'{r}_attempt_comparison.csv').mean_shift_baseline_sd.abs().max() for r in ['H4_full_age', 'H5_train_age']}
    effectrow = age[age.quantity.eq('natural_change_25_to_75') & age.parameter.eq('friend_value_probability_effect')].iloc[0]
    theta_truth = individual[individual.parameter.eq('theta') & individual.method.eq('hierarchical')]['true']
    fail = src.inventory[~src.inventory.accepted]
    agefrac = age[age.quantity.eq('latent_age_variance_fraction')].copy()
    agefrac[['mean', 'median', 'ci_low', 'ci_high']] *= 100
    fig_info = [
        ('01_behavior_age', 'Behavior and age', 'Bands are pointwise GEE 95% confidence intervals, standardized over empirical offers. The cross-sectional age association is not an individual aging trajectory.'),
        ('02_hierarchical_age', 'Joint hierarchical age effects', 'Curves transform the population location at each age (zero participant random effect), not the average across all participant random effects. Bands are pointwise 95% posterior credible intervals. The canonical effect turns friend value on versus off at belief P=.5, standardized over empirical offers; it is not the observed friend-minus-computer contrast. Variance fractions are on each parameter’s latent scale, not behavioral R². Sensitivity intervals show the age-75 minus age-25 canonical effect.'),
        ('03_heldout', 'Temporal predictive performance', 'Each participant has equal weight. Intervals are percentile intervals from 5,000 participant bootstrap samples, with seed 20260925; contrasts resample paired scores. These quantify participant sampling variability conditional on fitted models, not posterior credible intervals or refitting uncertainty. All five no-age models use the same 4,017 held-out choices. H5 training-age is excluded. Prediction updates beliefs as feedback arrives while keeping the training parameter posterior fixed.'),
        ('04_predictive_checks', 'Posterior predictive model checks', 'Dots and intervals summarize simulated datasets from full-data age models, including posterior uncertainty; dashed lines show observed equal-participant means. A held-out advantage does not ensure adequate absolute fit. Intervals are pointwise descriptive checks across several model/partner combinations.'),
        ('05_theta_identifiability', 'Bounds and hierarchical regularization', 'The right panel compares different estimators and constraints. Shrinkage and a finite hierarchical estimate do not prove that the individual raw θ parameter is identified.'),
        ('06_recovery', 'Matched parameter recovery', 'Every dot is one complete synthetic dataset (111 participants); short horizontal lines are means across five datasets per condition. The two estimators see identical data. Five datasets per condition are too few for precise calibration or power estimates, and this is not simulation-based calibration.')]
    h5ppc = ppc[ppc.model.eq('H5') & ppc.label.eq('friend')].iloc[0]
    h5computer = ppc[ppc.model.eq('H5') & ppc.label.eq('computer')].iloc[0]
    accepted_n = len(src.accepted)
    report = f'''# Accepted-fit review — 25 September 2026

**{accepted_n} of {len(src.inventory)} Linux posterior runs meet the existing diagnostic gate. This is a scoped review, not completion of the full second pass.** H4 full-age inference and H5 age-model held-out scoring are excluded. Their older summary files may still exist elsewhere in the repository; this report never reads them. Primary analyses use N=111 from ds005123 v1.1.3; ratings-dependent H4 would use N=103 and remains secondary because rating timing is unresolved.

The audit found no custom-versus-reference gradient discrepancy at the four tested saved states (maximum scaled error {grad.max_scaled_error.max():.2e}). The largest population-mean shifts from the previous attempt were {shifts['H4_full_age']:.3f} posterior SD for H4 and {shifts['H5_train_age']:.3f} SD for H5. Recorded states are not the unavailable intermediate trajectory states where the integrator failed. These checks are reassuring about implementation at the tested points; they do not establish that divergences are harmless. The zero-divergence requirement is unchanged, and no further sampling was run to build this report. See the [diagnostic audit](../diagnostic_review/README.md) and [Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html).

{table(fail, ['run', 'status', 'retained_draws', 'max_rhat', 'min_bulk_ess', 'divergences'])}

## What the accepted results show

- **Age explains a small, uncertain fraction of H5 latent parameter variation.** Posterior mean fractions are listed below; all linear age-slope 95% credible intervals include zero. This does not establish absence of age effects. Behavioral friend-advantage changes are also imprecise.
- **Predictive flexibility matters.** The no-age partner preference and separate friend/stranger value models outperform H5 on paired held-out log loss. Their difference from one another is small. This supports useful partner-specific structure without uniquely establishing a social-reward mechanism.
- **Raw θ remains sensitive to constraints.** Even a ceiling of 20 captures many MLEs. Hierarchical regularization improves recovery in the tested simulation regime, which is substantially lower in θ than the empirical estimates.
- **Absolute fit still needs scrutiny.** For H5, observed friend high-choice frequency is {h5ppc.observed:.3f}, compared with posterior predictive mean {h5ppc['mean']:.3f} and 95% interval [{h5ppc.ci_low:.3f}, {h5ppc.ci_high:.3f}]. The computer-partner mismatch is larger: observed {h5computer.observed:.3f} versus predicted {h5computer['mean']:.3f} [{h5computer.ci_low:.3f}, {h5computer.ci_high:.3f}].

### How much variability is associated with age?

H5 latent-scale age variance fractions, in percent (posterior mean, median, and 95% credible interval):

{table(agefrac, ['parameter', 'mean', 'median', 'ci_low', 'ci_high'])}

The canonical friend-value probability effect changes by {effectrow['mean']:.3f} from age 25 to 75, with 95% credible interval [{effectrow.ci_low:.3f}, {effectrow.ci_high:.3f}]. Its posterior probability of a positive change is {effectrow.probability_positive:.3f}. Bounded θ, quadratic age, and wider age-prior sensitivity fits also have intervals spanning zero. These are joint hierarchical models: participant parameters are partially pooled through correlated population distributions, with age in their population means.

Behavioral changes in the friend advantage from age 25 to 75 (probability units; GEE 95% confidence intervals):

{table(change, ['contrast', 'estimate', 'ci_low', 'ci_high'])}

### What predicts new choices?

No-age posterior predictive log loss (lower is better; each participant weighted equally):

{table(absolute, ['model', 'mean', 'ci_low', 'ci_high'])}

Paired differences, model minus reference (negative favors the model):

{table(pd.concat([contrasts, pref7]), ['model', 'reference', 'mean', 'ci_low', 'ci_high'])}

Adding age, paired age-minus-no-age loss:

{table(gains, ['model', 'mean', 'ci_low', 'ci_high'], 5)}

H5 age prediction is unavailable because its training-age fit failed diagnostics. No claim about H5’s predictive age gain is supported here. Small changes for the other four models should be read with their paired intervals, not inferred from overlapping marginal score intervals.

### What recovery does and does not establish

The 15 accepted recovery runs were aggregated afresh from per-run files, avoiding the historical aggregate. Mean participant RMSE for θ:

{table(summary[summary.parameter.eq('theta')], ['condition', 'method', 'datasets', 'mean_rmse', 'mean_coverage'])}

Across these simulations, generating θ has median {theta_truth.median():.2f} and maximum {theta_truth.max():.2f}; the empirical median participant H5 posterior mean is {matched['mean'].median():.2f}. Better recovery here does not validate the empirical high-θ regime. MLE has no interval coverage column because it produces point estimates. Population θ age-slope recovery is based on only five datasets per condition:

{table(popsum, ['condition', 'datasets', 'true_slope', 'mean_estimate', 'intervals_excluding_zero', 'truth_coverage'])}

Participant θ interval coverage is only {summary.loc[summary.condition.eq('positive') & summary.method.eq('hierarchical') & summary.parameter.eq('theta'), 'mean_coverage'].iloc[0]:.3f} in the positive-age condition despite lower RMSE; improved point recovery does not ensure calibrated uncertainty. The original full model-recovery extension remains deferred. This review does not replace it or claim a unique mechanism.

## Figures

'''
    for name, title, caption in fig_info:
        report += f'### {title}\n\n![{title}](figures/{name}.png)\n\n{caption}\n\n[Vector PDF](figures/{name}.pdf)\n\n'
    report += '''## Reproduction and limits

Run `.venv/bin/python scripts/19_build_accepted_review.py` from the repository root. No CmdStan installation, raw posterior draws, or Linux access is needed. The builder checks both live Linux completion status and per-run numerical diagnostics before admitting a hierarchical input. It also checks paired participant identities and trial counts, and records SHA-256 hashes of every input in [provenance.json](provenance.json). Outputs use the committed per-run summaries; posterior extraction itself was performed on Linux and is not repeated here.

[Fit inventory](tables/fit_inventory.csv) · [Age results](tables/h5_age.csv) · [Paired predictions](tables/heldout_paired_contrasts.csv) · [Recovery by dataset](tables/recovery_by_dataset.csv).

Passing sampler diagnostics are necessary checks, not proof of model adequacy or identifiability. Cross-sectional age associations are exploratory. Credible intervals and bootstrap comparisons are pointwise and are not multiplicity-adjusted. Ratings timing remains unresolved. The full second-pass finalizer remains blocked on two fits; the first-pass report/gallery and older aggregate tables are historical rather than the source of this review.
'''
    (out / 'README.md').write_text(report)
    import html
    cards = ''.join(f'<section><h2>{html.escape(title)}</h2><a href="figures/{name}.pdf"><img src="figures/{name}.png" alt="{html.escape(title)}"></a><p>{html.escape(caption)}</p></section>' for name, title, caption in fig_info)
    (out / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Accepted-fit review</title><style>body{font:17px/1.6 system-ui;background:#fafbf9;color:#20303b;max-width:1400px;margin:40px auto;padding:0 24px}img{width:100%;height:auto}section{margin:55px 0}p{max-width:1000px}a{color:#007f86}.status{padding:16px;border-left:5px solid #bc6c25;background:#fff4e6}</style><h1>Trust, learning, and social value</h1><p class="status">Accepted-fit review · 32 of 34 fits pass. H4 inference and H5 age-model held-out scoring are excluded. Full finalization remains incomplete.</p><p><a href="README.md">Numerical report and limitations</a> · Click a figure for its vector PDF.</p>'+cards+'</html>')
    provenance = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      accepted_runs=sorted(src.accepted), excluded_runs=fail.run.tolist(),
                      bootstrap_seed=20260925, bootstrap_replicates=5000,
                      versions=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
                      inputs=src.hashes)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Built {out}: {accepted_n} accepted fits; exclusions: {", ".join(fail.run)}')


def main():
    build(Path(__file__).resolve().parents[2])
