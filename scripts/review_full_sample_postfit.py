"""Summarize and plot the published post-fit results without accessing raw data."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/full_sample/hierarchical'
DEST = BASE / 'postfit_review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = {}
    def record(path):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    def check(path, expected):
        if sha(path) != expected:
            raise ValueError(f'Published evidence changed: {path}')
        record(path)
    ppc = BASE / 'ppc'
    state = json.loads((ppc / 'status.json').read_text())
    if state['status'] != 'complete' or len(state['runs']) != 9 or any(state['runs'].values()):
        raise ValueError('Expected nine completed predictive checks')
    record(ppc / 'status.json')
    for file, h in state['output_sha256'].items():
        check(ppc / file, h)
    for name in state['runs']:
        folder = ppc / 'fits' / name
        f = json.loads((folder / 'status.json').read_text())
        if f['status'] != 'complete' or f['participants'] != 343 or f['simulations'] != 200:
            raise ValueError(f'Unexpected PPC scope: {name}')
        record(folder / 'status.json')
        check(BASE / 'fits' / name / 'status.json', f['accepted_status_sha256'])
        check(BASE / 'fits' / name / 'manifest_summary.json', f['manifest_sha256'])
        for file, h in f['output_sha256'].items():
            check(folder / file, h)
    geo = BASE / 'geometry_review'
    gs = json.loads((geo / 'status.json').read_text())
    if gs['status'] != 'complete':
        raise ValueError('Geometry export incomplete')
    record(geo / 'status.json')
    for when in ['before', 'after']:
        check(geo / f'live_source_audit_{when}.json', gs[f'live_source_audit_{when}_sha256'])
    record(ppc / 'live_source_audit.json')
    geometry = []
    for name, entry in gs['fits'].items():
        check(BASE / 'fits' / name / 'status.json', entry['status_sha256'])
        check(BASE / 'fits' / name / 'manifest_summary.json', entry['manifest_sha256'])
        for file, h in entry['output_sha256'].items():
            check(geo / name / file, h)
        chains = pd.read_csv(geo / name / 'chain_diagnostics.tsv', sep='\t')
        neighborhood = pd.read_csv(geo / name / 'divergent_iteration_neighborhood.tsv', sep='\t')
        d = neighborhood[neighborhood.divergent__.eq(1)]
        if len(d) != 1 or chains.divergences.sum() != 1:
            raise ValueError('Expected one unresolved divergence per fit')
        geometry.append(dict(name=name, chain=int(d.chain.iloc[0]), draw=int(d.draw.iloc[0]),
                             retained_draws=int(chains.retained_draws.sum()), divergences=1,
                             max_depth_hits=int(chains.max_depth_hits.sum()), min_bfmi=chains.bfmi.min()))
    a = pd.read_csv(ppc / 'predictive_checks_all_models.tsv', sep='\t')
    focal = a[a.model.isin(['H7', 'HPreference']) & a.variant.eq('zero') & a.metric.eq('high_choice')].copy()
    DEST.mkdir(parents=True, exist_ok=True)
    focal.to_csv(DEST / 'focal_predictive_checks.tsv', sep='\t', index=False)
    pd.DataFrame(geometry).to_csv(DEST / 'remaining_diagnostics.tsv', sep='\t', index=False)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout='constrained')
    cells = ['friend_zero', 'friend_positive', 'stranger_zero', 'stranger_positive', 'computer_zero', 'computer_positive']
    partner = ['all', 'friend', 'stranger', 'computer']
    for model, color, offset in [('H7', '#176b87', -.12), ('HPreference', '#c36e24', .12)]:
        d = focal[focal.model.eq(model) & focal.history.eq('generative')]
        f = d[d.stratification.eq('partner_zero') & d.statistic.eq('mean')].set_index('group').loc[cells]
        y = np.arange(len(f)) + offset
        axes[0].hlines(y, 100*(f.ci_low-f.observed), 100*(f.ci_high-f.observed), color=color, lw=2)
        axes[0].scatter(100*(f.predicted_mean-f.observed), y, color=color, label=model+' + zero', zorder=3)
        for ax, statistic, factor in [(axes[1], 'between_participant_sd', 1.), (axes[2], 'all_high_fraction', 100.)]:
            f = d[d.stratification.isin(['all','partner']) & d.statistic.eq(statistic)].set_index('group').loc[partner]
            scale = factor/f.observed if statistic == 'between_participant_sd' else factor
            y = np.arange(len(f)) + offset
            ax.hlines(y, f.ci_low*scale, f.ci_high*scale, color=color, lw=2)
            ax.scatter(f.predicted_mean*scale, y, color=color, zorder=3)
            if statistic == 'all_high_fraction' and model == 'H7':
                ax.scatter(f.observed*100, np.arange(len(f)), marker='x', color='#222222', s=65, label='Observed', zorder=4)
    axes[0].axvline(0, color='.5', ls='--', lw=1)
    axes[0].set(yticks=range(6), yticklabels=['Friend / zero','Friend / positive','Stranger / zero','Stranger / positive','Computer / zero','Computer / positive'],
                xlabel='Predicted minus observed (percentage points)', title='Mean choice residuals')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[1].axvline(1, color='.5', ls='--', lw=1)
    axes[1].set(yticks=range(4), yticklabels=['All choices','Friend','Stranger','Computer'],
                xlabel='Predicted / observed participant SD', title='Variation between participants', xlim=(.5,1.1))
    axes[2].set(yticks=range(4), yticklabels=['All choices','Friend','Stranger','Computer'],
                xlabel='Participants always choosing high (%)', title='Consistent high investors')
    axes[2].legend(loc='lower right', fontsize=9)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Two leading models share residual misfit\nN = 343 · 200 generative replicates · 95% predictive intervals', fontsize=14)
    for ext in ['png','pdf']:
        fig.savefig(DEST / f'focal_model_adequacy.{ext}', dpi=180, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)
    output_files = ['focal_predictive_checks.tsv','remaining_diagnostics.tsv','focal_model_adequacy.png','focal_model_adequacy.pdf']
    record(Path(__file__).resolve())
    (DEST / 'status.json').write_text(json.dumps(dict(status='complete', new_sampling=False,
        evidence_sha256=evidence, output_sha256={f:sha(DEST/f) for f in output_files},
        interpretation='Descriptive full-data model adequacy; not a new heldout ranking or mechanism-identification test.'), indent=2)+'\n')
    print(f'Wrote post-fit review; verified/recorded {len(evidence)} evidence files.')


if __name__ == '__main__':
    main()
