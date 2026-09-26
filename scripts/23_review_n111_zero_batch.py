#!/usr/bin/env python3
"""Rebuild the read-only review of the first zero-option batch (e6eb3b3)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rf1_trust_socialvalue import n111_zero as z
from rf1_trust_socialvalue.n111_zero_retry import validate_plan
from rf1_trust_socialvalue.accepted_review import table


def main():
    z.gate_stage_a()
    _, states = validate_plan()
    rows = []; ppcs = []; parameters = []; comparisons = []
    hashes = {}
    def read(path):
        hashes[str(path)] = z.sha(path)
        return pd.read_csv(path)
    for entry in z.entries():
        name = entry['name']
        d = read(z.TABLE/f'zero_diagnostics_{name}.csv')
        rows.append(dict(run=name, status=states[name]['status'], max_rhat=d.R_hat.max(),
                         min_bulk_ess=d.ESS_bulk.min(), min_tail_ess=d.ESS_tail.min(),
                         divergences=int(d.divergences.max()), depth_hits=int(d.max_depth_hits.max()), min_bfmi=d.min_bfmi.min()))
        if states[name]['status'] != 'complete':
            continue
        if entry['training']:
            new = read(z.TABLE/f'zero_heldout_{name}.csv')
            old = read(Path(f'results/tables/heldout_{entry["model"]}_train_noage.csv'))
            for metric in ['log_loss', 'brier', 'accuracy']:
                comparisons.append(dict(model=entry['model'], metric=metric,
                    **z.paired_metric(new, old, metric, z.stable_seed(20260926, entry['model'], metric, 'paired'))))
        else:
            for label, target in [('ppc', ppcs), ('parameters', parameters)]:
                frame = read(z.TABLE/f'zero_{label}_{name}.csv')
                if not frame.run.eq(name).all():
                    raise ValueError(f'Output source mismatch: {name}')
                target.append(frame)
    diagnostics = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    expected = pd.read_csv(z.TABLE/'zero_option_model_comparison.csv').set_index(['model', 'metric'])
    actual = comparison.set_index(['model', 'metric'])
    cols = ['base_mean', 'zero_mean', 'mean_delta', 'ci_low', 'ci_high']
    np.testing.assert_allclose(actual[cols], expected.loc[actual.index, cols], atol=1e-12)
    ppc = pd.concat(ppcs, ignore_index=True)
    offers = ppc[ppc.stratification.eq('offer')].copy()
    offers['absolute_residual_high'] = offers.residual_high.abs()
    error = offers.groupby(['model', 'variant', 'prediction_type', 'zero_option']).agg(
        mean_absolute_offer_cell_residual=('absolute_residual_high', 'mean'), n_cells=('run', 'size')).reset_index()
    parameters = pd.concat(parameters, ignore_index=True)
    paired = parameters[parameters.variant.eq('zero')].merge(parameters[parameters.variant.eq('base')],
        on=['model', 'participant_id', 'parameter'], suffixes=('_zero', '_base'), validate='one_to_one')
    paired['change'] = paired.mean_zero-paired.mean_base
    shifts = paired.groupby(['model', 'parameter']).agg(n_subjects=('participant_id', 'size'),
        median_base_participant_mean=('mean_base', 'median'), median_zero_participant_mean=('mean_zero', 'median'),
        median_paired_change=('change', 'median')).reset_index()
    prefix = 'zero_initial_review_'
    for label, frame in [('diagnostics', diagnostics), ('offer_errors', error), ('parameter_shifts', shifts)]:
        frame.to_csv(z.TABLE/f'{prefix}{label}.csv', index=False)
    z.plot_comparison(ppc, comparison, partial=True)
    report = '''# Review of the first N=111 zero-option batch

Source: Linux results commit `e6eb3b3`, code `22249ce`. This review records the original batch even if a later explicitly reviewed retry succeeds. Rebuild with `python scripts/23_review_n111_zero_batch.py`; it reads committed tables and starts no sampling.

## Run outcome

All 12 fits finished sampling. Eleven passed the unchanged diagnostic gate. `N111_HPreference_zero_full` is excluded solely because 8/16,000 retained transitions reached maximum tree depth 12 (0.05%, all in chain 4). It has zero divergences, maximum Rhat 1.00459, minimum bulk ESS 666.857, minimum tail ESS 1551.26 and minimum BFMI .6422. The batch returned exit 2 as designed, rather than crashing. It ran from 12:36:50 to 15:54:46 UTC on 26 September (about 3 h 18 min).

The retained-draw diagnostics, rather than nonfatal LKJ proposal warnings, determine acceptance. No thresholds are relaxed. The two original exclusions H4_full_age and H5_train_age remain untouched.

## Temporal prediction

All four zero-option TRAINING fits passed, including HPreference. Each predicts the same 4,017 held-out choices in 111 participants as its accepted no-age baseline. Negative log-loss/Brier differences favor the extension; positive accuracy differences favor it. Intervals are paired participant bootstrap 95% intervals conditional on the fitted posteriors.

'''
    report += table(comparison, ['model', 'metric', 'mean_delta', 'ci_low', 'ci_high'], 5)+'\n\n'
    report += '''These are exploratory improvements. The same N=111 diagnostics, including held-out choices, motivated the extension; this is not independent confirmatory validation.

## Where prediction improves and where it does not

The following is the equal-cell mean absolute high-choice residual over nine partner × exact-offer cells in each offer class, using matched full-data no-age fits. It is descriptive, not a participant-weighted loss or a posterior contrast. HPreference has only a baseline here; no failed-fit inference is included. Full tables retain signed residuals, investment units and intervals.

'''
    report += table(error[error.prediction_type.eq('conditional_history')], ['model', 'variant', 'zero_option', 'mean_absolute_offer_cell_residual', 'n_cells'], 4)+'\n\n'
    report += '''H5 and H7 markedly improve zero-containing offers. The extension is not a complete repair: H5's positive-positive mean absolute offer-cell residual rises from .0599 to .0711; H7's falls overall from .0820 to .0703, but computer positive-positive underprediction grows from .0394 to .1089 when pooled across those offers. H8's overall offer errors decline, while its friend positive-positive underprediction increases and its computer zero-option predictions overshoot. Conditional and generative versions show similar changes.

## Existing parameters change

These are medians of participant posterior means, not population-location estimates. The median paired change is computed within participant and need not equal the difference between the two medians. They are descriptive fit-to-fit comparisons without a joint posterior contrast interval.

'''
    report += table(shifts, ['model', 'parameter', 'median_base_participant_mean', 'median_zero_participant_mean', 'median_paired_change'], 4)+'\n\n'
    report += '''H5's median participant theta estimate falls from 6.33 to 2.80 and H7's from 3.84 to 1.88, while kappa increases. This supports concern that the original value parameters partly compensated for the omitted zero-option feature; it does not identify a unique psychological mechanism. Median participant gamma means are 1.81 (H5), 1.80 (H7) and 1.52 (H8), on the additive logit scale. HPreference parameter stability is not yet available.

## Decision and bounded next step

The shared zero-option term is a promising candidate, with consistent temporal predictive gain and large H5/H7 zero-offer repair. Final retention remains pending: the no-damage criterion is mixed and the matched HPreference PPC/parameter comparison is missing. Do not claim that all partner models have been repaired.

One explicitly selected HPreference full-data retry is justified to resolve that missing comparison and provide an accepted empirical recovery generator. Only maximum tree depth changes, from 12 to 14; priors, model, seed, chains, warmup, draws and acceptance thresholds remain the same. It writes a separate cache, preserves the failed attempt, reuses the eleven accepted exported results, and cannot launch another model or automatic escalation. See [Linux retry instructions](../../docs/n111_zero_retry.md). If it fails, retain the limitation and review the recovery scope rather than automatically increasing compute again.

The realistic recovery screen has not run. It remains downstream of the feature decision; no new age analysis, ratings investigation, additional participants or imaging data are used.

![Partial zero-option comparison](figures/03_zero_option_comparison_partial.png)

The HPreference row deliberately omits the failed full-data extension while retaining its independently accepted training-fit prediction comparison.
'''
    (z.OUT/'zero_option_initial_review.md').write_text(report)
    (z.OUT/'zero_option_initial_review_provenance.json').write_text(json.dumps(dict(
        evidence_commit='e6eb3b3', input_sha256=hashes, generator_sha256=z.sha(Path(__file__)), sampling_started=False), indent=2)+'\n')
    print('First-batch evidence verified; review tables and partial figure generated without MCMC.')


if __name__ == '__main__':
    main()
