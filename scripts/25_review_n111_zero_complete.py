#!/usr/bin/env python3
"""Rebuild the accepted stage-B decision from committed tables; no sampling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rf1_trust_socialvalue import n111_recovery as r
from rf1_trust_socialvalue import n111_zero as z
from rf1_trust_socialvalue.accepted_review import table


def main():
    cfg = r.gate(); p = z.TABLE
    comparison = pd.read_csv(p/'zero_option_model_comparison.csv')
    if not comparison.status.eq('complete').all():
        raise ValueError('All four temporal comparisons must be accepted')
    for model in z.FOCAL:
        a = pd.read_csv(p/f'zero_heldout_N111_{model}_zero_train.csv')
        b = pd.read_csv(f'results/tables/heldout_{model}_train_noage.csv')
        for metric in ['log_loss','brier','accuracy']:
            result = z.paired_metric(a,b,metric,z.stable_seed(20260926,model,metric,'paired'))
            saved = comparison[(comparison.model==model)&(comparison.metric==metric)].iloc[0]
            for key,value in result.items():
                if not np.isclose(value,saved[key],atol=1e-12,rtol=1e-12):
                    raise ValueError('Temporal comparison does not regenerate')
    ppc = pd.read_csv(p/'zero_option_ppc_comparison.csv')
    for variant in ['zero','base']:
        for model in z.FOCAL:
            expected = cfg['source_runs'][variant+':'+model]
            if not ppc[ppc.model==model][f'run_{variant}'].eq(expected).all():
                raise ValueError('PPC comparison uses an excluded source')
    offers = ppc[ppc.stratification=='offer'].copy()
    offers['base_absolute_error'] = offers.residual_high_base.abs()
    offers['zero_absolute_error'] = offers.residual_high_zero.abs()
    errors = offers.groupby(['model','prediction_type','zero_option']).agg(
        n_cells=('offer_pair','size'),base_mean_absolute_cell_error=('base_absolute_error','mean'),
        zero_mean_absolute_cell_error=('zero_absolute_error','mean')).reset_index()
    errors.to_csv(p/'zero_option_final_offer_error_summary.csv',index=False)
    shifts = pd.read_csv(p/'zero_option_parameter_changes.csv').groupby(['model','parameter']).agg(
        n_subjects=('participant_id','size'),median_base_participant_mean=('mean_base','median'),
        median_zero_participant_mean=('mean_zero','median'),median_paired_change=('change_posterior_mean','median')).reset_index()
    shifts.to_csv(p/'zero_option_final_parameter_shift_summary.csv',index=False)
    text = '''# Zero-option decision after the accepted HPreference retry

**Retain the single shared zero-option term as a candidate feature for the larger sample, with unresolved residual misfit.** This is not a declaration that the model is fully adequate or that a particular psychological mechanism has been identified. No further choice extension is added in this closeout.

Source results: `cf9b33d`. All twelve required comparison sources pass the unchanged diagnostic gate. The isolated HPreference full-data depth-14 retry has Rhat max 1.00446, bulk ESS min 727.878, tail ESS min 1888.57, zero divergences, zero depth hits and BFMI min .6447. It finished in about 2 h 34 min. Its original depth-12 failed attempt remains preserved and excluded, as do H4_full_age and H5_train_age.

## Temporal prediction

The four training extensions predict the same 4,017 held-out choices in 111 participants as their original no-age baselines. Negative differences favor the extension; intervals are paired participant-bootstrap 95% intervals conditional on fitted posteriors.

'''
    text += table(comparison[comparison.metric.isin(['log_loss','brier'])], ['model','metric','mean_delta','ci_low','ci_high'],5)+'\n\n'
    text += '''All four improve both measures. The feature was motivated by diagnostics on these same N=111 data, including held-out trials, so this remains exploratory evidence rather than untouched confirmatory validation.

## Absolute fit and the no-damage qualification

Equal-cell mean absolute high-choice residual across the nine partner × exact-offer cells in each class, from matched full-data no-age fits, is shown below. These are descriptive errors, not posterior contrasts or participant-weighted losses.

'''
    text += table(errors[errors.prediction_type=='conditional_history'], ['model','zero_option','base_mean_absolute_cell_error','zero_mean_absolute_cell_error'],4)+'\n\n'
    text += '''For zero-containing offers, error drops from .2107 to .0298 in H5, .1770 to .0152 in H7 and .1730 to .0180 in HPreference. H8 improves less and retains substantial partner misfit. Conditional and generative checks give similar conclusions.

The strict “without worsening positive-positive trials” condition is **not uniformly met**. H5's positive-positive cell error rises .0599→.0711. Although H7 and HPreference improve on average across positive-positive cells, computer positive-positive underprediction grows: .0394→.1089 for H7 and .0189→.0950 for HPreference. Thus candidate retention rests on consistent temporal gain and large zero-offer repair, with this limitation explicitly carried forward. It must not be presented as complete PPC repair. No partner-specific zero feature or other extension is introduced.

## Parameter interpretation changes

The entries below summarize participant posterior means; they are not population-location parameters. Median paired changes need not equal the differences of medians, and no joint posterior uncertainty for a fit-to-fit difference is implied.

'''
    text += table(shifts,['model','parameter','median_base_participant_mean','median_zero_participant_mean','median_paired_change'],4)+'\n\n'
    text += '''H5 theta falls 6.33→2.80; H7 friend theta 3.84→1.88; HPreference friend preference 2.29→.91. Kappa increases in all three. Original partner/value parameters partly compensated for omitted choice structure. This strengthens the need for realistic recovery before assigning mechanism labels; it is not proof that social valuation is absent.

![Accepted zero-option comparison](figures/03_zero_option_comparison.png)

## Next finite step

The [recovery protocol](../../docs/n111_realistic_recovery.md) uses the zero-adjusted family primarily and the original family secondarily. It samples accepted posterior population hyperparameters, draws new correlated participant effects, retains the actual N=111 task schedules and missingness, and separately labels high-theta stress. It runs a finite nonhierarchical structural screen with wide fitting bounds and 32 starts, exporting participant- and dataset-level confusion plus temporal differences. It launches no hierarchical sampling automatically.

Realistic recovery and final synthesis remain pending. The accepted behavioral/model age estimates and unresolved ratings timing remain unchanged. No additional participants or imaging are accessed.
'''
    (z.OUT/'zero_option_decision.md').write_text(text)
    report = z.OUT/'zero_option_report.md'
    original = report.read_text()
    for marker in ['## Pending scientific decision', '## Scientific decision']:
        original = original.split(marker)[0]
    report.write_text(original+'## Scientific decision\n\nRetain the shared term as a candidate with unresolved residual misfit; see the [accepted feature decision](zero_option_decision.md) for the PPC tradeoffs, parameter shifts and finite recovery plan. The initial no-damage criterion is not uniformly met. Realistic recovery and final synthesis remain pending.\n')
    provenance=dict(evidence_commit='cf9b33d',inputs=cfg['evidence_sha256'],generator_sha256=z.sha(Path(__file__)),sampling_started=False)
    (z.OUT/'zero_option_decision_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    conclusions=[
        ('history_mismatch','answered','Main aggregate zero-offer error persists under actual histories; maximum held-out conditional/generative offer-cell difference .005252.'),
        ('zero_option','candidate_retained_with_limitations','All four temporal comparisons improve; H5/H7/HPreference zero-offer PPCs improve markedly; positive-positive cell damage is mixed, so models are not fully adequate.'),
        ('parameter_stability','answered','Partner values/preferences fall and kappa increases after gamma0; original parameter interpretation was sensitive to choice specification.'),
        ('mechanism_recovery','pending','Finite realistic structural screen prepared; no selection rates or identifiability conclusion available yet.'),
        ('age','frozen_uncertain','Age25→75 behavioral friend−computer .025 [-.150,.201], friend−stranger .002 [-.141,.146]; H5 friend-value probability −.050 [-.185,.078].'),
        ('ratings','timing_unresolved','103 primary participants have ratings; timing remains unverified; ratings models remain secondary.'),
        ('excluded_original_fits','unchanged','H4_full_age and H5_train_age remain excluded.'),
    ]
    pd.DataFrame(conclusions,columns=['question','status','conclusion']).assign(closeout_complete=False).to_csv(p/'n111_conclusions.csv',index=False)
    print('Accepted stage-B decision and paired comparison checks regenerated; recovery remains pending.')


if __name__=='__main__':
    main()
