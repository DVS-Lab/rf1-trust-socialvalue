"""Validate committed N=111 results and build the final synthesis; no fitting."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import n111_recovery as recovery
from . import n111_zero as zero
from .accepted_review import Inputs, diagnostic_pass, table
from .n111_diagnostics import verify_scope, sha, style, save_figure, FOCAL

OUT = Path('results/n111_wrapup')
TABLE = OUT/'tables'
DATASET_KEYS = ['variant', 'regime', 'generating', 'replicate']
CASE_KEYS = DATASET_KEYS+['participant_id']


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path); self.hashes[str(path)] = sha(path)
        return json.loads(path.read_text()) if path.suffix == '.json' else pd.read_csv(path)


def assert_same(actual, saved, keys, values):
    if actual.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ValueError('Duplicate summary keys')
    a = actual.set_index(keys).sort_index(); b = saved.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError('Summary key coverage differs')
    np.testing.assert_allclose(a[values].to_numpy(float), b[values].to_numpy(float), atol=1e-10, rtol=1e-10)


def reconstruct_confusion(frame):
    rows = []
    criteria = ['AICc', 'BIC', 'heldout_log_loss', 'heldout_brier']
    for unit in ['participant', 'dataset']:
        keys = CASE_KEYS if unit == 'participant' else DATASET_KEYS
        data = frame if unit == 'participant' else frame.groupby(DATASET_KEYS+['fitted'], as_index=False).agg(
            AICc=('AICc', 'sum'), BIC=('BIC', 'sum'), heldout_log_loss=('heldout_log_loss', 'mean'), heldout_brier=('heldout_brier', 'mean'))
        for metric in criteria:
            minimum = data.groupby(keys)[metric].transform('min')
            wins = (data[metric]-minimum <= 1e-6).astype(float)
            total = wins.groupby([data[k] for k in keys]).transform('sum')
            selected = data.assign(selected_weight=wins/total)
            summary = selected.groupby(['variant', 'regime', 'generating', 'fitted']).agg(
                selection_probability=('selected_weight', 'mean'), n_units=('selected_weight', 'size'), n_replicates=('replicate', 'nunique')).reset_index()
            rows.append(summary.assign(unit=unit, metric=metric))
    return pd.concat(rows, ignore_index=True)


def validate_recovery(frame, confusion, cfg, ids, trials):
    if len(frame) != 31968 or frame.duplicated(CASE_KEYS+['fitted']).any():
        raise ValueError('Incomplete or duplicate fast recovery fits')
    if set(frame.participant_id) != set(ids) or set(frame.fitted) != set(FOCAL) or not frame.starts.eq(32).all():
        raise ValueError('Recovery participants, candidates or starts differ')
    expected = {(v, regime, m, rep) for v in ['zero', 'base'] for m in FOCAL
                for regime, count in [('empirical', 8), ('high_theta_stress', 2 if m in ['H5', 'H7'] else 0)] for rep in range(count)}
    observed = set(frame[DATASET_KEYS].itertuples(index=False, name=None))
    if observed != expected or not frame.groupby(DATASET_KEYS).size().eq(444).all() or not frame.groupby(CASE_KEYS).size().eq(4).all():
        raise ValueError('Recovery dataset/candidate coverage mismatch')
    for _, group in frame.groupby(CASE_KEYS):
        if set(group.fitted) != set(FOCAL):
            raise ValueError('Candidate absent from a simulation')
    num = ['n', 'k', 'nll', 'AICc', 'BIC', 'train_n', 'heldout_n', 'heldout_log_loss', 'heldout_brier', 'near_best', 'train_near_best']
    if not np.isfinite(frame[num].to_numpy(float)).all() or (frame[['nll', 'heldout_log_loss']] < -1e-8).any().any():
        raise ValueError('Nonfinite or impossible recovery score')
    if not frame.heldout_brier.between(0, 1).all() or not frame.near_best.between(0, 32).all() or not frame.train_near_best.between(0, 32).all():
        raise ValueError('Invalid optimizer or Brier output')
    from .models import pack
    from .hierarchical import split_index
    for sub, group in trials[trials.participant_id.isin(ids)].groupby('participant_id'):
        a = pack(group); n = int((a[:, 3] >= 0).sum()); nt = int((a[:split_index(a), 3] >= 0).sum())
        f = frame[frame.participant_id == sub]
        if not f.n.eq(n).all() or not f.train_n.eq(nt).all() or not f.heldout_n.eq(n-nt).all():
            raise ValueError('Simulation changed observed trial counts or temporal split')
    for (model, variant), group in frame.groupby(['fitted', 'variant']):
        k = len(recovery.parameter_names(model, variant == 'zero'))
        if not group.k.eq(k).all():
            raise ValueError('Wrong candidate complexity')
    np.testing.assert_allclose(frame.AICc, 2*frame.nll+2*frame.k+2*frame.k*(frame.k+1)/(frame.n-frame.k-1), atol=1e-9)
    np.testing.assert_allclose(frame.BIC, 2*frame.nll+frame.k*np.log(frame.n), atol=1e-9)
    nll = frame.pivot(index=CASE_KEYS, columns='fitted', values='nll')
    if (nll.H7 > nll.H5+1e-4).any():
        raise ValueError('Nested likelihood check fails')
    actual = reconstruct_confusion(frame)
    assert_same(actual, confusion, ['variant', 'regime', 'unit', 'metric', 'generating', 'fitted'], ['selection_probability', 'n_units', 'n_replicates'])
    return actual


def boundary_details(frame, limits):
    rows = []
    keys = ['variant', 'regime', 'generating', 'fitted']
    for key, group in frame.groupby(keys):
        variant, _, _, model = key
        names = recovery.parameter_names(model, variant == 'zero')
        bounds = recovery.bounds_for(model, variant == 'zero', limits)
        for stage in ['fit', 'train']:
            for name, (lo, hi) in zip(names, bounds):
                values = group[stage+'_'+name].to_numpy()
                if not np.isfinite(values).all() or (values < lo-1e-7).any() or (values > hi+1e-7).any():
                    raise ValueError('Nonfinite or out-of-range fitted parameter')
                rows.append(dict(zip(keys, key), stage=stage, parameter=name, lower_bound=lo, upper_bound=hi,
                    lower_hit_rate=np.mean(values-lo < 1e-4*(hi-lo)), upper_hit_rate=np.mean(hi-values < 1e-4*(hi-lo)), n_cases=len(values)))
    return pd.DataFrame(rows)


def loss_tail_summary(frame):
    rows = []
    keys = ['variant', 'regime', 'generating', 'fitted']
    for key, group in frame.groupby(keys):
        x = group.heldout_log_loss.to_numpy()
        rows.append(dict(zip(keys, key), n_cases=len(x), median=np.median(x), q95=np.quantile(x, .95), maximum=x.max(),
                         fraction_above_5=np.mean(x > 5), fraction_above_10=np.mean(x > 10)))
    return pd.DataFrame(rows)


def plot_synthesis(behavior, comparison, age, model_age, confusion):
    plt = style(); fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), layout='constrained')
    ax = axes[0, 0]
    for i, row in behavior.iterrows():
        ax.hlines(i, 100*row.ci_low, 100*row.ci_high, color='#007f86', lw=2)
        ax.scatter(100*row.difference, i, color='#007f86', s=40)
    ax.axvline(0, color='#999999', ls='--')
    ax.set(yticks=range(len(behavior)), yticklabels=behavior.contrast.str.replace(' - ', ' − '),
           xlabel='Difference in high-choice probability (percentage points)', title='A  Friends elicit higher investment choices')
    ax.invert_yaxis()
    ax.text(0, -.23, 'Offer-standardized paired participant bootstrap · 95% intervals', transform=ax.transAxes, fontsize=9)
    ax = axes[0, 1]; loss = comparison[comparison.metric == 'log_loss'].set_index('model').loc[FOCAL]
    for i, (_, row) in enumerate(loss.iterrows()):
        ax.hlines(i, row.ci_low, row.ci_high, color='#007f86', lw=2); ax.scatter(row.mean_delta, i, color='#007f86', s=40)
    ax.axvline(0, color='#999999', ls='--')
    ax.set(yticks=range(4), yticklabels=FOCAL, xlabel='Zero extension − base log loss (lower is better)', title='B  One zero-option term improves prediction')
    ax.invert_yaxis(); ax.text(0, -.23, '4,017 temporal test choices · paired bootstrap 95% intervals', transform=ax.transAxes, fontsize=9)
    ax = axes[1, 0]
    labels = ['Behavior: friend − computer', 'Behavior: friend − stranger', 'H5: canonical friend-value effect']
    means = list(age.estimate)+[model_age['mean']]; lows = list(age.ci_low)+[model_age.ci_low]; highs = list(age.ci_high)+[model_age.ci_high]
    for i, (mean, lo, hi) in enumerate(zip(means, lows, highs)):
        color = '#64748b' if i < 2 else '#bc6c25'
        ax.hlines(i, 100*lo, 100*hi, color=color, lw=2); ax.scatter(100*mean, i, color=color, s=40)
    ax.axvline(0, color='#999999', ls='--')
    ax.set(yticks=range(3), yticklabels=labels, xlabel='Age 25→75 probability change (percentage points)', title='C  Age moderation remains uncertain')
    ax.invert_yaxis(); ax.text(0, -.23, 'Gray: GEE 95% CI · Orange: posterior 95% CrI\nDifferent estimands; cross-sectional associations.', transform=ax.transAxes, fontsize=9)
    ax = axes[1, 1]
    selected = ['H5', 'H7', 'HPreference', 'H8']
    data = confusion[(confusion.variant == 'zero')&(confusion.regime == 'empirical')&(confusion.unit == 'participant')&(confusion.metric == 'AICc')]
    mat = data.pivot(index='generating', columns='fitted', values='selection_probability').loc[['H7', 'HPreference'], selected]
    ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(2):
        for j in range(4):
            value = mat.iloc[i, j]; ax.text(j, i, f'{value:.0%}', ha='center', va='center', color='white' if value > .55 else '#222222', fontsize=13)
    ax.set(xticks=range(4), xticklabels=selected, yticks=range(2), yticklabels=['H7 generated', 'HPreference generated'], xlabel='Selected model', title='D  Mechanism labels are often confused')
    ax.text(0, -.23, 'Individual AICc · zero-adjusted empirical generators\n888 tasks per row; only 8 population draws.', transform=ax.transAxes, fontsize=9)
    fig.suptitle('N=111 developmental closeout\nUseful choice structure; limited evidence for a unique mechanism', fontsize=17)
    save_figure(fig, OUT/'figures/05_n111_synthesis')


def build():
    e = Evidence(); cfg = recovery.gate(); ids = verify_scope(Path.cwd())
    inventory = Inputs(Path.cwd())
    if 'H4_full_age' in inventory.accepted or 'H5_train_age' in inventory.accepted or 'H5_full_age' not in inventory.accepted:
        raise ValueError('Original accepted/excluded hierarchy changed')
    e.hashes.update(inventory.hashes)
    state = e.read(OUT/'realistic_recovery_status.json'); provenance = e.read(OUT/'realistic_recovery_provenance.json')
    errors = e.read(OUT/'realistic_recovery_errors.json')
    if state['status'] != 'screen_complete_scientific_review_pending' or state['exit_code'] != 0 or state['completed_cases'] != 7992 or state['failed_cases'] != 0 or errors:
        raise ValueError('Recovery screen is incomplete')
    saved_fingerprint = provenance['fingerprint']; body = {k:v for k,v in provenance.items() if k != 'fingerprint'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != saved_fingerprint or state['fingerprint'] != saved_fingerprint:
        raise ValueError('Recovery provenance fingerprint mismatch')
    if provenance['config_sha256'] != sha(recovery.CONFIG):
        raise ValueError('Recovery configuration changed')
    for path, digest in provenance['implementation_sha256'].items():
        local = Path('src/rf1_trust_socialvalue')/Path(path).name
        if sha(local) != digest:
            raise ValueError('Recovery implementation changed after run')
        e.hashes[str(local)] = digest
    for path, digest in cfg['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Accepted generator evidence changed')
        e.hashes[path] = digest
    if set(s['run'] for s in provenance['sources']) != set(cfg['source_runs'].values()):
        raise ValueError('Generator source set differs')
    for source in provenance['sources']:
        manifest = e.read(OUT/'sampler_evidence'/source['run']/'manifest_summary.json')
        if source['fingerprint'] != manifest['fingerprint'] or len(source['chains']) != 4:
            raise ValueError('Generator source fingerprint/chain coverage differs')
        if sorted(c['chain'] for c in source['chains']) != [1,2,3,4] or any(c['bfmi'] <= .3 or len(c['retained_indices']) != 2 for c in source['chains']):
            raise ValueError('Invalid balanced posterior draw audit')
    frame = e.read(TABLE/'realistic_model_recovery.csv'); confusion = e.read(TABLE/'realistic_model_recovery_confusion.csv')
    trials = e.read('results/tables/trial_table.csv')
    validate_recovery(frame, confusion, cfg, ids, trials)
    details = boundary_details(frame, provenance['fitting_upper_bounds'])
    details.to_csv(TABLE/'realistic_recovery_boundary_details.csv', index=False)
    tails = loss_tail_summary(frame); tails.to_csv(TABLE/'realistic_recovery_loss_tails.csv', index=False)
    ranges = e.read(TABLE/'realistic_recovery_generating_ranges.csv')
    if set(ranges[['variant','regime','model','replicate']].itertuples(index=False, name=None)) != set(frame[DATASET_KEYS].itertuples(index=False, name=None)):
        raise ValueError('Generating ranges lack a dataset')
    theta = ranges[ranges.parameter.str.startswith('theta')]
    if theta.maximum.max() >= provenance['fitting_upper_bounds']['theta'] or provenance['fitting_upper_bounds']['theta'] < 20:
        raise ValueError('Generating upper tail not covered by fitting bounds')
    stress = theta[(theta.regime == 'high_theta_stress')&(theta.parameter == 'theta')]
    if len(stress) != 8 or not np.allclose(stress.fraction_5_to_10,55/111) or not np.allclose(stress.fraction_above_10,56/111):
        raise ValueError('High-theta stress coverage differs')
    opt = e.read(TABLE/'realistic_recovery_optimizer_audit.csv')
    for field in ['boundary', 'train_boundary']:
        frame[field+'_any'] = frame[field].fillna('').ne('')
    rebuilt_opt = frame.groupby(['variant','regime','generating','fitted']).agg(n_cases=('participant_id','size'),
        full_boundary_rate=('boundary_any','mean'), training_boundary_rate=('train_boundary_any','mean'),
        median_near_best=('near_best','median'), median_train_near_best=('train_near_best','median')).reset_index()
    assert_same(rebuilt_opt,opt,['variant','regime','generating','fitted'],['n_cases','full_boundary_rate','training_boundary_rate','median_near_best','median_train_near_best'])
    behavior = e.read('results/tables/behavior_paired_bootstrap.csv')
    if not behavior.n.eq(111).all():
        raise ValueError('Behavioral contrasts must use primary N=111')
    # Verify the fixed contrast point estimates from equal-offer participant means.
    v = trials[trials.participant_id.isin(ids)&trials.valid_choice]
    cell = v.groupby(['participant_id','partner','offer_pair']).chose_high.mean().groupby(['participant_id','partner']).mean().unstack()
    for _, row in behavior.iterrows():
        a,b = row.contrast.split(' - ')
        if not np.isclose((cell[a]-cell[b]).mean(),row.difference,atol=1e-12):
            raise ValueError('Behavior contrast point estimate differs')
    age = e.read('results/tables/age_partner_change_25_to_75.csv')
    age = age.set_index('contrast').loc[['friend - computer','friend - stranger']].reset_index()
    original_age = e.read('results/tables/age_H5_full_age.csv')
    model_age = original_age[(original_age.parameter == 'friend_value_probability_effect')&(original_age.quantity == 'natural_change_25_to_75')].iloc[0]
    comparison = e.read(TABLE/'zero_option_model_comparison.csv')
    zero_ppc = e.read(TABLE/'zero_option_final_offer_error_summary.csv')
    shifts = e.read(TABLE/'zero_option_final_parameter_shift_summary.csv')
    temporal = []
    for model in ['H2']+FOCAL:
        old = e.read(f'results/tables/heldout_{model}_train_noage.csv')
        diag = e.read(f'results/tables/diagnostics_{model}_train_noage.csv')
        if not diagnostic_pass(diag) or model+'_train_noage' not in inventory.accepted:
            raise ValueError('Unaccepted no-age prediction source')
        temporal.append(dict(model=model,variant='base',mean_log_loss=old.log_loss.mean()))
        if model in FOCAL:
            new = e.read(TABLE/f'zero_heldout_N111_{model}_zero_train.csv')
            for metric in ['log_loss','brier','accuracy']:
                check = zero.paired_metric(new,old,metric,zero.stable_seed(20260926,model,metric,'paired'))
                row = comparison[(comparison.model==model)&(comparison.metric==metric)].iloc[0]
                for key,value in check.items():
                    if not np.isclose(value,row[key],atol=1e-12,rtol=1e-12):
                        raise ValueError('Zero-option paired score mismatch')
            temporal.append(dict(model=model,variant='zero',mean_log_loss=new.log_loss.mean()))
    pd.DataFrame(temporal).to_csv(TABLE/'n111_final_temporal_summary.csv',index=False)
    plot_synthesis(behavior,comparison,age,model_age,confusion)
    central = confusion[(confusion.variant=='zero')&(confusion.regime=='empirical')&confusion.generating.isin(['H7','HPreference'])]
    central.to_csv(TABLE/'n111_central_recovery.csv',index=False)
    for n, stem in enumerate(['where_models_miss','conditional_vs_generative','zero_option_comparison','realistic_recovery','n111_synthesis'],1):
        for ext in ['png','pdf','svg']:
            path=OUT/'figures'/f'{n:02d}_{stem}.{ext}'
            if not path.exists():
                raise ValueError(f'Missing final figure: {path}')
    report(behavior,comparison,age,model_age,central,opt,tails,ranges,temporal)
    conclusions = [
        ('behavior','complete','Friend advantage is robust: offer-standardized high-choice difference +.293 versus computer and +.249 versus stranger.'),
        ('history_mismatch','complete','Main zero-offer discrepancy persists under actual histories; maximum held-out conditional/generative offer-cell difference .005252.'),
        ('zero_option','candidate_retained_with_limitations','All four temporal comparisons improve and partner-model zero-offer PPCs improve markedly, but some positive-positive cells worsen.'),
        ('parameter_stability','complete','Partner/value parameters fall and kappa rises after gamma0; original interpretation was choice-specification sensitive.'),
        ('mechanism_recovery','complete_with_limitations','Substantial criterion-dependent H7/HPreference confusion; no reliable unique-mechanism conclusion. MLE boundary behavior and only eight empirical population draws limit inference.'),
        ('hierarchical_confirmation','not_warranted_in_scoped_closeout','Fast screen does not show robust mechanism separation; stop under the user brief rather than launch further hierarchical simulations.'),
        ('age','frozen_uncertain','Age25→75 behavioral friend−computer .025 [-.150,.201], friend−stranger .002 [-.141,.146]; H5 canonical friend-value probability −.050 [-.185,.078].'),
        ('ratings','timing_unresolved','103 primary participants have usable ratings; timing remains unverified and ratings models secondary.'),
        ('excluded_original_fits','unchanged','H4_full_age and H5_train_age remain excluded; original HPreference zero full depth-12 attempt also remains excluded, replaced only by accepted depth-14 source.'),
    ]
    pd.DataFrame(conclusions,columns=['question','status','conclusion']).assign(closeout_complete=True).to_csv(TABLE/'n111_conclusions.csv',index=False)
    decision = dict(status='N111_closeout_complete', recovery_evidence_commit='46ad9c2', feature='retain_candidate_with_residual_misfit',
        hierarchical_confirmation='not_warranted_in_scoped_closeout', rationale='Strong, asymmetric and criterion-dependent confusion; frequent MLE boundaries and extreme temporal losses limit mechanistic interpretation. Stop after the fast screen under the scoped brief.',
        further_sampling_requested=False, additional_participants_accessed=False, excluded_original_runs=['H4_full_age','H5_train_age'], n_primary=111,
        recovery_cases=7992, recovery_datasets=72, failed_cases=0,
        builder_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (OUT/'closeout_status.json').write_text(json.dumps(decision,indent=2)+'\n')
    e.hashes[str(recovery.CONFIG)] = sha(recovery.CONFIG)
    (OUT/'closeout_provenance.json').write_text(json.dumps(dict(input_sha256=e.hashes,builder_sha256=sha(Path(__file__)),
        checks=['accepted_sources','fixed_sample_and_splits','recovery_coverage','AICc_BIC_formulae','nested_likelihood','all_confusion_rates_reconstructed','optimizer_summary','wide_parameter_coverage','paired_temporal_scores','required_figures'],
        posterior_sampling_started=False,raw_chains_rechecked_locally=False),indent=2)+'\n')
    print('N=111 closeout complete. All 7992 cases validated; no further hierarchical sampling requested.',flush=True)


def report(behavior,comparison,age,model_age,central,opt,tails,ranges,temporal):
    text = '''# N=111 developmental-analysis closeout

**N=111 closeout complete.** The fixed sample diagnostics, one generic choice extension and realistic structural recovery are complete. The recovery screen shows insufficiently reliable separation of the leading mechanisms to justify more hierarchical simulations in this closeout. No additional participant data were accessed, and no further sampling is requested.

The fixed primary sample contains 8,442 presented decisions and 8,251 valid choices: 4,234 training and 4,017 held out.

This endpoint supersedes the earlier stage reports as the current synthesis. The original accepted-fit report, caches, thresholds and excluded runs are preserved. This is a developmental analysis of ds005123 v1.1.3, not a preregistered full-sample analysis or proof of structural nonidentifiability.

![N=111 synthesis](figures/05_n111_synthesis.png)

## 1. What behavioral pattern is robust?

Participants choose higher investments substantially more often with friends. The table uses the primary N=111, averaging offer-specific choice rates equally within each participant before computing paired contrasts. Intervals resample participants; the small stranger–computer contrast should not be given the same evidential emphasis as the friend advantage. Original broad descriptive tables use N=113 and are not substituted for this primary sample.

'''
    text += table(behavior,['contrast','n','difference','ci_low','ci_high'],3)+'\n\n'
    text += '''## 2. What does hierarchical modeling establish?

The empirical models are jointly fitted, with participant parameters partially pooled through correlated population distributions. Partner-specific models predict held-out choices better than monetary RL: the original no-age H2 log loss is .679, versus approximately .574 for H7/HPreference. H7 and HPreference were nearly tied; predictive gain does not determine whether the effect is reciprocation-specific value or a broader investment preference. New zero-adjusted models also retain hierarchical partial pooling, including one generic participant gamma0 common across partners.

Mean temporal log loss, with each participant weighted equally, is shown below. These are posterior predictive scores for 4,017 held-out choices; training parameters remain fixed while beliefs update from observed prior feedback. Model-selection uncertainty should be assessed with paired comparisons, not these means alone.

'''
    text += table(pd.DataFrame(temporal),['model','variant','mean_log_loss'],4)+'\n\n'
    text += '''## 3. What caused the main predictive mismatch?

The strongest discrepancy was structured by zero-containing versus positive-positive offers. Under the accepted no-age training posteriors, held-out zero-minus-positive residual contrasts remain about .18 for friends, .29–.30 for strangers and .20 for computers when using actual feedback histories. The largest conditional-versus-generative difference across held-out partner × offer means is .005252. Thus simulated feedback-history compounding does not explain the main aggregate error. These contrasts do not match investment amounts and do not identify a psychological mechanism.

Add exactly `gamma0 * I(low_option == 0)` to the existing choice logit, shared across partners and not multiplied by kappa. Retain it as a **candidate zero-option avoidance / positive-investment feature**, not a stronger mechanistic label. All four temporal comparisons improve:

'''
    text += table(comparison[comparison.metric.isin(['log_loss','brier'])],['model','metric','mean_delta','ci_low','ci_high'],4)+'\n\n'
    text += '''Negative values favor the extension. Intervals are paired participant-bootstrap 95% intervals conditional on fitted posteriors. Because N=111 diagnostics including test choices motivated the extension, this evaluation remains exploratory rather than untouched confirmatory validation.

Matched full-data no-age mean absolute zero-offer cell residuals drop .211→.030 (H5), .177→.015 (H7), and .173→.018 (HPreference). **The no-damage condition is not uniformly met:** some positive-positive cells worsen, particularly computer trials; H5's mean absolute positive-positive cell error rises .060→.071. Candidate retention therefore carries a residual-misfit qualification. No second extension is added.

The choice feature changes parameter interpretation: median participant posterior-mean theta falls 6.33→2.80 in H5 and 3.84→1.88 in H7; HPreference's friend preference falls 2.29→.91, while kappa increases. These are descriptive changes between fits, not jointly estimated posterior contrasts. They support concern that original partner/value parameters partly compensated for omitted choice structure.

## 4. What remains ambiguous mechanistically?

The finite screen completed 72 full N=111 simulated datasets, all 7,992 participant cases and 63,936 full/training candidate fits, with 32 starts each and no unresolved optimizer errors. It used accepted posterior population hyperparameter draws, new correlated participant effects, and the actual task schedules/missingness. Zero-adjusted models were primary; originals were secondary. Each empirical generator used eight balanced population draws, not 888 independent population draws.

Empirical generating friend theta reached 12.66 in zero-adjusted H5 and 22.61 in base H5. Separate stress datasets deliberately assigned friend theta in 5–10 and 10–20. The shared theta fitting bound expanded to 24.87, above every realized generator; kappa extended to 100 and signed preference to ±24.87. Generators were not clipped. Stress interventions are not posterior draws and are never pooled with empirical simulations.

Primary zero-adjusted, empirical-generating selection rates:

'''
    for unit in ['participant','dataset']:
        text += f'### {unit.capitalize()}-level AICc\n\n'
        matrix = central[(central.unit==unit)&(central.metric=='AICc')].pivot(index='generating',columns='fitted',values='selection_probability').reindex(index=['H7','HPreference'],columns=['H5','H7','HPreference','H8']).reset_index()
        text += table(matrix,['generating','H5','H7','HPreference','H8'],4)+'\n\n'
    text += '''Participant rates pool 888 simulated tasks per generating mechanism across eight population draws. Whole-dataset rates are based on only eight N=111 datasets and sum individual AICc values; they are not hierarchical marginal likelihoods.

H7 was selected in only 2/8 H7-generated datasets by AICc; HPreference was selected in the other 6/8. HPreference was selected in all 8/8 HPreference-generated datasets. This asymmetry is not reliable separation of both mechanisms. BIC selected H7 in 2/8 H7 datasets, HPreference in 3/8 and H5 in 3/8. Individual AICc also confused H7 with H8 in 12.3% of cases, and HPreference with H8 in 9.4%; confusion with simpler H5 was larger. Separate high-theta H7 stress still selected H7 in only one of two datasets and HPreference in the other, in each family. Two stress replicates cannot establish a precise rate.

Temporal MLE rankings are particularly unstable. Whole-dataset log loss selected H5 in 5/8 and H8 in 3/8 H7-generated datasets; for HPreference generators it selected H5 in 6/8 and H8 in 2/8. This does **not** establish superior mechanistic truth of H5/H8: independent short-task training fits frequently hit boundaries and make extremely confident wrong predictions. In the primary empirical screen the largest participant mean test loss is about 727.9. The correctly specified H7 fit hits some parameter boundary in 69.5% of full and 78.8% of training cases; for correctly specified HPreference these rates are 51.1% and 65.5%. Many starts reach comparable optima, so numerical convergence does not resolve weak estimation. AICc/BIC also deserve caution where regularity assumptions fail at boundaries.

**Decision: do not launch hierarchical confirmation in this scoped closeout.** The fast screen fails to show robust, criterion-consistent H7/HPreference separation, and the user brief explicitly stops expensive confirmation when confusion is strong. This is a limitation of the present recovery evidence and estimation scheme, not proof that every possible hierarchical analysis or task design must fail. Strong mechanistic labels are not justified by the current data/model comparison. Preserve reciprocation value, generic partner preference and asymmetric learning as competing accounts; do not infer a unique social-reward mechanism.

## 5. What can N=111 say about age?

Existing age results remain unchanged; no additional age parameterizations, groups, interactions or power simulations were run. Cross-sectional age-25-to-75 changes in high-choice probability:

'''
    text += table(age,['contrast','estimate','ci_low','ci_high'],3)+'\n\n'
    text += f"The accepted H5 canonical friend-value probability effect changes by {model_age['mean']:.3f}, with posterior 95% credible interval [{model_age.ci_low:.3f}, {model_age.ci_high:.3f}]. This is a model-implied effect at standardized conditions, not the observed friend–computer contrast. These intervals admit meaningful effects in either direction; they do not establish absence of age moderation. H5 latent age-variance fractions are not percentages of total behavioral variability.\n\n"
    text += '''## Ratings and excluded runs

Ratings are available for 103 primary participants, but pre/post timing remains unresolved from the available files. Ratings-dependent models stay secondary. H4_full_age and H5_train_age remain excluded; no further attempts were made to obtain 34/34 accepted original fits. The original zero-option HPreference depth-12 attempt remains excluded, with only its explicitly reviewed, accepted depth-14 replacement used in the final comparison. Passing sampler checks do not establish model adequacy.

## Final figures and reproducibility

1. [Where the models miss](figures/01_where_models_miss.png): partner × exact offer.
2. [Conditional versus generative](figures/02_conditional_vs_generative.png): actual versus simulated history residuals.
3. [Zero-option comparison](figures/03_zero_option_comparison.png): accepted matched PPCs and paired temporal gain.
4. [Targeted recovery](figures/04_realistic_recovery.png): primary empirical confusion, individual and dataset levels.
5. [N=111 synthesis](figures/05_n111_synthesis.png): behavior, prediction, age uncertainty and mechanism confusion.

Each has PNG, vector PDF and SVG versions in [figures](figures/). Required machine-readable results are in [tables](tables/), including predictive_residuals.csv, conditional_vs_generative.csv, zero_option_model_comparison.csv, zero_option_parameter_summary.csv, realistic_model_recovery.csv, realistic_model_recovery_confusion.csv and n111_conclusions.csv. Additional audits expose parameter-boundary rates and extreme test-loss tails.

Run `python scripts/26_finalize_n111.py` from the repo root to rebuild this synthesis without posterior sampling or Linux raw chains. The builder independently reconstructs every confusion rate, checks simulation coverage and temporal splits, verifies AICc/BIC formulae and nested likelihoods, recomputes paired zero-option comparisons, and checks accepted generator fingerprints and range coverage. Local validation uses committed per-case outputs and Linux chain-audit hashes; it does not re-read unavailable Linux raw draws. [Provenance](closeout_provenance.json) records inputs and checks, and [closeout status](closeout_status.json) records the stopping decision. The earlier accepted-fit report and phase-specific reports remain historical evidence.

[Decision and handoff record](../../docs/n111_wrapup_decisions.md) · [Recovery protocol](../../docs/n111_realistic_recovery.md) · [Detailed feature decision](zero_option_decision.md).

## What should carry forward to the full dataset

Carry forward the robust friend-related behavior, fixed task/feedback conventions, correlated partial pooling, separate actual-history and generative checks, and the common zero-option term as a candidate needing independent evaluation. Treat raw value/preference estimates as sensitive to choice specification. Preserve competing mechanism explanations and the demonstrated recovery limitations, including boundary-sensitive short-task prediction. Keep age uncertainty explicit and resolve ratings timing before stronger ratings-based inference. This is a descriptive handoff, not a new analysis plan: no additional participants have been accessed or authorized for this phase.
'''
    (OUT/'README.md').write_text(text)
    review = '# Recovery interpretation and stopping decision\n\nThe complete N=111 [synthesis](README.md) is authoritative. The screen completed all 7,992 cases without unresolved optimizer errors. H7/HPreference separation is asymmetric and criterion-dependent, with frequent MLE boundaries and extreme held-out losses. The finite closeout stops without hierarchical confirmation; no mathematical nonidentifiability claim is made.\n\n'
    review += table(central[central.metric.isin(['AICc','BIC'])],['unit','metric','generating','fitted','selection_probability','n_units'],4)+'\n\n'
    review += 'Read [boundary details](tables/realistic_recovery_boundary_details.csv) and [loss tails](tables/realistic_recovery_loss_tails.csv) alongside [all confusion rates](tables/realistic_model_recovery_confusion.csv). Eight empirical population draws and two social-value stress replicates per family are sufficient for this stopping decision but not precise recovery probabilities.\n'
    (OUT/'realistic_recovery_review.md').write_text(review)


if __name__ == '__main__':
    build()
