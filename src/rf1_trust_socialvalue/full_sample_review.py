"""Review published diagnostics and compare only accepted run-1 posteriors."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import full_sample_sampling as s
from . import full_sample_batch as batch


def resolved_entries(root):
    original=json.loads((root/'config/full_sample_batch.json').read_text())
    retry=json.loads((root/'config/full_sample_batch_retry.json').read_text())
    replacements={e['source_name']:e for e in retry['retries']}
    return [replacements.get(e['name'],dict(e,subset=e.get('subset','full'))) for e in original['reuse']+batch.new_entries(original)]


def accepted_scores(root):
    out=root/'results/full_sample/hierarchical';rows=[];frames=[];evidence={}
    thresholds=json.loads((root/'config/full_sample_linux2.json').read_text())['hierarchical']['acceptance']
    for entry in resolved_entries(root):
        folder=out/'fits'/entry['name'];status=json.loads((folder/'status.json').read_text())
        manifest=json.loads((folder/'manifest_summary.json').read_text())
        if manifest['fingerprint']!=status['fingerprint']:raise ValueError('Manifest target mismatch')
        files={'diagnostics.tsv':status['diagnostics_sha256'],**status.get('summary_hashes',{}),**status.get('trace_hashes',{})}
        if 'heldout_sha256' in status:files['heldout_participants.tsv']=status['heldout_sha256']
        for file,h in files.items():
            if s.fs.sha(folder/file)!=h:raise ValueError('Published output changed: '+str(folder/file))
        for file in list(files)+['status.json','manifest_summary.json']:
            evidence[str((folder/file).relative_to(root))]=s.fs.sha(folder/file)
        d=pd.read_csv(folder/'diagnostics.tsv',sep='\t')
        passed=bool(len(d) and np.isfinite(d[['R_hat','ESS_bulk','ESS_tail']]).all().all()
            and (d.R_hat<thresholds['rhat_less_than']).all() and (d.ESS_bulk>=thresholds['minimum_bulk_ess']).all()
            and (d.ESS_tail>=thresholds['minimum_tail_ess']).all() and status['divergences']==thresholds['divergences']
            and status['max_depth_hits']==thresholds['treedepth_hits'] and np.isfinite(status['min_bfmi'])
            and status['min_bfmi']>thresholds['bfmi_greater_than'])
        if passed!=(status.get('status')=='complete' and status['passed']):raise ValueError('Diagnostic/status disagreement')
        rows.append(dict(name=entry['name'],model=entry['model'],variant=entry['variant'],subset=entry['subset'],accepted=passed,
            **{key:status[key] for key in ['max_rhat','min_bulk_ess','min_tail_ess','divergences','max_depth_hits','min_bfmi','sampling_seconds']}))
        if passed and entry['subset']=='train':
            f=pd.read_csv(folder/'heldout_participants.tsv',sep='\t')
            if set(f.name)!={entry['name']} or f.participant_id.duplicated().any():raise ValueError('Score identity mismatch')
            if set(f.participant_id)!=set(manifest['meta']['ids']):raise ValueError('Scoring/fitting participant mismatch')
            f['name']=f"Train_{entry['model']}_{entry['variant']}_noage"
            frames.append(f)
    return pd.DataFrame(rows),pd.concat(frames,ignore_index=True),evidence


def comparisons(scores,seed=20260926,iterations=2000):
    wide=scores.pivot(index='participant_id',columns='name',values='log_loss').sort_index()
    counts=scores.pivot(index='participant_id',columns='name',values='n').reindex(wide.index)
    if wide.isna().any().any() or counts.isna().any().any() or not counts.eq(counts.iloc[:,0],axis=0).all().all():
        raise ValueError('Models must score identical participants and choice counts')
    if not np.isfinite(wide.to_numpy()).all():raise ValueError('Nonfinite predictive score')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(wide),(iterations,len(wide)))
    def contrast(a,b):
        delta=(wide[a]-wide[b]).to_numpy();lo,hi=np.quantile(delta[indices].mean(axis=1),[.025,.975])
        return dict(delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi))
    baseline='Train_H2_base_noage';summary=[];matched=[]
    for name in wide:
        f=scores[scores.name==name]
        summary.append(dict(name=name,model=f.model.iloc[0],variant=f.variant.iloc[0],participants=len(wide),choices=int(f.n.sum()),
            log_loss=float(wide[name].mean()),brier=float(f.brier.mean()),**contrast(name,baseline)))
    for model in s.MODELS:
        zero=f'Train_{model}_zero_noage';base=f'Train_{model}_base_noage'
        if zero in wide and base in wide:matched.append(dict(model=model,**contrast(zero,base)))
    focal=None
    if {'Train_H7_zero_noage','Train_HPreference_zero_noage'}.issubset(wide.columns):
        focal=contrast('Train_H7_zero_noage','Train_HPreference_zero_noage')
    return pd.DataFrame(summary),pd.DataFrame(matched),focal


def run(root):
    root=Path(root).resolve();dest=root/'results/full_sample/hierarchical/accepted_review';dest.mkdir(parents=True,exist_ok=True)
    diagnostics,scores,evidence=accepted_scores(root);summary,matched,focal=comparisons(scores)
    diagnostics.to_csv(dest/'diagnostic_inventory.tsv',sep='\t',index=False)
    summary.to_csv(dest/'heldout_available_models.tsv',sep='\t',index=False)
    matched.to_csv(dest/'zero_option_available_pairs.tsv',sep='\t',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    ordered=summary.sort_values('log_loss');fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    y=np.arange(len(ordered));ax.hlines(y,ordered.ci_low,ordered.ci_high,color='#176b87',linewidth=2)
    ax.scatter(ordered.delta,y,color='#176b87',zorder=3);ax.axvline(0,color='.5',linestyle='--')
    ax.set(yticks=y,yticklabels=[f"{m} {'+ zero option' if v=='zero' else 'base'}" for m,v in zip(ordered.model,ordered.variant)],
        xlabel='Run-2 log loss minus H2 base (lower is better)',
        title=f'Accepted training fits: {len(summary)} of 10 models\n304 participants · paired 95% bootstrap intervals')
    ax.invert_yaxis();fig.savefig(dest/'heldout_available_models.png',dpi=180,bbox_inches='tight',pad_inches=.2);fig.savefig(dest/'heldout_available_models.pdf',bbox_inches='tight',pad_inches=.2);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');y=np.arange(len(matched))
    ax.hlines(y,matched.ci_low,matched.ci_high,color='#176b87',linewidth=2);ax.scatter(matched.delta,y,color='#176b87',zorder=3)
    ax.axvline(0,color='.5',linestyle='--');ax.set(yticks=y,yticklabels=matched.model,
        xlabel='Run-2 log loss: zero-option model minus its base\nNegative values favor adding the zero-option term',
        title='Four accepted matched comparisons\nH5 omitted: base training fit has not passed diagnostics')
    fig.savefig(dest/'zero_option_available_pairs.png',dpi=180,bbox_inches='tight',pad_inches=.2);fig.savefig(dest/'zero_option_available_pairs.pdf',bbox_inches='tight',pad_inches=.2);plt.close(fig)
    missing=diagnostics[~diagnostics.accepted].name.tolist()
    state=dict(status='partial_accepted_results',source_commit=s.fs.git_sha(root),accepted_fits=int(diagnostics.accepted.sum()),total_fits=len(diagnostics),
        accepted_training_models=len(summary),missing_fits=missing,paired_n=int(summary.participants.iloc[0]),heldout_choices=int(summary.choices.iloc[0]),
        h7_zero_minus_preference_zero=focal,bootstrap_seed=20260926,bootstrap_iterations=2000,evidence_sha256=evidence,
        interpretation='Online run-2 conditional prediction using fixed run-1 posteriors. Equal-participant per-trial log loss; paired participant bootstrap. Available-model comparisons, not a completed 20-fit analysis or mechanism identification.')
    lines=['# Accepted full-cohort results after the targeted retry','',
        f"{state['accepted_fits']}/{len(diagnostics)} fits pass all predeclared diagnostics; {len(summary)}/10 training models can be compared on 304 participants and 12,494 run-2 choices.",'',
        'Two fits remain unaccepted: `Train_H5_base_noage_retry1` and `Full_HPreference_base_noage_retry1`. Each has exactly one divergent transition among 32,000 retained draws; all Rhat, ESS, treedepth and BFMI checks pass. The zero-divergence threshold remains unchanged. No scores or scientific parameter summaries from these fits enter this report.','',
        '| Model | Variant | Mean log loss | Difference from H2 base | 95% paired interval |','|---|---|---:|---:|---|']
    for _,r in ordered.iterrows():lines.append(f'| {r.model} | {r.variant} | {r.log_loss:.4f} | {r.delta:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |')
    lines+=['','## Adding the zero-option term','','| Model | Zero minus base log loss | 95% paired interval |','|---|---:|---|']
    for _,r in matched.iterrows():lines.append(f'| {r.model} | {r.delta:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] |')
    if focal:lines+=['',f"H7 zero minus HPreference zero: {focal['delta']:.5f}, 95% paired interval [{focal['ci_low']:.5f}, {focal['ci_high']:.5f}]."]
    lines+=['','Intervals describe participant sampling variation and are not adjusted for multiple comparisons. Prediction uses observed feedback online, with run-1 posterior parameters fixed; it is not a joint run-2 marginal likelihood. Similar predictive scores cannot establish equivalent or distinguishable psychological mechanisms.',
        '', 'The two failed fits need targeted geometry inspection before another sampling change. The saved thinned global traces show similar chain locations but cannot rule out local geometry problems. Inspect all participant-level states at the divergent iterations and their chain context from existing Linux2 posterior files; this requires no new sampling.',
        '', 'This results-only review verifies published diagnostics, manifests, summaries and heldout-score hashes. It does not claim to revalidate the ignored Linux2 canonical data or raw posterior CSVs locally.']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
    state['output_sha256']={p.name:s.fs.sha(p) for p in dest.iterdir() if p.is_file() and p.name!='status.json'}
    s.save(dest/'status.json',state);print(json.dumps({k:v for k,v in state.items() if k not in ['evidence_sha256','output_sha256']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');a=p.parse_args();run(a.root)


if __name__=='__main__':main()
