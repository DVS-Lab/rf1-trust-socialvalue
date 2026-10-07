"""Portable review of published amount-comparison results; never loads raw draws."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/full_sample/hierarchical'
SOURCE=BASE/'amount_comparison'
DEST=SOURCE/'review'


def main():
    evidence={}
    def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
    def record(f):evidence[str(f.relative_to(ROOT))]=sha(f)
    def check(f,h):
        if sha(f)!=h:raise ValueError('Published evidence changed: '+str(f))
        record(f)
    parent=json.loads((SOURCE/'status.json').read_text());record(SOURCE/'status.json')
    for f,h in parent['comparison_sha256'].items():check(SOURCE/f,h)
    implementation=json.loads((SOURCE/'implementation_status.json').read_text());record(SOURCE/'implementation_status.json')
    if implementation['status']!='passed':raise ValueError('Implementation checks did not pass')
    for f,h in implementation['source_hashes'].items():check(ROOT/f,h)
    check(SOURCE/'implementation_checks.tsv',implementation['check_sha256'])
    check(SOURCE/'extension_prior_screen.tsv',implementation['prior_screen_sha256'])
    limits=json.loads((ROOT/'config/full_sample_linux2.json').read_text())['hierarchical']['acceptance']
    rows=[];frames=[]
    for folder in sorted((SOURCE/'fits').iterdir()):
        state=json.loads((folder/'status.json').read_text());record(folder/'status.json')
        manifest=json.loads((folder/'manifest_summary.json').read_text());record(folder/'manifest_summary.json')
        if manifest['fingerprint']!=state['fingerprint']:raise ValueError('Fit fingerprint mismatch')
        for file,h in state['output_sha256'].items():check(folder/file,h)
        d=pd.read_csv(folder/'diagnostics.tsv',sep='\t')
        accepted=bool(np.isfinite(d[['R_hat','ESS_bulk','ESS_tail']]).all().all() and
                      (d.R_hat<limits['rhat_less_than']).all() and (d.ESS_bulk>=limits['minimum_bulk_ess']).all() and
                      (d.ESS_tail>=limits['minimum_tail_ess']).all() and state['divergences']==0 and state['max_depth_hits']==0 and
                      np.isfinite(state['min_bfmi']) and state['min_bfmi']>limits['bfmi_greater_than'])
        if accepted!=state['passed'] or accepted!=(state['status']=='complete'):raise ValueError('Diagnostic/status disagreement')
        rows.append(dict(name=folder.name,accepted=accepted,**{k:state[k] for k in ['max_rhat','min_bulk_ess','min_tail_ess','divergences','max_depth_hits','min_bfmi','sampling_seconds']}))
        for when in ['before','after']:record(folder/f'live_source_audit_{when}.json')
        if accepted:frames.append(pd.read_csv(folder/'heldout_participants.tsv',sep='\t'))
    for name in ['Train_H7_zero_noage','Train_HPreference_zero_noage']:
        folder=BASE/'fits'/name;state=json.loads((folder/'status.json').read_text());record(folder/'status.json')
        if state['status']!='complete' or not state['passed']:raise ValueError('Unaccepted original model')
        check(folder/'heldout_participants.tsv',state['heldout_sha256'])
        frames.append(pd.read_csv(folder/'heldout_participants.tsv',sep='\t'))
    scores=pd.concat(frames,ignore_index=True);wide=scores.pivot(index='participant_id',columns='name',values='log_loss').sort_index()
    counts=scores.pivot(index='participant_id',columns='name',values='n').reindex(wide.index)
    if wide.isna().any().any() or not counts.eq(counts.iloc[:,0],axis=0).all().all():raise ValueError('Unmatched participants/choices')
    if len(wide)!=304 or counts.iloc[:,0].sum()!=12494:raise ValueError('Unexpected score cohort')
    rng=np.random.default_rng(20260930);ix=rng.integers(0,len(wide),(2000,len(wide)));contrasts=[]
    for name in wide:
        for baseline in ['Train_H7_zero_noage','Train_HPreference_zero_noage']:
            delta=(wide[name]-wide[baseline]).to_numpy();lo,hi=np.quantile(delta[ix].mean(axis=1),[.025,.975])
            contrasts.append(dict(name=name,baseline=baseline,mean_log_loss=wide[name].mean(),delta=delta.mean(),ci_low=lo,ci_high=hi,participants=304,choices=12494,fraction_improved=np.mean(delta<0)))
    DEST.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(DEST/'diagnostic_inventory.tsv',sep='\t',index=False)
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(DEST/'accepted_contrasts.tsv',sep='\t',index=False)
    q=contrasts[contrasts.baseline.eq('Train_H7_zero_noage')].sort_values('mean_log_loss');y=np.arange(len(q))
    labels={'Train_H7_zero_noage':'Original H7 + zero','Train_HPreference_zero_noage':'Original HPreference + zero','Train_HPreference_zero_bias_v1':'HPreference + zero + bias'}
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');ax.hlines(y,q.ci_low,q.ci_high,color='#176b87',lw=2);ax.scatter(q.delta,y,color='#176b87',zorder=3)
    ax.axvline(0,color='.5',ls='--');ax.set(yticks=y,yticklabels=[labels.get(n,n) for n in q.name],xlabel='Run-2 log loss minus original H7 + zero (lower is better)',
        title='One accepted extension improves run-2 prediction\n304 participants · paired 95% bootstrap intervals · exploratory')
    ax.invert_yaxis()
    for ext in ['png','pdf']:fig.savefig(DEST/f'accepted_prediction_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    name='Train_HPreference_zero_bias_v1'
    if name not in set(scores.name):raise ValueError('Expected accepted bias model is unavailable')
    data=pd.read_csv(SOURCE/'fits'/name/'heldout_cells.tsv',sep='\t');cells=[]
    for (strat,group),g in data.groupby(['stratification','group']):
        if g.participant_id.duplicated().any():raise ValueError('Multiple participant rows in cell')
        residual=(g.observed-g.predicted).to_numpy();samples=rng.integers(0,len(g),(2000,len(g)))
        lo,hi=np.quantile(residual[samples].mean(axis=1),[.025,.975])
        cells.append(dict(stratification=strat,group=group,participants=len(g),observed=g.observed.mean(),predicted=g.predicted.mean(),residual=residual.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(cells).to_csv(DEST/'accepted_bias_run2_cells.tsv',sep='\t',index=False)
    record(Path(__file__).resolve())
    (DEST/'status.json').write_text(json.dumps(dict(status='partial_review',results_commit='854b523',new_sampling=False,
        accepted_new_fits=int(sum(r['accepted'] for r in rows)),total_new_fits=len(rows),evidence_sha256=evidence,
        output_sha256={f.name:sha(f) for f in DEST.iterdir() if f.suffix in ['.tsv','.png','.pdf']},
        interpretation='One accepted bias extension; amount versus bias unavailable. Paired bootstrap conditions on fitted predictions; exploratory reuse of evaluation data.'),indent=2)+'\n')
    print('Reviewed',len(rows),'new fits; accepted',sum(r['accepted'] for r in rows),'; verified/recorded',len(evidence),'evidence files.')


if __name__=='__main__':main()
