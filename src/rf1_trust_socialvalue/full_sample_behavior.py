"""Descriptive first milestone, participant uncertainty, and planned age GEE."""
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from patsy import build_design_matrices, dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .full_sample import PARTNERS, sha, git_sha, save_json, verify, parity


def interval(values, rng, boot):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if len(v)<2:return (float(np.mean(v)) if len(v) else np.nan,np.nan,np.nan,len(v))
    draws=rng.choice(v,(boot,len(v)),replace=True).mean(axis=1)
    lo,hi=np.quantile(draws,[.025,.975])
    return float(v.mean()),float(lo),float(hi),len(v)


def summarize(t,c):
    rng=np.random.default_rng(c['seed']);boot=c['bootstrap_iterations'];rows=[]
    t=t.copy();t['previous_feedback']=t.last_observed_reciprocation.map({1.:'recip',0.:'defect'}).fillna('none_observed')
    sets=[('partner',['partner']),('offer',['partner','offer_pair']),('zero',['partner','zero_option_present']),
          ('time',['partner','run','trial_bin']),('recent_feedback',['partner','previous_feedback']),
          ('cohort',['cohort_tag','partner']),('cohort_zero',['cohort_tag','partner','zero_option_present']),
          ('cohort_offer',['cohort_tag','partner','offer_pair'])]
    for label,cols in sets:
        for key,g in t.groupby(cols,observed=True):
            if not isinstance(key,tuple):key=(key,)
            for metric,col in [('investment','chosen_amount'),('high_choice','chose_high'),('response_time','response_time'),('response_fraction','valid_choice')]:
                means=g.groupby('participant_id')[col].mean().dropna()
                est,lo,hi,n=interval(means,rng,boot)
                rows.append(dict(grouping=label,**dict(zip(cols,key)),metric=metric,estimate=est,ci_low=lo,ci_high=hi,
                    n_participants=n,n_presented=len(g),n_valid=int(g.valid_choice.sum()),uncertainty='participant bootstrap; equally weighted participant means'))
    summary=pd.DataFrame(rows)
    contrasts=[]
    # Paired participant uncertainty preserves within-person dependence.
    for cohort, q in [('full_primary',t)]+list(t.groupby('cohort_tag')):
        for metric in ['chose_high','chosen_amount']:
            means=q.groupby(['participant_id','partner'])[metric].mean().unstack().reindex(columns=PARTNERS)
            for a,b in [('friend','computer'),('friend','stranger'),('stranger','computer')]:
                estimate,lo,hi,n=interval((means[a]-means[b]).dropna(),rng,boot)
                contrasts.append(dict(cohort=cohort,metric=metric,contrast=a+' - '+b,estimate=estimate,ci_low=lo,ci_high=hi,n_participants=n))
        for partner,g in q.groupby('partner'):
            z=g.groupby(['participant_id','zero_option_present']).chose_high.mean().unstack().reindex(columns=[False,True])
            estimate,lo,hi,n=interval((z[True]-z[False]).dropna(),rng,boot)
            contrasts.append(dict(cohort=cohort,metric='chose_high',contrast=partner+': zero - positive-positive',estimate=estimate,ci_low=lo,ci_high=hi,n_participants=n))
    return summary,pd.DataFrame(contrasts)


def age_model(t,c,dest):
    """Continuous age; robust participant-cluster covariance; fixed empirical target."""
    d=t[t.valid_choice & t.age.notna()].copy()
    if d.participant_id.nunique()<10:
        return dict(status='not_estimable',reason='fewer than 10 participants with age')
    ages=d.groupby('participant_id').age.first(); center=ages.mean(); scale=ages.std(ddof=0)
    if scale<=0:return dict(status='not_estimable',reason='no age variation')
    d['age_z']=(d.age-center)/scale
    d['chose_high']=d.chose_high.astype(float)
    d['partner']=pd.Categorical(d.partner,categories=['computer','stranger','friend'])
    formula='chose_high ~ C(partner)*age_z + C(offer_pair) + C(run) + trial_scaled + C(partner):trial_scaled'
    # Single-level categorical run terms automatically drop. Simplification only
    # on a rank-deficient design, never based on the size/sign of an age result.
    model=smf.gee(formula,groups='participant_id',data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable())
    simplification=None
    if np.linalg.matrix_rank(model.exog)<model.exog.shape[1]:
        formula=formula.replace(' + C(run)','')
        simplification='Removed redundant run term after design-rank check.'
        model=smf.gee(formula,groups='participant_id',data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable())
    if np.linalg.matrix_rank(model.exog)<model.exog.shape[1]:
        return dict(status='not_estimable',reason='rank deficiency persists after removing redundant run term',formula=formula)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');m=model.fit(maxiter=c['age_behavior']['max_iterations'])
    if not m.converged or not np.isfinite(m.params).all() or not np.isfinite(m.cov_params()).all().all():
        return dict(status='not_estimable',reason='GEE convergence/covariance failure',warnings=[str(w.message) for w in caught])
    design=dmatrix(formula.split('~',1)[1],d,return_type='dataframe')
    if set(design.columns)!=set(m.params.index):
        raise ValueError('GEE and marginal design columns disagree')
    design_info=design.design_info
    column_order=[design.columns.get_loc(name) for name in m.params.index]
    if not np.allclose(design.to_numpy()[:,column_order],m.model.exog):
        raise ValueError('GEE and marginal design coding disagree')
    ci=m.conf_int()
    pd.DataFrame(dict(term=m.params.index,estimate=m.params.to_numpy(),se=m.bse.to_numpy(),ci_low=ci[0].to_numpy(),ci_high=ci[1].to_numpy())).to_csv(dest/'age_partner_behavior.tsv',sep='\t',index=False)
    ref=d[['participant_id','offer_pair','run','trial_scaled']].copy()
    ref['weight']=1/d.groupby('participant_id').participant_id.transform('size').to_numpy()/len(ages)
    ref=ref.groupby(['offer_pair','run','trial_scaled'],as_index=False).weight.sum()
    grid=np.unique(np.r_[np.linspace(ages.min(),ages.max(),41),c['age_behavior']['contrast_ages']])
    cov=m.cov_params().to_numpy();cache={};rows=[]
    for age in grid:
        for partner in PARTNERS:
            q=ref.copy();q['age_z']=(age-center)/scale;q['partner']=pd.Categorical([partner]*len(q),categories=['computer','stranger','friend'])
            x=np.asarray(build_design_matrices([design_info],q)[0])[:,column_order];prob=expit(x@m.params)
            value=float(ref.weight@prob);gradient=(ref.weight.to_numpy()*prob*(1-prob))@x
            cache[age,partner]=(value,gradient)
            se=np.sqrt(max(0,float(gradient@cov@gradient)))
            rows.append(dict(age=age,contrast=partner,estimate=value,ci_low=max(0,value-1.96*se),ci_high=min(1,value+1.96*se),extrapolation=bool(age<ages.min() or age>ages.max())))
        for a,b in [('friend','computer'),('friend','stranger'),('stranger','computer')]:
            value=cache[age,a][0]-cache[age,b][0];gradient=cache[age,a][1]-cache[age,b][1]
            cache[age,a+' - '+b]=(value,gradient);se=np.sqrt(max(0,float(gradient@cov@gradient)))
            rows.append(dict(age=age,contrast=a+' - '+b,estimate=value,ci_low=value-1.96*se,ci_high=value+1.96*se,extrapolation=bool(age<ages.min() or age>ages.max())))
    marginal=pd.DataFrame(rows);marginal.to_csv(dest/'age_partner_marginal.tsv',sep='\t',index=False)
    changes=[];a0,a1=c['age_behavior']['contrast_ages']
    for contrast in ['friend - computer','friend - stranger','stranger - computer']:
        val=cache[a1,contrast][0]-cache[a0,contrast][0];g=cache[a1,contrast][1]-cache[a0,contrast][1];se=np.sqrt(max(0,float(g@cov@g)))
        changes.append(dict(contrast=contrast,age_from=a0,age_to=a1,estimate=val,ci_low=val-1.96*se,ci_high=val+1.96*se))
    pd.DataFrame(changes).to_csv(dest/'age_partner_change.tsv',sep='\t',index=False)
    interactions=[]
    for a,b in [('friend','computer'),('friend','stranger'),('stranger','computer')]:
        w=pd.Series(0.,index=m.params.index)
        for partner,sign in [(a,1),(b,-1)]:
            if partner!='computer':w[f'C(partner)[T.{partner}]:age_z']+=sign
        val=float(w@m.params);se=np.sqrt(max(0,float(w@cov@w)))
        interactions.append(dict(contrast=a+' - '+b,estimate=val,ci_low=val-1.96*se,ci_high=val+1.96*se))
    pd.DataFrame(interactions).to_csv(dest/'age_partner_interactions.tsv',sep='\t',index=False)
    return dict(status='complete',formula=formula,simplification=simplification,n_participants=len(ages),n_choices=len(d),
        age_center=float(center),age_scale=float(scale),uncertainty='GEE participant-cluster robust covariance; delta-method marginal intervals',
        warnings=[str(w.message) for w in caught],quadratic_sensitivity='prespecified for next phase; not run at integration gate')


def figures(summary,contrasts,dest,age_dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(friend='#007c91',stranger='#dd9651',computer='#706ba8')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,metric,ylabel in [(axes[0],'investment','Mean investment'),(axes[1],'high_choice','Probability of higher choice')]:
        q=summary[(summary.grouping=='partner')&(summary.metric==metric)].set_index('partner').reindex(PARTNERS)
        ax.bar(PARTNERS,q.estimate,color=[colors[p] for p in PARTNERS],alpha=.8)
        ax.errorbar(PARTNERS,q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt='none',color='#303030',capsize=4)
        ax.set_ylabel(ylabel)
    q=summary[(summary.grouping=='offer')&(summary.metric=='high_choice')]
    offers=sorted(q.offer_pair.unique(),key=lambda s:tuple(map(float,s.split('-'))))
    for partner in PARTNERS:
        g=q[q.partner==partner].set_index('offer_pair').reindex(offers)
        axes[2].plot(offers,g.estimate,'o-',color=colors[partner],label=partner)
        axes[2].fill_between(offers,g.ci_low,g.ci_high,color=colors[partner],alpha=.12)
    axes[2].set(xlabel='Exact offer pair',ylabel='Probability of higher choice',ylim=(0,1));axes[2].legend()
    fig.suptitle('Full-cohort Trust behavior · equal participant weighting · 95% participant bootstrap intervals')
    dest.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'01_behavior_by_partner_offer.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for ax,cohort in zip(axes,['in_original_openneuro_release','later_participant']):
        q=summary[(summary.grouping=='cohort_zero')&(summary.metric=='high_choice')&(summary.cohort_tag==cohort)]
        for partner in PARTNERS:
            g=q[q.partner==partner].sort_values('zero_option_present')
            if len(g):ax.errorbar(g.zero_option_present.astype(int),g.estimate,yerr=[g.estimate-g.ci_low,g.ci_high-g.estimate],fmt='o-',color=colors[partner],label=partner,capsize=3)
        ax.set(xticks=[0,1],xticklabels=['Both positive','Zero available'],ylim=(0,1),ylabel='Probability of higher choice',title=cohort.replace('_',' '))
    axes[0].legend();fig.suptitle('Internal cohort check · descriptive offer differences, not a causal zero-option effect')
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'02_zero_option_cohort_check.{ext}',dpi=170)
    plt.close(fig)
    if (age_dest/'age_partner_marginal.tsv').exists():
        a=pd.read_csv(age_dest/'age_partner_marginal.tsv',sep='\t')
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        for partner in PARTNERS:
            q=a[a.contrast==partner];axes[0].plot(q.age,q.estimate,color=colors[partner],label=partner);axes[0].fill_between(q.age,q.ci_low,q.ci_high,color=colors[partner],alpha=.15)
        for name,color in [('friend - computer',colors['friend']),('friend - stranger',colors['stranger']),('stranger - computer',colors['computer'])]:
            q=a[a.contrast==name];axes[1].plot(q.age,q.estimate,color=color,label=name);axes[1].fill_between(q.age,q.ci_low,q.ci_high,color=color,alpha=.15)
        for ax in axes:ax.set_xlabel('Age (years)');ax.legend(fontsize=9)
        axes[0].set(ylabel='Marginal high-choice probability',ylim=(0,1));axes[1].set_ylabel('Partner contrast');axes[1].axhline(0,color='gray',lw=.8)
        fig.suptitle('Age × partner · equal participant standardization · robust GEE 95% intervals')
        for ext in ['png','pdf','svg']:fig.savefig(dest/f'03_age_partner_behavior.{ext}',dpi=170)
        plt.close(fig)


def run(c):
    out=Path(c['output']);dest=out/'tables';diagnostics=out/'diagnostics';diagnostics.mkdir(exist_ok=True)
    # Invalidate any earlier success before reading inputs or producing replacements.
    save_json(out/'milestone_status.json',dict(status='running',hierarchical_launch_authorized=False))
    prov=verify(c)
    t=pd.read_csv(Path(c['work'])/'canonical_trials.tsv',sep='\t',dtype={'trial_id':str,'session':str})
    m=pd.read_csv(dest/'cohort_manifest.tsv',sep='\t');a=pd.read_csv(dest/'data_audit.tsv',sep='\t')
    for f in dest.glob('age_partner_*.tsv'):f.unlink()
    for f in (out/'figures').glob('03_age_partner_behavior.*'):f.unlink()
    p,overlap=parity(t,c);p.to_csv(dest/'openneuro_linux2_parity.tsv',sep='\t',index=False)
    unexplained=int(p.classification.eq('new unexplained discrepancy').sum())
    summary,contrasts=summarize(t,c)
    summary.to_csv(dest/'behavior_summary.tsv',sep='\t',index=False)
    contrasts.to_csv(dest/'behavior_contrasts.tsv',sep='\t',index=False)
    heterogeneity=a.groupby('cohort_tag').agg(n=('participant_id','size'),age_mean=('age','mean'),age_sd=('age','std'),age_min=('age','min'),age_max=('age','max'),miss_fraction=('miss_fraction','mean'))
    heterogeneity.to_csv(dest/'cohort_heterogeneity.tsv',sep='\t')
    age=dict(status='blocked_by_parity') if unexplained else age_model(t,c,dest)
    save_json(diagnostics/'age_behavior_status.json',age)
    figures(summary,contrasts,out/'figures',dest)
    ages=a.age.dropna()
    stats=dict(total_canonical_participants=int(m.loc[m.events_sha256.notna(),'participant_id'].nunique()),
        primary_n=len(a),sensitivity_n=int(a.include_sensitivity.sum()),two_run_primary_n=int(a.n_runs.eq(2).sum()),
        total_valid_choices=int(t.valid_choice.sum()),age_n=len(ages),age_missing_n=int(a.age.isna().sum()),
        age_min=float(ages.min()) if len(ages) else None,age_max=float(ages.max()) if len(ages) else None,
        age_mean=float(ages.mean()) if len(ages) else None,age_sd=float(ages.std()) if len(ages)>1 else None,
        source_excluded_trust_n=int(m.loc[m.source_excluded,'participant_id'].nunique()),
        source_exclusion_inventory_n=len(pd.read_csv(c['source_exclusions'],sep='\t')),
        qc_review_runs=int(m.response_qc_status.eq('review').sum()),qc_review_participants=int(m.loc[m.response_qc_status.eq('review'),'participant_id'].nunique()),
        n111_overlap_n=overlap,unexplained_parity_fields=unexplained,
        expected_canonical_corrections=int(p.classification.eq('expected canonical correction').sum()),
        expected_public_omissions=int(p.classification.eq('expected public-data omission').sum()))
    save_json(out/'cohort_summary.json',stats)
    text=['# Linux2 full-sample integration', '', 'N=111 remains frozen at `'+c['frozen_n111_commit']+'`.', '',
        'Status: **'+('parity review required' if unexplained else 'first milestone ready for scientific review')+'**. No hierarchical models launched.', '',
        '## Cohort and migration', '', *['- '+k+': '+str(v) for k,v in stats.items()], '',
        '## Behavior', '', 'Estimates weight participants equally; intervals resample participants. Zero-versus-positive-positive contrasts are descriptive and confounded with offer amounts. Cohort tags are internal heterogeneity checks, not separate primary samples.', '']
    for r in contrasts[contrasts.cohort.eq('full_primary')].itertuples():
        text.append(f'- {r.metric}, {r.contrast}: {r.estimate:.3f} (95% CI {r.ci_low:.3f}, {r.ci_high:.3f}; N={r.n_participants}).')
    text+=['', 'Age model status: '+age['status']+'.', '',
        'Review `tables/cohort_heterogeneity.tsv`, `tables/behavior_contrasts.tsv`, and `figures/02_zero_option_cohort_check.png` before pooling for computational interpretation.', '',
        '## Provenance', '', '- Upstream SHA at freeze: `'+prov['upstream_git_sha']+'`',
        '- Analysis SHA at execution: `'+git_sha(c['_root'])+'`', '- Refreshed QC hash: `'+sha(c['qc_provenance'])+'`', '',
        'Ratings inventory/export is deferred to upstream work and does not gate this milestone.',
        'Stop for review here. The planned next phase includes H2/H5/H7/HPreference/H8, each with and without generic gamma0; representative no-age fits first, then run-1 → run-2 prediction.']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    (diagnostics/'parity_report.md').write_text('# Canonical migration audit\n\n'+f'{overlap} N111 primary participants overlap; {unexplained} unexplained field discrepancies.\n\n'+
        'Joins use participant, session, run and source trial_id. All shared-release participants with primary canonical data are audited. New canonical runs absent from the historical trial table are public omissions; all within-run or observed-field differences require a value- and hash-bound reviewed resolution. Historical appended sessions are never silently reassigned.\n')
    verify(c)
    status=dict(status='blocked_parity' if unexplained else 'ready_for_review',generated_at=datetime.now(timezone.utc).isoformat(),
        hierarchical_launch_authorized=False,analysis_git_sha=git_sha(c['_root']),cohort_summary=stats,
        output_hashes={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='milestone_status.json'})
    save_json(out/'milestone_status.json',status)
    print(json.dumps(stats,indent=2),flush=True)
    if unexplained:raise RuntimeError(f'{unexplained} unexplained parity fields; review saved audit before modeling')
    print('First milestone ready for review. No hierarchical fits launched.',flush=True)
