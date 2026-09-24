"""Explicit bound sensitivities; the first-pass tables remain immutable references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from .models import specification, pack
from .fitting import load_inputs, one_fit
from .recovery import _recover_subject, _safe_corr

OUT = Path('results/tables')


def audit():
    fits = pd.read_csv(OUT/'model_fits.csv')
    rows = []
    for model, g in fits.groupby('model'):
        # These files were generated at the original bounds, irrespective of new defaults.
        old_name = model + '_theta5' if model in ['M4','M5','M6','M7'] else model
        _, names, _, bounds = specification(old_name)
        for p, (lo, hi) in zip(names, bounds):
            x = g[p].to_numpy()
            masks = {'lower':np.isclose(x,lo,atol=1e-8,rtol=0),
                     'upper':np.isclose(x,hi,atol=1e-8,rtol=0),
                     'within_1pct_lower':x <= lo+.01*(hi-lo),
                     'within_1pct_upper':x >= hi-.01*(hi-lo)}
            row = dict(model=model,parameter=p,n=len(x),mean=x.mean(),median=np.median(x),
                       minimum=x.min(),maximum=x.max(),lower_bound=lo,upper_bound=hi)
            for key, mask in masks.items():
                row[key+'_n'] = int(mask.sum()); row[key+'_percent'] = 100*mask.mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'parameter_bound_audit.csv',index=False)


def bound_fits(config):
    audit()
    t, _, ratings = load_inputs()
    names = [f'M{m}_theta10' for m in [4,5,6,7]] + ['M3_phi10','M3_logitprior']
    tasks = [(sub,f,name) for sub,f in t.groupby('participant_id') for name in names
             if not name.startswith(('M3','M4')) or sub in ratings]
    legacy=pd.read_csv(OUT/'model_fits.csv').set_index(['participant_id','model'])
    result = Parallel(n_jobs=config['jobs'],verbose=10)(delayed(one_fit)(sub,f,name,ratings,config,False,legacy.loc[(sub,name.split('_')[0])].to_dict()) for sub,f,name in tasks)
    parent=pd.DataFrame([r[0] for r in result]).set_index(['participant_id','model'])
    tasks20=[(sub,f,f'M{m}_theta20') for sub,f in t.groupby('participant_id') for m in [4,5,6,7] if m!=4 or sub in ratings]
    result += Parallel(n_jobs=config['jobs'],verbose=10)(delayed(one_fit)(sub,f,name,ratings,config,False,parent.loc[(sub,name.replace('theta20','theta10'))].to_dict()) for sub,f,name in tasks20)
    new = pd.DataFrame([r[0] for r in result])
    pd.concat([r[1] for r in result]).to_csv(OUT/'bound_predictions.csv',index=False)
    new.to_csv(OUT/'bound_fits.csv',index=False)
    old = pd.read_csv(OUT/'model_fits.csv'); old = old[old.model.isin(['M4','M5','M6','M7'])].copy()
    old['model'] += '_theta5'
    all_fits = pd.concat([old,new[new.model.str.contains('_theta')]],ignore_index=True)
    all_fits['base_model'] = all_fits.model.str.split('_').str[0]
    all_fits['theta_upper'] = all_fits.model.str.extract(r'theta(\d+)').astype(int)
    for p in ['theta','theta_stranger']:
        all_fits[p+'_at_upper'] = np.isclose(all_fits[p],all_fits.theta_upper,atol=1e-8,rtol=0)
        all_fits[p+'_near_upper'] = all_fits[p] >= .99*all_fits.theta_upper
    all_fits['kappa_theta'] = all_fits.kappa*all_fits.theta
    all_fits.to_csv(OUT/'theta_bound_sensitivity.csv',index=False)
    revised = pd.read_csv(OUT/'model_fits.csv')
    revised = revised[~revised.model.isin(['M4','M5','M6','M7'])]
    primary = new[new.model.str.endswith('_theta10')].copy()
    primary['model'] = primary.model.str.split('_').str[0]
    revised = pd.concat([revised,primary],ignore_index=True)
    revised['theta_upper'] = np.where(revised.model.isin(['M4','M5','M6','M7']),10,np.nan)
    revised.to_csv(OUT/'model_fits_theta10.csv',index=False)
    sats = []
    for sub, values in ratings.items():
        if sub not in t.participant_id.unique(): continue
        for partner,s in zip(['friend','stranger','computer'],values):
            threshold = (1-1e-9)/s if s>0 else np.inf
            sats.append(dict(participant_id=sub,partner=partner,normalized_rating=s,
                             saturation_phi=threshold if np.isfinite(threshold) else np.nan,
                             prior_constant_zero=s==0,
                             flat_fraction_0_10=1 if s==0 else max(0,10-threshold)/10))
    pd.DataFrame(sats).to_csv(OUT/'phi_saturation.csv',index=False)
    # Held-out comparisons must use the revised bounds as well.
    tasks=[(sub,f,f'M{m}_theta10') for sub,f in t.groupby('participant_id') for m in [4,5,6,7] if m!=4 or sub in ratings]
    result=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(one_fit)(sub,f,name,ratings,config,True) for sub,f,name in tasks)
    pd.DataFrame([x[0] for x in result]).to_csv(OUT/'heldout_theta10.csv',index=False)
    pd.concat([x[1] for x in result]).to_csv(OUT/'heldout_theta10_predictions.csv',index=False)


def bound_recovery(config):
    t,_,ratings=load_inputs()
    f=pd.read_csv(OUT/'bound_fits.csv');f=f[f.model.str.endswith('_theta10')]
    tasks=[(r['participant_id'],t[t.participant_id.eq(r['participant_id'])],r['model'],r,ratings.get(r['participant_id'],np.zeros(3))) for r in f.to_dict('records')]
    result=Parallel(n_jobs=config['jobs'],verbose=10)(delayed(_recover_subject)(*args,config) for args in tasks)
    rows=pd.DataFrame([r for batch in result for r in batch])
    rows.to_csv(OUT/'parameter_recovery_theta10.csv',index=False)
    old=pd.read_csv(OUT/'parameter_recovery.csv');old=old[old.model.isin(['M4','M5','M6','M7'])].copy();old['model']+='_theta5'
    both=pd.concat([old,rows],ignore_index=True);summ=[]
    for (model,p,it),g in both.groupby(['model','parameter','iteration']):
        error=g.recovered-g.true
        upper=dict(zip(specification(model)[1],specification(model)[3]))[p][1]
        wide=both[(both.model==model)&(both.iteration==it)].pivot(index='participant_id',columns='parameter',values='recovered')
        corr=wide.theta.corr(wide.kappa) if 'theta' in wide else np.nan
        summ.append(dict(model=model,parameter=p,iteration=it,n=len(g),pearson=_safe_corr(g.true,g.recovered,stats.pearsonr),
                         spearman=_safe_corr(g.true,g.recovered,stats.spearmanr),rmse=np.sqrt(np.mean(error**2)),bias=error.mean(),
                         boundary_rate=g.boundary.mean(),upper_rate=np.isclose(g.recovered,upper,atol=1e-4).mean(),theta_kappa_correlation=corr))
    pd.DataFrame(summ).groupby(['model','parameter']).mean(numeric_only=True).drop(columns='iteration').to_csv(OUT/'theta_recovery_summary.csv')
    recovery_tradeoffs()


def figures():
    import matplotlib.pyplot as plt
    from .plotting import style,save,heading
    style();d=pd.read_csv(OUT/'theta_bound_sensitivity.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,(m,p) in zip(axes.flat,[('M4','theta'),('M5','theta'),('M6','theta'),('M7','theta'),('M7','theta_stranger')]):
        g=d[d.base_model.eq(m)].pivot(index='participant_id',columns='theta_upper',values=p)
        for _,row in g.iterrows():ax.plot([5,10,20],row,alpha=.17,color='#157F86',lw=.7)
        ax.plot([5,10,20],g.median(),color='#203A4F',marker='o',lw=2,label='Median')
        ax.plot([5,10,20],[5,10,20],ls='--',color='#D59437',label='Ceiling')
        ax.set(title=f'{m}: {p}',xlabel='Upper bound',ylabel='Fitted value',xticks=[5,10,20],ylim=(-.5,20.5))
    ax=axes.flat[-1];g=d[d.base_model.eq('M5')]
    for b,c in zip([5,10,20],['#73829A','#157F86','#D59437']):
        q=g[g.theta_upper.eq(b)];ax.scatter(q.theta,q.kappa,s=15,alpha=.5,color=c,label=f'Bound {b}')
    ax.set(xlabel='M5 theta',ylabel='M5 kappa',yscale='log');ax.legend()
    heading(fig,'Do social-value estimates follow the ceiling?','Same participants and task logic; 100 starts per new fit. Lines connect individual estimates.')
    save(fig,'14_theta_bound_sensitivity')


def recovery_tradeoffs():
    new=pd.read_csv(OUT/'parameter_recovery_theta10.csv')
    old=pd.read_csv(OUT/'parameter_recovery.csv');old=old[old.model.isin(['M4','M5','M6','M7'])].copy();old['model']+='_theta5'
    both=pd.concat([old,new],ignore_index=True);rows=[]
    for model,g in both.groupby('model'):
        rec=g.pivot(index=['participant_id','iteration'],columns='parameter',values='recovered')
        truth=g.pivot(index=['participant_id','iteration'],columns='parameter',values='true');err=rec-truth
        for p in ['theta','theta_stranger']:
            if p not in rec:continue
            correlations=[]
            for _,q in rec.groupby(level=0):
                if q[p].std()>1e-8 and q.kappa.std()>1e-8:correlations.append(q[p].corr(q.kappa))
            rows.append(dict(model=model,parameter=p,n_simulated_fits=len(rec),recovered_correlation=rec[p].corr(rec.kappa),
                             error_correlation=err[p].corr(err.kappa),median_within_participant_correlation=np.median(correlations),n_informative_participants=len(correlations)))
    pd.DataFrame(rows).to_csv(OUT/'theta_kappa_recovery_tradeoff.csv',index=False)
