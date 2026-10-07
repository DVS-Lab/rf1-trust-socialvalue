"""Age summaries, generative checks and bounded recovery for the closeout."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr
from . import full_sample_closeout as q
from . import full_sample_amount as a
from . import full_sample_sampling as s
from . import full_sample_residuals as residuals


@njit(cache=True)
def simulate(arr, par, code, uniforms):
    """Preserve offers, schedules and missingness; update from simulated investments."""
    result=arr.copy();belief=np.full(3,.5)
    for i in range(len(arr)):
        if arr[i,3]<0:
            result[i,4]=0
            continue
        partner=int(arr[i,0]);p=belief[partner];lo=arr[i,1];hi=arr[i,2]
        bonus=par[partner+2] if code==7 and partner<2 else 0.
        pref=par[partner+2] if code==9 and partner<2 else 0.
        logit=par[1]*(hi-lo)*(-1+1.5*p+p*bonus+pref)+par[4]*(lo==0)+par[5]
        if len(par)==7:logit+=par[6]*(hi*hi-lo*lo)/64
        prob=1/(1+np.exp(-logit)) if logit>=0 else np.exp(logit)/(1+np.exp(logit))
        choice=int(uniforms[i]<prob);result[i,3]=choice
        feedback=(hi if choice else lo)>0;result[i,4]=feedback
        if feedback:belief[partner]+=par[0]*(arr[i,5]-p)
    return result


def parameter_tables(fit,natural,data,meta,entry,cfg,dest):
    rows=[]
    for i,sub in enumerate(meta['ids']):
        for j,name in enumerate(meta['parameter_names']):
            rows.append(dict(participant_id=sub,parameter=name,**s.summarize(natural[:,i,j])))
    pd.DataFrame(rows).to_csv(dest/'participant_parameters.tsv',sep='\t',index=False)
    population=[]
    for field in ['mu','tau','mu_ext','tau_ext']:
        values=fit.stan_variable(field)
        for j in range(values.shape[1]):population.append(dict(parameter=f'{field}[{j+1}]',**s.summarize(values[:,j])))
    pd.DataFrame(population).to_csv(dest/'population_parameters.tsv',sep='\t',index=False)
    if not data['A']:return
    slopes=np.c_[fit.stan_variable('beta')[:,:,0],fit.stan_variable('beta_ext')[:,:,0]]
    mu=np.c_[fit.stan_variable('mu'),fit.stan_variable('mu_ext')]
    tau=np.c_[fit.stan_variable('tau'),fit.stan_variable('tau_ext')]
    variance=np.var((np.asarray(meta['age_years'])-cfg['age_center'])/cfg['age_scale'],ddof=0)
    rows=[];curves=[]
    for j,name in enumerate(meta['parameter_names']):
        explained=slopes[:,j]**2*variance
        fraction=explained/(explained+tau[:,j]**2)
        for quantity,values in [('latent_slope_per_20_years',slopes[:,j]),('latent_age_variance_fraction',fraction)]:
            rows.append(dict(parameter=name,quantity=quantity,probability_positive=float(np.mean(values>0)),**s.summarize(values)))
        for age in range(20,90):
            eta=mu[:,j]+slopes[:,j]*(age-cfg['age_center'])/cfg['age_scale']
            values=s.transform(eta[:,None],[data['kind'][j] if j<5 else 4])[:,0]
            curves.append(dict(parameter=name,age=age,curve='zero_random_effect_population_location',**s.summarize(values)))
    pd.DataFrame(rows).to_csv(dest/'age_effects.tsv',sep='\t',index=False)
    pd.DataFrame(curves).to_csv(dest/'age_curves.tsv',sep='\t',index=False)


def predictive_tables(arrays,meta,natural,entry,cfg,dest):
    selected=np.linspace(0,len(natural)-1,cfg['ppc_simulations']).astype(int);cells={}
    ages=np.asarray(meta['age_years'])
    for i,(sub,arr) in enumerate(arrays.items()):
        pars=natural[selected,i];rng=np.random.default_rng(s.stable_seed(cfg['seed'],sub,'closeout_ppc'))
        conditional=np.asarray([a.engine(arr,p,s.SPECS[entry['model']][1])[:,0] for p in pars])
        replicated=np.asarray([simulate(arr,p,s.SPECS[entry['model']][1],u)[:,3] for p,u in zip(pars,rng.random((len(pars),len(arr))))])
        for strat,group,mask in residuals.groups(arr):
            if not mask.any():continue
            for history,values in [('conditional',conditional),('generative',replicated)]:
                cell=cells.setdefault((history,strat,group),dict(observed=[],predicted=[],ages=[],choices=0))
                cell['observed'].append(arr[mask,3].mean());cell['predicted'].append(values[:,mask].mean(axis=1));cell['ages'].append(ages[i]);cell['choices']+=mask.sum()
    rows=[];age_rows=[]
    for (history,strat,group),cell in cells.items():
        obs=np.asarray(cell['observed']);pred=np.asarray(cell['predicted'])
        rows+=a.ppc.summarize_cell((history,strat,group,'high_choice'),obs,pred,int(cell['choices']))
        row=residuals.cell_summary(obs,pred,np.asarray(cell['ages']),s.stable_seed(cfg['seed'],history,group),cfg['bootstrap'])
        age_rows.append(dict(history=history,stratification=strat,group=group,**row))
    pd.DataFrame(rows).to_csv(dest/'predictive_checks.tsv',sep='\t',index=False)
    pd.DataFrame(age_rows).to_csv(dest/'age_predictive_checks.tsv',sep='\t',index=False)


def lkj_cholesky(rng,k,eta=2.):
    """Onion construction of an LKJ(eta) correlation Cholesky factor."""
    L=np.zeros((k,k));L[0,0]=1.
    for i in range(1,k):
        radius=np.sqrt(rng.beta(i/2,eta+(k-i-1)/2));direction=rng.normal(size=i);direction/=np.linalg.norm(direction)
        L[i,:i]=radius*direction;L[i,i]=np.sqrt(1-radius*radius)
    return L


def prior_screen(t,cfg,phase,c):
    entry=dict(model='HPreference',extension='bias',age=True,prior_sd=.5)
    data,meta,arrays=q.model_data(t,entry,phase,cfg);x=np.asarray(data['age'])[:,0]
    rows=[];rng=np.random.default_rng(s.stable_seed(cfg['seed'],'joint_prior'))
    for prior_sd in [cfg['age_prior_sd'],cfg['wide_age_prior_sd']]:
        for rep in range(cfg['prior_simulations']):
            mu=rng.normal(data['mu_location'],data['mu_scale']);tau=abs(rng.normal(0,.8,5));beta=rng.normal(0,prior_sd,5)
            eta=mu+x[:,None]*beta+(np.diag(tau)@lkj_cholesky(rng,5)@rng.normal(size=(5,len(x)))).T
            b=rng.normal(0,1)+x*rng.normal(0,prior_sd)+abs(rng.normal(0,.8))*rng.normal(size=len(x))
            natural=np.c_[s.transform(eta,data['kind']),b];observed=[];probs=[]
            for i,arr in enumerate(arrays.values()):
                generated=simulate(arr,natural[i],9,rng.random(len(arr)));valid=arr[:,3]>=0
                observed.append(generated[valid,3].mean());probs.extend(a.engine(generated,natural[i],9)[valid,0])
            rows.append(dict(age_prior_sd=prior_sd,replicate=rep,mean_high_choice=np.mean(observed),
                between_participant_sd=np.std(observed,ddof=1),all_high_fraction=np.mean(np.asarray(observed)==1),
                all_low_fraction=np.mean(np.asarray(observed)==0),fraction_probability_below_01=np.mean(np.asarray(probs)<.01),
                fraction_probability_above_99=np.mean(np.asarray(probs)>.99)))
    frame=pd.DataFrame(rows)
    if not np.isfinite(frame.to_numpy()).all():raise ValueError('Nonfinite joint prior simulation')
    frame.to_csv(q.paths(c)[0]/'joint_prior_screen.tsv',sep='\t',index=False)


def age_anchor(cfg,old,phase,c,t):
    entry=next(e for e in q.entries() if e['name']=='Full_HPreference_zero_bias_age_closeout1')
    out,work=q.paths(c);dest=out/'fits'/entry['name'];folder=work/'fits'/entry['name']
    status=json.loads((dest/'status.json').read_text());portable=json.loads((dest/'manifest_summary.json').read_text())
    if status['status']!='complete' or not status['passed']:raise ValueError('Recovery requires accepted full age fit')
    data,meta,_=q.model_data(t,entry,phase,cfg)
    fingerprint=s.digest(dict(data=data,meta=meta,entry=entry,config=cfg,sources=q.sources(c),recovery_anchor=None))
    manifest=q.authenticated_cache(folder,fingerprint)
    if manifest is None or status['fingerprint']!=fingerprint or portable['fingerprint']!=fingerprint:
        raise ValueError('Recovery anchor target mismatch')
    if {Path(f).name:h for f,h in manifest['posterior_sha256'].items()}!=portable['posterior_sha256']:
        raise ValueError('Recovery anchor portable hashes mismatch')
    fit=s.load_chains(manifest['csv_files']);_,info=q.diagnostics(fit,cfg,c)
    if not info['passed']:raise ValueError('Recovery anchor diagnostics failed')
    return fit,dict(fingerprint=fingerprint,posterior_sha256=portable['posterior_sha256']),data,meta


def synthetic_trials(t,rep,cfg,old,phase,c):
    fit,anchor,data,meta=age_anchor(cfg,old,phase,c,t)
    mu=np.median(fit.stan_variable('mu'),axis=0);tau=np.median(fit.stan_variable('tau'),axis=0)
    # Mean correlation is positive definite; construct its Cholesky, not median elements of L.
    L=np.linalg.cholesky(fit.stan_variable('Omega').mean(axis=0))
    mu_ext=np.median(fit.stan_variable('mu_ext'),axis=0);tau_ext=np.median(fit.stan_variable('tau_ext'),axis=0)
    del fit
    beta=np.zeros(6) if rep<4 else np.array([.3,-.3,.3,-.3,.3,-.3])*(-1 if rep%2 else 1)
    rng=np.random.default_rng(s.stable_seed(cfg['seed'],'recovery',rep));n=len(meta['ids']);x=np.asarray(data['age'])[:,0]
    eta=mu+x[:,None]*beta[:5]+(np.diag(tau)@L@rng.normal(size=(5,n))).T
    bias=mu_ext[0]+x*beta[5]+tau_ext[0]*rng.normal(size=n)
    natural=np.c_[s.transform(eta,data['kind']),bias]
    frames=[]
    for i,sub in enumerate(meta['ids']):
        frame=t[t.participant_id.eq(sub)].sort_values(['run','trial_in_run']).copy()
        generated=simulate(s.pack(frame),natural[i],9,rng.random(len(frame)));valid=generated[:,3]>=0
        frame['chose_high']=np.where(valid,generated[:,3],np.nan)
        frame['feedback_observed']=generated[:,4].astype(bool)
        frame['observed_reciprocation']=np.where(generated[:,4]>0,generated[:,5],np.nan)
        frames.append(frame)
    truth=dict(replicate=rep,scenario='null' if rep<4 else 'known_effect',beta=beta.tolist(),natural=natural.tolist(),
        ids=meta['ids'],parameter_names=meta['parameter_names'],anchor=anchor,
        note='Task-conditioned recovery with population locations/scales anchored to the accepted fit; fresh participant effects. Four null and four signed 0.3 latent-unit/20-year scenarios; not SBC.')
    return pd.concat(frames,ignore_index=True),truth,anchor


def recovery_tables(fit,natural,truth,meta,dest):
    if meta['ids']!=truth['ids'] or natural.shape[1:]!=np.asarray(truth['natural']).shape:
        raise ValueError('Recovery identity mismatch')
    slopes=np.c_[fit.stan_variable('beta')[:,:,0],fit.stan_variable('beta_ext')[:,:,0]];rows=[];subjects=[]
    for j,name in enumerate(meta['parameter_names']):
        summary=s.summarize(slopes[:,j]);actual=truth['beta'][j]
        rows.append(dict(parameter=name,truth=actual,covered=summary['ci_low']<=actual<=summary['ci_high'],
                         excludes_zero=summary['ci_low']>0 or summary['ci_high']<0,**summary))
        actual_subject=np.asarray(truth['natural'])[:,j];estimated=natural[:,:,j].mean(axis=0)
        lo,hi=np.quantile(natural[:,:,j],[.025,.975],axis=0)
        subjects.append(dict(parameter=name,rank_correlation=float(spearmanr(actual_subject,estimated).statistic),
                             rmse=float(np.sqrt(np.mean((actual_subject-estimated)**2))),
                             interval_coverage=float(np.mean((actual_subject>=lo)&(actual_subject<=hi)))))
    pd.DataFrame(rows).to_csv(dest/'recovery_age.tsv',sep='\t',index=False)
    pd.DataFrame(subjects).to_csv(dest/'recovery_participants.tsv',sep='\t',index=False)


def accepted_tables(folder):
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='complete' or not status['passed']:return status,{}
    for file,h in status['output_sha256'].items():
        if s.fs.sha(folder/file)!=h:raise ValueError('Published closeout table changed: '+str(folder/file))
    return status,{f:pd.read_csv(folder/f,sep='\t') for f in status['output_sha256']}


def paired_contrast(left,right,seed,boot):
    x=left.set_index('participant_id').sort_index();y=right.set_index('participant_id').sort_index()
    if x.index.duplicated().any() or y.index.duplicated().any() or not x.index.equals(y.index) or not np.array_equal(x.n,y.n):
        raise ValueError('Unmatched heldout participants/counts')
    delta=(x.log_loss-y.log_loss).to_numpy()
    if not np.isfinite(delta).all():raise ValueError('Nonfinite heldout loss')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(delta),(boot,len(delta)))
    lo,hi=np.quantile(delta[indices].mean(axis=1),[.025,.975])
    return dict(participants=len(delta),choices=int(x.n.sum()),delta_log_loss=float(delta.mean()),ci_low=lo,ci_high=hi)


def report(cfg,c):
    out,_=q.paths(c);inventory=[];tables={};recovery=[];participant_recovery=[];age=[];curves=[];predictive=[]
    for entry in q.entries()+q.recovery_entries():
        folder=out/'fits'/entry['name']
        if not (folder/'status.json').exists():
            inventory.append(dict(name=entry['name'],status='not_run'));continue
        state,frames=accepted_tables(folder);inventory.append({key:state.get(key) for key in ['name','status','divergences','max_rhat','min_bulk_ess','min_tail_ess','min_bfmi','max_depth_hits']})
        tables[entry['name']]=frames
        for file,collection in [('recovery_age.tsv',recovery),('recovery_participants.tsv',participant_recovery),('age_effects.tsv',age),('age_curves.tsv',curves),('predictive_checks.tsv',predictive)]:
            if file in frames:collection.append(frames[file].assign(name=entry['name']))
    pd.DataFrame(inventory).to_csv(out/'diagnostic_inventory.tsv',sep='\t',index=False)
    score_frames={name:f['heldout_participants.tsv'] for name,f in tables.items() if 'heldout_participants.tsv' in f}
    oldfolder=a.paths(c)[0]/'fits'/q.ACCEPTED;st=json.loads((oldfolder/'status.json').read_text());score=oldfolder/'heldout_participants.tsv'
    if st['status']!='complete' or s.fs.sha(score)!=st['output_sha256'][score.name]:raise ValueError('Accepted bias reference changed')
    score_frames[q.ACCEPTED]=pd.read_csv(score,sep='\t')
    contrasts=[]
    pairs=[('Train_H7_zero_amount_v1_retry1','Train_H7_zero_bias_v1_retry1'),
           ('Train_HPreference_zero_amount_v1_retry1',q.ACCEPTED),
           ('Train_H7_zero_bias_v1_retry1',q.ACCEPTED),
           ('Train_HPreference_zero_bias_age_closeout1',q.ACCEPTED)]
    for left,right in pairs:
        if left in score_frames and right in score_frames:
            contrasts.append(dict(left=left,right=right,**paired_contrast(score_frames[left],score_frames[right],cfg['seed'],cfg['bootstrap'])))
    contrast=pd.DataFrame(contrasts,columns=['left','right','participants','choices','delta_log_loss','ci_low','ci_high'])
    contrast.to_csv(out/'heldout_contrasts.tsv',sep='\t',index=False)
    summary=[]
    for name,frame in score_frames.items():
        if len(frame)!=304 or int(frame.n.sum())!=12494:raise ValueError('Unexpected paired heldout cohort')
        summary.append(dict(name=name,participants=len(frame),choices=int(frame.n.sum()),log_loss=frame.log_loss.mean(),brier=frame.brier.mean()))
    pd.DataFrame(summary).to_csv(out/'heldout_models.tsv',sep='\t',index=False)
    for collection,file in [(age,'age_effects_all.tsv'),(curves,'age_curves_all.tsv'),(recovery,'recovery_age_all.tsv'),(participant_recovery,'recovery_participants_all.tsv'),(predictive,'predictive_checks_all.tsv')]:
        if collection:pd.concat(collection,ignore_index=True).to_csv(out/file,sep='\t',index=False)
    render(out,contrast,age,curves,predictive,recovery)
    accepted=sum(row.get('status')=='complete' for row in inventory)
    text=f'''# Full-cohort closeout: results awaiting scientific review

{accepted} of 16 scoped fits are accepted/completed. See diagnostic_inventory.tsv; failed fits do not contribute scientific summaries. Three amount-stage retries, five age/control/sensitivity fits, and eight simulated-data fits were planned. Missing comparisons remain unanswered.

Age is included jointly in all six population parameter locations: learning rate (logit), inverse temperature (log), friend and stranger preferences, zero-option coefficient, and general choice bias. Age is (years - 50)/20, with independent Normal(0, 0.5) slope priors. The wider-prior fit uses SD 1. Population random effects and likelihood otherwise retain the accepted model. Sensitivity excludes the five participants flagged by the frozen N=338 rule.

Age variance fraction is beta^2 Var(age_z)/(beta^2 Var(age_z)+tau^2), calculated separately per latent parameter in each posterior draw. It is a model-based, cross-sectional variance decomposition, not a fraction of observed choice variance explained or a causal aging effect. Curves transform the population location at zero random effect; they are not population-marginal means. No-age/full-age latent tau comparisons alone are not a valid variance decomposition.

Run-2 comparisons use the same 304 people and 12,494 choices. Negative left-minus-right log loss favors the left model; intervals bootstrap whole people 2,000 times, conditioning on fitted predictions. This is exploratory reuse of run 2 after previous residual inspection, not untouched validation or prediction for new participants. Full fits use all 343 participants. The preferred bias model anchors the age analysis; a supported amount extension would require an additional age robustness fit before attributing effects to specific parameters.

Predictive checks generate choices and feedback histories on the actual offers and schedules, preserving observed missingness. Age predictive checks distinguish conditional expected behavior from generated choices. Joint prior checks draw all parameters and include the wider age prior; inspect extreme rates rather than interpreting a finite simulation as scientific prior approval.

Recovery uses four zero-age and four known signed age-effect datasets (0.3 latent units per 20 years), with fresh participant effects and population locations/scales anchored to the accepted full-age fit. This is a targeted identification screen, not simulation-based calibration or a precise false-positive/coverage estimate. Inspect each parameter's slope bias, interval coverage and participant rank recovery; do not promote weakly recovered parameters to mechanistic claims. Any divergence or poor mixing remains a failure. No automatic further retries.

Finish criteria: assess amount-versus-bias contrasts; inspect exact-offer and age posterior predictive checks; compare primary age slopes against the N=338 and wider-prior fits; review recovery before interpreting parameters. A passed sampler is necessary but does not establish adequate model fit. Persistent misspecification should limit conclusions rather than trigger an open-ended model search.
'''
    (out/'README.md').write_text(text)


def render(out,contrast,age,curves,predictive,recovery):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    labels={'alpha':'Learning rate','kappa':'Choice sensitivity','preference_friend':'Friend preference',
            'preference_stranger':'Stranger preference','gamma0':'Zero-option effect','choice_bias':'General choice bias'}
    if len(contrast):
        contrast_labels=[]
        for left,right in zip(contrast.left,contrast.right):
            if 'age_closeout' in left: label='Preference + bias: age minus no age'
            elif 'amount' in left: label=('H7' if '_H7_' in left else 'Preference')+': amount minus bias'
            else: label='Bias models: H7 minus preference'
            contrast_labels.append(label)
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained');y=np.arange(len(contrast))
        ax.hlines(y,contrast.ci_low,contrast.ci_high);ax.scatter(contrast.delta_log_loss,y);ax.axvline(0,color='.5',ls='--')
        ax.set(yticks=y,yticklabels=contrast_labels,xlabel='Run-2 log loss difference (negative favors left)',title='Exploratory matched comparisons · 95% participant bootstrap intervals');save(fig,'heldout_comparisons')
    primary='Full_HPreference_zero_bias_age_closeout1'
    if age:
        frame=pd.concat(age,ignore_index=True);f=frame[(frame.name==primary)&frame.quantity.eq('latent_slope_per_20_years')]
        if len(f):
            fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');y=np.arange(len(f))
            for ax,quantity,label in zip(axes,['latent_slope_per_20_years','latent_age_variance_fraction'],['Latent slope per 20 years','Age fraction of latent between-person variance']):
                z=frame[(frame.name==primary)&frame.quantity.eq(quantity)].set_index('parameter').reindex(f.parameter)
                ax.hlines(y,z.ci_low,z.ci_high,color='#176b87');ax.scatter(z['mean'],y,color='#176b87');ax.set(yticks=y,yticklabels=[labels.get(p,p) for p in f.parameter],xlabel=label)
                if 'slope' in quantity:ax.axvline(0,color='.5',ls='--')
                else:ax.set_xlim(0,1)
            fig.suptitle('Hierarchical age associations · 343 participants · 95% posterior intervals');save(fig,'hierarchical_age')
            sensitivity=frame[frame.quantity.eq('latent_slope_per_20_years')&frame.name.isin([primary,'Sensitivity_HPreference_zero_bias_age_closeout1','Full_HPreference_zero_bias_age_wideprior_closeout1'])]
            fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
            for offset,(name,z) in zip([-.2,0,.2],sensitivity.groupby('name',sort=True)):
                z=z.set_index('parameter').reindex(f.parameter)
                points=ax.scatter(z['mean'],y+offset,label={primary:'Primary (N=343)','Sensitivity_HPreference_zero_bias_age_closeout1':'QC sensitivity (N=338)','Full_HPreference_zero_bias_age_wideprior_closeout1':'Wider age prior (N=343)'}[name])
                ax.hlines(y+offset,z.ci_low,z.ci_high,color=points.get_facecolor()[0])
            ax.axvline(0,color='.5',ls='--');ax.set(yticks=y,yticklabels=[labels.get(p,p) for p in f.parameter],xlabel='Latent age slope per 20 years');ax.legend(fontsize=8);save(fig,'age_sensitivity')
    if curves:
        f=pd.concat(curves,ignore_index=True);f=f[f.name==primary]
        if len(f):
            fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
            for ax,(parameter,z) in zip(axes.flat,f.groupby('parameter',sort=False)):
                ax.fill_between(z.age,z.ci_low,z.ci_high,alpha=.2,color='#176b87')
                ax.plot(z.age,z['median'],color='#176b87');ax.set(title=labels.get(parameter,parameter),xlabel='Age (years)',ylabel='Natural parameter scale')
            fig.suptitle('Population locations by age · zero random effect · 95% posterior intervals')
            save(fig,'population_age_curves')
    if predictive:
        f=pd.concat(predictive,ignore_index=True);f=f[(f.name==primary)&f.history.eq('generative')&f.stratification.eq('partner_offer')&f.statistic.eq('mean')]
        if len(f):
            fig,ax=plt.subplots(figsize=(12,5),layout='constrained');x=np.arange(len(f))
            ax.vlines(x,f.ci_low,f.ci_high,color='#176b87');ax.plot(x,f.predicted_mean,'o',label='Replicated mean + 95% interval');ax.plot(x,f.observed,'x',label='Observed');ax.set(xticks=x,xticklabels=f.group.str.replace('_',' · '),ylim=(-.03,1.03),ylabel='High-choice rate',title='Full age model: exact-offer generative checks');ax.tick_params(axis='x',rotation=75);ax.legend();save(fig,'exact_offer_predictive_checks')
    if recovery:
        f=pd.concat(recovery,ignore_index=True);fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for ax,(parameter,z) in zip(axes.flat,f.groupby('parameter',sort=True)):
            x=np.arange(len(z));ax.vlines(x,z.ci_low,z.ci_high,color='#176b87');ax.plot(x,z['mean'],'o');ax.plot(x,z.truth,'x',color='#bd3d32');ax.set(title=labels.get(parameter,parameter),xticks=x,xticklabels=['0' if v==0 else f'{v:+.1f}' for v in z.truth],xlabel='True age slope',ylabel='Recovered slope');ax.axhline(0,color='.7',ls='--')
        fig.suptitle('Targeted recovery · accepted simulations only · 95% posterior intervals');save(fig,'age_recovery')
