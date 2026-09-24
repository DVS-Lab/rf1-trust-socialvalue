"""Second-pass figures, each generated from committed small result tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plotting import style,save,heading,COLORS
from .hierarchical import TABLE,collect


def figures():
    style();collect()
    if (TABLE/'hierarchical_parameter_summary.csv').exists():
        d=pd.read_csv(TABLE/'hierarchical_parameter_summary.csv');d=d[d.run.eq('H5_full_age')]
        mle=pd.read_csv(TABLE/'model_fits_theta10.csv');mle=mle[mle.model.eq('M5')].set_index('participant_id')
        fig,axs=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
        for ax,p in zip(axs,['alpha','kappa','theta']):
            q=d[d.parameter.eq(p)].set_index('participant_id').join(mle[[p]])
            ax.vlines(q[p],q.ci_low,q.ci_high,alpha=.18,color='#157F86',lw=.7)
            ax.scatter(q[p],q['mean'],s=16,alpha=.7,color='#157F86')
            lim=max(q[p].max(),q['mean'].max());ax.plot([0,lim],[0,lim],ls='--',color='#73829A')
            ax.set(xlabel='Individual MLE (theta ceiling 10)',ylabel='Hierarchical posterior mean',title=p)
        heading(fig,'What does partial pooling change?','H5 linear-age hierarchy; vertical lines show individual 95% credible intervals. Different priors and estimators contribute to the change.')
        save(fig,'16_hierarchical_shrinkage')
    if (TABLE/'hierarchical_age_curves.csv').exists():
        curves=pd.read_csv(TABLE/'hierarchical_age_curves.csv');d=curves[curves.run.eq('H5_full_age')]
        fig,axs=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
        for ax,p in zip(axs,['alpha','kappa','theta']):
            q=d[d.parameter.eq(p)];ax.plot(q.age,q['median'],color='#157F86');ax.fill_between(q.age,q.ci_low,q.ci_high,alpha=.18,color='#157F86');ax.set(xlabel='Age (years)',ylabel=p,title=p)
        heading(fig,'Age effects estimated inside the hierarchy','H5: parameters for a participant with zero latent deviation from the age-specific mean; posterior medians and 95% credible intervals.')
        save(fig,'17_hierarchical_age_parameters')
        fig,axs=plt.subplots(1,2,figsize=(11.8,5),layout='constrained')
        for run,color,label in [('H5_full_age','#157F86','Positive, unbounded theta'),('H5_full_age_bounded','#D59437','Theta bounded at 10')]:
            q=curves[(curves.run==run)&(curves.parameter=='friend_value_probability_effect')]
            if len(q):axs[0].plot(q.age,q['median'],color=color,label=label);axs[0].fill_between(q.age,q.ci_low,q.ci_high,alpha=.13,color=color)
        axs[0].set(xlabel='Age (years)',ylabel='Friend-value probability effect',ylim=(0,1));axs[0].legend()
        individuals=pd.read_csv(TABLE/'hierarchical_parameter_summary.csv');q=individuals[(individuals.run=='H5_full_age')&(individuals.parameter=='friend_value_probability_effect')]
        axs[1].vlines(q.age,q.ci_low,q.ci_high,color='#157F86',alpha=.18,lw=.7);axs[1].scatter(q.age,q['median'],color='#157F86',s=17,alpha=.6)
        axs[1].set(xlabel='Age (years)',ylabel='Individual friend-value effect',ylim=(0,1))
        heading(fig,'Social value expressed on the choice scale','Turn the friend-value term on versus off at P=0.5, averaged over observed offer gaps. Left: typical-participant age curve; right: individual posteriors.')
        save(fig,'18_friend_value_probability_effect')
    if (TABLE/'hierarchical_predictive_summary.csv').exists():
        d=pd.read_csv(TABLE/'hierarchical_predictive_summary.csv');d=d[d.run.str.endswith('_full_age')]
        models=[m for m in ['H2','H5','H8','HPreference','H7','H4'] if m in d.model.unique()]
        fig,axs=plt.subplots(1,3,figsize=(12,5),layout='constrained')
        for j,p in enumerate(['friend','stranger','computer']):
            q=d[(d.category=='partner')&(d.label==p)&(d.measure=='high_probability')].set_index('model').reindex(models)
            axs[j].errorbar(np.arange(len(models)),q['mean'],yerr=[q['mean']-q.ci_low,q.ci_high-q['mean']],fmt='o',color=COLORS[p],capsize=4,label='Posterior predictive')
            axs[j].scatter(np.arange(len(models)),q.observed,marker='_',s=130,color='#203A4F',label='Observed')
            axs[j].set(title=p.title(),xticks=range(len(models)),xticklabels=models,ylim=(0,1),ylabel='High-choice probability');axs[j].tick_params(axis='x',rotation=40)
        axs[0].legend(loc='lower left');heading(fig,'Can the joint models reproduce partner differences?','Complete simulated datasets include posterior uncertainty, actual schedules, missing trials and choice-dependent feedback. H4 uses 103 participants; others use 111.')
        save(fig,'19_hierarchical_predictive_checks')
        fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for ax,m in zip(axs.flat,models):
            q=d[(d.model==m)&(d.category=='age_slope')]
            for j,p in enumerate(['friend','stranger','computer']):
                g=q[q.label==p].iloc[0];ax.errorbar(j,g['mean'],yerr=[[g['mean']-g.ci_low],[g.ci_high-g['mean']]],fmt='o',capsize=3,color=COLORS[p]);ax.scatter(j,g.observed,marker='_',s=100,color='#203A4F')
            ax.axhline(0,color='#73829A',lw=.7);ax.set(title=m,xticks=[0,1,2],xticklabels=['Friend','Stranger','Computer'],ylabel='High-choice slope per SD age')
        heading(fig,'Do posterior simulations reproduce age patterns?','Discrepancy check: continuous-age slopes of participant mean choices; points and 95% predictive intervals versus observed horizontal marks.')
        save(fig,'22_hierarchical_age_predictive_checks')
    if (TABLE/'hierarchical_heldout_summary.csv').exists():
        h=pd.read_csv(TABLE/'hierarchical_heldout_participant.csv');old=pd.read_csv(TABLE/'heldout_fits.csv');new=pd.read_csv(TABLE/'heldout_theta10.csv')
        old=old[~old.model.isin(['M4','M5','M6','M7'])];new['model']=new.model.str.split('_').str[0];mle=pd.concat([old,new]+([pd.read_csv(TABLE/'heldout_preference.csv')] if (TABLE/'heldout_preference.csv').exists() else []))
        pairs=[('H2','M2'),('H5','M5'),('H8','M8'),('HPreference','preference'),('H7','M7')]
        fig,axs=plt.subplots(1,2,figsize=(12,5.3),layout='constrained');labels=[]
        for j,(hm,mm) in enumerate(pairs):
            q=h[h.run.eq(hm+'_train_age')];r=mle[mle.model.eq(mm)]
            if not len(q):continue
            labels.append(hm)
            axs[0].scatter(j-.13,q.log_loss.mean(),color='#157F86',s=50,label='Hierarchical + age' if j==0 else None)
            no=h[h.run.eq(hm+'_train_noage')]
            if len(no):axs[0].scatter(j,no.log_loss.mean(),color='#D59437',s=45,label='Hierarchical, no age' if j==0 else None)
            if len(r):
                axs[0].scatter(j+.13,r.heldout_log_loss.mean(),color='#73829A',s=50,label='Individual MLE' if j==0 else None)
                z=q.set_index('participant_id').log_loss-r.set_index('participant_id').heldout_log_loss
                axs[1].scatter(np.full(len(z),j)+np.random.default_rng(j).uniform(-.12,.12,len(z)),z,color='#157F86',s=12,alpha=.35)
                axs[1].scatter(j,z.mean(),marker='D',color='#203A4F',s=35)
        axs[0].set(xticks=range(len(pairs)),xticklabels=[p[0] for p in pairs],ylabel='Mean held-out log loss (lower is better)');axs[0].legend(fontsize=8)
        axs[1].axhline(0,color='#73829A',lw=.8);axs[1].set(xticks=range(len(pairs)),xticklabels=[p[0] for p in pairs],ylabel='Hierarchical minus MLE log loss',yscale='symlog');axs[1].text(.03,.03,'Negative values favor hierarchy\nSymmetric log scale retains extreme errors',transform=axs[1].transAxes,fontsize=8)
        heading(fig,'Does partial pooling improve future-choice prediction?','Joint training-only posterior; parameters frozen during held-out scoring, beliefs continue updating. MLE social-value bounds are 10.')
        save(fig,'20_hierarchical_heldout')
    if (TABLE/'hierarchical_recovery_by_dataset.csv').exists():
        d=pd.read_csv(TABLE/'hierarchical_recovery_by_dataset.csv');fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for ax,p in zip(axs.flat[:4],['alpha','kappa','theta','friend_value_probability_effect']):
            for j,condition in enumerate(['zero','positive','negative']):
                for method,shift,color in [('hierarchical',-.12,'#157F86'),('MLE_theta10',.12,'#D59437')]:
                    q=d[(d.level=='participant')&(d.parameter==p)&(d.condition==condition)&(d.method==method)]
                    ax.scatter(j+shift+np.linspace(-.04,.04,len(q)),q.rmse,color=color,s=25,alpha=.65,label=method if j==0 else None)
            ax.set(title=p.replace('_',' '),xticks=[0,1,2],xticklabels=['Zero','Positive','Negative'],xlabel='True theta age slope',ylabel='Participant RMSE');ax.legend(fontsize=7)
        ax=axs.flat[4];q=d[(d.level=='population')&(d.parameter=='beta_theta')].sort_values(['condition','replicate'])
        for j,row in enumerate(q.to_dict('records')):
            ax.errorbar(j,row['mean'],yerr=[[row['mean']-row['ci_low']],[row['ci_high']-row['mean']]],fmt='o',color='#157F86',capsize=2);ax.scatter(j,row['true'],marker='_',color='#D59437',s=80)
        ax.set(xlabel='Simulated dataset',ylabel='Theta age coefficient',title='Age-slope recovery');ax.axhline(0,color='#73829A',lw=.7)
        axs.flat[5].axis('off');axs.flat[5].text(0,.9,'Paired recovery design\n\nSame ages and task schedules\nSame simulated choices\nSame missing-trial patterns\n\n5 datasets per age condition\n111 participants per dataset\n\nSmall simulation count: descriptive\ncoverage, not calibrated SBC.',va='top',fontsize=11)
        heading(fig,'Does hierarchy improve individual-difference measurement?','Each point is a complete simulated dataset; MLE and hierarchy see exactly the same choices. Lower RMSE indicates better recovery.')
        save(fig,'21_hierarchical_recovery')


def trace_figure(fit,model,name):
    from .hierarchical import SPECS
    params=SPECS[model][2];variables=[]
    for j,p in enumerate(params):variables.append((f'mu[{j+1}]',f'mean: {p}'))
    if 'beta[1,1]' in fit.column_names:
        for j,p in enumerate(params):variables.append((f'beta[{j+1},1]',f'age: {p}'))
    draws=fit.draws();fig,axs=plt.subplots(len(variables),1,figsize=(11,1.5*len(variables)+1.3),layout='constrained')
    for ax,(v,label) in zip(np.atleast_1d(axs),variables):
        j=fit.column_names.index(v)
        for c in range(draws.shape[1]):ax.plot(draws[:,c,j],alpha=.55,lw=.4)
        ax.set(ylabel=label)
    np.atleast_1d(axs)[-1].set(xlabel='Retained iteration')
    heading(fig,f'{name}: posterior trace check','Four independent chains; inspect alongside R-hat, ESS, divergences and BFMI in the diagnostic tables.')
    save(fig,'trace_'+name)
