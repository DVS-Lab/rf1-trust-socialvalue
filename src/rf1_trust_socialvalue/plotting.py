"""Publication figures: subject-level data, uncertainty, and explicit model limits."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .data import PARTNERS
from .models import LABELS

OUT=Path('results/figures');TABLE=Path('results/tables')
COLORS={'friend':'#157F86','stranger':'#D59437','computer':'#73829A'}
MODEL_COLORS={'M0':'#B8BEC7','M1':'#8F9CAA','M2':'#566980','M3':'#AB789B','M4':'#955C91','M5':'#168188','M6':'#74A398','M7':'#215D72','M8':'#D28B38'}
PRIMARY=['M0','M1','M2','M5','M6','M7','M8']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A8AFB6','axes.labelcolor':'#243242',
        'text.color':'#243242','xtick.color':'#52606D','ytick.color':'#52606D','figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white','grid.color':'#E6E9ED','grid.linewidth':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    OUT.mkdir(parents=True,exist_ok=True)


def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=320,bbox_inches='tight',metadata={'Creator':'rf1-trust-socialvalue'} if ext=='pdf' else None)
    plt.close(fig)


def heading(fig,title,subtitle):
    import textwrap
    width,height=fig.get_size_inches()
    fig.get_layout_engine().set(rect=(0,0,1,1-.95/height))
    fig.suptitle(title,x=.055,ha='left',y=1-.08/height,fontsize=19,fontweight='bold')
    fig.text(.055,1-.46/height,textwrap.fill(subtitle,width=int(width*14)),ha='left',va='top',fontsize=9.5,color='#647184')


def paired(ax,table,value,summary,scale=1):
    mat=table.pivot(index='participant_id',columns='partner',values=value).reindex(columns=PARTNERS)
    rng=np.random.default_rng(24)
    for _,row in mat.iterrows():ax.plot(range(3),row.to_numpy()*scale,color='#B9C2CA',lw=.45,alpha=.28,zorder=1)
    for i,p in enumerate(PARTNERS):
        vals=mat[p].dropna()*scale
        ax.scatter(i+rng.uniform(-.055,.055,len(vals)),vals,s=10,alpha=.35,color=COLORS[p],edgecolors='none',zorder=2)
        q=summary[summary.partner.eq(p)].iloc[0]
        ax.errorbar(i,q['mean']*scale,yerr=np.array([[q['mean']-q.ci_low],[q.ci_high-q['mean']]])*scale,color='#203A4F',marker='o',ms=8,capsize=5,lw=2,zorder=4)
    ax.set_xticks(range(3),['Friend','Stranger','Computer']);ax.grid(axis='y',zorder=0)


def behavior_figures():
    b=pd.read_csv(TABLE/'behavior_participant.csv');s=pd.read_csv(TABLE/'behavior_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(10.8,5),layout='constrained')
    paired(axs[0],b,'investment',s[s.measure.eq('investment')]);axs[0].set(ylabel='Mean investment ($)',ylim=(-.2,8.2))
    paired(axs[1],b,'high_probability',s[s.measure.eq('high_probability')]);axs[1].set(ylabel='Probability of choosing the higher offer',ylim=(-.03,1.03));axs[1].yaxis.set_major_formatter(PercentFormatter(1))
    heading(fig,'Trust is highest with friends','113 participants · paired individual means · dark points: mean and 95% participant-bootstrap CI')
    save(fig,'01_behavior_by_partner')
    d=pd.read_csv(TABLE/'behavior_offer.csv');pairs=sorted(d.offer_pair.unique(),key=lambda s:tuple(map(int,s.split('-'))))
    fig,ax=plt.subplots(figsize=(10.5,5),layout='constrained');rng=np.random.default_rng(32)
    for j,p in enumerate(PARTNERS):
        means=[];lo=[];hi=[]
        for pair in pairs:
            v=d[(d.partner==p)&(d.offer_pair==pair)].high_probability.to_numpy();draw=v[rng.integers(0,len(v),(3000,len(v)))].mean(axis=1)
            means.append(v.mean());q=np.quantile(draw,[.025,.975]);lo.append(v.mean()-q[0]);hi.append(q[1]-v.mean())
        ax.errorbar(np.arange(len(pairs))+(j-1)*.16,means,yerr=[lo,hi],fmt='o-',lw=1.8,ms=6,capsize=3,color=COLORS[p],label=p.title())
    ax.set(xticks=range(len(pairs)),xticklabels=['\\$'+p.replace('-',' vs \\$') for p in pairs],ylabel='Probability of choosing the higher offer',ylim=(0,1),xlabel='Offered investments');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y');ax.legend(ncol=3,loc='lower left')
    heading(fig,'The friend difference spans the offered amounts','Participant-weighted means and 95% bootstrap CIs · the $2 vs $8 pair is offered twice as often')
    save(fig,'02_offer_pair_behavior')
    r=pd.read_csv(TABLE/'ratings.csv');r=r[r.trait.eq(2)];fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    mat=r.pivot(index='participant_id',columns='partner',values='rating').reindex(columns=PARTNERS)
    for _,row in mat.iterrows():ax.plot(range(3),row,color='#CAD1D7',lw=.45,alpha=.3)
    for i,p in enumerate(PARTNERS):
        vals=mat[p].dropna();ax.scatter(i+rng.uniform(-.1,.1,len(vals)),vals,s=14,alpha=.45,color=COLORS[p]);ax.plot(i,vals.mean(),'o',ms=9,color='#243242')
    ax.set(xticks=range(3),xticklabels=[p.title() for p in PARTNERS],ylim=(-5.5,5.5),ylabel='Trustworthiness rating (−5 to +5)');ax.axhline(0,c='#AAB4BD',lw=.7)
    heading(fig,'Ratings favor friends, but their timing is unverified',f'{len(mat)} participants with one complete rating set · rating-weighted models are secondary')
    save(fig,'03_trustworthiness_ratings')
    d=pd.read_csv(TABLE/'behavior_time.csv');fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for p in PARTNERS:
        z=d[d.partner.eq(p)].groupby('trial_bin').high_probability.agg(['mean','sem'])
        ax.plot(z.index,z['mean'],'o-',color=COLORS[p],label=p.title(),lw=2);ax.fill_between(z.index,z['mean']-1.96*z['sem'],z['mean']+1.96*z['sem'],color=COLORS[p],alpha=.12)
    ax.set(xticks=range(1,7),xticklabels=['1–14','15–28','29–42','43–56','57–70','71–84'],ylim=(0,1),xlabel='Global decision number',ylabel='Probability of choosing the higher offer');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.axvline(3.5,color='#A8AFB6',ls='--',lw=.8);ax.text(3.55,.08,'Run boundary',fontsize=9,color='#647184');ax.legend(ncol=3);ax.grid(axis='y')
    heading(fig,'Friend-directed trust persists over time','Participant-weighted trajectories ±1.96 SEM · later bins contain participants with a second run')
    save(fig,'04_behavior_over_time')


def comparison_figures():
    d=pd.read_csv(TABLE/'model_deltas.csv');d=d[d['sample'].eq('primary')]
    fig,axs=plt.subplots(1,3,figsize=(14,5.4),layout='constrained');rng=np.random.default_rng(19)
    for ax,metric in zip(axs,['AIC','AICc','BIC']):
        for j,m in enumerate(PRIMARY):
            vals=d[(d.model==m)&(d.metric==metric)].delta
            ax.scatter(j+rng.uniform(-.2,.2,len(vals)),vals,s=9,alpha=.25,c=MODEL_COLORS[m],edgecolors='none')
            ax.plot(j,vals.median(),'_',color='#182F43',ms=17,mew=2)
        ax.set(xticks=range(len(PRIMARY)),xticklabels=PRIMARY,ylabel=f'Δ{metric} from each participant’s best model',title=metric);ax.grid(axis='y');ax.set_ylim(bottom=-2)
    heading(fig,'Model fit varies substantially across participants','Primary rating-free sample: N=111 · each dot is a participant · horizontal marks show medians · lower is better')
    save(fig,'05_model_comparison')
    f=pd.read_csv(TABLE/'model_fits.csv');fig,axs=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    for ax,param,title in zip(axs,['alpha','kappa','theta'],['Learning rate α','Inverse temperature κ','Reciprocation value θ']):
        models=['M2','M5','M7','M8'] if param!='theta' else ['M4','M5','M7']
        for j,m in enumerate(models):
            vals=f[f.model.eq(m)][param].dropna()
            ax.scatter(j+rng.uniform(-.18,.18,len(vals)),vals,s=13,alpha=.45,c=MODEL_COLORS[m],edgecolors='none');ax.plot(j,vals.median(),'_',color='#243242',ms=22,mew=2)
        ax.set(xticks=range(len(models)),xticklabels=models,title=title);ax.grid(axis='y')
        if param=='alpha':ax.set_ylim(-.03,1.03)
        if param=='theta':ax.set_ylim(-.15,5.15);ax.text(.02,.95,'M7: friend bonus',transform=ax.transAxes,va='top',fontsize=9)
        if param=='kappa':ax.set_yscale('symlog',linthresh=.1);ax.set_ylabel('Symlog scale; linear near zero')
    heading(fig,'Parameter estimates often reach their allowed bounds','Participant maximum-likelihood estimates · M4 is secondary · recovery determines which differences are interpretable')
    save(fig,'06_model_parameters')
    h=pd.read_csv(TABLE/'heldout_fits.csv');fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    order=PRIMARY+['M3','M4']
    for j,m in enumerate(order):
        g=h[h.model.eq(m)];v=g.heldout_log_loss
        axs[0].scatter(j+rng.uniform(-.18,.18,len(v)),v,s=10,alpha=.3,c=MODEL_COLORS[m]);axs[0].plot(j,v.mean(),'D',ms=5,c='#243242')
        axs[1].scatter(j+rng.uniform(-.18,.18,len(v)),g.heldout_brier,s=10,alpha=.3,c=MODEL_COLORS[m]);axs[1].plot(j,g.heldout_brier.mean(),'D',ms=5,c='#243242')
    axs[0].axhline(np.log(2),ls='--',lw=.8,c='#7B8591');axs[0].set(ylabel='Held-out log loss (lower is better)',yscale='symlog');axs[1].set(ylabel='Held-out Brier score (lower is better)')
    for ax in axs:ax.set_xticks(range(len(order)),order);ax.grid(axis='y')
    heading(fig,'Good in-sample fit does not guarantee prediction','Frozen training parameters; online feedback updates · diamonds: participant mean · M3/M4: rating-complete sample')
    save(fig,'12_heldout_prediction')


def trajectory_figure():
    t=pd.read_csv(TABLE/'trial_table.csv');f=pd.read_csv(TABLE/'model_fits.csv');p=pd.read_csv(TABLE/'model_fits_predictions.csv')
    eligible=f[(f.model=='M7')&(f.n>=75)&f.alpha.between(.03,.8)].copy();target=eligible.pseudo_r2_mcfadden.quantile(.75);sub=eligible.iloc[np.argmin(abs(eligible.pseudo_r2_mcfadden-target))].participant_id
    d=t[t.participant_id.eq(sub)].merge(p[(p.participant_id==sub)&(p.model=='M7')],on=['participant_id','global_trial'])
    fig,axs=plt.subplots(4,1,figsize=(12,8),sharex=True,layout='constrained',height_ratios=[.7,.8,1.2,1.4])
    for j,partner in enumerate(PARTNERS):
        g=d[d.partner.eq(partner)];color=COLORS[partner]
        axs[0].scatter(g.global_trial,np.full(len(g),2-j),s=22,color=color)
        seen=g[g.feedback_observed];axs[1].scatter(seen.global_trial,seen.observed_reciprocation,s=24,color=color)
        axs[2].plot(g.global_trial,g.belief,'o-',color=color,lw=1,ms=3,label=partner.title())
        axs[3].scatter(g.global_trial,g.probability,color=color,s=22)
    axs[0].set(yticks=[0,1,2],yticklabels=['Computer','Stranger','Friend'])
    axs[1].set(yticks=[0,1],yticklabels=['Defection','Reciprocation'],ylim=(-.3,1.3))
    axs[2].set(ylabel='Belief before feedback',ylim=(-.04,1.04));axs[2].legend(ncol=3,loc='lower right')
    axs[3].scatter(d.global_trial,d.chose_high.astype(float),marker='|',s=50,color='#243242',label='Actual choice');axs[3].set(ylabel='P(higher offer)',xlabel='Global decision number',ylim=(-.07,1.07));axs[3].legend(loc='center left')
    for ax in axs:ax.axvline(42.5,ls='--',c='#B5BEC8',lw=.8);ax.grid(axis='y')
    heading(fig,f'An illustrative partner-value trajectory · {sub}','M7 · Illustration selected near the 75th fit percentile among N≥75 and .03≤α≤.8 · beliefs shown at partner encounters')
    save(fig,'07_representative_trajectory')


def predictive_figures():
    d=pd.read_csv(TABLE/'predictive_checks.csv');models=['M2','M5','M7','M8','M4'];fig,axs=plt.subplots(1,3,figsize=(13,4.8),sharey=True,layout='constrained')
    for c,ax in enumerate(axs):
        for j,m in enumerate(models):
            row=d[(d.model==m)&(d.category=='partner')&(d.label==str(c))&(d.measure=='high_probability')].iloc[0]
            ax.errorbar(j,row.predicted,yerr=[[row.predicted-row.sim_low],[row.sim_high-row.predicted]],fmt='o',capsize=4,color=MODEL_COLORS[m],ms=7)
            ax.plot(j,row.observed,'_',ms=20,mew=2,c='#26384B')
        ax.set(xticks=range(len(models)),xticklabels=models,title=PARTNERS[c].title(),ylim=(0,1));ax.grid(axis='y');ax.yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_ylabel('Probability of choosing the higher offer')
    heading(fig,'Can the models reproduce partner-dependent behavior?','Points and intervals: simulated means and 95% conditional simulation ranges · dark ticks: observed means · M4 is secondary')
    save(fig,'08_observed_vs_simulated')
    fig,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained',sharey=True)
    pairs=['0-2','0-4','0-8','2-4','2-8','4-8']
    for c,ax in enumerate(axs):
        for j,m in enumerate(['M2','M7','M8']):
            q=d[(d.model==m)&(d.category=='offer')&(d.measure=='high_probability')].set_index('label').reindex([f'{c}:{pair}' for pair in pairs])
            ax.plot(range(6),q.predicted,'o-',color=MODEL_COLORS[m],ms=4,lw=1.5,label=m)
            if j==0:ax.plot(range(6),q.observed,'k_',ms=16,mew=2,label='Observed')
        ax.set(xticks=range(6),xticklabels=pairs,title=PARTNERS[c].title(),ylim=(0,1),xlabel='Offer pair ($)');ax.grid(axis='y')
    axs[0].set_ylabel('Probability of choosing the higher offer');axs[-1].legend()
    heading(fig,'Predictive checks across offer pairs','Participant-weighted observed choices and stochastic simulation means; fitted parameters remain fixed')
    save(fig,'13_predictive_offer_pairs')


def recovery_figures():
    r=pd.read_csv(TABLE/'parameter_recovery.csv');s=pd.read_csv(TABLE/'recovery_summary.csv')
    choices=[('M5','alpha'),('M5','kappa'),('M5','theta'),('M7','alpha'),('M7','kappa'),('M7','theta')]
    fig,axs=plt.subplots(2,3,figsize=(12,7.8),layout='constrained');rng=np.random.default_rng(1)
    for ax,(m,p) in zip(axs.flat,choices):
        g=r[(r.model==m)&(r.parameter==p)];q=s[(s.model==m)&(s.parameter==p)].iloc[0]
        sel=rng.choice(len(g),min(1800,len(g)),replace=False);g=g.iloc[sel]
        ax.scatter(g.true,g.recovered,s=7,alpha=.13,color=MODEL_COLORS[m],edgecolors='none')
        lim=max(g.true.max(),g.recovered.max());ax.plot([0,lim],[0,lim],ls='--',lw=.8,c='#607081')
        ax.set(xlabel=f'Generating {p}',ylabel=f'Recovered {p}',title=f'{m} · mean r = {q.pearson:.2f} · RMSE = {q.rmse:.2f}')
    heading(fig,'Recovery limits the precision of individual parameters','50 stochastic repetitions per participant and model · points sampled for visibility · true values are empirical fitted parameters')
    save(fig,'09_parameter_recovery')
    d=pd.read_csv(TABLE/'model_recovery_confusion.csv');models=list(d.generating.unique());models=sorted(models,key=lambda m:(m not in PRIMARY,PRIMARY.index(m) if m in PRIMARY else m))
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for ax,metric in zip(axs,['AICc','BIC']):
        mat=d[d.metric.eq(metric)].pivot(index='generating',columns='fitted',values='selection_probability').reindex(index=models,columns=models)
        im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=1)
        for i in range(len(models)):
            for j in range(len(models)):
                val=mat.iloc[i,j]
                ax.text(j,i,f'{100*val:.0f}',ha='center',va='center',fontsize=8,color='white' if val>.55 else '#243242')
        ax.set(xticks=range(len(models)),xticklabels=models,yticks=range(len(models)),yticklabels=models,xlabel='Selected model',ylabel='Generating model',title=metric)
        ax.tick_params(axis='x',rotation=60)
    fig.colorbar(im,ax=axs,shrink=.65,label='Selection probability')
    heading(fig,'Distinct mechanisms can produce overlapping choices','Model recovery on actual schedules · values are selection percentages · ties share credit · rating-complete participants')
    save(fig,'10_model_recovery')


def age_figure():
    f=pd.read_csv(TABLE/'model_fits.csv');s=pd.read_csv(TABLE/'sample_audit.csv');d=f[f.model.eq('M5')].merge(s[['participant_id','age']],on='participant_id')
    fig,ax=plt.subplots(figsize=(9,5.1),layout='constrained');rng=np.random.default_rng(6)
    ax.scatter(d.age,d.theta+rng.uniform(-.025,.025,len(d)),s=28,alpha=.6,color=MODEL_COLORS['M5'],edgecolor='white',lw=.4)
    x=np.linspace(d.age.min(),d.age.max(),100);p=np.polyfit(d.age,d.theta,1);ax.plot(x,np.polyval(p,x),color='#26394C',lw=2)
    ax.set(xlabel='Age (years)',ylabel='Friend-specific value θ (M5)',ylim=(-.15,5.15));ax.grid(axis='y')
    heading(fig,'Age–parameter relationships remain exploratory','Unpooled participant estimates · line: descriptive linear association · boundary estimates and recovery limit interpretation')
    save(fig,'11_age_and_social_value')


def overview():
    b=pd.read_csv(TABLE/'behavior_participant.csv');s=pd.read_csv(TABLE/'behavior_summary.csv');h=pd.read_csv(TABLE/'heldout_summary.csv');mc=pd.read_csv(TABLE/'model_comparison.csv');rec=pd.read_csv(TABLE/'recovery_summary.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    paired(axs[0,0],b,'high_probability',s[s.measure.eq('high_probability')]);axs[0,0].set(ylim=(-.04,1.04),ylabel='P(higher offer)',title='A  ·  Strong friend preference');axs[0,0].yaxis.set_major_formatter(PercentFormatter(1))
    z=mc[(mc['sample']=='primary')&(mc.metric=='AICc')].set_index('model').reindex(PRIMARY)
    axs[0,1].barh(PRIMARY,z.best_share,color=[MODEL_COLORS[m] for m in PRIMARY]);axs[0,1].invert_yaxis();axs[0,1].set(xlabel='Share of participants best fit by each model',title='B  ·  In-sample support is heterogeneous');axs[0,1].xaxis.set_major_formatter(PercentFormatter(1))
    z=h.set_index('model').reindex(PRIMARY);axs[1,0].plot(z.log_loss,range(len(PRIMARY)),'o',color='#26394C');axs[1,0].set(yticks=range(len(PRIMARY)),yticklabels=PRIMARY,xlabel='Mean held-out log loss · lower is better',title='C  ·  Prediction favors asymmetric learning');axs[1,0].invert_yaxis();axs[1,0].axvline(np.log(2),ls='--',lw=1,color='#9AA6B3');axs[1,0].grid(axis='x')
    z=rec[(rec.model=='M5')].set_index('parameter').reindex(['alpha','kappa','theta']);axs[1,1].bar(['α','κ','θ'],z.pearson,color=['#7899A6','#497A8C','#168188'],width=.55);axs[1,1].set(ylim=(-.05,1),ylabel='Mean true–recovered Pearson correlation',title='D  ·  Individual estimates need caution');axs[1,1].axhline(.7,ls=':',lw=1,color='#A1ABB5');axs[1,1].grid(axis='y')
    heading(fig,'Trust, learning, and social value','ds005123 v1.1.3 · behavioral N=113; primary modeling N=111 · rating timing unverified · no imaging data used')
    save(fig,'00_results_overview')


def make_figures():
    style();behavior_figures();comparison_figures();trajectory_figure();predictive_figures();recovery_figures();age_figure();overview()
    print('Created 14 figures in PNG, PDF, and SVG',flush=True)
