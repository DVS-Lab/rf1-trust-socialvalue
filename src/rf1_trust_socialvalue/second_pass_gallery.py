"""Offline results gallery and final overview, generated only after analyses finish."""
from pathlib import Path
from html import escape
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plotting import style,save,heading
from .hierarchical import TABLE


def overview():
    style();root=Path('results/figures')
    for ext in ['png','pdf','svg']:
        old=root/f'00_results_overview.{ext}';backup=root/f'first_pass_00_results_overview.{ext}'
        if old.exists() and not backup.exists():shutil.copyfile(old,backup)
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    b=pd.read_csv(TABLE/'theta_bound_sensitivity.csv');b=b[b.base_model.eq('M5')]
    q=b.groupby('theta_upper').theta_at_upper.mean()
    axs[0,0].bar(q.index.astype(str),q,color=['#73829A','#157F86','#D59437'],width=.55)
    for j,v in enumerate(q):axs[0,0].text(j,v+.025,f'{v:.0%}',ha='center',fontweight='bold')
    axs[0,0].set(title='M5 estimates still follow the ceiling',xlabel='Theta upper bound',ylabel='Fraction exactly at the upper bound',ylim=(0,.7))
    a=pd.read_csv(TABLE/'age_partner_marginal_effects.csv')
    for contrast,color in [('friend - computer','#157F86'),('friend - stranger','#D59437')]:
        q=a[a.contrast.eq(contrast)];axs[0,1].plot(q.age,q.estimate,label=contrast.title(),color=color);axs[0,1].fill_between(q.age,q.ci_low,q.ci_high,color=color,alpha=.15)
    axs[0,1].set(title='Friend advantage across age',xlabel='Age (years)',ylabel='Difference in high-choice probability',ylim=(0,.5));axs[0,1].legend(fontsize=8)
    h=pd.read_csv(TABLE/'hierarchical_heldout_summary.csv');models=['H2','H5','H8','HPreference','H7']
    for offset,suffix,color,label in [(-.12,'_age','#157F86','Age in hierarchy'),(.12,'_noage','#D59437','No age')]:
        values=[h.loc[h.run.eq(m+'_train'+suffix),'log_loss'].iloc[0] for m in models]
        axs[1,0].scatter(np.arange(len(models))+offset,values,s=55,color=color,label=label)
    axs[1,0].set(title='Predicting later choices',xticks=range(len(models)),xticklabels=models,ylabel='Mean log loss (lower is better)');axs[1,0].legend(fontsize=8)
    r=pd.read_csv(TABLE/'hierarchical_recovery_summary.csv');r=r[r.level.eq('participant')]
    labels=['alpha','kappa','theta','friend_value_probability_effect'];ratios=[]
    for p in labels:
        q=r[r.parameter.eq(p)];ratios.append(q.loc[q.method.eq('hierarchical'),'rmse'].mean()/q.loc[q.method.eq('MLE_theta10'),'rmse'].mean())
    axs[1,1].bar(['Learning','Kappa','Theta','Choice effect'],ratios,color='#157F86',width=.55);axs[1,1].axhline(1,color='#73829A',ls='--')
    axs[1,1].set(title='Paired recovery on identical simulated data',ylabel='Hierarchical / MLE RMSE',ylim=(0,max(1.2,max(ratios)*1.15)))
    axs[1,1].text(.02,.95,'Below 1 favors hierarchy\n5 datasets × 3 age-effect conditions\nLow/moderate simulated theta',transform=axs[1,1].transAxes,va='top',fontsize=8)
    heading(fig,'Bounds, age, and partial pooling','OpenNeuro ds005123 v1.1.3 · 111 primary participants · unchanged audited task logic · mechanism requires prediction and recovery evidence.')
    save(fig,'00_results_overview')


def gallery():
    root=Path('results');old=root/'first_pass_index.html'
    if not old.exists():
        html=(root/'index.html').read_text().replace('analysis_summary.md','first_pass_analysis_summary.md').replace('figures/00_results_overview','figures/first_pass_00_results_overview')
        old.write_text(html)
    titles={'00':'Second-pass overview','14':'Theta bounds: 5, 10, and 20','15':'Behavioral age × partner analysis',
            '16':'MLE versus partially pooled estimates','17':'Age effects inside H5','18':'Friend value on the choice scale',
            '19':'Posterior predictive checks','20':'Temporal prediction','21':'Paired parameter and age-effect recovery','22':'Age-pattern predictive checks'}
    cards=[]
    for code,title in titles.items():
        files=sorted((root/'figures').glob(code+'_*.png'))
        if not files:continue
        file=files[0];stem=file.stem
        cards.append(f'<article id="f{code}"><h2>{escape(title)}</h2><a href="figures/{file.name}"><img src="figures/{file.name}" alt="{escape(title)}" loading="lazy"></a><p><a href="figures/{stem}.pdf">Vector PDF</a> · <a href="figures/{stem}.svg">SVG</a> · <a href="figures/{stem}.png">PNG</a></p></article>')
    diag=pd.read_csv(TABLE/'hierarchical_diagnostics.csv');current=diag[~diag.run.str.contains('attempt')]
    runs=current.groupby('run').passed.all();passing=int(runs.sum())
    h=pd.read_csv(TABLE/'hierarchical_heldout_summary.csv');best=h.sort_values('log_loss').iloc[0]
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trust: bounds, age, and partial pooling</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f1f4f6;color:#243748;font:16px/1.65 system-ui,sans-serif}}header,main,footer{{max-width:1160px;margin:auto;padding:30px 28px}}header{{padding-top:55px}}.eyebrow{{color:#157f86;font-weight:700;font-size:12px;letter-spacing:.15em;text-transform:uppercase}}h1{{font-size:clamp(32px,5vw,54px);line-height:1.12;letter-spacing:-.03em}}h2{{font-size:23px}}a{{color:#147d84;text-underline-offset:4px}}article,details{{background:white;border:1px solid #dce4e9;border-radius:12px;padding:24px;margin:25px 0}}img{{width:100%;height:auto}}.stats{{display:flex;flex-wrap:wrap;gap:16px}}.stat{{padding:16px 22px;background:white;border-radius:10px;min-width:170px}}.stat b{{display:block;font-size:28px}}.note{{border-left:4px solid #d59437;background:#fff8ed;padding:16px 22px}}nav a{{display:inline-block;margin:6px 18px 6px 0}}summary{{cursor:pointer;font-weight:650}}footer{{font-size:13px;color:#637382}}</style>
    <header><div class="eyebrow">Second-pass scientific analysis</div><h1>Trust across age.<br>What does partial pooling change?</h1>
    <p>Explore the bound audit, continuous-age behavioral analysis, joint Bayesian models, future-choice prediction, and paired recovery.</p>
    <div class="stats"><div class="stat"><b>111</b>primary participants</div><div class="stat"><b>52%</b>M5 estimates at theta=10</div><div class="stat"><b>{passing}/{len(runs)}</b>final runs passing diagnostics</div><div class="stat"><b>15</b>paired recovery datasets</div></div>
    <p class="note">Theta's scale remains sensitive to its ceiling. Age intervals and predictive comparisons matter more than the direction of a point estimate. Rating-based H4 remains secondary because rating timing is unverified.</p>
    <nav><a href="analysis_summary.md">Full report</a><a href="hierarchical_provenance.json">Run provenance</a><a href="tables/hierarchical_heldout_summary.csv">Prediction table</a><a href="tables/hierarchical_age_effects.csv">Age effects</a><a href="tables/hierarchical_recovery_summary.csv">Recovery table</a><a href="first_pass_index.html">Archived first pass</a></nav></header><main>'''
    text+='\n'.join(cards)
    traces=sorted((root/'figures').glob('trace_*.png'))
    text+='<details><summary>Sampler trace figures</summary><ul>'+''.join(f'<li><a href="figures/{p.name}">{escape(p.stem.removeprefix("trace_"))}</a></li>' for p in traces)+'</ul></details>'
    text+='</main><footer>OpenNeuro ds005123 v1.1.3 · DOI 10.18112/openneuro.ds005123.v1.1.3 · Behavioral data only. PNG, SVG, and vector PDF figures are available above. Raw data and sampler chains are excluded from the publication.</footer></html>'
    (root/'index.html').write_text(text)
