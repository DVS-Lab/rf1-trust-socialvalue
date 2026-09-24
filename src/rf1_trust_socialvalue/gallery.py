"""Portable, offline figure gallery with links to vector figures and result tables."""
from pathlib import Path
from html import escape
import pandas as pd


def build_gallery():
    root=Path('results')
    captions={
        '00':'Main findings at a glance', '01':'Trust behavior by partner',
        '02':'Choices across offered amounts', '03':'Trustworthiness ratings: timing unverified',
        '04':'Behavior over time', '05':'Participant-level model comparisons',
        '06':'Parameter estimates and boundaries', '07':'Illustrative trialwise model trajectory',
        '08':'Observed and simulated behavior', '09':'Parameter recovery',
        '10':'Model recovery and competing explanations', '11':'Exploratory age analysis',
        '12':'Held-out prediction', '13':'Predictive checks across offers'}
    cards=[]
    for p in sorted((root/'figures').glob('*.png')):
        title=captions.get(p.name[:2],p.stem.replace('_',' ').capitalize());stem=p.stem
        cards.append(f'<article id="figure-{p.name[:2]}"><h2>{escape(title)}</h2><a href="figures/{p.name}"><img src="figures/{p.name}" alt="{escape(title)}" loading="lazy"></a><p class="downloads"><a href="figures/{stem}.png">PNG</a> <a href="figures/{stem}.pdf">Vector PDF</a> <a href="figures/{stem}.svg">SVG</a></p></article>')
    table=pd.read_csv(root/'tables/model_comparison.csv');table=table[(table['sample']=='primary')&(table.metric=='AICc')][['model','n','mean','best_share']]
    table.columns=['Model','N','Mean AICc','Best-fit share'];table['Mean AICc']=table['Mean AICc'].round(2);table['Best-fit share']=table['Best-fit share'].map(lambda x:f'{x:.1%}')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trust, learning, and social value</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f1f4f6;color:#233648;font:16px/1.65 system-ui,sans-serif}header,main,footer{max-width:1150px;margin:auto;padding:34px 28px}header{padding-top:60px;padding-bottom:15px}.eyebrow{color:#137e84;text-transform:uppercase;letter-spacing:.15em;font-size:12px;font-weight:750}h1{font-size:clamp(32px,5vw,54px);line-height:1.12;margin:14px 0 20px;letter-spacing:-.03em}h2{font-size:23px;line-height:1.3}p{max-width:900px}a{color:#167d85;text-underline-offset:4px}.lead{font-size:21px}.stats{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.stat{background:white;border:1px solid #dce4e9;border-radius:12px;padding:16px 24px;min-width:175px}.stat b{font-size:29px;display:block}.note{padding:18px 22px;border-left:4px solid #d59437;background:#fff8ed}.actions{display:flex;gap:20px;flex-wrap:wrap;margin-top:26px}article{background:#fff;margin:28px 0;padding:20px;border-radius:14px;border:1px solid #dde4e8}article h2{margin:5px 8px 22px}img{width:100%;height:auto;display:block}.downloads{margin:15px 8px 0;display:flex;gap:20px;font-size:13px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px 18px;border-bottom:1px solid #e2e8ec;text-align:left!important}th{background:#edf3f5}footer{font-size:13px;color:#6b7887}summary{cursor:pointer;font-weight:650}details{padding:20px;background:#fff;border:1px solid #dde4e8;border-radius:12px}nav{font-size:14px;margin:22px 0}nav a{margin-right:16px}
    </style><header><div class="eyebrow">Behavioral & computational analysis · September 2026</div><h1>Trust, learning,<br>and social value</h1><p class="lead">Friends elicit greater trust. The mechanism is less certain.</p><p>Partner-value models capture the observed choices, while asymmetric learning predicts future choices better on average. Generic partner preferences and nonlinear monetary utility remain important competing explanations.</p><div class="stats"><div class="stat"><b>113</b>behavioral participants</div><div class="stat"><b>111</b>primary model participants</div><div class="stat"><b>50</b>recovery repetitions per fit</div><div class="stat"><b>14</b>scientific figures</div></div><p class="note"><b>Interpretation matters.</b> Rating timing is unverified. Individual parameter estimates often hit bounds, and recovery is imperfect. Better fit does not uniquely demonstrate a social-reward mechanism.</p><div class="actions"><a href="analysis_summary.md">Full analysis report</a><a href="provenance.json">Provenance</a><a href="tables/model_fits.csv">Participant model fits</a><a href="tables/recovery_summary.csv">Recovery summary</a></div><nav><a href="#figure-01">Behavior</a><a href="#figure-05">Model comparisons</a><a href="#figure-09">Recovery</a><a href="#figure-12">Prediction</a></nav></header><main>'''
    doc+='\n'.join(cards[:1])+f'<details><summary>First-pass AICc comparison (theta≤5)</summary><p>Mean criterion, individual best-fit frequency, and held-out performance answer different questions.</p>{table.to_html(index=False,border=0)}</details>'+'\n'.join(cards[1:])
    doc+='</main><footer>OpenNeuro ds005123 v1.1.3 · DOI 10.18112/openneuro.ds005123.v1.1.3 · Behavioral data only; no imaging content used. Complete methods, exclusions, equations, and limitations are in the analysis report.</footer></html>'
    (root/'index.html').write_text(doc)
