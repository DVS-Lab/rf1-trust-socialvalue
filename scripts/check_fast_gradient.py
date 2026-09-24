"""Exact analytic-gradient validation against independent numerical derivatives."""
from pathlib import Path
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from rf1_trust_socialvalue.hierarchical import stan_model,inputs,SPECS
from rf1_trust_socialvalue.models import trajectory

stan_model()
source=Path('stan/hierarchical_fast.stan').read_text().split('\ndata {')[0]
source += "\ndata { int T; int K; int code; array[T] int y; array[T] int partner; vector[T] gap; array[T] int feedback; vector[T] outcome; vector[3] ratings; }\nparameters { vector[K] par; }\nmodel { target += rl_fast_lpmf(y | par,code,1,T,partner,gap,feedback,outcome,ratings); }\n"
file=Path('work/hierarchical/fast_gradient_check.stan')
if not file.exists() or file.read_text()!=source:file.write_text(source)
m=CmdStanModel(stan_file='work/hierarchical/fast_gradient_check.stan',user_header=str(Path('stan/rl_fast.hpp').resolve()),stanc_options={'allow-undefined':True})
arrays=inputs('H5')[2];rng=np.random.default_rng(9401);rows=[]
for model,(base,code,names,kinds) in SPECS.items():
    for case,sub in enumerate(list(arrays)[::23]):
        a=arrays[sub];a=a[a[:,3]>=0]
        par=np.array([rng.uniform(.1,.9) if k==1 else rng.uniform(.1,1.5) if k==2 else rng.uniform(.1,8) if k==3 else rng.uniform(-2,2) for k in kinds])
        ratings=rng.uniform(0,1,3)
        d=dict(T=len(a),K=len(par),code=code,y=a[:,3].astype(int).tolist(),partner=(a[:,0]+1).astype(int).tolist(),gap=(a[:,2]-a[:,1]).tolist(),feedback=a[:,4].astype(int).tolist(),outcome=a[:,5].tolist(),ratings=ratings.tolist())
        result=m.log_prob(params={'par':par.tolist()},data=d,jacobian=False,sig_figs=16).iloc[0]
        def logp(x):return -trajectory(a,base,dict(zip(names,x)),ratings)[:,3].sum()
        exact=result.iloc[1:].to_numpy(float);numerical=[]
        for j in range(len(par)):
            h=1e-5;plus=par.copy();minus=par.copy();plus[j]+=h;minus[j]-=h
            numerical.append((logp(plus)-logp(minus))/(2*h))
        ll_error=abs(result['lp__']-logp(par));grad_error=np.max(abs(exact-numerical));relative=np.max(abs(exact-numerical)/(1+abs(exact)))
        assert ll_error<1e-9,(model,ll_error)
        assert relative<2e-7,(model,par,exact,numerical,relative)
        rows.append(dict(model=model,case=case,participant_id=sub,likelihood_error=ll_error,max_gradient_error=grad_error,max_scaled_gradient_error=relative))
p=Path('results/tables/analytic_gradient_validation.csv');pd.DataFrame(rows).to_csv(p,index=False)
print(pd.DataFrame(rows).groupby('model')[['likelihood_error','max_gradient_error','max_scaled_gradient_error']].max().to_string())
