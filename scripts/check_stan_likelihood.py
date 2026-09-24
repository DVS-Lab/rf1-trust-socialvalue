"""Compile the SAME Stan likelihood function and compare it against the audited engine."""
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel
from rf1_trust_socialvalue.hierarchical import stan_model,SPECS,WORK
from rf1_trust_socialvalue.models import trajectory

stan_model()
source=Path('stan/hierarchical_shared.stan').read_text().split('\ndata {')[0]
source+='''
data { int T; int K; vector[K] par; int code;
 array[T] int partner; vector[T] gap; array[T] int feedback;
 vector[T] outcome; vector[3] ratings; }
generated quantities { vector[T] logits=rl_logits(par,code,1,T,partner,gap,feedback,outcome,ratings); }
'''
p=WORK/'likelihood_check.stan'
if not p.exists() or p.read_text()!=source:p.write_text(source)
sm=CmdStanModel(stan_file=str(p.resolve()))
# Include both feedback signs, zero-choice censoring, a run change and all partners.
a=np.array([[0,0,2,1,1,1,1,1],[1,2,4,1,1,0,0,1],[0,0,8,0,0,0,1,1],[2,2,8,0,1,1,0,2],[0,2,8,1,1,0,0,2],[1,0,4,0,0,1,1,2],[0,2,4,1,1,1,1,2]],float)
ratings=np.array([.9,.1,.5]);results=[]
for model,(base,code,names,kinds) in SPECS.items():
    params=dict(alpha=.23,kappa=.61,theta=1.4,theta_stranger=.8,preference_stranger=-.4,alpha_negative=.67)
    data=dict(T=len(a),K=len(names),par=[params[k] for k in names],code=code,partner=(a[:,0]+1).astype(int).tolist(),gap=(a[:,2]-a[:,1]).tolist(),feedback=a[:,4].astype(int).tolist(),outcome=a[:,5].tolist(),ratings=ratings.tolist())
    fit=sm.sample(data=data,chains=1,iter_sampling=1,fixed_param=True,seed=24,output_dir=str((WORK/'likelihood_check_draws').resolve()),show_progress=False,sig_figs=16)
    z=fit.stan_variable('logits')[0];expected=trajectory(a,base,params,ratings)[:,1]
    error=np.max(np.abs(1/(1+np.exp(-z))-expected));assert error<1e-12,(model,error)
    results.append(f'{model}: maximum probability error {error:.3g}')
Path('results/stan_likelihood_validation.txt').write_text('\n'.join(results)+'\n')
print('\n'.join(results))
