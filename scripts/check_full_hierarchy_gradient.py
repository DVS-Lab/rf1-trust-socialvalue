"""Full posterior/gradient parity: priors, transforms, covariance and age included."""
from pathlib import Path
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from rf1_trust_socialvalue.hierarchical import stan_model,inputs,WORK

fast=stan_model();source=Path('stan/hierarchical_shared.stan').read_text()
reference_file=WORK/'full_reference_validation.stan'
if not reference_file.exists() or reference_file.read_text()!=source:reference_file.write_text(source)
reference=CmdStanModel(stan_file=str(reference_file.resolve()))
rows=[];rng=np.random.default_rng(8742)
cases=[(m,{}) for m in ['H2','H5','H8','HPreference','H7','H4']]
cases += [('H5',dict(age_terms=0)),('H5',dict(age_terms=2)),('H5',dict(bounded=True))]
for model,options in cases:
    d=inputs(model,**options)[0];k=d['K'];corr=np.eye(k)+.1*(np.ones((k,k))-np.eye(k))
    pars=dict(mu=d['mu_location'],tau=[.8]*k,beta=np.full((k,d['A']),.1).tolist(),
              L=np.linalg.cholesky(corr).tolist(),z=rng.normal(0,.3,(k,d['N'])).tolist())
    a=reference.log_prob(params=pars,data=d,jacobian=True,sig_figs=16).iloc[0]
    b=fast.log_prob(params=pars,data=d,jacobian=True,sig_figs=16).iloc[0]
    assert list(a.index)==list(b.index)
    error=np.max(np.abs(a-b));relative=np.max(np.abs(a-b)/(1+np.abs(a)))
    assert relative<1e-9,(model,options,error,relative)
    rows.append(dict(model=model,age_terms=d['A'],bounded=d['bounded_theta'],parameters=len(a)-1,
                     max_absolute_error=error,max_scaled_error=relative))
pd.DataFrame(rows).to_csv('results/tables/full_hierarchy_gradient_validation.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
