"""Resumable full/train fits; sampling failures are retained, never concealed."""
from pathlib import Path
import json
import pandas as pd
from rf1_trust_socialvalue.hierarchical import run_one,sample,run_name,collect
from rf1_trust_socialvalue.hierarchical_figures import trace_figure

jobs=[(m,False,1,False,1) for m in ['HPreference','H7','H4','H2','H5','H8']]
jobs += [('H5',False,1,True,1),('H5',False,2,False,1),('H5',False,1,False,1.5)]
jobs += [(m,True,a,False,1) for m in ['H2','H5','H8','HPreference','H7'] for a in [1,0]]
for model,train,age,bounded,prior in jobs:
    name=run_name(model,train,age,bounded,prior_scale=prior)
    if age==2:
        d=pd.read_csv('results/tables/diagnostics_H5_full_age.csv')
        if not d.passed.all():raise RuntimeError('Resolve linear H5 convergence before quadratic age fit')
    print('START '+name,flush=True)
    run_one(model,train,age,bounded,prior)
    if not train:
        fit,*_=sample(model,train,age,bounded,prior_scale=prior)
        trace_figure(fit,model,name)
    print('COMPLETE '+name,flush=True)
collect()
