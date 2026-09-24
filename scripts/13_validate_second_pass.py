"""Validate the expanded analysis without conflating it with historical results."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--hierarchical',action='store_true');args=p.parse_args()
t=Path('results/tables')
b=pd.read_csv(t/'theta_bound_sensitivity.csv')
assert len(b)==436*3
assert set(b.theta_upper)=={5,10,20}
for bound,g in b.groupby('theta_upper'):
    assert g.theta.between(0,bound+1e-8).all()
    assert g.loc[g.base_model.eq('M7'),'theta_stranger'].between(0,bound+1e-8).all()
assert b.converged.all()
ll=b.pivot(index=['participant_id','base_model'],columns='theta_upper',values='log_likelihood')
assert (ll[10]-ll[5]>=-1e-4).all()
assert (ll[20]-ll[10]>=-1e-4).all()
r=pd.read_csv(t/'model_fits_theta10.csv');assert len(r)==983
assert r.loc[r.model.isin(['M4','M5','M6','M7']),'theta_upper'].eq(10).all()
a=pd.read_csv(t/'age_partner_gee.csv');assert a.n_subjects.eq(111).all()
assert np.isfinite(a[['estimate','se','ci_low','ci_high']]).all().all()
bootstrap=pd.read_csv(t/'age_partner_bootstrap.csv');assert bootstrap.successful_replicates.min()>=475
rec=pd.read_csv(t/'parameter_recovery_theta10.csv')
assert rec.groupby(['participant_id','model','parameter']).iteration.nunique().eq(50).all()
old=pd.read_csv(t/'model_fits.csv').query("model=='M3'").set_index('participant_id')
new=pd.read_csv(t/'bound_fits.csv').query("model=='M3_phi10'").set_index('participant_id')
assert (new.log_likelihood-old.log_likelihood>=-1e-4).all()
grad=pd.read_csv(t/'analytic_gradient_validation.csv')
assert grad.likelihood_error.max()<1e-9 and grad.max_scaled_gradient_error.max()<2e-7
full=pd.read_csv(t/'full_hierarchy_gradient_validation.csv')
assert full.max_scaled_error.max()<1e-9
if args.hierarchical:
    expected={m+'_full_age' for m in ['H2','H5','H8','HPreference','H7','H4']}
    expected|={'H5_full_age_bounded','H5_full_quadratic','H5_full_age_prior1.5'}
    expected|={m+'_train_'+a for m in ['H2','H5','H8','HPreference','H7'] for a in ['age','noage']}
    expected|={f'H5_recovery_{c}_{i}' for c in ['zero','positive','negative'] for i in range(5)}
    d=pd.read_csv(t/'hierarchical_diagnostics.csv');assert expected<=set(d.run)
    d=d[d.run.isin(expected)];failed=d[~d.passed]
    assert d.loc[d.run.eq('H4_full_age'),'n_subjects'].eq(103).all()
    assert d.loc[~d.run.eq('H4_full_age'),'n_subjects'].eq(111).all()
    assert d.chains.ge(4).all() and d.warmup_per_chain.ge(2000).all() and d.draws_per_chain.ge(2000).all()
    assert not len(failed),f'Diagnostics failed: {failed.run.unique().tolist()}'
    for name in ['hierarchical_parameter_summary','hierarchical_age_effects','hierarchical_predictive_summary']:
        tab=pd.read_csv(t/f'{name}.csv');assert np.isfinite(tab[['mean','ci_low','ci_high']]).all().all()
    h=pd.read_csv(t/'hierarchical_heldout_summary.csv');assert len(h)==10 and h.n_subjects.eq(111).all()
    assert np.isfinite(h[['log_loss','brier','accuracy']]).all().all()
    assert h.n_trials.eq(4017).all()
    participants=pd.read_csv(t/'hierarchical_heldout_participant.csv')
    assert not participants.duplicated(['run','participant_id']).any()
    assert participants.groupby('run').participant_id.nunique().eq(111).all()
    recovery=pd.read_csv(t/'hierarchical_recovery_summary.csv');assert recovery.datasets.eq(5).all()
print('Second-pass validation passed'+(' including all full/train/recovery posterior diagnostics.' if args.hierarchical else '.'))
