"""Fail if results are incomplete or contain invalid probabilities/fit metrics."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd


def main():
    p=Path('results/tables');config=json.load(open('config/analysis.json'))
    fits=pd.read_csv(p/'model_fits.csv');held=pd.read_csv(p/'heldout_fits.csv');rob=pd.read_csv(p/'robustness_fits.csv')
    for f in [fits,held,rob]:
        assert f.converged.all()
        assert np.isfinite(f[['log_likelihood','AIC','AICc','BIC']]).all().all()
        assert f.log_likelihood.le(1e-7).all()
        assert f.groupby('participant_id').n.nunique().eq(1).all()
    for name in ['model_fits_predictions','heldout_fits_predictions','robustness_fits_predictions']:
        d=pd.read_csv(p/f'{name}.csv');assert d.probability.between(0,1).all()
    r=pd.read_csv(p/'parameter_recovery.csv')
    assert r.converged.all() and np.isfinite(r[['true','recovered']]).all().all()
    assert r.groupby(['participant_id','model','parameter']).iteration.nunique().eq(config['recovery_iterations']).all()
    mr=pd.read_csv(p/'model_recovery.csv');assert mr.converged.all()
    assert np.allclose(mr.groupby(['participant_id','generating','iteration','metric']).selected_weight.sum(),1)
    confusion=pd.read_csv(p/'model_recovery_confusion.csv')
    assert np.allclose(confusion.groupby(['metric','generating']).selection_probability.sum(),1)
    assert {'preference','M2_power','M5_power'}.issubset(set(confusion.generating))
    for ext in ['png','pdf','svg']:
        assert len(list(Path('results/figures').glob('*.'+ext)))==14
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    assert not any(x.startswith('data/') or x.endswith(('.nii','.nii.gz','.img','.hdr')) for x in tracked)
    # Installed imaging entries must remain unfetched annex symlinks.
    root=Path('data/ds005123')
    images=list(root.glob('sub-*/**/*.nii.gz'))
    assert images and all(x.is_symlink() and not x.exists() for x in images)
    print(f'Validated {len(fits)} core fits, {len(held)} held-out fits, {len(rob)} robustness fits; {len(images)} imaging placeholders remain unfetched.')

if __name__=='__main__':main()
