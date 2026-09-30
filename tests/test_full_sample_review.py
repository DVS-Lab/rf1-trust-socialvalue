from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample_review as review
from rf1_trust_socialvalue import full_sample_geometry as geometry


def scores():
    rows=[]
    for model in ['H2','H5','H7','HPreference','H8']:
        for variant in ['base','zero']:
            if (model,variant)==('H5','base'):continue
            for i in range(2):rows.append(dict(name=f'Train_{model}_{variant}_noage',model=model,variant=variant,participant_id=f'sub-{i}',
                n=2 if i==0 else 40,log_loss=(.2 if i==0 else .8)-(.1 if variant=='zero' else 0),brier=.2))
    return pd.DataFrame(rows)


def test_partial_comparison_is_paired_equal_weight_and_does_not_invent_missing_model():
    summary,matched,focal=review.comparisons(scores(),iterations=200)
    assert len(summary)==9 and set(matched.model)=={'H2','H7','HPreference','H8'}
    assert summary.loc[summary.name=='Train_H2_base_noage','log_loss'].iloc[0]==pytest.approx(.5)
    np.testing.assert_allclose(matched.delta,-.1)
    assert focal['delta']==pytest.approx(0)


@pytest.mark.parametrize('change',['missing_subject','count_mismatch','nonfinite'])
def test_partial_comparison_rejects_incomparable_scores(change):
    f=scores()
    if change=='missing_subject':f=f.iloc[1:]
    if change=='count_mismatch':f.loc[0,'n']=3
    if change=='nonfinite':f.loc[0,'log_loss']=np.inf
    with pytest.raises(ValueError):review.comparisons(f,iterations=10)


def test_published_result_inventory_excludes_failed_fits():
    root=Path(__file__).resolve().parents[1]
    diagnostics,frame,hashes=review.accepted_scores(root)
    assert len(diagnostics)==20 and diagnostics.accepted.sum()==18
    assert frame.name.nunique()==9 and frame.participant_id.nunique()==304
    assert frame.groupby('name').n.sum().eq(12494).all()
    assert 'Train_H5_base_noage' not in set(frame.name)
    assert len(hashes)>50


def test_geometry_exports_correct_participant_mapping_and_saved_draw_context():
    names=['divergent__','energy__','mu[1]','z[1,2]','natural[2,1]','L[1,1]']
    a=np.zeros((20,2,len(names)));a[:,:,1]=np.arange(20)[:,None]
    a[:,:,2]=np.arange(20)[:,None];a[:,:,3]=np.arange(20)[:,None]*2
    a[:,:,4]=np.arange(20)[:,None]*3;a[:,:,5]=1;a[10,1,0]=1
    meta=dict(ids=['sub-a','sub-b'],parameter_names=['alpha'])
    result=geometry.parameter_context(a,names,meta)
    assert len(result)==3 and set(result.chain)=={2} and set(result.draw)=={11}
    assert result[result.stan_parameter=='natural[2,1]'].participant_id.iloc[0]=='sub-b'
    assert result[result.stan_parameter=='z[1,2]'].parameter.iloc[0]=='latent_z_alpha'
    assert result[result.stan_parameter=='mu[1]'].empirical_percentile.iloc[0]==pytest.approx(.55)
    neighborhood=geometry.draw_neighborhood(a,names,radius=2)
    assert neighborhood.draw.tolist()==[9,10,11,12,13]
    assert neighborhood.divergent__.sum()==1
