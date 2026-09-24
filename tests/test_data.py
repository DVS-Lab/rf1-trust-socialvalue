import pandas as pd
import numpy as np
import pytest
from rf1_trust_socialvalue.data import reconstruct

def frames():
    raw=pd.DataFrame(dict(cLeft=[0,2,0],cRight=[2,4,8],Partner=[3,2,1],Reciprocate=[1,0,1],resp=[0,4,999],highlow=['low','high','999'],rt=[1,2,0],onset=[4,10,20]))
    events=pd.DataFrame(dict(trial_type=['choice_friend','choice_stranger','outcome_stranger_defect','missed_trial'],cLow=[0,2,2,0],cHigh=[2,4,4,8],trust_value=[0,4,4,np.nan],choice=['low','high','high',np.nan],response_time=[1,2,2,3],onset=[4,10,14,20]))
    return raw,events

def test_one_row_per_decision_and_misses():
    r,e=frames();t=reconstruct(r,e,'test',1)
    assert len(t)==3 and t.valid_choice.sum()==2
    assert t.feedback_observed.tolist()==[False,True,False]
    assert np.isnan(t.observed_reciprocation.iloc[0])

def test_unexplained_mismatch_raises():
    r,e=frames();e.loc[0,'trust_value']=2
    with pytest.raises(ValueError,match='selected amount'):reconstruct(r,e,'test',1)

def test_unexpected_zero_feedback_raises():
    r,e=frames();e.loc[0,'trial_type']='choice_stranger'
    with pytest.raises(AssertionError):reconstruct(r,e,'test',1)
