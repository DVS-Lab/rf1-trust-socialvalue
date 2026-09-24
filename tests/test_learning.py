import numpy as np
from rf1_trust_socialvalue.models import trajectory
from test_models import toy

def test_partner_updates_and_zero_censoring():
    r=trajectory(toy(),'M2',{'alpha':.5,'kappa':1},np.ones(3))
    assert np.allclose(r[:,0],[.5,.5,.75,.75])

def test_missed_trial_does_not_update():
    a=toy();a[0,3]=-1;a[0,4]=0
    r=trajectory(a,'M2',{'alpha':.5,'kappa':1},np.ones(3))
    assert r[2,0]==.5 and r[0,3]==0

def test_asymmetric_updates():
    a=toy();a[:,0]=0
    r=trajectory(a,'M8',{'alpha':.8,'alpha_negative':.2,'kappa':1},np.ones(3))
    assert np.allclose(r[:,0],[.5,.9,.72,.72])

def test_reset_and_carry():
    p={'alpha':.5,'kappa':1,'theta':0}
    assert trajectory(toy(),'M5_reset',p,np.ones(3))[-1,0]==.5
    assert trajectory(toy(),'M5',p,np.ones(3))[-1,0]==.75
