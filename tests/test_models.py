import numpy as np
import pytest
from rf1_trust_socialvalue.models import monetary_value, trajectory, simulate

def toy():
    return np.array([[0,0,2,1,1,1,1,1],[1,2,4,1,1,0,0,1],[0,0,8,0,0,0,1,1],[0,2,8,1,1,0,0,2]],float)

def test_payoffs():
    assert monetary_value(0,.5)==8
    assert monetary_value(8,.5)==6
    assert monetary_value(8,2/3)==8

def test_probabilities():
    a=toy();r=trajectory(a,'M1',{'kappa':2},np.ones(3))
    assert r[0,1]==pytest.approx(1/(1+np.exp(1)))
    assert np.allclose(trajectory(a,'M0',{},np.ones(3))[:,1],.5)

def test_social_value_scales_with_gap():
    a=toy();f={'alpha':0,'kappa':1,'theta':2}
    r=trajectory(a,'M5',f,np.ones(3))
    assert r[0,1]==pytest.approx(1/(1+np.exp(-1.5)))
    assert r[3,1]==pytest.approx(1/(1+np.exp(-4.5)))

def test_power_one_matches_linear():
    a=toy();p={'alpha':.2,'kappa':.7,'theta':1}
    assert np.allclose(trajectory(a,'M5',p,np.ones(3)),trajectory(a,'M5_power',dict(p,rho=1),np.ones(3)))

def test_simulation_feedback_depends_on_simulated_amount():
    a=toy();a[:,3]=0;a[:,4]=0;a[:,1]=0
    b,res=simulate(a,'M5',{'alpha':.5,'kappa':20,'theta':5},np.ones(3),np.random.default_rng(1))
    assert b[0,3]==1 and b[0,4]==1
    assert res[2,0]==.75

def test_side_bias_is_independent_of_offer_orientation():
    a=toy();a[:,1:3]=[2,4]
    p={'alpha':0,'kappa':1,'theta':0,'side_bias':2}
    r=trajectory(a,'M5_side',p,np.ones(3))
    assert r[0,1]>r[1,1]

def test_lapse_keeps_probabilities_inside_lapse_bounds():
    a=toy();r=trajectory(a,'M5_lapse',{'alpha':0,'kappa':20,'theta':5,'lapse':.2},np.ones(3))
    assert np.all((r[:,1]>=.1)&(r[:,1]<=.9+1e-12))

def test_feedback_can_become_hidden_in_simulation():
    a=toy();a[:,1]=0;a[:,3]=1;a[:,4]=1
    b,_=simulate(a,'M1',{'kappa':20},np.ones(3),np.random.default_rng(17))
    assert np.all(b[:,3]==0) and np.all(b[:,4]==0)

def test_rating_prior_clips_and_is_partner_specific():
    a=toy();r=trajectory(a,'M3',{'alpha':0,'kappa':1,'phi':2},np.array([.9,.2,0]))
    assert r[0,0]==pytest.approx(1-1e-9)
    assert r[1,0]==pytest.approx(.4)
