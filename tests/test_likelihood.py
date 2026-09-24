import numpy as np
from rf1_trust_socialvalue.models import trajectory, specification, value_gradient, objective
from rf1_trust_socialvalue.fitting import fit
from test_models import toy

def test_bernoulli_likelihood():
    a=toy();r=trajectory(a,'M2',{'alpha':.4,'kappa':.3},np.ones(3));p=r[:,1];y=a[:,3]
    assert np.allclose(r[:,3],-y*np.log(p)-(1-y)*np.log1p(-p))

def test_stable_extreme_likelihood():
    a=toy();r=trajectory(a,'M5',{'alpha':0,'kappa':1000,'theta':5},np.ones(3))
    assert np.isfinite(r).all() and r[:,3].min()>=0

def test_gradient():
    code,_,idx,_=specification('M5');x=np.array([.3,.7,1.2]);a=toy();ratings=np.ones(3)
    _,g=value_gradient(x,idx,a,code,ratings,False)
    for j in range(3):
        h=np.zeros(3);h[j]=1e-6
        expected=(objective(x+h,idx,a,code,ratings,False)-objective(x-h,idx,a,code,ratings,False))/2e-6
        assert np.isclose(g[j],expected,atol=1e-5)

def test_null_and_information_criteria():
    a=np.tile(toy(),(3,1));r=fit(a,'M0',np.ones(3))
    assert r['pseudo_r2_fareri']==0 and r['pseudo_r2_mcfadden']==0
    assert r['AIC']==r['AICc']==r['BIC']

def test_future_outcomes_do_not_affect_past_predictions():
    a=toy();b=a.copy();b[2:,5]=1-b[2:,5];p={'alpha':.5,'kappa':1}
    assert np.allclose(trajectory(a,'M2',p,np.ones(3))[:3,1],trajectory(b,'M2',p,np.ones(3))[:3,1])
