import numpy as np
import pytest
from rf1_trust_socialvalue.models import trajectory,specification,pack
from rf1_trust_socialvalue.hierarchical import inputs,split_index,friend_effect,transform


def test_bounds_are_explicit():
    for model in ['M4','M5','M6','M7']:
        for upper in [5,10,20]:
            _,names,_,bounds=specification(f'{model}_theta{upper}')
            assert dict(zip(names,bounds))['theta']==(0,upper)
            if model=='M7':assert dict(zip(names,bounds))['theta_stranger']==(0,upper)
    assert specification('M5')[3][-1]==(0,5)  # Unqualified names retain the archived first pass.
    assert specification('M5_kappa100')[3][1]==(1e-5,100)


def test_logistic_prior_neutral_and_signed_rating():
    a=np.array([[c,2,8,1,0,0,1,1] for c in range(3)],float)
    tr=trajectory(a,'M3_logitprior',dict(alpha=.3,kappa=.5,phi=2),np.array([1,.5,0]))
    assert tr[0,0]==pytest.approx(1/(1+np.exp(-2)))
    assert tr[1,0]==.5
    assert tr[2,0]==pytest.approx(1/(1+np.exp(2)))


def test_choice_scale_effect_matches_engine():
    par=np.array([[.2,.4,2.]])
    gaps=np.array([2,4,6,8]);weights=np.array([.2,.2,.4,.2])
    a=np.array([[0,0,g,1,0,0,1,1] for g in gaps],float)
    on=trajectory(a,'M5',dict(alpha=.2,kappa=.4,theta=2),np.zeros(3))[:,1]
    off=trajectory(a,'M5',dict(alpha=.2,kappa=.4,theta=0),np.zeros(3))[:,1]
    assert friend_effect(par,gaps,weights)[0]==pytest.approx((on-off)@weights)


def test_train_data_never_include_test_choices():
    data,meta,arrays,_,_=inputs('H5',training=True)
    cursor=0
    for sub,a in arrays.items():
        train=a[:split_index(a)];valid=train[:,3]>=0;n=valid.sum()
        assert data['y'][cursor:cursor+n]==train[valid,3].astype(int).tolist()
        assert data['feedback'][cursor:cursor+n]==train[valid,4].astype(int).tolist()
        cursor+=n
    assert cursor==data['T'] and data['N']==111


def test_omitting_missed_trials_preserves_likelihood():
    a=np.array([[0,0,8,1,1,1,0,1],[0,2,8,-1,0,0,0,1],[0,0,8,0,0,0,1,2],[0,2,8,1,1,0,1,2]],float)
    p=dict(alpha=.3,kappa=.7,theta=2)
    full=trajectory(a,'M5',p,np.zeros(3));compact=trajectory(a[a[:,3]>=0],'M5',p,np.zeros(3))
    assert np.allclose(full[a[:,3]>=0],compact)


def test_population_transform_has_no_theta_ceiling():
    x=transform(np.array([[-1,-2,30.]]),[1,2,3])
    assert x[0,2]>20
    assert transform(np.array([[-1,-2,30.]]),[1,2,3],True)[0,2]<=10
