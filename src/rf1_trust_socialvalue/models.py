"""Trialwise RL, stable likelihoods, and choice-dependent feedback simulations.

Natural bounded parameters include exact no-learning limits. The finite kappa
cap is explicit and tested with a wider-cap sensitivity analysis.
"""
import re
import numpy as np
from numba import njit

INDEX={'alpha':0,'kappa':1,'theta':2,'theta_stranger':3,'phi':4,'alpha_negative':5,'lapse':6,'side_bias':7,'rho':8,'preference_stranger':9}
BASE=np.array([.2,1.,0.,0.,1.,.2,0.,0.,1.,0.])
CORE={
 'M0':(0,[]), 'M1':(1,['kappa']), 'M2':(2,['alpha','kappa']),
 'M3':(3,['alpha','kappa','phi']), 'M4':(4,['alpha','kappa','theta']),
 'M5':(5,['alpha','kappa','theta']), 'M6':(6,['alpha','kappa','theta']),
 'M7':(7,['alpha','kappa','theta','theta_stranger']),
 'M8':(8,['alpha','alpha_negative','kappa']),
 'preference':(9,['alpha','kappa','theta','preference_stranger']),
}
LABELS={'M0':'Random','M1':'Fixed EV','M2':'RL EV','M3':'Rating prior','M4':'Rating value','M5':'Friend value','M6':'Human value','M7':'Partner values','M8':'Asymmetric RL','preference':'Partner preferences'}


def specification(name):
    base=name.split('_')[0]
    code,names=CORE[base];names=names.copy()
    if name=='M3_logitprior':code=10
    match=re.search(r'_theta(5|10|20)(?:_|$)',name)
    theta_upper=float(match.group(1)) if match else 5.
    if 'lapse' in name:names+=['lapse']
    if 'side' in name:names+=['side_bias']
    if 'power' in name:names+=['rho']
    bounds=[]
    for p in names:
        if p.startswith('alpha'):b=(0.,1.)
        elif p=='kappa':b=(1e-5,100. if ('kappa100' in name or 'wide' in name) else 20.)
        elif p=='lapse':b=(0.,.2)
        elif p=='side_bias':b=(-5.,5.)
        elif p=='rho':b=(.2,3.)
        elif p=='phi':b=(0.,10. if name in ['M3_phi10','M3_logitprior'] else 5.)
        elif base=='preference' or 'signed' in name:b=(-5.,5.)
        else:b=(0.,theta_upper)
        bounds.append(b)
    return code,names,np.array([INDEX[p] for p in names],dtype=np.int64),bounds


def pack(frame):
    """Columns: partner,low,high,choice,feedback,programmed outcome,side,run."""
    a=np.column_stack([frame.partner.map({'friend':0,'stranger':1,'computer':2}),frame.c_low,frame.c_high,
        frame.chose_high.fillna(-1).astype(float),frame.feedback_observed.astype(float),
        frame.scheduled_reciprocation,frame.high_is_right.astype(float),frame.run])
    assert np.isfinite(a).all()
    return np.ascontiguousarray(a,dtype=float)


@njit(cache=True)
def _sigmoid(z):
    if z>=0:return 1./(1.+np.exp(-z))
    e=np.exp(z);return e/(1.+e)


@njit(cache=True)
def engine(a, pars, code, ratings, reset=False, simulate=False, uniforms=np.zeros(1)):
    """Return pre-feedback beliefs, choice probabilities, simulated choices, trial NLL."""
    n=len(a);out=np.zeros((n,5));p=np.full(3,.5)
    if code==3:
        for c in range(3):p[c]=min(pars[4]*ratings[c],1.-1e-9)
    elif code==10:
        for c in range(3):p[c]=_sigmoid(pars[4]*(2*ratings[c]-1))
    for t in range(n):
        if reset and t>0 and a[t,7]!=a[t-1,7]:
            p[:]=.5
            if code==3:
                for c in range(3):p[c]=min(pars[4]*ratings[c],1.-1e-9)
        c=int(a[t,0]);lo=a[t,1];hi=a[t,2];belief=p[c]
        bonus=0.
        if code==4:bonus=pars[2]*ratings[c]
        elif code==5 and c==0:bonus=pars[2]
        elif code==6 and c<2:bonus=pars[2]
        elif code==7:
            if c==0:bonus=pars[2]
            elif c==1:bonus=pars[3]
        rho=pars[8]
        dv=(1-belief)*((8-hi)**rho-(8-lo)**rho)+belief*((8+.5*hi)**rho-(8+.5*lo)**rho)
        dv+=belief*(hi-lo)*bonus
        if code==9:
            if c==0:dv+=(hi-lo)*pars[2]
            elif c==1:dv+=(hi-lo)*pars[9]
        z=pars[1]*dv+pars[7]*(2*a[t,6]-1)
        if code==0:z=0.
        prob=pars[6]*.5+(1-pars[6])*_sigmoid(z)
        ch=a[t,3]
        if simulate and ch>=0:ch=1. if uniforms[t]<prob else 0.
        amount=lo if ch==0 else hi
        feedback=a[t,4]>0
        if simulate:feedback=ch>=0 and amount>0
        nll=0.
        if ch>=0:
            if pars[6]==0:nll=np.logaddexp(0.,z)-ch*z
            else:nll=-ch*np.log(max(prob,1e-300))-(1-ch)*np.log(max(1-prob,1e-300))
        out[t,0]=belief;out[t,1]=prob;out[t,2]=ch;out[t,3]=nll;out[t,4]=1. if feedback else 0.
        if feedback and code>=2:
            err=a[t,5]-belief
            alpha=pars[0]
            if code==8 and err<0:alpha=pars[5]
            p[c]=belief+alpha*err
    return out


@njit(cache=True)
def objective(x,indices,a,code,ratings,reset):
    p=BASE.copy()
    for j in range(len(x)):p[indices[j]]=x[j]
    return engine(a,p,code,ratings,reset)[:,3].sum()


@njit(cache=True)
def value_gradient(x,indices,a,code,ratings,reset):
    value=objective(x,indices,a,code,ratings,reset)
    grad=np.zeros(len(x))
    for j in range(len(x)):
        step=1e-5*max(1.,abs(x[j]));xp=x.copy();xm=x.copy()
        xp[j]+=step;xm[j]-=step
        grad[j]=(objective(xp,indices,a,code,ratings,reset)-objective(xm,indices,a,code,ratings,reset))/(2*step)
    return value,grad


def trajectory(a,name,params,ratings):
    code,names,indices,_=specification(name);p=BASE.copy()
    for key in names:p[INDEX[key]]=params[key]
    return engine(a,p,code,ratings,'reset' in name)


def simulate(a,name,params,ratings,rng):
    code,names,indices,_=specification(name);p=BASE.copy()
    for key in names:p[INDEX[key]]=params[key]
    res=engine(a,p,code,ratings,'reset' in name,True,rng.random(len(a)))
    b=a.copy();b[:,3]=res[:,2];b[:,4]=res[:,4]
    return b,res


def monetary_value(x,p):
    return 8-x+1.5*x*p
