"""Economical constant-bound Wiener pilot; noise SD is fixed to 1.

This is a numerical prototype, not an approved association-analysis model.
The event audit and actual-schedule recovery must precede scientific use.
"""
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize


@njit(cache=True)
def upper_probability(a, z, v):
    if abs(v) < 1e-8:
        return z
    return math.expm1(-2*v*a*z)/math.expm1(-2*v*a)


@njit(cache=True)
def lower_density(t, a, z, v):
    if t <= 0:
        return 0.
    tau=t/(a*a)
    if tau < .2:
        s=0.
        for k in range(-10,11):
            w=z+2*k
            s+=w*math.exp(-w*w/(2*tau))
        base=s/math.sqrt(2*math.pi*tau**3)/(a*a)
    else:
        s=0.
        for k in range(1,101):
            s+=k*math.sin(k*math.pi*z)*math.exp(-.5*(k*math.pi)**2*tau)
        base=math.pi*s/(a*a)
    return max(0.,base*math.exp(-v*a*z-.5*v*v*t))


@njit(cache=True)
def lower_cdf(t, a, z, v):
    if t <= 0:
        return 0.
    tail=0.
    for k in range(1,301):
        lam=.5*(v*v+(k*math.pi/a)**2)
        tail+=k*math.sin(k*math.pi*z)*math.exp(-lam*t)/lam
    f=1-upper_probability(a,z,v)-math.exp(-v*a*z)*math.pi/(a*a)*tail
    return max(0.,min(1.,f))


@njit(cache=True)
def retained_mass(a,z,v,t0,lo,hi):
    upper=lower_cdf(hi-t0,a,1-z,-v)-lower_cdf(lo-t0,a,1-z,-v)
    lower=lower_cdf(hi-t0,a,z,v)-lower_cdf(lo-t0,a,z,v)
    return upper+lower


def names(model, history=False, motor=False):
    result=["a","t0","v_friend","v_stranger","v_computer","beta_gap"]
    result += ["z_shared"] if model=="shared" else ["z_friend","z_stranger","z_computer"]
    if history: result += ["beta_history"]
    if motor: result += ["motor_logit_bias"]
    return result


def bounds(model, config, history=False, motor=False, wide=False):
    b=config["bounds"]
    result=[tuple(b["a"]),tuple(b["t0"])] + [tuple(b["drift"])]*3+[tuple(b["offer_slope"])]
    result += [(.02,.98) if wide else tuple(b["z"])]*(1 if model=="shared" else 3)
    if history: result += [(-3.,3.)]
    if motor: result += [(-2.,2.)]
    return result


@njit(cache=True)
def nll_core(theta, partner,gap,high_right,history,choice,rt,model_shared,
             add_history,add_motor,lo,hi,contamination,prior_sd):
    a,t0=theta[0],theta[1]
    total=0.
    for i in range(len(rt)):
        z=theta[6] if model_shared else theta[6+partner[i]]
        index=7 if model_shared else 9
        v=theta[2+partner[i]]+theta[5]*gap[i]
        if add_history:
            v+=theta[index]*history[i]; index+=1
        if add_motor:
            eta=math.log(z/(1-z))+theta[index]*(2*high_right[i]-1)
            z=1/(1+math.exp(-eta))
        f=lower_density(rt[i]-t0,a,1-z,-v) if choice[i] else lower_density(rt[i]-t0,a,z,v)
        mass=retained_mass(a,z,v,t0,lo,hi)
        if mass <= 1e-12:
            return 1e20
        # Contaminants are generated before censoring, then the entire mixture
        # is conditioned on retention. Their conditional fraction is not fixed.
        f=((1-contamination)*f+contamination/(2*(hi-lo)))/((1-contamination)*mass+contamination)
        total-=math.log(max(1e-300,f))
    if prior_sd>0:
        for k in range(6,7 if model_shared else 9):
            # Explicit normal penalty on the natural-z scale, truncated by bounds.
            # MAP sensitivity only; not a hierarchical posterior or uncertainty interval.
            total+=.5*((theta[k]-.5)/prior_sd)**2
    return total


def arrays(d):
    lookup={"friend":0,"stranger":1,"computer":2}
    h=d.previous_partner_feedback.fillna(.5).to_numpy(float)-.5
    return (d.partner.map(lookup).to_numpy(np.int64),
        ((d.c_high-d.c_low).to_numpy(float)/8-.5),
        d.high_is_right.to_numpy(float),h,d.chose_high.to_numpy(np.int64),
        d.response_time.to_numpy(float))


def fit(d, model, config, seed, *, history=False,motor=False,wide=False,prior_sd=0.):
    if model not in {"shared","partner"}:
        raise ValueError("Unknown candidate")
    if not d.ddm_valid.all() or not len(d):
        raise ValueError("Fit input must contain audited valid decisions only")
    x=arrays(d); rng=np.random.default_rng(seed)
    b=bounds(model,config,history,motor,wide)
    # t0 can exceed the fastest retained RT under the fixed contaminant model;
    # that possibility is intentional and must be reported in sensitivity checks.
    objective=lambda t:nll_core(np.asarray(t),*x,model=="shared",history,motor,
        config["rt_min_seconds"],config["rt_max_seconds"],.01,prior_sd)
    solutions=[]
    for k in range(config["multistarts"]):
        init=np.array([1.4,.18,.4,0.,0.,-.5]+[.5]*(1 if model=="shared" else 3)+[0.]*(history+motor))
        if k:
            init+=rng.normal(0,.15,len(init))
        init=np.array([np.clip(v,lo+1e-4,hi-1e-4) for v,(lo,hi) in zip(init,b)])
        opt=minimize(objective,init,method="L-BFGS-B",bounds=b,
                     options={"maxiter":500,"ftol":1e-10,"gtol":1e-5})
        solutions.append(opt)
    best=min(solutions,key=lambda o:o.fun)
    return dict(parameters=dict(zip(names(model,history,motor),map(float,best.x))),
        objective=float(best.fun),success=bool(best.success and set(d.partner)=={"friend","stranger","computer"}),message=str(best.message),
        complete_partner_support=set(d.partner)=={"friend","stranger","computer"},
        starts_success=sum(bool(o.success) for o in solutions),
        starts_agree=sum(abs(o.fun-best.fun)<.01 for o in solutions),
        starts=len(solutions),boundary=any(min(v-lo,hi-v)<.005*(hi-lo) for v,(lo,hi) in zip(best.x,b)),
        prior_sd=prior_sd,wide_bounds=wide,history=history,motor=motor)


@njit(cache=True)
def simulate_paths(partner,gap,high_right,history,theta,shared,dt,seed,add_history=False,add_motor=False):
    """Euler simulation independent of the series likelihood, with absorbing bounds.

    A half-step temporal correction is used. dt convergence must be checked.
    For history sensitivity, this helper conditions on supplied history; a fully
    generative learning simulation is a separate required expansion gate.
    """
    np.random.seed(seed)
    n=len(partner); rt=np.empty(n); choice=np.empty(n,dtype=np.int64)
    for i in range(n):
        a,t0=theta[0],theta[1]
        z=theta[6] if shared else theta[6+partner[i]]
        j=7 if shared else 9
        v=theta[2+partner[i]]+theta[5]*gap[i]
        if add_history:
            v+=theta[j]*history[i]; j+=1
        if add_motor:
            z=1/(1+math.exp(-(math.log(z/(1-z))+theta[j]*(2*high_right[i]-1))))
        y=a*z; t=0.
        while 0 < y < a and t<30.:
            y+=v*dt+math.sqrt(dt)*np.random.randn(); t+=dt
        choice[i]=int(y>=a); rt[i]=max(0.,t-.5*dt)+t0
    return choice,rt


@njit(cache=True)
def simulate_learning(partner,gap,low,high,scheduled,run,response_mask,theta,dt,seed,lo,hi):
    """Stress test: true drifting beliefs, with feedback revealed only if invested.

    Beliefs reset to .5 on each run; fixed alpha=.3 and drift history gain=2.
    The fitted simple model deliberately omits this time-varying mechanism.
    """
    np.random.seed(seed)
    q=np.full(3,.5); n=len(partner)
    choices=np.empty(n,dtype=np.int64); rt=np.empty(n)
    previous=-1
    for i in range(n):
        if run[i]!=previous:
            q[:]=.5; previous=run[i]
        p=partner[i]; a=theta[0]; z=theta[6+p]
        v=theta[2+p]+theta[5]*gap[i]+2*(q[p]-.5)
        y=a*z; t=0.
        while 0<y<a and t<30.:
            y+=v*dt+math.sqrt(dt)*np.random.randn(); t+=dt
        choices[i]=int(y>=a); rt[i]=t-.5*dt+theta[1]
        if np.random.rand()<.01:
            choices[i]=int(np.random.rand()<.5)
            rt[i]=lo+(hi-lo)*np.random.rand()
        amount=high[i] if choices[i] else low[i]
        if response_mask[i] and amount>0 and lo<=rt[i]<=hi:
            q[p]+=.3*(scheduled[i]-q[p])
    return choices,rt
