"""Independent full-cohort Wiener simulation and posterior/recovery summaries."""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import expit
from scipy.stats import spearmanr
from audit import PARTNERS,add_history
from hier_data import WORK,PUBLIC,save


@njit(cache=True)
def path_draws(person,partner,gap,zero_offer,side,trial_position,run,low,high,scheduled,mask,
               a_scale,t0_scale,eta,beta_zero,beta_side,beta_history,Z,H,lo,hi,lapse,dt,seed,conditional):
    np.random.seed(seed); n=len(person)
    choices=np.zeros(n,dtype=np.int64); rt=np.full(n,np.nan); retained=np.zeros(n,dtype=np.bool_)
    feedback=np.full(3,.5); previous_person=-1; previous_run=-1
    for j in range(n):
        i=person[j]; p=partner[j]
        if i!=previous_person or run[j]!=previous_run:
            feedback[:]=.5; previous_person=i;previous_run=run[j]
        if not mask[j]:continue
        a=np.exp(eta[i,0])*a_scale[p]; t0=hi/(1+np.exp(-eta[i,1]))*t0_scale[p]
        zi=6 if Z==1 else 6+p
        z=1/(1+np.exp(-(eta[i,zi]+beta_side*side[j])))
        v=eta[i,2+p]+eta[i,5]*gap[j]+beta_zero*zero_offer[j]
        if H:
            v+=beta_history[0]*(feedback[p]-.5)+beta_history[1]*trial_position[j]
        attempts=10000 if conditional else 1
        for attempt in range(attempts):
            if np.random.rand()<lapse:
                choice=int(np.random.rand()<.5); total=lo+(hi-lo)*np.random.rand()
            else:
                y=a*z; elapsed=0.
                while 0<y<a and elapsed+t0<=hi+dt:
                    y+=v*dt+np.sqrt(dt)*np.random.randn();elapsed+=dt
                choice=int(y>=a); total=elapsed-.5*dt+t0
                if 0<y<a:total=np.inf
            if lo<total<=hi:
                retained[j]=True;rt[j]=total;choices[j]=choice;break
        if retained[j]:
            amount=high[j] if choices[j] else low[j]
            if amount>0:feedback[p]=scheduled[j]
    return choices,rt,retained


def simulate(trials,eta,beta_zero,beta_side,beta_history,c,Z,seed,conditional=False,dt=None,a_scale=None,t0_scale=None):
    t=trials.sort_values(["participant_id","run","trial_in_run"]).copy()
    ids=sorted(t.participant_id.unique());lookup={s:i for i,s in enumerate(ids)}
    pmap={p:i for i,p in enumerate(PARTNERS)}
    if len(ids)!=len(eta):raise ValueError("Simulation population/schedule mismatch")
    lo=max(0.,c["raw_rt_min_seconds"]-c["cue_seconds"]);hi=c["response_window_seconds_from_cue"]-c["cue_seconds"]
    ch,rt,valid=path_draws(t.participant_id.map(lookup).to_numpy(np.int64),t.partner.map(pmap).to_numpy(np.int64),
        ((t.c_high-t.c_low)/8-.5).to_numpy(float),t.c_low.eq(0).to_numpy(float),
        (2*t.high_is_right.astype(int)-1).to_numpy(float),((t.trial_in_run-21.5)/42).to_numpy(float),
        t.run.to_numpy(np.int64),t.c_low.to_numpy(float),t.c_high.to_numpy(float),
        t.scheduled_reciprocation.to_numpy(float),t.analysis_valid.to_numpy(np.bool_),
        np.ones(3) if a_scale is None else np.asarray(a_scale,float),np.ones(3) if t0_scale is None else np.asarray(t0_scale,float),
        np.asarray(eta,float),float(beta_zero),float(beta_side),np.asarray(beta_history,float),Z,len(beta_history),
        lo,hi,c["contaminant_probability"],dt or c["simulation_dt_seconds"],seed,conditional)
    if conditional and np.any(t.analysis_valid.to_numpy(bool)&~valid):
        raise RuntimeError("Conditional simulator failed to obtain a retained response; inspect model/prior support")
    t["analysis_valid"]=valid;t["ddm_valid"]=valid;t["decision_rt"]=rt
    t["response_time"]=rt+c["cue_seconds"];t["chose_high"]=np.where(valid,ch,np.nan)
    t["chosen_amount"]=np.where(valid,np.where(ch,t.c_high,t.c_low),np.nan)
    t["observed_reciprocation"]=np.where(valid&t.chosen_amount.gt(0),t.scheduled_reciprocation,np.nan)
    t["feedback_observed"]=valid&t.chosen_amount.gt(0)
    return add_history(t)


def posterior_arrays(fit):
    result={n:fit.stan_variable(n) for n in ["eta","mu","tau","L_Omega","beta_zero","beta_side","a","t0","z","z_difference"]}
    try:result["beta_history"]=fit.stan_variable("beta_history")
    except ValueError:result["beta_history"]=np.zeros((len(result["eta"]),0))
    return result


def new_population(draws,index,n,c,regime,rng):
    mu=draws["mu"][index].copy();tau=draws["tau"][index].copy();L=draws["L_Omega"][index]
    Z=len(mu)-6
    if regime in c["recovery_z_logit_sd"]:tau[6:]=c["recovery_z_logit_sd"][regime]
    eta=mu[None,:]+rng.normal(size=(n,len(mu)))@(np.diag(tau)@L).T
    if regime in ["drift_only_null","partner_boundary_stress","partner_nondecision_stress"]:
        # Exact within-person equality is an explicit stress test at zero contrast variance.
        common=rng.normal(float(mu[6:].mean()),float(tau[6:].mean()),n)
        eta[:,6:]=common[:,None]
        if regime=="drift_only_null":eta[:,2]+=.5;eta[:,3]-=.5
    return eta,float(draws["beta_zero"][index]),float(draws["beta_side"][index]),draws["beta_history"][index].copy()


def truth_table(eta,ids,c):
    Z=eta.shape[1]-6;hi=c["response_window_seconds_from_cue"]-c["cue_seconds"]
    z=expit(eta[:,6:]);z=np.repeat(z,3,axis=1) if Z==1 else z
    return pd.DataFrame(dict(participant_id=ids,a=np.exp(eta[:,0]),t0=hi*expit(eta[:,1]),
        v_friend=eta[:,2],v_stranger=eta[:,3],v_computer=eta[:,4],beta_gap=eta[:,5],
        z_friend=z[:,0],z_stranger=z[:,1],z_computer=z[:,2],z_difference=z[:,0]-z[:,1]))


def participant_posterior(draws,ids):
    out=[]
    variables={"a":draws["a"],"t0":draws["t0"],"z_friend":draws["z"][:,:,0],
        "z_stranger":draws["z"][:,:,1],"z_computer":draws["z"][:,:,2],"z_difference":draws["z_difference"],
        "v_friend":draws["eta"][:,:,2],"v_stranger":draws["eta"][:,:,3],"v_computer":draws["eta"][:,:,4],"beta_gap":draws["eta"][:,:,5]}
    for name,x in variables.items():
        lo,med,hi=np.quantile(x,[.025,.5,.975],axis=0)
        for i,pid in enumerate(ids):
            out.append(dict(participant_id=pid,parameter=name,mean=float(x[:,i].mean()),median=float(med[i]),
                sd=float(x[:,i].std()),q025=float(lo[i]),q975=float(hi[i]),interval_width=float(hi[i]-lo[i])))
    return pd.DataFrame(out)


def recovery_summary(posterior,truth,cohort,regime,replicate,fit_name):
    wide=truth.melt(id_vars="participant_id",var_name="parameter",value_name="truth")
    p=posterior.merge(wide,on=["participant_id","parameter"],validate="one_to_one")
    columns=[k for k in ["participant_id","preregistered_rule_pass","ddm_valid","simulated_rule_pass","simulated_valid"] if k in cohort]
    p=p.merge(cohort[columns],on="participant_id",validate="many_to_one")
    p["covered"]=p.truth.ge(p.q025)&p.truth.le(p.q975)
    p["excludes_zero"]=p.q025.gt(0)|p.q975.lt(0)
    rows=[]
    groups=[("full",np.ones(len(p),bool)),("preregistered_subset",p.preregistered_rule_pass),
            ("observed_constant_or_missing_roster",~p.preregistered_rule_pass),
            ("low_trial_count",p.ddm_valid.le(cohort.ddm_valid.quantile(.25)))]
    if "simulated_rule_pass" in p:groups.append(("simulated_constant_or_missing",~p.simulated_rule_pass))
    if "simulated_valid" in p:groups.append(("simulated_no_observations",p.simulated_valid.eq(0)))
    for group,mask in groups:
        for parameter,g in p[mask].groupby("parameter"):
            variable=g.truth.std()>1e-10 and g["mean"].std()>1e-10
            rows.append(dict(fit=fit_name,regime=regime,replicate=replicate,group=group,parameter=parameter,n=len(g),
                pearson=float(g.truth.corr(g["mean"])) if variable else None,
                spearman=float(spearmanr(g.truth,g["mean"]).statistic) if variable else None,
                rmse=float(np.sqrt(np.mean((g["mean"]-g.truth)**2))),bias=float((g["mean"]-g.truth).mean()),
                coverage=float(g.covered.mean()),mean_interval_width=float(g.interval_width.mean()),
                zero_exclusion_fraction=float(g.excludes_zero.mean()),truth_sd=float(g.truth.std())))
    return p,pd.DataFrame(rows)


def ppc_summary(trials):
    t=trials[trials.analysis_valid]
    result=[]
    # Participant-weighted summaries; exact offer/choice RT checks included.
    for keys in [["participant_id","partner"],["participant_id","partner","c_low","c_high"],
                 ["participant_id","partner","chose_high"]]:
        if len(keys)==2:kind="partner"
        elif "chose_high" in keys:kind="partner_choice"
        else:kind="partner_offer"
        cells=t.groupby(keys).agg(high=("chose_high","mean"),rt10=("decision_rt",lambda x:x.quantile(.1)),
            rt50=("decision_rt","median"),rt90=("decision_rt",lambda x:x.quantile(.9)),n=("decision_rt","size")).reset_index()
        for row in cells.to_dict("records"):
            row["cell_kind"]=kind;result.append(row)
    return pd.DataFrame(result)
