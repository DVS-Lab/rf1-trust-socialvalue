"""Bounded actual-schedule recovery and eight-person observed-data pilot.

Only compact aggregate results are published. Simulated truths, fitted values,
participant IDs, choices and RTs remain ignored under work/ddm/. No hierarchy
or external-outcome model is reachable from this entry point.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import importlib.metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit import ROOT, PARTNERS, dump, sha, private_dir, variability, add_history
from model import fit, arrays, names, simulate_paths, simulate_learning


def draw_truth(rng, sd, shared=False, drift_only=False):
    a=rng.uniform(.9,2.0); t0=rng.uniform(.15,.4)
    v=rng.normal([.5,0.,-.15],[.45,.45,.45])
    z=np.clip(.5+rng.normal(0,sd,1 if shared else 3),.12,.88)
    if drift_only:
        z[:]=.5; v[0]+=.7; v[1]-=.7
    return np.r_[a,t0,v,rng.normal(-.5,.35),z]


def simulate(d,theta,shared,config,seed,*,learning=False,dt=None):
    """Preserve actual schedule and original response/RT rejection mask.

    Additional simulated fast responses and deadline timeouts are excluded too.
    Conditional likelihood explicitly normalizes to the retained RT interval.
    """
    x=arrays(d.fillna({"chose_high":0.}))
    step=dt or config["simulation_dt_seconds"]
    if learning:
        c,rt=simulate_learning(x[0],x[1],d.c_low.to_numpy(float),d.c_high.to_numpy(float),
            d.scheduled_reciprocation.to_numpy(float),d.run.to_numpy(np.int64),d.ddm_valid.to_numpy(np.bool_),
            theta,step,seed,config["rt_min_seconds"],config["rt_max_seconds"])
    else:
        c,rt=simulate_paths(*x[:4],theta,shared,step,seed)
    rng=np.random.default_rng(seed+1)
    contam=(rng.random(len(d))<.01) if not learning else np.zeros(len(d),dtype=bool)
    c[contam]=rng.integers(0,2,contam.sum())
    rt[contam]=rng.uniform(config["rt_min_seconds"],config["rt_max_seconds"],contam.sum())
    s=d.copy()
    s["chose_high"]=c.astype(float); s["response_time"]=rt
    s["chosen_amount"]=np.where(c,s.c_high,s.c_low)
    s["ddm_valid"]=d.ddm_valid & (rt>=config["rt_min_seconds"]) & (rt<=config["rt_max_seconds"])
    s.loc[~s.ddm_valid,"chose_high"]=np.nan
    s["observed_reciprocation"]=np.where(s.ddm_valid & s.chosen_amount.gt(0),s.scheduled_reciprocation,np.nan)
    return add_history(s)


def select_ids(people, config):
    eligible=people.loc[people.eligible.astype(str).str.lower().eq("true")].copy()
    if len(eligible)<8:
        raise ValueError("Fewer than eight eligible participants; stop before recovery fitting")
    # Deterministic counts-based stratification includes low-information schedules.
    eligible=eligible.sort_values(["ddm_valid","participant_id"])
    n=min(config["pilot_participants"],len(eligible))
    return eligible.iloc[np.unique(np.linspace(0,len(eligible)-1,n).round().astype(int))].participant_id.tolist()


def spread_ids(ids,n):
    return [ids[k] for k in np.unique(np.linspace(0,len(ids)-1,min(n,len(ids))).round().astype(int))]


def record_fit(s,model,config,seed,label,pid,rep,theta,**kwargs):
    valid=s[s.ddm_valid].copy()
    missing,constant=variability(s,"ddm_valid")
    result=fit(valid,model,config,seed,**kwargs)
    row=dict(scenario=label,participant_id=pid,replicate=rep,model=model,
        n=len(valid),post_sim_choice_rule_excluded=bool(missing or constant),
        success=result["success"],starts_agree=result["starts_agree"],starts=result["starts"],
        boundary=result["boundary"],objective=result["objective"])
    row.update({"estimate_"+k:v for k,v in result["parameters"].items()})
    if theta is not None:
        true_shared=len(theta)==7
        row.update({"true_"+k:float(v) for k,v in zip(names("shared" if true_shared else "partner"),theta)})
        z=np.repeat(theta[6],3) if true_shared else theta[6:9]
        row.update(true_z_mean=float(z.mean()),true_z_friend=float(z[0]),true_z_stranger=float(z[1]),
                   true_z_computer=float(z[2]),true_z_difference=float(z[0]-z[1]))
    p=result["parameters"]
    if model=="shared":
        row["estimate_z_mean"]=p["z_shared"]
    else:
        row["estimate_z_mean"]=np.mean([p["z_"+k] for k in PARTNERS])
        row["estimate_z_difference"]=p["z_friend"]-p["z_stranger"]
    return row,result


def metrics(records):
    rows=[]
    d=pd.DataFrame(records)
    for (scenario,model,rep),g in d.groupby(["scenario","model","replicate"]):
        for target in ["a","t0","z_mean","z_friend","z_stranger","z_difference","v_friend","v_stranger"]:
            t="true_"+target; e="estimate_"+target
            if t not in g or e not in g: continue
            if model=="shared" and target=="z_mean" and scenario!="true_shared_z":
                continue  # A shared-z pseudo-parameter has no unique true value under heterogeneous z.
            h=g[[t,e]].dropna()
            if not len(h): continue
            correlation=float(h[t].corr(h[e])) if h[t].std()>1e-8 and h[e].std()>1e-8 else None
            rows.append(dict(scenario=scenario,model=model,replicate=int(rep),parameter=target,n=len(h),
                correlation=correlation,rmse=float(np.sqrt(np.mean((h[e]-h[t])**2))),
                bias=float((h[e]-h[t]).mean()),
                success_fraction=float(g.success.mean()),boundary_fraction=float(g.boundary.mean()),
                multistart_agreement_fraction=float((g.starts_agree==g.starts).mean()),
                choice_rule_excluded_fraction=float(g.post_sim_choice_rule_excluded.mean())))
    return pd.DataFrame(rows)


def recovery_plots(d, public):
    fig,axs=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
    for col,scenario in enumerate(["small","moderate","large"]):
        g=d[d.scenario.eq(scenario)&d.model.eq("partner")]
        for row,target in enumerate(["z_friend","z_difference"]):
            ax=axs[row,col]; ax.scatter(g["true_"+target],g["estimate_"+target],s=14,alpha=.6)
            lo,hi=(0,1) if row==0 else (-.7,.7)
            ax.plot([lo,hi],[lo,hi],"--",color="gray",lw=1)
            ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="True "+target,ylabel="Estimated "+target,title=scenario)
    fig.suptitle("Actual-schedule pilot recovery • fresh simulated parameters")
    fig.savefig(public/"parameter_recovery.png",dpi=180); plt.close(fig)
    g=d[d.scenario.eq("moderate")&d.model.eq("partner")]
    targets=["z_friend","z_difference","v_friend","a","t0"]
    truth=g[["true_"+p for p in targets]]
    errors=g[["estimate_"+p for p in targets]].to_numpy()-truth.to_numpy()
    values=np.empty((len(targets),len(targets)))
    for i in range(len(targets)):
        for j in range(len(targets)):
            values[i,j]=np.corrcoef(errors[:,i],errors[:,j])[0,1]
    fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True)
    im=ax.imshow(values,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(len(targets)),targets,rotation=35); ax.set_yticks(range(len(targets)),targets)
    ax.set_title("Recovery-error correlations; not posterior correlations")
    fig.colorbar(im,ax=ax); fig.savefig(public/"recovery_confounding.png",dpi=180); plt.close(fig)


def observed_pilot(trials, ids, config, private, public):
    ids=spread_ids(ids,config["observed_pilot_participants"])
    rows=[]; predictive=[]; rng=np.random.default_rng(config["seed"]+90000)
    for k,pid in enumerate(ids[:config["observed_pilot_participants"]]):
        d=trials[trials.participant_id.eq(pid)].copy()
        for model in ["shared","partner"]:
            row,result=record_fit(d,model,config,int(rng.integers(1,2**30)),"observed",pid,0,None)
            rows.append(row)
            theta=np.array([result["parameters"][p] for p in names(model)])
            for rep in range(40):
                s=simulate(d,theta,model=="shared",config,int(rng.integers(1,2**30)))
                for partner,g in s[s.ddm_valid].groupby("partner"):
                    predictive.append(dict(participant_id=pid,model=model,replicate=rep,partner=partner,
                        high=float(g.chose_high.mean()),rt10=float(g.response_time.quantile(.1)),
                        rt50=float(g.response_time.median()),rt90=float(g.response_time.quantile(.9))))
        print(f"observed pilot {k+1}/{min(len(ids),config['observed_pilot_participants'])}",flush=True)
    pd.DataFrame(rows).to_csv(private/"observed_pilot_fits.tsv",sep="\t",index=False)
    pred=pd.DataFrame(predictive); pred.to_csv(private/"observed_predictive_draws.tsv",sep="\t",index=False)
    # Plug-in predictive check: label explicitly. Parameter uncertainty not integrated.
    obs=trials[trials.participant_id.isin(ids[:config["observed_pilot_participants"]])&trials.ddm_valid]
    obs=obs.groupby(["participant_id","partner"]).agg(high=("chose_high","mean"),
        rt10=("response_time",lambda s:s.quantile(.1)),rt50=("response_time","median"),
        rt90=("response_time",lambda s:s.quantile(.9))).reset_index()
    # Participant weighted, with each replicate summarized before interval calculation.
    agg=pred.groupby(["model","replicate","partner"])[["high","rt10","rt50","rt90"]].mean().reset_index()
    fig,axs=plt.subplots(1,4,figsize=(12,3.5),constrained_layout=True)
    for ax,measure in zip(axs,["high","rt10","rt50","rt90"]):
        for i,partner in enumerate(PARTNERS):
            actual=obs.loc[obs.partner.eq(partner),measure].mean()
            ax.scatter(i,actual,color="black",marker="x",zorder=4)
            for offset,(model,color) in zip([-.12,.12],[("shared","#35678b"),("partner","#a66438")]):
                v=agg.loc[agg.partner.eq(partner)&agg.model.eq(model),measure]
                if len(v):
                    low,med,hi=v.quantile([.05,.5,.95])
                    ax.errorbar(i+offset,med,yerr=[[med-low],[hi-med]],fmt="o",color=color)
        ax.set_xticks(range(3),PARTNERS,rotation=30); ax.set_title(measure)
    fig.suptitle("Plug-in predictive checks • black observed; blue shared z; orange partner z")
    fig.savefig(public/"pilot_predictive_checks.png",dpi=180); plt.close(fig)
    pd.DataFrame(rows).groupby("model").agg(fits=("success","size"),success_fraction=("success","mean"),
        boundary_fraction=("boundary","mean")).to_csv(public/"observed_pilot_diagnostics.tsv",sep="\t")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--private-output",type=Path,default=ROOT/"work/ddm")
    args=ap.parse_args(); config_path=Path(__file__).parent/"config.json"
    config=json.loads(config_path.read_text()); private=private_dir(args.private_output)
    public=ROOT/"results/ddm_feasibility"
    audit=json.loads((public/"audit_summary.json").read_text())
    if json.loads((public/"audit_status.json").read_text())["status"]!="canonical_audit_complete":
        raise ValueError("Latest canonical audit did not complete")
    if audit["ddm_eligible_participants"] is None:
        raise ValueError("Actual canonical audit must finish before any recovery pilot")
    trials_path=private/"canonical_trials.tsv"; people_path=private/"participant_eligibility.tsv"
    state=json.loads((private/"audit_state.json").read_text())
    for key,path in [("canonical_trials_sha256",trials_path),("participant_eligibility_sha256",people_path),
                     ("config_sha256",config_path),("audit_source_sha256",Path(__file__).parent/"audit.py")]:
        if sha(path)!=state[key]:
            raise ValueError("Audited input/config changed; rerun canonical audit: "+key)
    trials=pd.read_csv(trials_path,sep="\t",dtype={"participant_id":str,"session":str})
    people=pd.read_csv(people_path,sep="\t",dtype={"participant_id":str})
    ids=select_ids(people,config); records=[]; rng=np.random.default_rng(config["seed"])
    start=time.monotonic()
    dump(public/"pilot_status.json",dict(status="running",association_analysis=False))
    dump(private/"selected_participants.json",ids)
    # Fresh independent parameter vectors per schedule and replicate, never observed-fit truths.
    for scenario,sd in config["individual_z_sd"].items():
        for rep in range(config["datasets_per_scenario"]):
            for k,pid in enumerate(ids):
                d=trials[trials.participant_id.eq(pid)].copy()
                truth=draw_truth(rng,sd)
                s=simulate(d,truth,False,config,int(rng.integers(1,2**30)))
                for model in ["shared","partner"]:
                    row,_=record_fit(s,model,config,int(rng.integers(1,2**30)),scenario,pid,rep,truth)
                    records.append(row)
                pd.DataFrame(records).to_csv(private/"recovery_fits.tsv",sep="\t",index=False)
                print(f"recovery {scenario} dataset {rep+1} person {k+1}/{len(ids)}",flush=True)
    # Small targeted stress screen, declared before data: false z contrast from drift,
    # unmodeled learning, wider bounds, z-prior strength, and Euler dt sensitivity.
    for k,pid in enumerate(spread_ids(ids,config["stress_participants"])):
        d=trials[trials.participant_id.eq(pid)].copy(); truth=draw_truth(rng,.08)
        seed=int(rng.integers(1,2**30)); s=simulate(d,truth,False,config,seed)
        row,_=record_fit(s,"partner",config,seed,"stress_baseline",pid,0,truth); records.append(row)
        for label,kwargs in [("wide_bounds",dict(wide=True)),("z_prior_sd_0.10",dict(prior_sd=.1)),("z_prior_sd_0.25",dict(prior_sd=.25))]:
            row,_=record_fit(s,"partner",config,seed,label,pid,0,truth,**kwargs); records.append(row)
        for label,learn,step in [("unmodeled_learning",True,None),("half_simulation_dt",False,config["simulation_dt_seconds"]/2)]:
            sim=simulate(d,truth,False,config,seed,learning=learn,dt=step)
            row,_=record_fit(sim,"partner",config,seed,label,pid,0,truth); records.append(row)
        theta=draw_truth(rng,.0,drift_only=True)
        sim=simulate(d,theta,False,config,seed)
        row,_=record_fit(sim,"partner",config,seed,"drift_only",pid,0,theta); records.append(row)
        # A true shared-z generator tests whether shared-z recovery is favorable only under nesting.
        theta_shared=draw_truth(rng,.08,shared=True)
        sim=simulate(d,theta_shared,True,config,seed)
        for model in ["shared","partner"]:
            row,_=record_fit(sim,model,config,seed,"true_shared_z",pid,0,theta_shared); records.append(row)
        print(f"stress screen {k+1}/{min(len(ids),config['stress_participants'])}",flush=True)
    d=pd.DataFrame(records); d.to_csv(private/"recovery_fits.tsv",sep="\t",index=False)
    summary=metrics(records); summary.to_csv(public/"recovery_summary.tsv",sep="\t",index=False)
    recovery_plots(d,public)
    observed_pilot(trials,ids,config,private,public)
    g=config["pilot_gates"]
    moderate=summary[summary.scenario.eq("moderate")&summary.model.eq("partner")&summary.parameter.isin(["z_friend","z_stranger","z_difference"])]
    passed=bool(len(moderate) and (moderate.correlation>=g["z_and_contrast_correlation_min"]).all() and
        (moderate.rmse<=g["z_and_contrast_rmse_max"]).all() and
        (moderate.boundary_fraction<=g["boundary_fraction_max"]).all() and
        (moderate.success_fraction>=g["optimization_success_fraction_min"]).all() and
        (moderate.multistart_agreement_fraction>=g["multistart_agreement_fraction_min"]).all())
    dump(public/"pilot_status.json",dict(status="pilot_complete_requires_scientific_review",core_screen_passed=passed,
        elapsed_seconds=time.monotonic()-start,participants=len(ids),simulated_fits=len(records),
        inputs={"canonical_trials_sha256":sha(trials_path),"participant_eligibility_sha256":sha(people_path),"config_sha256":sha(config_path)},
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        software={p:importlib.metadata.version(p) for p in ["numpy","scipy","pandas","matplotlib","numba"]},
        python=platform.python_version(),hierarchical_fit=False,posterior_predictive_checks=False,
        plug_in_predictive_checks=True,association_analysis=False,
        limitations=["No posterior uncertainty, interval coverage, hierarchical recovery, or group-effect inference.",
            "Fixed empirical missingness mask assumes noninformative omissions; censoring/lapse mechanisms require review.",
            "Monte Carlo pilot is too small to certify false-positive rates or approve association analyses.",
            "Shared-z fits to partner-varying truths are deliberate misspecification checks.",
            "Recovery-error correlations are not posterior parameter correlations.",
            "Learning stress uses fixed alpha and drift gain; no joint RL-DDM is fitted."]))
    print("Pilot complete. Review recovery, predictive checks, stress tests, and report before expansion.",flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        public=ROOT/"results/ddm_feasibility"
        if public.is_dir():
            dump(public/"pilot_status.json",dict(status="failed",error_type=type(error).__name__,association_analysis=False))
        raise
