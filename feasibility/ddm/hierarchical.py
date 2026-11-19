"""Full-cohort hierarchical DDM orchestration. Scientific sampling runs on Linux2.

Local `check` verifies the native Wiener backend and gradients on synthetic data.
Participant parameters/chains never leave ignored work/ddm_hierarchical/.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from audit import ROOT,PARTNERS,sha,add_history
from hier_data import (HERE,WORK,PUBLIC,config,save,digest,source_hashes,model_data,model_kind,
    prepare,load_prepared,parameter_names)
from hier_simulation import (posterior_arrays,new_population,truth_table,participant_posterior,
    recovery_summary,simulate,ppc_summary)
from model import lower_density,lower_cdf,retained_mass


def now():return datetime.now(timezone.utc).isoformat()


def cmdstan(c):
    import cmdstanpy
    version=c["cmdstan_version"]
    candidates=[Path(os.environ["CMDSTAN"])] if os.environ.get("CMDSTAN") else []
    candidates += [ROOT/"work/cmdstan"/("cmdstan-"+version),
        ROOT/"work/full_sample/hierarchical/cmdstan"/("cmdstan-"+version),WORK/"cmdstan"/("cmdstan-"+version)]
    found=[p for p in candidates if (p/"bin/stanc").is_file()]
    if not found:raise RuntimeError("Existing CmdStan "+version+" required. Set CMDSTAN to its installation; no installation is launched automatically.")
    cmdstanpy.set_cmdstan_path(str(found[0].resolve()))
    if cmdstanpy.cmdstan_version()!=tuple(map(int,version.split(".")[:2])):
        raise RuntimeError("CmdStan version differs from configuration")
    return cmdstanpy


def compile_model(c,name="hierarchical"):
    cp=cmdstan(c);source=HERE/(name+".stan")
    folder=WORK/"build"/sha(source)[:16];folder.mkdir(parents=True,exist_ok=True)
    target=folder/source.name
    if not target.exists():shutil.copy(source,target)
    if sha(target)!=sha(source):raise ValueError("Build source changed")
    return cp.CmdStanModel(stan_file=str(target),cpp_options={"STAN_THREADS":"true","PRECOMPILED_HEADERS":"false"},
                          stanc_options={"O1":True})


def synthetic_fixture(n=5):
    rng=np.random.default_rng(27);rows=[]
    for i in range(n):
        for trial in range(1,37):
            p=PARTNERS[(trial-1)%3];low,high=[(0,2),(0,8),(2,4),(2,8),(4,8),(0,4)][(trial-1)//3%6]
            choice=1 if i==0 else 0 if i==1 else int(rng.integers(0,2))
            valid=not(i==2 and trial%4==0)
            rows.append(dict(participant_id=f"synthetic-{i}",session="01",run=1,trial_in_run=trial,
                partner=p,c_low=low,c_high=high,c_left=low,c_right=high,high_is_right=True,
                chose_high=float(choice) if valid else np.nan,chosen_amount=(high if choice else low) if valid else np.nan,
                decision_rt=float(rng.uniform(.4,1.8)) if valid else np.nan,
                response_time=float(rng.uniform(.9,2.3)) if valid else np.nan,
                analysis_valid=valid,ddm_valid=valid,valid_choice=valid,valid_recorded_choice=valid,
                missed=not valid,scientific_run_retained=True,
                scheduled_reciprocation=int(trial%2),observed_reciprocation=float(trial%2) if valid and (high if choice else low)>0 else np.nan))
    return add_history(pd.DataFrame(rows))


def initial_values(data,seed):
    rng=np.random.default_rng(seed);K=6+data["Z"]
    return dict(mu=(np.asarray(data["mu_location"])+rng.normal(0,.04,K)).tolist(),
        tau=(np.asarray(data["tau_scale"])*.5).tolist(),L_Omega=np.eye(K).tolist(),
        raw=rng.normal(0,.05,(K,data["N"])).tolist(),beta_zero=0.,beta_side=0.,beta_history=[0.]*data["H"])


def reference_loglik(data,pars):
    eta=(np.asarray(pars["mu"])[:,None]+(np.diag(pars["tau"])@np.asarray(pars["L_Omega"]))@np.asarray(pars["raw"])).T
    result=0.
    for i in range(data["N"]):
        a=np.exp(eta[i,0]);t0=data["hi"]*expit(eta[i,1])
        for row in range(data["first"][i]-1,data["last"][i]):
            cell=data["cell"][row]-1;p=data["partner"][cell]-1
            zi=6 if data["Z"]==1 else 6+p
            z=expit(eta[i,zi]+pars["beta_side"]*data["side"][cell])
            v=eta[i,2+p]+eta[i,5]*data["gap"][cell]+pars["beta_zero"]*data["zero_offer"][cell]
            v+=np.dot(data["history"][cell],pars["beta_history"])
            dt=data["rt"][row]-t0
            f=lower_density(dt,a,1-z,-v) if data["y"][row] else lower_density(dt,a,z,v)
            m=retained_mass(a,z,v,t0,data["lo"],data["hi"]);eps=data["lapse"]
            result+=np.log((1-eps)*f+eps/(2*(data["hi"]-data["lo"])))-np.log((1-eps)*m+eps)
    return float(result)


def check_implementation(c):
    PUBLIC.mkdir(parents=True,exist_ok=True); model=compile_model(c);probe=compile_model(c,"wiener_probe")
    rng=np.random.default_rng(c["seed"]);M=30
    probe_data=dict(M=M,rt=(rng.uniform(.03,2.,M)+.15).tolist(),a=rng.uniform(.7,2.5,M).tolist(),
        t0=[.15]*M,z=rng.uniform(.15,.85,M).tolist(),drift=rng.uniform(-2.,2.,M).tolist())
    pfolder=WORK/"checks"/now().replace(":","");pfolder.mkdir(parents=True,exist_ok=True)
    p=probe.sample(data=probe_data,chains=1,iter_warmup=0,iter_sampling=1,fixed_param=True,adapt_engaged=False,
        seed=c["seed"],output_dir=str(pfolder),show_progress=False,refresh=1)
    expected_pdf=[];expected_cdf=[]
    for rt,a,t0,z,v in zip(probe_data["rt"],probe_data["a"],probe_data["t0"],probe_data["z"],probe_data["drift"]):
        expected_pdf.append(np.log(lower_density(rt-t0,a,1-z,-v)))
        expected_cdf.append(lower_cdf(rt-t0,a,1-z,-v))
    pdf_error=float(np.max(np.abs(p.stan_variable("log_density")[0]-expected_pdf)))
    cdf_error=float(np.max(np.abs(np.exp(p.stan_variable("log_cdf")[0])-expected_cdf)))
    if pdf_error>1e-5 or cdf_error>1e-6:raise ValueError("Native Wiener probability check failed")
    results=[]
    for name in ["shared","partner","partner_history"]:
        data,_=model_data(synthetic_fixture(),c,name);data["likelihood_only"]=1
        for state in range(3):
            pars=initial_values(data,c["seed"]+state)
            if state==1:pars["mu"][6]+=1.0;pars["mu"][2]-=.7;pars["beta_side"]=-.3
            if state==2:pars["mu"][1]+=1.2;pars["beta_zero"]=.4
            stan=model.log_prob(params=pars,data=data,jacobian=False,sig_figs=14).iloc[0].to_numpy(float)
            ref=reference_loglik(data,pars);error=abs(stan[0]-ref)
            if error>1e-5:raise ValueError("Stan/reference joint likelihood disagreement")
            for k in [0,1,2,6]:
                delta=1e-5;plus=json.loads(json.dumps(pars));minus=json.loads(json.dumps(pars))
                plus["mu"][k]+=delta;minus["mu"][k]-=delta
                fd=(reference_loglik(data,plus)-reference_loglik(data,minus))/(2*delta)
                if not np.isclose(stan[k+1],fd,atol=2e-4,rtol=2e-4):raise ValueError("Stan/reference gradient disagreement")
            results.append(dict(model=name,state=state,loglik_absolute_error=float(error)))
    guarded=dict(data);bad=initial_values(guarded,c["seed"]);bad["mu"][0]=np.log(100.)
    save(pfolder/"guard_data.json",guarded);save(pfolder/"guard_parameters.json",bad)
    rejected=subprocess.run([model.exe_file,"log_prob","constrained_params="+str(pfolder/"guard_parameters.json"),
        "jacobian=false","data","file="+str(pfolder/"guard_data.json"),"output","file="+str(pfolder/"guard.csv")],
        capture_output=True,text=True,timeout=15)
    (pfolder/"guard.log").write_text(rejected.stdout+rejected.stderr)
    guard_passed = rejected.returncode == 0 and bool(np.isneginf(
        pd.read_csv(pfolder/"guard.csv",comment="#").iloc[0,0]))
    if not guard_passed:raise ValueError("Extreme-proposal numerical guard failed")
    status=dict(status="passed",cmdstan_version=c["cmdstan_version"],density_max_absolute_log_error=pdf_error,
        cdf_max_absolute_error=cdf_error,likelihood_gradient_states=results,numerical_guard_check=guard_passed,sources=source_hashes(),
        interpretation="Numerical implementation verification on synthetic data, not RF1 recovery or sampler convergence.")
    save(PUBLIC/"implementation_status.json",status)
    return model


def sampler_settings(c):
    return {k:c[k] for k in ["chains","parallel_chains","threads_per_chain","warmup","draws","adapt_delta","max_treedepth"]}


def diagnostics(fit,c):
    s=fit.summary();method=fit.method_variables();d=c["diagnostics"]
    # Deterministic shared-z contrasts and correlation diagonals are not MCMC variables.
    random=s[s.StdDev.gt(1e-12)].copy()
    required=[k for k in random.index if k!="lp__"]
    check=random.loc[required]
    rhat=check.R_hat.to_numpy(float);bulk=check.ESS_bulk.to_numpy(float);tail=check.ESS_tail.to_numpy(float)
    energy=np.asarray(method["energy__"]);bfmi=np.mean(np.diff(energy,axis=0)**2,axis=0)/np.var(energy,axis=0)
    result=dict(divergences=int(np.sum(method["divergent__"])),
        treedepth_hits=int(np.sum(np.asarray(method["treedepth__"])>=c["max_treedepth"])),
        max_rhat=float(np.nanmax(rhat)),min_bulk_ess=float(np.nanmin(bulk)),min_tail_ess=float(np.nanmin(tail)),
        min_bfmi=float(np.nanmin(bfmi)),nonfinite_diagnostics=int((~np.isfinite(rhat)|~np.isfinite(bulk)|~np.isfinite(tail)).sum()),
        sampled_parameters_checked=len(required))
    a=fit.stan_variable("a");t0=fit.stan_variable("t0")
    # The response-window value is constant across the full and recovery datasets.
    hi=c["response_window_seconds_from_cue"]-c["cue_seconds"]
    ratio=a*a/(hi-t0)
    result["max_numerical_domain_ratio"]=float(np.max(ratio))
    result["near_numerical_guard_fraction"]=float(np.mean(ratio>500))
    result["passed"]=bool(result["divergences"]<=d["divergences_max"] and result["treedepth_hits"]<=d["treedepth_hits_max"]
        and result["max_rhat"]<d["rhat_max"] and result["min_bulk_ess"]>=d["ess_bulk_min"]
        and result["min_tail_ess"]>=d["ess_tail_min"] and result["min_bfmi"]>d["bfmi_min"]
        and result["nonfinite_diagnostics"]==0)
    return result,s


def fit_dataset(trials,c,spec,name,seed):
    if platform.system()!="Linux":raise RuntimeError("Scientific full-cohort sampling runs on Linux2; use check/test for local synthetic validation")
    binding=json.loads((WORK/"data_binding.json").read_text())
    if source_hashes()!=binding["sources"] or sha(c["_path"])!=binding["config_sha256"]:
        raise ValueError("Source/config changed after preparation; preserve existing jobs and prepare the reviewed version")
    cp=cmdstan(c);data,meta=model_data(trials,c,spec)
    settings=sampler_settings(c);source=source_hashes()
    core={k:source[k] for k in ["hierarchical.stan","hierarchical.py","hier_data.py","hier_simulation.py","audit.py","model.py"]}
    payload=dict(data_sha256=digest(data),sources=core,settings=settings,cmdstan_version=c["cmdstan_version"],seed=seed)
    fingerprint=digest(payload);folder=WORK/"fits"/name/fingerprint[:16];public=PUBLIC/"fits"/name
    folder.mkdir(parents=True,exist_ok=True);public.mkdir(parents=True,exist_ok=True)
    state=folder/"cache.json";started=time.monotonic()
    if state.is_file():
        manifest=json.loads(state.read_text())
        if manifest["fingerprint"]!=fingerprint:raise ValueError("Cache fingerprint mismatch")
        for path,h in manifest["csv_sha256"].items():
            if sha(path)!=h:raise ValueError("Saved posterior CSV changed")
        fit=cp.from_csv(list(manifest["csv_sha256"]))
    else:
        if list(folder.glob("*.csv")):raise RuntimeError("Partial chains preserved; use a new explicit attempt/config rather than overwrite")
        model=compile_model(c);save(folder/"data.json",data);save(folder/"metadata.json",meta)
        save(folder/"target.json",dict(fingerprint=fingerprint,**payload))
        fit=model.sample(data=str(folder/"data.json"),seed=seed,chains=c["chains"],parallel_chains=c["parallel_chains"],
            force_one_process_per_chain=True,
            threads_per_chain=c["threads_per_chain"],iter_warmup=c["warmup"],iter_sampling=c["draws"],
            adapt_delta=c["adapt_delta"],max_treedepth=c["max_treedepth"],sig_figs=12,
            inits=[initial_values(data,seed+i) for i in range(c["chains"])],
            output_dir=str(folder),show_progress=False,show_console=True,refresh=100)
        if source_hashes()!=binding["sources"]:raise ValueError("Source changed during sampling; preserve chains for explicit review")
        manifest=dict(fingerprint=fingerprint,csv_sha256={str(Path(p).resolve()):sha(p) for p in fit.runset.csv_files})
        save(state,manifest)
    quality,summary=diagnostics(fit,c);summary.to_csv(folder/"diagnostics.tsv",sep="\t")
    status=dict(name=name,specification=spec,status="accepted" if quality["passed"] else "diagnostic_failed",**quality,
        participants=data["N"],valid_decisions=data["T"],informative_participants=meta["informative_participants"],
        fingerprint=fingerprint,settings=settings,sources=core,cohort_trials_sha256=sha(WORK/"trials.tsv"),
        participant_manifest_sha256=digest(meta["ids"]),elapsed_seconds=time.monotonic()-started,
        association_analysis=False,participant_subsampling=False)
    save(public/"status.json",status);save(folder/"status.json",status)
    return fit,meta,status,folder


def publish_accepted(fit,meta,status,folder,c,trials,participants):
    if not status["passed"]:return None
    draws=posterior_arrays(fit);p=participant_posterior(draws,meta["ids"])
    p.to_csv(folder/"participant_parameters.tsv",sep="\t",index=False)
    indices=range(len(meta["parameter_names"]))
    pop=[]
    for k in indices:
        for kind in ["mu","tau"]:
            x=draws[kind][:,k]
            pop.append(dict(parameter=kind+"_"+meta["parameter_names"][k],mean=float(x.mean()),sd=float(x.std()),
                q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975))))
    z=draws["z_difference"]
    for name,x in [("cohort_mean_z_difference",z.mean(axis=1)),("cohort_sd_z_difference",z.std(axis=1))]:
        pop.append(dict(parameter=name,mean=float(x.mean()),sd=float(x.std()),q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975))))
    public=PUBLIC/"fits"/meta["name"]
    # meta name is specification; observed/recovery run names can differ.
    public=PUBLIC/"fits"/status["name"]
    pd.DataFrame(pop).to_csv(public/"population_parameters.tsv",sep="\t",index=False)
    pair=[]
    for i,pid in enumerate(meta["ids"]):
        for label,x,y in [("z_friend_vs_drift_friend",draws["z"][:,i,0],draws["eta"][:,i,2]),
                          ("z_stranger_vs_drift_stranger",draws["z"][:,i,1],draws["eta"][:,i,3]),
                          ("a_vs_t0",draws["a"][:,i],draws["t0"][:,i])]:
            pair.append(dict(participant_id=pid,pair=label,correlation=float(np.corrcoef(x,y)[0,1])))
    corr=pd.DataFrame(pair);corr.to_csv(folder/"posterior_parameter_correlations.tsv",sep="\t",index=False)
    corr.groupby("pair").correlation.agg(["mean","median","min","max"]).to_csv(public/"posterior_correlation_summary.tsv",sep="\t")
    q=p.merge(participants[["participant_id","preregistered_rule_pass","ddm_valid"]],on="participant_id",validate="many_to_one")
    q.groupby(["parameter","preregistered_rule_pass"]).agg(n=("mean","size"),median_sd=("sd","median"),
        median_interval_width=("interval_width","median")).to_csv(public/"uncertainty_by_eligibility.tsv",sep="\t")
    return draws


def posterior_predictive(draws,meta,trials,participants,c,name):
    public=PUBLIC/"fits"/name;private=WORK/"predictive"/name;private.mkdir(parents=True,exist_ok=True)
    fingerprint=json.loads((public/"status.json").read_text())["fingerprint"]
    save(public/"predictive_status.json",dict(status="running",fit_fingerprint=fingerprint))
    observed=ppc_summary(trials);observed.to_csv(private/"observed_cells.tsv",sep="\t",index=False)
    rng=np.random.default_rng(c["seed"]+6300);indices=rng.choice(len(draws["eta"]),min(c["ppc_draws"],len(draws["eta"])),replace=False)
    aggregate=[];selection=[]
    for k,index in enumerate(indices):
        s=simulate(trials,draws["eta"][index],draws["beta_zero"][index],draws["beta_side"][index],draws["beta_history"][index],
                   c,meta["name"]=="shared" and 1 or 3,int(rng.integers(1,2**30)),conditional=True)
        cells=ppc_summary(s)
        cells=cells.merge(participants[["participant_id","preregistered_rule_pass"]],on="participant_id",validate="many_to_one")
        for group,mask in [("full",np.ones(len(cells),bool)),("preregistered_subset",cells.preregistered_rule_pass),
                           ("constant_or_missing_partner",~cells.preregistered_rule_pass)]:
            keys=["cell_kind","partner","c_low","c_high","chose_high"]
            # Missing keys identify coarser cells and are retained as NaN groups.
            for key,g in cells[mask].groupby(keys,dropna=False):
                row=dict(zip(keys,key));row.update(group=group,replicate=k,
                    **{metric:float(g[metric].mean()) for metric in ["high","rt10","rt50","rt90"]})
                aggregate.append(row)
        # Unconditional censoring check is separate from fixed-mask conditional PPC.
        u=simulate(trials,draws["eta"][index],draws["beta_zero"][index],draws["beta_side"][index],draws["beta_history"][index],
                   c,meta["name"]=="shared" and 1 or 3,int(rng.integers(1,2**30)),conditional=False)
        selection.append(dict(replicate=k,retained_fraction=float(u.analysis_valid.sum()/max(1,trials.analysis_valid.sum()))))
        if (k+1)%10==0:print(f"{name}: full-cohort posterior predictive {k+1}/{len(indices)}",flush=True)
    pd.DataFrame(aggregate).to_csv(public/"posterior_predictive_cells.tsv",sep="\t",index=False)
    pd.DataFrame(selection).to_csv(public/"predicted_retention.tsv",sep="\t",index=False)
    save(public/"predictive_status.json",dict(status="complete",posterior_draws=len(indices),full_cohort=True,
        fit_fingerprint=fingerprint,integrates_parameter_uncertainty=True,feedback="generative; reveal only after a retained positive investment"))


def execution_failure(name, error):
    state = dict(name=name, status="execution_failed", error_type=type(error).__name__,
                 association_analysis=False)
    match = re.search(r"retcodes=\[([-\d, ]+)\]", str(error))
    if match:
        state["chain_exit_codes"] = [int(code) for code in match[1].split(",")]
    return state


def full_fits(c):
    t,p,binding=load_prepared(c);check_implementation(c);rows=[]
    save(PUBLIC/"fit_status.json",dict(status="running",models=[]))
    names=list(c["models"])
    image=WORK/"imaging_membership.tsv"
    if image.is_file() and pd.read_csv(image,sep="\t").imaging_included.any():names.append("partner_imaging")
    for k,name in enumerate(names):
        frames=t;people=p
        if name=="partner_imaging":
            m=pd.read_csv(image,sep="\t",dtype={"participant_id":str});ids=set(m.loc[m.imaging_included,"participant_id"])
            frames=t[t.participant_id.isin(ids)].copy();people=p[p.participant_id.isin(ids)].copy()
        try:
            fit,meta,status,folder=fit_dataset(frames,c,name,name,c["seed"]+k)
            rows.append(status)
            draws=publish_accepted(fit,meta,status,folder,c,frames,people)
            if draws is not None:posterior_predictive(draws,meta,frames,people,c,name)
        except Exception as e:
            status=execution_failure(name, e)
            save(PUBLIC/"fits"/name/"status.json",status);rows.append(status)
            print(f"{name} failed: {type(e).__name__}: {e}",flush=True)
    save(PUBLIC/"fit_status.json",dict(status="complete" if all(r.get("passed",False) for r in rows) else "incomplete",
        models=rows,full_cohort=True,participant_subsampling=False))
    return all(r.get("passed",False) for r in rows)


def prior_predictive(c):
    t,p,b=load_prepared(c);data,meta=model_data(t,c,"partner");rng=np.random.default_rng(c["seed"]+4200);rows=[];rejections=0
    # Joint prior population draws, including random correlations, without data-fitted truths.
    from scipy.stats import wishart
    for k in range(c.get("prior_predictive_draws",100)):
        while True:
            K=6+data["Z"];mu=rng.normal(data["mu_location"],data["mu_scale"])
            tau=np.abs(rng.normal(0,data["tau_scale"]))
            # LKJ(eta=2) correlation via normalized Wishart(df=K+3).
            w=wishart.rvs(df=K+3,scale=np.eye(K),random_state=rng);std=np.sqrt(np.diag(w));corr=w/std[:,None]/std[None,:]
            L=np.linalg.cholesky(corr);eta=mu+rng.normal(size=(len(meta["ids"]),K))@(np.diag(tau)@L).T
            a=np.exp(eta[:,0]);t0=data["hi"]*expit(eta[:,1])
            ratio=a*a/(data["hi"]-t0)
            if np.isfinite(ratio).all() and ratio.max()<1000 and a.min()>=1e-6:break
            rejections+=1
            if rejections>10000:raise ValueError("Prior mass repeatedly exceeds the numerical domain; review priors")
        s=simulate(t,eta,rng.normal(),rng.normal(0,.5),[],c,3,int(rng.integers(1,2**30)))
        valid=s[s.analysis_valid]
        rows.append(dict(replicate=k,retained_fraction=float(s.analysis_valid.sum()/max(1,t.analysis_valid.sum())),
            high_fraction=float(valid.chose_high.mean()) if len(valid) else None,
            median_decision_rt=float(valid.decision_rt.median()) if len(valid) else None,
            extreme_participant_fraction=float(valid.groupby("participant_id").chose_high.mean().isin([0.,1.]).mean()) if len(valid) else None))
        if (k+1)%10==0:print(f"full-cohort prior predictive {k+1}",flush=True)
    pd.DataFrame(rows).to_csv(PUBLIC/"prior_predictive.tsv",sep="\t",index=False)
    save(PUBLIC/"prior_predictive_status.json",dict(status="complete",datasets=len(rows),
        numerical_domain_rejected_populations=rejections,full_cohort=True,
        note="Joint prior draws conditioned on the documented, generous numerical CDF evaluation domain."))


def recovery_inputs(draws, ids, c):
    # Draw populations and simulation seeds in the same order for every worker count.
    rng = np.random.default_rng(c["seed"] + 91000)
    regimes=[(s,c["recovery_datasets_per_regime"]) for s in c["recovery_z_logit_sd"]]+[("drift_only_null",c.get("recovery_null_datasets",2))]
    if c.get("recovery_misspecification_stress",True):regimes += [("partner_boundary_stress",1),("partner_nondecision_stress",1)]
    for regime,repetitions in regimes:
        for rep in range(repetitions):
            name=f"Recovery_{regime}_{rep+1}";index=int(rng.integers(len(draws["eta"])))
            for rejected_populations in range(10000):
                eta,bzero,bside,bhistory=new_population(draws,index,len(ids),c,regime,rng)
                a=np.exp(eta[:,0]);hi=c["response_window_seconds_from_cue"]-c["cue_seconds"]
                ratio=a*a/(hi-hi*expit(eta[:,1]))
                if np.isfinite(ratio).all() and ratio.max()<1000 and a.min()>=1e-6:break
            else:raise ValueError("Recovery populations repeatedly exceed numerical evaluation domain")
            seed=int(rng.integers(1,2**30))
            yield (name, regime, rep, index, seed, rejected_populations, eta, bzero, bside, bhistory)


def recovery_worker(job, t, p, ids, primary_fingerprint, c):
    name, regime, rep, index, seed, rejected_populations, eta, bzero, bside, bhistory = job
    population = []
    stress={}
    if regime=="partner_boundary_stress":stress["a_scale"]=[1.1,.9,1.]
    if regime=="partner_nondecision_stress":stress["t0_scale"]=[.9,1.,1.]
    s=simulate(t,eta,bzero,bside,bhistory,c,3,seed,**stress)
    truth=truth_table(eta,ids,c)
    # Misspecified common-a/t0 fits have no unique true common parameter.
    if regime=="partner_boundary_stress":truth=truth.drop(columns="a")
    if regime=="partner_nondecision_stress":truth=truth.drop(columns="t0")
    cohort=p.copy()
    simulated_pass={}
    for pid,g in s.groupby("participant_id"):
        variability=g[g.analysis_valid].groupby("partner").chose_high.nunique()
        simulated_pass[pid]=all(variability.get(partner,0)>1 for partner in ["friend","stranger"])
    cohort["simulated_rule_pass"]=cohort.participant_id.map(simulated_pass)
    simulated_n=s.groupby("participant_id").analysis_valid.sum()
    cohort["simulated_valid"]=cohort.participant_id.map(simulated_n).fillna(0).astype(int)
    private=WORK/"recovery"/name;private.mkdir(parents=True,exist_ok=True)
    truth.to_csv(private/"truth.tsv",sep="\t",index=False)
    save(private/"generator.json",dict(index=index,seed=seed,primary_fingerprint=primary_fingerprint,numerical_domain_rejected_populations=rejected_populations,
        regime=regime,misspecification=stress,full_participants=len(ids),sources=source_hashes()))
    sf,sm,ss,sfolder=fit_dataset(s,c,"partner",name,seed+1)
    if not ss["passed"]:
        return ss, None, []
    sd=publish_accepted(sf,sm,ss,sfolder,c,s,p)
    post=participant_posterior(sd,sm["ids"])
    paired,summary=recovery_summary(post,truth,cohort,regime,rep,name)
    paired.to_csv(private/"recovery_participants.tsv",sep="\t",index=False)
    delta=sd["z_difference"]
    for label,values,true in [("cohort_mean_z_difference",delta.mean(axis=1),truth.z_difference.mean()),
                              ("cohort_sd_z_difference",delta.std(axis=1),truth.z_difference.std(ddof=0))]:
        lo,hi=np.quantile(values,[.025,.975])
        boundary=label=="cohort_sd_z_difference" and true<1e-12
        population.append(dict(fit=name,regime=regime,replicate=rep,parameter=label,truth=float(true),
            estimate=float(values.mean()),q025=float(lo),q975=float(hi),covered=None if boundary else bool(lo<=true<=hi),
            truth_on_variance_boundary=boundary))
    return ss, summary, population


def full_recovery(c, workers=1):

    t, p, _ = load_prepared(c)
    fit, meta, status, _ = fit_dataset(t, c, "partner", "partner", c["seed"] + c["models"].index("partner"))
    if not status["passed"]:
        raise RuntimeError("Primary full-cohort posterior failed diagnostics; recovery generator unavailable")
    ids = meta["ids"]
    draws = posterior_arrays(fit)
    jobs = list(recovery_inputs(draws, ids, c))
    del draws, fit
    return run_recovery_jobs(jobs, workers, recovery_worker, (t, p, ids, status["fingerprint"], c))


def run_recovery_jobs(jobs, workers, worker, args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    states, summaries, population = {}, {}, {}

    def publish(stage):
        rows = [states[j[0]] for j in jobs if j[0] in states]
        if summaries:
            pd.concat([summaries[j[0]] for j in jobs if j[0] in summaries], ignore_index=True).to_csv(
                PUBLIC / "recovery_summary.tsv", sep="\t", index=False)
            pd.DataFrame([row for j in jobs for row in population.get(j[0], [])]).to_csv(
                PUBLIC / "recovery_population.tsv", sep="\t", index=False)
        save(PUBLIC / "recovery_status.json", dict(status=stage, fits=rows, full_cohort=True,
             participant_subsampling=False, datasets_planned=len(jobs), workers=workers))

    def collect(name, get_result):
        try:
            state, summary, pop = get_result()
        except Exception as error:
            print(f"{name} failed: {type(error).__name__}: {error}", flush=True)
            state, summary, pop = execution_failure(name, error), None, []
        states[name] = state
        if summary is not None:
            summaries[name], population[name] = summary, pop
        publish("running")

    publish("running")
    if workers == 1:
        for job in jobs:
            collect(job[0], lambda: worker(job, *args))
    else:
        # One worker owns each dataset's chains and outputs. Only the parent
        # updates cohort-wide tables. Spawn avoids inheriting native thread pools.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(worker, j, *args): j[0] for j in jobs}
            for future in as_completed(futures):
                collect(futures[future], future.result)
    passed = len(states) == len(jobs) and all(row.get("passed", False) for row in states.values())
    publish("complete" if passed else "incomplete")
    return passed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["prepare","check","prior","primary","fit","recovery","report","all"])
    ap.add_argument("--config",type=Path)
    ap.add_argument("--upstream",type=Path,default=Path(os.environ.get("DDM_UPSTREAM","/ZPOOL/data/projects/rf1-sra-linux2")))
    ap.add_argument("--task-source",type=Path,default=os.environ.get("DDM_TASK_SOURCE"))
    ap.add_argument("--approved-trust-imaging-manifest",type=Path)
    ap.add_argument("--recovery-workers", type=int, default=1,
                    help="Independent recovery datasets sampled concurrently after an accepted primary fit")
    args=ap.parse_args();c=config(args.config)
    workers=args.recovery_workers
    if workers < 1 or (workers != 1 and args.stage not in ("all","recovery")):
        ap.error("--recovery-workers must be positive and applies only to all/recovery")
    if workers > 1 and hasattr(os,"sched_getaffinity"):
        needed=workers*c["parallel_chains"]*c["threads_per_chain"]
        if needed > len(os.sched_getaffinity(0)):
            ap.error(f"{workers} recovery workers require an allocation of {needed} CPUs with these settings")
    WORK.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    save(PUBLIC/"run_status.json",dict(status="running",stage=args.stage,started=now()))
    ok=True
    try:
        if args.stage in ["prepare","all"]:prepare(args.upstream,c,args.task_source,args.approved_trust_imaging_manifest)
        if args.stage=="check":check_implementation(c)
        if args.stage in ["prior","all"]:prior_predictive(c)
        if args.stage=="primary":
            trials, people, _ = load_prepared(c)
            check_implementation(c)
            seed = c["seed"] + c["models"].index("partner")
            try:
                fit, meta, status, folder = fit_dataset(trials, c, "partner", "partner", seed)
            except Exception as error:
                save(PUBLIC/"fits/partner/status.json", execution_failure("partner", error))
                raise
            ok = status["passed"]
            if ok:
                draws = publish_accepted(fit, meta, status, folder, c, trials, people)
                posterior_predictive(draws, meta, trials, people, c, "partner")
        if args.stage in ["fit","all"]:ok=full_fits(c) and ok
        if args.stage in ["recovery","all"]:
            try:ok=full_recovery(c,workers) and ok
            except Exception as e:
                save(PUBLIC/"recovery_status.json",dict(status="blocked_primary_posterior_or_execution",error_type=type(e).__name__))
                print(f"Recovery unavailable: {type(e).__name__}: {e}",flush=True);ok=False
    except Exception as error:
        save(PUBLIC/"run_status.json",dict(status="failed",stage=args.stage,error_type=type(error).__name__))
        raise
    finally:
        if args.stage not in ["check"] and (PUBLIC/"cohort_summary.json").is_file():
            from hier_report import report
            report(c)
    save(PUBLIC/"run_status.json",dict(status="complete" if ok else "incomplete",stage=args.stage,finished=now()))
    if not ok:raise SystemExit(2)


if __name__=="__main__":main()
