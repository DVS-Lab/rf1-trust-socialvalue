"""Full-cohort data contract for hierarchical DDMs; no outcome covariates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.special import logit
from audit import ROOT,PARTNERS,sha,read,contract,canonical_trials,add_history,tables

HERE=Path(__file__).parent
WORK=ROOT/"work/ddm_hierarchical"
PUBLIC=ROOT/"results/ddm_hierarchical"
TIMING_REFERENCE_SHA="111ea88bfcf20a539093e155d48f3a995c9299f12a25b995d82244659eef65be"
SOURCES=["hierarchical.json","hierarchical.stan","wiener_probe.stan","hier_data.py",
         "hierarchical.py","hier_simulation.py","hier_report.py","audit.py","model.py"]


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def config(path=None):
    path=Path(path or HERE/"hierarchical.json").resolve()
    c=json.loads(path.read_text()); c["_path"]=str(path)
    if c["session"]!="01" or c["choice_variability_exclusion"] or c["fit_cohort"]!="all_certified_behavioral_participants":
        raise ValueError("This phase fits all certified session-01 behavior, retaining constant responders")
    if not 0<c["contaminant_probability"]<.1 or c["cue_seconds"]>=c["response_window_seconds_from_cue"]:
        raise ValueError("Invalid timing/contaminant specification")
    if c["chains"]<4 or c["parallel_chains"]>c["chains"]:
        raise ValueError("At least four chains required")
    return c


def source_hashes():
    result={name:sha(HERE/name) for name in SOURCES}
    result["src/rf1_trust_socialvalue/full_sample.py"]=sha(ROOT/"src/rf1_trust_socialvalue/full_sample.py")
    return result


def timing_source(upstream,c,explicit=None):
    candidates=[Path(explicit)] if explicit else [
        upstream.parent/"rf1-sra/stimuli/Scan-Investment_Game/investment_game.py",
        upstream/"stimuli/Scan-Investment_Game/investment_game.py",
        ROOT/"data/ds005123/sourcedata/Scan-Investment_Game/investment_game.py"]
    found=[p for p in candidates if p.is_file()]
    if not found:
        raise FileNotFoundError("Task code required for timing verification; set DDM_TASK_SOURCE or pass --task-source. No raw logs are needed.")
    path=found[0]; text=path.read_text(); compact=re.sub(r"\s+","",text)
    duration=re.search(r"(?m)^decision_dur\s*=\s*([0-9.]+)\s*(?:#.*)?$",text)
    loops=list(re.finditer(r"whiletimer\.getTime\(\)<([^:]+):",compact))
    cue=[m for m in loops if re.fullmatch(r"[0-9.]+",m.group(1))]
    if not duration or len(cue)!=1:
        raise ValueError("Unrecognized task timing; inspect task code before fitting")
    first=cue[0]; end=compact.find("whiletimer.getTime()<decision_dur:",first.end())
    body=compact[first.end():end]
    if end<0 or "resp_text_left.draw()" in body or "resp_text_left.draw()" not in compact[end:]:
        raise ValueError("Cannot verify partner-only cue before offers")
    if "trial_onset=globalClock.getTime()" not in compact[:first.start()] or "rt=resp_onset-trial_onset" not in compact:
        raise ValueError("Cannot verify RT clock origin")
    if float(duration.group(1))!=c["response_window_seconds_from_cue"] or float(first.group(1))!=c["cue_seconds"]:
        raise ValueError("Task timing differs from hierarchical config; resolve explicitly")
    return dict(path=str(path.resolve()),sha256=sha(path),cue_seconds=float(first.group(1)),
                response_window_seconds_from_cue=float(duration.group(1)),
                identical_to_released_reference=sha(path)==TIMING_REFERENCE_SHA,
                interpretation="Current script verified; historical per-run executable versions are not independently certified.")


def model_kind(name):
    allowed={"shared":(1,0,1.),"partner":(3,0,1.),"partner_history":(3,2,1.),"partner_wide_prior":(3,0,2.),"partner_imaging":(3,0,1.)}
    if name not in allowed:raise ValueError("Unknown hierarchical DDM specification")
    return allowed[name]


def parameter_names(Z):
    return ["log_a","logit_t0_fraction","v_friend","v_stranger","v_computer","beta_gap"]+(
        ["logit_z_shared"] if Z==1 else ["logit_z_friend","logit_z_stranger","logit_z_computer"])


def reviewed_run_decisions(upstream):
    """Preserve structured upstream holds; public settled decisions are a fallback.

    IDs and participant-specific rationales are written only to ignored work/.
    No QC outlier is turned into a behavioral exclusion here.
    """
    rows=[];sources={}
    for rel in ["qc/run_disposition.tsv","code/task_availability_holds.tsv"]:
        path=upstream/rel
        if not path.is_file():continue
        d=read(path);sources[rel]=sha(path)
        if not {"subject","session","task","run","reason"}.issubset(d):raise ValueError("Invalid upstream run-decision schema")
        d=d[d.task.eq("trust")&d.session.str.zfill(2).eq("01")].copy()
        for r in d.to_dict("records"):
            decision=r.get("decision","review")
            if decision not in {"include","exclude","review","unavailable"}:raise ValueError("Unknown upstream run decision")
            rows.append(dict(participant_id="sub-"+r["subject"].removeprefix("sub-"),run=int(r["run"]),
                decision=decision,source=rel,reason=r["reason"]))
    docs=upstream/"docs/open-decisions.md"
    if docs.is_file():
        sources["docs/open-decisions.md"]=sha(docs)
        text=docs.read_text()
        pattern=r"- `(\d+)` received the wrong friend image\.[\s\S]*?Trust run (\d+) for their respective task analyses\."
        for pid,run in re.findall(pattern,text):
            key=("sub-"+pid,int(run))
            # A later explicit structured adjudication is authoritative.
            if not any((r["participant_id"],r["run"])==key for r in rows):
                rows.append(dict(participant_id=key[0],run=key[1],decision="exclude",source="docs/open-decisions.md",
                    reason="Settled upstream task-stimulus validity decision; preserve other runs."))
    d=pd.DataFrame(rows,columns=["participant_id","run","decision","source","reason"])
    if len(d):
        for _,g in d.groupby(["participant_id","run"]):
            if len(set(g.decision)-{"include"})>1:raise ValueError("Conflicting upstream Trust run decisions")
    d.to_csv(WORK/"reviewed_run_decisions.tsv",sep="\t",index=False)
    return d,sources


def model_data(trials,c,name):
    """Cell-cached normalizers; full participant roster, even if a person has no retained RT."""
    Z,H,scale=model_kind(name); ids=sorted(trials.participant_id.unique()); K=6+Z
    first=[];last=[];cfirst=[];clast=[];cells=[];rts=[];ys=[];cell_index=[]
    lookup={p:i+1 for i,p in enumerate(PARTNERS)}
    for pid in ids:
        g=trials[trials.participant_id.eq(pid)].sort_values(["run","trial_in_run"])
        valid=g[g.analysis_valid]
        first.append(len(rts)+1);cfirst.append(len(cells)+1); mapping={}
        for r in valid.itertuples():
            history=(float(0 if pd.isna(r.previous_partner_feedback) else r.previous_partner_feedback-.5),
                     (float(r.trial_in_run)-21.5)/42) if H else ()
            key=(lookup[r.partner],float((r.c_high-r.c_low)/8-.5),float(r.c_low==0),float(2*r.high_is_right-1),*history)
            if key not in mapping:
                cells.append(key);mapping[key]=len(cells)
            cell_index.append(mapping[key]);rts.append(float(r.decision_rt));ys.append(int(r.chose_high))
        last.append(len(rts));clast.append(len(cells))
    hi=c["response_window_seconds_from_cue"]-c["cue_seconds"]
    lo=max(0.,c["raw_rt_min_seconds"]-c["cue_seconds"])
    a=np.asarray(cells,float).reshape((-1,4+H))
    locations=[np.log(1.5),logit(.5/hi),.5,0.,-.1,-.5]+[0.]*Z
    data=dict(N=len(ids),T=len(rts),C=len(cells),Z=Z,H=H,grainsize=c["grainsize"],person=list(range(1,len(ids)+1)),
        first=first,last=last,cfirst=cfirst,clast=clast,cell=cell_index,
        partner=a[:,0].astype(int).tolist(),gap=a[:,1].tolist(),zero_offer=a[:,2].tolist(),side=a[:,3].tolist(),
        history=a[:,4:].tolist(),rt=rts,y=ys,lo=lo,hi=hi,lapse=c["contaminant_probability"],
        mu_location=list(map(float,locations)),mu_scale=[.6,1.,1.5,1.5,1.5,1.]+[.75]*Z,
        tau_scale=[.4,.7,1.,1.,1.,.75]+[.6]*Z,prior_scale=scale,prior_only=0,likelihood_only=0)
    if any(not lo<r<=hi for r in rts):raise ValueError("Decision RT outside model support")
    return data,dict(ids=ids,parameter_names=parameter_names(Z),name=name,
        informative_participants=sum(b>=a for a,b in zip(first,last)),full_participants=len(ids),trials=len(rts),cells=len(cells))


def imaging_annotations(upstream,trials,manifest=None,decisions=None):
    ids=sorted(trials.participant_id.unique()); people=pd.DataFrame({"participant_id":ids})
    result={"scientific_imaging_eligibility":"unresolved_no_approved_Trust_manifest","imaging_members":None}
    qcpath=upstream/"qc/run_qc.tsv"
    if qcpath.is_file():
        qc=read(qcpath)
        if {"subject","session","task","run"}.issubset(qc):
            qc=qc[qc.task.eq("trust")&qc.session.str.removeprefix("ses-").str.zfill(2).eq("01")].copy()
            qc["participant_id"]="sub-"+qc.subject.str.removeprefix("sub-")
            full_reference=qc.copy()
            qc=qc[qc.participant_id.isin(ids)]
            if qc.duplicated(["participant_id","session","run"]).any():raise ValueError("Duplicate upstream Trust imaging QC row")
            qc.to_csv(WORK/"imaging_qc_annotations.tsv",sep="\t",index=False)
            result.update(qc_table_sha256=sha(qcpath),qc_matched_runs=len(qc),qc_matched_participants=int(qc.participant_id.nunique()),
                qc_interpretation="Recorded imaging QC flags are annotations, not newly adjudicated exclusions. Readiness is not scientific eligibility.")
            threshold_path=upstream/"qc/thresholds.tsv";policy_path=upstream/"qc/qc_policy.json"
            if threshold_path.is_file() and policy_path.is_file():
                policy=json.loads(policy_path.read_text())
                if policy.get("threshold_grouping")!=["task","session"] or policy.get("iqr_multiplier")!=1.5:
                    raise ValueError("Upstream imaging thresholds do not use the settled task/session policy")
                thresholds=read(threshold_path)
                thresholds=thresholds[thresholds.task.eq("trust")&thresholds.session.eq("01")]
                limits={}
                for metric,fence in [("fd_mean","upper_fence"),("tsnr","lower_fence")]:
                    found=thresholds[thresholds.metric.eq(metric)]
                    if len(found)!=1:raise ValueError("Missing/duplicate Trust imaging threshold")
                    r=found.iloc[0];values=pd.to_numeric(full_reference[metric],errors="coerce").dropna().to_numpy(float)
                    if not np.isfinite(values).all():raise ValueError("Nonfinite canonical QC values")
                    q1,q3=np.quantile(values,[.25,.75],method="linear");iqr=q3-q1
                    expected={"q1":q1,"q3":q3,"iqr":iqr,"lower_fence":q1-1.5*iqr,"upper_fence":q3+1.5*iqr}
                    if int(r["n"])!=len(values) or any(not np.isclose(float(r[k]),v,rtol=1e-11,atol=1e-11) for k,v in expected.items()):
                        raise ValueError("Saved Trust imaging fence does not reproduce full canonical task/session QC")
                    limits[metric]=float(r[fence])
                for col in ["fd_mean","tsnr","brain_coverage_pct"]:qc[col]=pd.to_numeric(qc[col],errors="coerce")
                qc["qc_numeric_complete"]=np.isfinite(qc[["fd_mean","tsnr","brain_coverage_pct"]]).all(axis=1)
                qc["qc_numeric_complete"] &= qc.fd_mean.ge(0)&qc.tsnr.gt(0)&qc.brain_coverage_pct.between(0,100)
                qc["high_fd"]=qc.fd_mean.gt(limits["fd_mean"])
                qc["low_tsnr"]=qc.tsnr.lt(limits["tsnr"])
                qc["low_coverage"]=qc.brain_coverage_pct.lt(95.)
                for saved_flag,current in [("fd_mean_outlier","high_fd"),("tsnr_outlier","low_tsnr")]:
                    if saved_flag in qc:
                        finite=qc.qc_numeric_complete
                        if not qc.loc[finite,saved_flag].str.lower().eq(qc.loc[finite,current].map({True:"true",False:"false"})).all():
                            raise ValueError("Saved Trust imaging flag disagrees with its metric/fence")
                qc["settled_hold"]=False
                if decisions is not None:
                    held={(r.participant_id,int(r.run)) for r in decisions.itertuples() if r.decision!="include"}
                    qc["settled_hold"]=[(pid,int(run)) in held for pid,run in zip(qc.participant_id,qc.run)]
                behavior_runs=set(zip(trials.loc[trials.scientific_run_retained,"participant_id"],trials.loc[trials.scientific_run_retained,"run"].astype(int)))
                qc["behavior_run_available"]=[(pid,int(run)) in behavior_runs for pid,run in zip(qc.participant_id,qc.run)]
                qc["imaging_included"]=qc.qc_numeric_complete & ~qc.high_fd & ~qc.low_tsnr & ~qc.low_coverage & ~qc.settled_hold & qc.behavior_run_available
                qc.to_csv(WORK/"imaging_run_selection.tsv",sep="\t",index=False)
                counts=qc.groupby("participant_id").imaging_included.sum()
                people["imaging_included"]=people.participant_id.map(counts).fillna(0).gt(0)
                people["imaging_usable_runs"]=people.participant_id.map(counts).fillna(0).astype(int)
                people.to_csv(WORK/"imaging_membership.tsv",sep="\t",index=False)
                result.update(scientific_imaging_eligibility="derived_Trust_run_QC_intersection_preserving_upstream_decisions",
                    imaging_members=int(people.imaging_included.sum()),passing_runs=int(qc.imaging_included.sum()),
                    threshold_sha256=sha(threshold_path),policy_sha256=sha(policy_path),thresholds=limits,
                    minimum_coverage_percent=95.,high_fd_runs=int(qc.high_fd.sum()),low_tsnr_runs=int(qc.low_tsnr.sum()),
                    low_coverage_runs=int(qc.low_coverage.sum()),missing_qc_runs=int((~qc.qc_numeric_complete).sum()),
                    interpretation="Separate imaging sensitivity: passing Trust runs only, retain good single runs, no age restriction, no TEDANA-count exclusion. This is not certification of completed imaging contrasts.")
    if manifest:
        path=Path(manifest); m=pd.read_csv(path,sep="\t" if path.suffix==".tsv" else ",",dtype=str,keep_default_na=False)
        col="participant_id" if "participant_id" in m else "subject"
        if col not in m or not {"session","run"}.issubset(m):raise ValueError("Approved Trust imaging manifest requires subject/participant_id, session, run")
        if "task" in m and not m.task.eq("trust").all():raise ValueError("Imaging manifest is not Trust-specific")
        m["participant_id"]="sub-"+m[col].str.removeprefix("sub-")
        m["session"]=m.session.str.removeprefix("ses-").str.zfill(2)
        m=m[m.session.eq("01")].copy()
        if m.duplicated(["participant_id","session","run"]).any():raise ValueError("Duplicate approved imaging run")
        if "include" in m:
            if not m.include.isin(["true","false","1","0"]).all():raise ValueError("Ambiguous imaging include values")
            m=m[m.include.isin(["true","1"])]
        matches=set(m.participant_id)&set(ids)
        if not matches:raise ValueError("Approved imaging list has no behavioral overlap")
        people["imaging_included"]=people.participant_id.isin(matches)
        people.to_csv(WORK/"imaging_membership.tsv",sep="\t",index=False)
        result.update(scientific_imaging_eligibility="user_supplied_approved_Trust_run_manifest",
                      manifest_sha256=sha(path),imaging_members=len(matches),
                      interpretation="At least one approved imaging run; behavioral parameters still use all certified behavior for these participants.")
    save(PUBLIC/"imaging_status.json",result)
    return result


def prepare(upstream,c,task_source=None,imaging_manifest=None):
    upstream=Path(upstream).resolve();WORK.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    save(PUBLIC/"preparation_status.json",dict(status="running"))
    # Preparation never writes to the older pilot or RL result directories.
    timing=timing_source(upstream,c,task_source)
    elig,j=contract(upstream,c["session"])
    audit_config=json.loads((HERE/"config.json").read_text())
    t,events=canonical_trials(upstream,j,audit_config); t=add_history(t)
    t["decision_rt"]=t.response_time-c["cue_seconds"]
    t["analysis_valid"]=t.valid_recorded_choice & t.response_time.ge(c["raw_rt_min_seconds"]) & t.decision_rt.gt(0) & t.response_time.le(c["response_window_seconds_from_cue"])
    decisions,decision_sources=reviewed_run_decisions(upstream)
    held={(r.participant_id,int(r.run)) for r in decisions.itertuples() if r.decision!="include"}
    t["scientific_run_retained"]=[(pid,int(run)) not in held for pid,run in zip(t.participant_id,t.run)]
    t["analysis_valid"] &= t.scientific_run_retained
    t["valid_recorded_choice"] &= t.scientific_run_retained
    t["ddm_valid"]=t.analysis_valid
    p,valid=tables(t,WORK)
    p["preregistered_rule_pass"]=p.eligible
    p["full_cohort_included"]=True
    for partner in PARTNERS:
        for direction,value in [("always_high",1),("always_low",0)]:
            by=t[t.valid_recorded_choice&t.partner.eq(partner)].groupby("participant_id").chose_high.agg(["size","min","max"])
            yes=set(by.index[(by["size"]>0)&by["min"].eq(value)&by["max"].eq(value)])
            p[partner+"_"+direction]=p.participant_id.isin(yes)
    p.to_csv(WORK/"participants.tsv",sep="\t",index=False)
    t.to_csv(WORK/"trials.tsv",sep="\t",index=False)
    save(WORK/"events_manifest.json",events)
    timing_private=dict(timing); timing_public={k:v for k,v in timing.items() if k!="path"}
    save(WORK/"task_timing.json",timing_private)
    image=imaging_annotations(upstream,t,imaging_manifest or c["imaging_run_manifest"],decisions)
    counts=dict(participants=len(p),certified_runs=len(j),scientifically_retained_runs=int(t[t.scientific_run_retained].groupby(["participant_id","run"]).ngroups),
        presented_decisions=len(t),recorded_choices=int(t.valid_choice.sum()),
        valid_model_decisions=int(t.analysis_valid.sum()),misses=int(t.missed.sum()),
        preregistered_rule_pass=int(p.preregistered_rule_pass.sum()),
        choice_rule_excluded=int(p.choice_rule_excluded.sum()),
        participants_without_retained_RT=int(p.ddm_valid.eq(0).sum()),
        decision_rt_under_200ms=int((t.analysis_valid&t.decision_rt.lt(.2)).sum()),
        decision_rt_under_200ms_policy="Retained with the prespecified contaminant mixture; not a participant exclusion.",
        beyond_nominal_window=int((t.valid_recorded_choice&t.response_time.gt(c["response_window_seconds_from_cue"])).sum()),
        settled_run_holds_applied=len(set(zip(t.loc[~t.scientific_run_retained,"participant_id"],t.loc[~t.scientific_run_retained,"run"]))),
        excluded_choice_directions={k:int(p[k].sum()) for k in p if k.endswith(("always_high","always_low"))},
        timing=timing_public,imaging=image)
    binding=dict(trials_sha256=sha(WORK/"trials.tsv"),participants_sha256=sha(WORK/"participants.tsv"),
        config_sha256=sha(c["_path"]),upstream_git_sha=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip(),
        events_manifest_sha256=sha(WORK/"events_manifest.json"),timing_sha256=timing["sha256"],
        upstream_decision_sources=decision_sources,sources=source_hashes())
    save(WORK/"data_binding.json",binding);save(PUBLIC/"cohort_summary.json",counts)
    save(PUBLIC/"data_provenance.json",binding)
    save(PUBLIC/"preparation_status.json",dict(status="complete",full_cohort=True,participant_subsampling=False))
    return t,p


def load_prepared(c):
    binding=json.loads((WORK/"data_binding.json").read_text())
    for key,p in [("trials_sha256",WORK/"trials.tsv"),("participants_sha256",WORK/"participants.tsv"),("config_sha256",Path(c["_path"]))]:
        if sha(p)!=binding[key]:raise ValueError("Prepared data/config changed; rerun prepare and preserve prior runs")
    if source_hashes()!=binding["sources"]:raise ValueError("Implementation changed; rerun prepare to bind the new source")
    t=pd.read_csv(WORK/"trials.tsv",sep="\t",dtype={"participant_id":str,"session":str})
    p=pd.read_csv(WORK/"participants.tsv",sep="\t",dtype={"participant_id":str})
    return t,p,binding
