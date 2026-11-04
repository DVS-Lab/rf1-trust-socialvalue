"""Full-roster, task-clock, validity, imaging and recovery contracts; synthetic only."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).parent))
import hier_data as hd
import hier_report as hr
from hierarchical import synthetic_fixture,reference_loglik,initial_values
from hier_simulation import simulate,truth_table,recovery_summary

C=hd.config()
TASK="""decision_dur=3
timer.reset()
trial_onset=globalClock.getTime()
while timer.getTime() < .5:
    shareStim.draw()
while timer.getTime() < decision_dur:
    resp_text_left.draw()
    resp_text_right.draw()
    rt=resp_onset-trial_onset
"""


def test_full_roster_includes_constant_responders_and_people_without_retained_rts():
    t=synthetic_fixture(5);t.loc[t.participant_id.eq("synthetic-4"),"analysis_valid"]=False
    data,meta=hd.model_data(t,C,"partner")
    assert data["N"]==5 and meta["informative_participants"]==4
    assert data["last"][-1]<data["first"][-1] and data["clast"][-1]<data["cfirst"][-1]
    assert "synthetic-0" in meta["ids"] and "synthetic-1" in meta["ids"]
    assert data["T"]==t.analysis_valid.sum()
    assert data["hi"]==2.5


def test_cell_cache_preserves_motor_mapping_and_history():
    t=synthetic_fixture(3);t["high_is_right"]=np.arange(len(t))%2==0
    plain,_=hd.model_data(t,C,"partner");history,_=hd.model_data(t,C,"partner_history")
    assert history["C"]>=plain["C"] and plain["C"]<=plain["T"]
    assert len(history["history"][0])==2
    assert set(plain["side"])=={-1.,1.}
    pars=initial_values(history,31)
    assert np.isfinite(reference_loglik(history,pars))


def test_timing_verifies_clock_origin_and_rejects_mismatched_duration(tmp_path):
    source=tmp_path/"task.py";source.write_text(TASK)
    result=hd.timing_source(tmp_path,C,source)
    assert result["cue_seconds"]==.5 and result["response_window_seconds_from_cue"]==3
    source.write_text(TASK.replace("decision_dur=3","decision_dur=4"))
    with pytest.raises(ValueError,match="differs"):
        hd.timing_source(tmp_path,C,source)


def test_simulation_preserves_mask_and_does_not_reveal_masked_feedback():
    t=synthetic_fixture(4);mask=t.trial_in_run.le(5)
    t.loc[mask,"analysis_valid"]=False
    alt=t.copy();alt.loc[mask,"scheduled_reciprocation"]=1-t.loc[mask,"scheduled_reciprocation"]
    data,_=hd.model_data(t,C,"partner_history")
    eta=np.tile(np.asarray(data["mu_location"]),(4,1))
    a=simulate(t,eta,0.,0.,[.8,.2],C,3,31,conditional=True)
    b=simulate(alt,eta,0.,0.,[.8,.2],C,3,31,conditional=True)
    assert a.analysis_valid.tolist()==t.analysis_valid.tolist()
    assert np.allclose(a.decision_rt,b.decision_rt,equal_nan=True)
    assert np.allclose(a.chose_high,b.chose_high,equal_nan=True)
    assert not a.loc[a.chosen_amount.eq(0),"feedback_observed"].any()


def build_upstream(tmp_path):
    root=tmp_path/"upstream";elig=[];qc=[];image=[]
    for i in range(4):
        pid=f"sub-{900000+i}"
        for run in [1,2]:
            rows=[]
            for trial in range(1,13):
                partner=["friend","stranger","computer"][(trial-1)%3]
                high= i<2 or trial%2==1
                amount=8 if high else 0; rt=.55 if trial==1 else 1.4+.02*trial
                common=dict(trial_id=str(trial),partner=partner,cLow="0",cHigh="8",cLeft="0",cRight="8",
                    choice="high" if high else "low",trust_value=str(amount),response_time=str(rt),
                    scheduled_reciprocation="recip",reciprocate="n/a")
                rows.append(dict(common,trial_type="choice_"+partner,onset=str(trial*5),duration=str(rt)))
                if amount:
                    rows.append(dict(common,trial_type="outcome_"+partner+"_recip",onset=str(trial*5+3),duration="1",reciprocate="recip"))
            rel=Path(pid)/"ses-01/func"/f"{pid}_ses-01_task-trust_run-{run}_events.tsv"
            path=root/"bids"/rel;path.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(rows).to_csv(path,sep="\t",index=False)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            elig.append(dict(participant_id=pid,session="01",run=str(run),events_path=str(rel),events_sha256=sha,
                source_excluded="false",structural_status="pass",structural_reasons=""))
            qc.append(dict(subject=pid[4:],session="01",task="trust",run=str(run),events_path="bids/"+str(rel),
                events_sha256=sha,expected_trials=12,response_trials=12,misses=0,miss_pattern="."*12))
            image.append(dict(subject=pid[4:],session="01",task="trust",run=str(run),
                fd_mean=1. if i==3 else .1+.005*i,tsnr=40.+i,brain_coverage_pct=94.9 if i==0 and run==1 else 95.))
    q=root/"qc/trust_analysis";q.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(elig).to_csv(q/"run_eligibility.tsv",sep="\t",index=False)
    (q/"source_exclusions.tsv").write_text("participant_id\n")
    (q/"provenance.json").write_text(json.dumps(dict(run_eligibility_sha256=hd.sha(q/"run_eligibility.tsv"),
        source_exclusions_sha256=hd.sha(q/"source_exclusions.tsv"),cohort_validation={"status":"passed"})))
    (root/"qc/events/results").mkdir(parents=True)
    pd.DataFrame(qc).to_csv(root/"qc/events/results/run_response_qc.tsv",sep="\t",index=False)
    m=pd.DataFrame(image);thresholds=[]
    for metric,direction in [("fd_mean","upper"),("tsnr","lower")]:
        q1,q3=m[metric].quantile([.25,.75]);iqr=q3-q1
        thresholds.append(dict(task="trust",session="01",metric=metric,n=len(m),q1=q1,q3=q3,iqr=iqr,lower_fence=q1-1.5*iqr,upper_fence=q3+1.5*iqr))
    m.to_csv(root/"qc/run_qc.tsv",sep="\t",index=False)
    pd.DataFrame(thresholds).to_csv(root/"qc/thresholds.tsv",sep="\t",index=False)
    (root/"qc/qc_policy.json").write_text(json.dumps(dict(threshold_grouping=["task","session"],iqr_multiplier=1.5)))
    (root/"qc/run_disposition.tsv").write_text("subject\tsession\ttask\trun\tdecision\treason\tevidence\n"+
        f"{900001}\t01\ttrust\t1\texclude\tSynthetic settled hold\tfixture\n")
    source=tmp_path/"task.py";source.write_text(TASK)
    return root,source


def test_prepare_full_cohort_applies_settled_run_hold_without_losing_person(tmp_path,monkeypatch):
    upstream,source=build_upstream(tmp_path)
    work=tmp_path/"private";public=tmp_path/"public"
    monkeypatch.setattr(hd,"WORK",work);monkeypatch.setattr(hd,"PUBLIC",public)
    monkeypatch.setattr(hd.subprocess,"check_output",lambda *a,**k:"synthetic-commit\n")
    t,p=hd.prepare(upstream,C,source)
    assert len(p)==4 and p.full_cohort_included.all()
    held=t.participant_id.eq(f"sub-{900001}")&t.run.eq(1)
    assert not t.loc[held,"analysis_valid"].any()
    assert t.loc[t.participant_id.eq(f"sub-{900001}")&t.run.eq(2),"analysis_valid"].all()
    assert (t.analysis_valid&t.decision_rt.lt(.2)).any()
    data,meta=hd.model_data(t,C,"partner")
    assert data["N"]==4
    summary=json.loads((public/"cohort_summary.json").read_text())
    assert summary["settled_run_holds_applied"]==1
    assert summary["imaging"]["imaging_members"]>=1
    selected=pd.read_csv(work/"imaging_run_selection.tsv",sep="\t",dtype={"participant_id":str})
    assert not selected.loc[selected.participant_id.eq(f"sub-{900000}")&selected.run.eq(1),"imaging_included"].any()
    assert selected.loc[selected.participant_id.eq(f"sub-{900000}")&selected.run.eq(2),"imaging_included"].all()
    monkeypatch.setattr(hr,"WORK",work);monkeypatch.setattr(hr,"PUBLIC",public)
    hr.report(C)
    assert "**4 participants**" in (public/"ddm_hierarchical_report.md").read_text()


def test_documented_settled_stimulus_exclusion_fallback_is_run_specific(tmp_path,monkeypatch):
    (tmp_path/"docs").mkdir()
    (tmp_path/"docs/open-decisions.md").write_text(
        f"- `{900099}` received the wrong friend image. This invalidates both Shared Reward\n"
        "  runs and Trust run 1 for their respective task analyses. Trust run 2 is not implicated.")
    work=tmp_path/"private";work.mkdir();monkeypatch.setattr(hd,"WORK",work)
    decisions,_=hd.reviewed_run_decisions(tmp_path)
    assert len(decisions)==1 and decisions.iloc[0].run==1 and decisions.iloc[0].decision=="exclude"


def test_recovery_reports_simulated_constant_responders_instead_of_dropping_them():
    t=synthetic_fixture(4);data,meta=hd.model_data(t,C,"partner")
    eta=np.tile(data["mu_location"],(4,1));truth=truth_table(eta,meta["ids"],C)
    post=truth.melt(id_vars="participant_id",var_name="parameter",value_name="mean")
    post["q025"]=post["mean"]-.1;post["q975"]=post["mean"]+.1;post["interval_width"]=.2
    cohort=pd.DataFrame(dict(participant_id=meta["ids"],preregistered_rule_pass=[False,False,True,True],
        ddm_valid=[12,24,36,36],simulated_rule_pass=[False,True,False,True],simulated_valid=[0,24,36,36]))
    pairs,summary=recovery_summary(post,truth,cohort,"test",0,"test")
    assert summary[summary.group.eq("full")].n.eq(4).all()
    assert summary[summary.group.eq("simulated_constant_or_missing")].n.eq(2).all()
    assert summary[summary.group.eq("simulated_no_observations")].n.eq(1).all()
    assert summary.coverage.eq(1).all()
