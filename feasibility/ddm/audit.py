"""Read-only Trust readiness audit. Private tables always go under ignored work/.

Run from repository root. The existing strict canonical loader is reused without
calling full_sample.preflight(), freeze(), behavior(), or any association code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rf1_trust_socialvalue.full_sample import load_run

PARTNERS = ["friend", "stranger", "computer"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def private_dir(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "work"):
        raise ValueError("Participant data must stay under this repository's ignored work/ directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def contract(upstream, session):
    """Use the upstream certification; do not inspect raw sources or demographics."""
    qc = upstream / "qc"
    prov = json.loads((qc / "trust_analysis/provenance.json").read_text())
    for key, path in [
        ("run_eligibility_sha256", qc / "trust_analysis/run_eligibility.tsv"),
        ("source_exclusions_sha256", qc / "trust_analysis/source_exclusions.tsv"),
    ]:
        if sha(path) != prov[key]:
            raise ValueError("Uncertified upstream input: " + key)
    if prov["cohort_validation"]["status"] != "passed":
        raise ValueError("Canonical cohort certification did not pass")
    elig = read(qc / "trust_analysis/run_eligibility.tsv")
    exclusions = set(read(qc / "trust_analysis/source_exclusions.tsv").participant_id)
    q = read(qc / "events/results/run_response_qc.tsv")
    q = q[q.task.eq("trust") & q.session.eq(session)].copy()
    q["participant_id"] = "sub-" + q.subject
    elig = elig[elig.session.eq(session)].copy()
    if elig.duplicated(["participant_id", "session", "run"]).any():
        raise ValueError("Duplicate eligibility run")
    if not elig.source_excluded.eq(elig.participant_id.isin(exclusions).map({True:"true", False:"false"})).all():
        raise ValueError("Source exclusions disagree with eligibility contract")
    accepted = elig[elig.structural_status.eq("pass") & elig.source_excluded.eq("false") & elig.events_sha256.ne("")]
    j = accepted.merge(q, on=["participant_id", "session", "run"], suffixes=("", "_qc"), how="left", validate="one_to_one")
    if not j.events_sha256.eq(j.events_sha256_qc).all():
        raise ValueError("Trust QC event hashes disagree with eligibility")
    if not ("bids/" + j.events_path).eq(j.events_path_qc).all():
        raise ValueError("QC event paths disagree with eligibility")
    for col in ["response_trials", "misses", "expected_trials"]:
        j[col] = pd.to_numeric(j[col], errors="raise").astype(int)
    j["recorded_responses"] = j.response_trials - j.misses
    if (j.recorded_responses < 0).any() or not j.apply(lambda r: len(r.miss_pattern)==r.response_trials and r.miss_pattern.count("M")==r.misses, axis=1).all():
        raise ValueError("QC counts disagree with missingness masks")
    return elig, j


def metadata_summary(j):
    nrun = j.groupby("participant_id").size()
    responses = j.groupby("participant_id").recorded_responses.sum()
    return {
        "evidence": "published upstream QC snapshot; current BIDS files not yet verified",
        "participants_with_certified_runs": int(j.participant_id.nunique()),
        "certified_runs": len(j),
        "participants_by_run_count": {str(k): int(v) for k,v in nrun.value_counts().sort_index().items()},
        "decision_opportunities": int(j.response_trials.sum()),
        "misses": int(j.misses.sum()),
        "miss_fraction": float(j.misses.sum()/j.response_trials.sum()),
        "recorded_responses_before_rt_filter": int(j.recorded_responses.sum()),
        "responses_per_participant": {str(k):float(v) for k,v in responses.describe().items()},
        "run_lengths": {str(k):int(v) for k,v in j.response_trials.value_counts().sort_index().items()},
        "ddm_eligible_participants": None,
        "choice_variability_exclusions": None,
        "sub_200ms_fraction": None,
        "partner_choice_counts": None,
        "observed_rt_summary": None,
    }


def response_flags(d, config):
    d = d.copy()
    rt = d.response_time
    d["missed"] = ~d.valid_choice
    d["placeholder_rt"] = d.valid_choice & rt.isin(config["rt_placeholder_values"])
    d["valid_recorded_choice"] = d.valid_choice & ~d.placeholder_rt
    d["fast_rt"] = d.valid_choice & ~d.placeholder_rt & rt.lt(config["rt_min_seconds"])
    d["late_rt"] = d.valid_choice & ~d.placeholder_rt & rt.gt(config["rt_max_seconds"])
    d["ddm_valid"] = d.valid_choice & np.isfinite(rt) & ~d.placeholder_rt & ~d.fast_rt & ~d.late_rt
    d["choice_is_right"] = np.where(d.valid_recorded_choice, d.chosen_amount.eq(d.c_right).astype(float), np.nan)
    return d


def variability(d, valid_column):
    by = d[d[valid_column]].groupby("partner").chose_high.agg(["size", "nunique"])
    absent = [p for p in ["friend", "stranger"] if p not in by.index]
    constant = [p for p in ["friend", "stranger"] if p in by.index and by.loc[p,"nunique"] == 1]
    return absent, constant


def canonical_trials(upstream, j, config):
    bids = upstream / "bids"
    known=set(read(upstream/"qc/trust_analysis/run_eligibility.tsv").events_path)
    # Detect newly acquired/unlisted runs instead of silently reusing an old N.
    for path in bids.glob(f"sub-*/ses-{config['session']}/func/*task-trust*_events.tsv"):
        if re.fullmatch(r"sub-\d+_ses-\d+_task-trust_run-\d+_events\.tsv",path.name):
            if str(path.relative_to(bids)) not in known:
                raise ValueError("Uncertified canonical Trust run; refresh the upstream handoff")
        elif re.fullmatch(r"sub-\d+_ses-\d+_task-trust_run-\d+_part-(mag|phase)_events\.tsv",path.name) and read(path).empty:
            continue
        else:
            raise ValueError("Unknown or nonempty noncanonical Trust events filename")
    data, manifest = [], []
    for r in j.itertuples():
        path = bids / r.events_path
        if not path.resolve().is_relative_to(bids.resolve()):
            raise ValueError("Events path escapes canonical BIDS root")
        if not path.is_file():
            raise FileNotFoundError("Canonical event files unavailable; first missing file: " + str(path))
        if sha(path) != r.events_sha256:
            raise ValueError("Live event hash changed; refresh upstream certification before auditing")
        expected = f"{r.participant_id}_ses-{r.session}_task-trust_run-{r.run}_events.tsv"
        if path.name != expected or path.parent != bids/r.participant_id/("ses-"+r.session)/"func":
            raise ValueError("Canonical event identity mismatch")
        events = read(path)
        d = load_run(events, r.participant_id, r.session, r.run)
        if len(d) != r.response_trials or int((~d.valid_choice).sum()) != r.misses:
            raise ValueError("Live decision/miss counts disagree with upstream QC")
        data.append(response_flags(d, config))
        manifest.append(dict(events_path=r.events_path, sha256=r.events_sha256))
    if not data:
        raise ValueError("No certified Trust runs")
    return pd.concat(data, ignore_index=True), manifest


def add_history(d):
    d = d.sort_values(["participant_id", "session", "run", "trial_in_run"]).copy()
    group = d.groupby(["participant_id", "session", "run", "partner"], sort=False)
    d["previous_partner_choice"] = group.chose_high.shift(1)
    d["previous_partner_feedback"] = group.observed_reciprocation.transform(lambda s: s.ffill().shift(1))
    d["partner_trial_index"] = group.cumcount()+1
    return d


def tables(d, out):
    records = []
    for pid, g in d.groupby("participant_id", sort=True):
        missing, constant = variability(g, "valid_recorded_choice")
        ddm_missing, ddm_constant = variability(g, "ddm_valid")
        row = dict(participant_id=pid, runs=g.run.nunique(), presented=len(g),
                   recorded_choices=int(g.valid_recorded_choice.sum()), ddm_valid=int(g.ddm_valid.sum()),
                   missed=int(g.missed.sum()), under_200ms=int(g.fast_rt.sum()),
                   placeholder_rt=int(g.placeholder_rt.sum()), beyond_deadline=int(g.late_rt.sum()),
                   choice_rule_excluded=bool(constant), missing_partner=bool(missing),
                   ddm_rule_excluded=bool(ddm_constant), ddm_missing_partner=bool(ddm_missing),
                   eligible=not bool(missing or constant or ddm_missing or ddm_constant),
                   exclusion_reason=";".join(["constant_"+p for p in constant]+["missing_"+p for p in missing]+
                                             ["ddm_constant_"+p for p in ddm_constant]+["ddm_missing_"+p for p in ddm_missing]))
        for partner in PARTNERS:
            cell = g[g.partner.eq(partner)]
            row[partner+"_valid"] = int(cell.ddm_valid.sum())
            row[partner+"_high"] = int(cell.loc[cell.ddm_valid,"chose_high"].sum())
            row[partner+"_low"] = row[partner+"_valid"]-row[partner+"_high"]
        records.append(row)
    participants = pd.DataFrame(records)
    participants.to_csv(out/"participant_eligibility.tsv", sep="\t", index=False)
    byrun = d.groupby(["participant_id","session","run","partner"]).agg(
        presented=("ddm_valid","size"), recorded_choices=("valid_recorded_choice","sum"),
        ddm_valid=("ddm_valid","sum"), missed=("missed","sum"), under_200ms=("fast_rt","sum"))
    byrun.to_csv(out/"participant_partner_run.tsv", sep="\t")
    d.to_csv(out/"canonical_trials.tsv", sep="\t", index=False)
    valid = d[d.ddm_valid]
    rt = valid.groupby(["partner","chose_high","c_low","c_high"]).response_time.agg(
        n="size", median="median", q10=lambda s:s.quantile(.1), q90=lambda s:s.quantile(.9))
    rt.to_csv(out/"rt_offer_choice_cells.tsv", sep="\t")
    return participants, valid


def plots_metadata(j, public):
    fig, axes = plt.subplots(1,3,figsize=(11,3.5), constrained_layout=True)
    counts = j.groupby("participant_id").size().value_counts().sort_index()
    axes[0].bar(counts.index.astype(str),counts.values,color="#35678b")
    axes[0].set(xlabel="Certified runs",ylabel="Participants")
    axes[1].hist(j.groupby("participant_id").recorded_responses.sum(),bins=np.arange(0,90,5),color="#35678b")
    axes[1].set(xlabel="Recorded responses per participant",ylabel="Participants")
    axes[2].hist(j.misses/j.response_trials,bins=np.linspace(0,1,21),color="#a66438")
    axes[2].set(xlabel="Miss fraction per run",ylabel="Runs")
    fig.suptitle("Published QC snapshot • recorded responses before RT filtering",fontsize=11)
    fig.savefig(public/"qc_readiness.png",dpi=170); plt.close(fig)


def plots_observed(valid, public):
    if valid.empty:
        fig,ax=plt.subplots(figsize=(7,3),constrained_layout=True)
        ax.text(.5,.5,"No RT-valid decisions after configured filters",ha="center",va="center")
        ax.axis("off")
        for name in ["choice_rt_distributions.png","rt_by_offer_choice_partner.png"]:
            fig.savefig(public/name,dpi=170)
        plt.close(fig)
        return
    fig, axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for p,color in zip(PARTNERS,["#35678b","#a66438","#777777"]):
        rt=valid.loc[valid.partner.eq(p),"response_time"].sort_values()
        axes[0].plot(rt,np.arange(1,len(rt)+1)/len(rt),label=p,color=color)
    axes[0].set(xlabel="RT (s)",ylabel="Cumulative fraction"); axes[0].legend()
    for c in [0,1]:
        rt=valid.loc[valid.chose_high.eq(c),"response_time"].sort_values()
        axes[1].plot(rt,np.arange(1,len(rt)+1)/len(rt),label=["low","high"][c])
    axes[1].set(xlabel="RT (s)",ylabel="Cumulative fraction"); axes[1].legend()
    means=valid.groupby(["partner","c_low","c_high"]).chose_high.mean().unstack(0)
    means.index=[f"{int(a)}/{int(b)}" for a,b in means.index]
    means.plot.bar(ax=axes[2],rot=45)
    axes[2].set(xlabel="Offered investments",ylabel="High-choice fraction",ylim=(0,1))
    fig.savefig(public/"choice_rt_distributions.png",dpi=170); plt.close(fig)
    cells=valid.groupby(["partner","chose_high","c_low","c_high"]).response_time.median().unstack([2,3])
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    im=ax.imshow(cells,aspect="auto",vmin=.2,vmax=3)
    ax.set_xticks(np.arange(cells.shape[1]),[f"{int(a)}/{int(b)}" for a,b in cells.columns])
    ax.set_yticks(np.arange(len(cells)),[f"{p}: {'high' if c else 'low'}" for p,c in cells.index])
    ax.set_xlabel("Offered investments"); fig.colorbar(im,ax=ax,label="Median RT (s)")
    fig.savefig(public/"rt_by_offer_choice_partner.png",dpi=170); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream",type=Path,required=True)
    ap.add_argument("--metadata-only",action="store_true")
    ap.add_argument("--private-output",type=Path,default=ROOT/"work/ddm")
    ap.add_argument("--public-output",type=Path,default=ROOT/"results/ddm_feasibility")
    args=ap.parse_args()
    config=json.loads((Path(__file__).parent/"config.json").read_text())
    private=private_dir(args.private_output)
    public=args.public_output.resolve()
    if public != ROOT/"results/ddm_feasibility":
        raise ValueError("Public results must stay isolated under results/ddm_feasibility")
    public.mkdir(parents=True,exist_ok=True)
    dump(public/"audit_status.json",dict(status="running"))
    upstream=args.upstream.resolve()
    elig,j=contract(upstream,config["session"])
    summary=metadata_summary(j)
    j.to_csv(private/"certified_run_inventory.tsv",sep="\t",index=False)
    participant=j.groupby("participant_id").agg(certified_runs=("run","size"),
        opportunities=("response_trials","sum"),misses=("misses","sum"),
        recorded_responses_before_rt_filter=("recorded_responses","sum"))
    participant["ddm_eligibility"]="unassessed_requires_canonical_events"
    participant["partner_trial_counts"]="unavailable"
    roster=elig.groupby("participant_id").agg(source_excluded=("source_excluded",lambda s:s.eq("true").any()),
        unresolved_runs=("structural_status",lambda s:s.ne("pass").sum()))
    participant=roster.join(participant,how="left")
    for col in ["certified_runs","opportunities","misses","recorded_responses_before_rt_filter"]:
        participant[col]=participant[col].fillna(0).astype(int)
    participant["ddm_eligibility"]=participant.ddm_eligibility.fillna("no_certified_events")
    participant.loc[participant.source_excluded,"ddm_eligibility"]="upstream_source_excluded"
    participant["partner_trial_counts"]=participant.partner_trial_counts.fillna("unavailable")
    participant.to_csv(private/"participant_available_trials.tsv",sep="\t")
    plots_metadata(j,public)
    if not args.metadata_only:
        d,manifest=canonical_trials(upstream,j,config)
        d=add_history(d)
        participants,valid=tables(d,private)
        summary.update(evidence="canonical BIDS events verified against upstream certification",
            ddm_eligible_participants=int(participants.eligible.sum()),
            choice_variability_exclusions=int(participants.choice_rule_excluded.sum()),
            missing_partner_participants=int(participants.missing_partner.sum()),
            post_rt_variability_exclusions=int(participants.ddm_rule_excluded.sum()),
            ddm_valid_decisions=len(valid),
            under_200ms=int(d.fast_rt.sum()),
            sub_200ms_fraction=float(d.fast_rt.sum()/d.valid_choice.sum()),
            placeholder_rt=int(d.placeholder_rt.sum()),
            beyond_deadline=int(d.late_rt.sum()),
            partner_choice_counts=valid.groupby("partner").chose_high.agg(["size","sum"]).rename(columns={"size":"valid","sum":"high"}).astype(int).to_dict("index"),
            observed_rt_summary={p:{str(k):float(v) for k,v in g.response_time.describe(percentiles=[.1,.5,.9]).items()} for p,g in valid.groupby("partner")})
        dump(private/"events_manifest.json",manifest)
        plots_observed(valid,public)
    dump(public/"audit_summary.json",summary)
    provenance={"upstream_git_sha":subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip(),
        "metadata_only":args.metadata_only,"config_sha256":sha(Path(__file__).parent/"config.json"),
        "audit_source_sha256":sha(__file__),"loader_source_sha256":sha(ROOT/"src/rf1_trust_socialvalue/full_sample.py"),
        "inputs":{str(p.relative_to(upstream)):sha(p) for p in [upstream/"qc/trust_analysis/run_eligibility.tsv",upstream/"qc/trust_analysis/source_exclusions.tsv",upstream/"qc/trust_analysis/provenance.json",upstream/"qc/events/results/run_response_qc.tsv"]}}
    dump(public/"audit_provenance.json",provenance)
    if not args.metadata_only:
        dump(private/"audit_state.json",dict(canonical_trials_sha256=sha(private/"canonical_trials.tsv"),
            participant_eligibility_sha256=sha(private/"participant_eligibility.tsv"),
            config_sha256=sha(Path(__file__).parent/"config.json"),audit_source_sha256=sha(__file__)))
    dump(public/"audit_status.json",dict(status="metadata_only" if args.metadata_only else "canonical_audit_complete"))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        public=ROOT/"results/ddm_feasibility"
        if public.is_dir():
            # Do not publish paths or participant IDs from exception strings.
            dump(public/"audit_status.json",dict(status="failed",error_type=type(error).__name__))
        raise
