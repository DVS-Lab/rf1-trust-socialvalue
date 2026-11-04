"""Aggregate-only full-cohort hierarchical DDM report; no outcome associations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hier_data import WORK,PUBLIC,HERE,sha,source_hashes,config,digest
from audit import PARTNERS


def plots_predictive(name):
    folder=PUBLIC/"fits"/name;private=WORK/"predictive"/name
    if not (folder/"predictive_status.json").is_file():return False
    pstatus=json.loads((folder/"predictive_status.json").read_text())
    state=json.loads((folder/"status.json").read_text())
    if pstatus.get("status")!="complete" or pstatus.get("fit_fingerprint")!=state.get("fingerprint"):return False
    if not (folder/"posterior_predictive_cells.tsv").is_file() or not (private/"observed_cells.tsv").is_file():return False
    pred=pd.read_csv(folder/"posterior_predictive_cells.tsv",sep="\t")
    obs=pd.read_csv(private/"observed_cells.tsv",sep="\t")
    fig,axs=plt.subplots(1,4,figsize=(12,3.6),constrained_layout=True)
    for ax,metric in zip(axs,["high","rt10","rt50","rt90"]):
        for i,partner in enumerate(PARTNERS):
            o=obs[obs.cell_kind.eq("partner")&obs.partner.eq(partner)][metric].mean()
            values=pred[pred.cell_kind.eq("partner")&pred.partner.eq(partner)&pred.group.eq("full")][metric]
            if len(values):
                lo,med,hi=values.quantile([.025,.5,.975])
                ax.errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt="o",color="#35678b")
                ax.scatter(i,o,marker="x",color="black",zorder=4)
        ax.set_xticks(range(3),PARTNERS,rotation=25);ax.set_title(metric)
    fig.suptitle(name+": full-cohort posterior predictive checks; RT measured from offer availability")
    fig.savefig(folder/"posterior_predictive.png",dpi=180);plt.close(fig)
    return True


def compare_estimates():
    base=PUBLIC/"fits/partner/status.json"
    if not base.is_file():return []
    records=[]
    def load(name):
        status_path=PUBLIC/"fits"/name/"status.json"
        if not status_path.is_file():return None
        status=json.loads(status_path.read_text())
        if status.get("status")!="accepted":return None
        folder=WORK/"fits"/name/status["fingerprint"][:16]
        path=folder/"participant_parameters.tsv"
        return pd.read_csv(path,sep="\t") if path.is_file() else None
    primary=load("partner")
    if primary is None:return []
    for name in ["partner_history","partner_wide_prior","partner_imaging"]:
        candidate=load(name)
        if candidate is None:continue
        joined=primary.merge(candidate,on=["participant_id","parameter"],suffixes=("_primary","_alternative"),validate="one_to_one")
        joined.to_csv(WORK/(name+"_paired_sensitivity.tsv"),sep="\t",index=False)
        for parameter,g in joined.groupby("parameter"):
            differences=(g.mean_alternative-g.mean_primary).abs()
            records.append(dict(comparison=name,parameter=parameter,n=len(g),median_absolute_change=float(differences.median()),
                q90_absolute_change=float(differences.quantile(.9)),mean_interval_width_primary=float(g.interval_width_primary.mean()),
                mean_interval_width_alternative=float(g.interval_width_alternative.mean()),
                pearson=float(g.mean_primary.corr(g.mean_alternative)) if g.mean_primary.std()>1e-12 and g.mean_alternative.std()>1e-12 else None))
    if records:pd.DataFrame(records).to_csv(PUBLIC/"sensitivity_comparison.tsv",sep="\t",index=False)
    return records


def recovery_plot():
    path=PUBLIC/"recovery_summary.tsv"
    if not path.is_file():return
    d=pd.read_csv(path,sep="\t")
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for parameter,color in [("z_friend","#35678b"),("z_stranger","#a6643c"),("z_difference","#54845f")]:
        g=d[d.group.eq("full")&d.parameter.eq(parameter)&d.regime.isin(["small","moderate","large"])]
        for ax,metric in zip(axs,["pearson","rmse","coverage"]):
            for i,regime in enumerate(["small","moderate","large"]):
                values=g[g.regime.eq(regime)][metric]
                if len(values):ax.scatter(np.full(len(values),i)+{"z_friend":-.12,"z_stranger":0.,"z_difference":.12}[parameter],values,color=color,label=parameter if i==0 else None)
            ax.set_xticks(range(3),["small","moderate","large"]);ax.set_title(metric)
    axs[0].axhline(.8,ls="--",color="gray",lw=1);axs[0].set_ylim(-.1,1.05)
    axs[2].axhline(.95,ls="--",color="gray",lw=1);axs[2].set_ylim(0,1.05);axs[0].legend(fontsize=8)
    fig.suptitle("Full-cohort hierarchical recovery; each point is a complete cohort dataset")
    fig.savefig(PUBLIC/"hierarchical_recovery.png",dpi=180);plt.close(fig)


def report(c_config=None):
    summary=PUBLIC/"cohort_summary.json"
    if not summary.is_file():return
    c=json.loads(summary.read_text());fit_path=PUBLIC/"fit_status.json";recovery_path=PUBLIC/"recovery_status.json"
    settings=c_config or config();sources=source_hashes()
    provenance=json.loads((PUBLIC/"data_provenance.json").read_text())
    f=json.loads(fit_path.read_text()) if fit_path.is_file() else {"status":"not_run","models":[]}
    r=json.loads(recovery_path.read_text()) if recovery_path.is_file() else {"status":"not_run"}
    rows=[]
    # Read status from disk: a global stage can be interrupted after a completed model.
    for path in sorted((PUBLIC/"fits").glob("*/status.json")):
        state=json.loads(path.read_text())
        current=all(sources.get(k)==v for k,v in state.get("sources",{}).items()) and bool(state.get("sources"))
        current=current and state.get("cohort_trials_sha256")==provenance.get("trials_sha256")
        current=current and all(state.get("settings",{}).get(k)==settings[k] for k in
            ["chains","parallel_chains","threads_per_chain","warmup","draws","adapt_delta","max_treedepth"])
        if path.parent.name=="partner_imaging" and (WORK/"imaging_membership.tsv").is_file():
            m=pd.read_csv(WORK/"imaging_membership.tsv",sep="\t",dtype={"participant_id":str})
            current=current and state.get("participant_manifest_sha256")==digest(sorted(m.loc[m.imaging_included,"participant_id"]))
        if not current:state=dict(state,status="stale_prior_run",passed=False)
        if not path.parent.name.startswith("Recovery_"):rows.append(state)
    predictive_current=set()
    for state in rows:
        if state.get("status")=="accepted" and plots_predictive(state["name"]):predictive_current.add(state["name"])
    if all(s.get("status")!="stale_prior_run" for s in rows):compare_estimates()
    else:f=dict(f,status="stale_prior_run")
    if r.get("status")=="complete":
        if any((s.get("sources") and any(sources.get(k)!=v for k,v in s["sources"].items())) or
               s.get("cohort_trials_sha256")!=provenance.get("trials_sha256") for s in r.get("fits",[])):
            r=dict(r,status="stale_prior_run")
        else:recovery_plot()
    lines=["# Full-cohort hierarchical Trust DDM assessment","",
        "## Status and interpretation","",
        "This analysis retains the complete certified behavioral cohort, including constant responders and valid single-run participants. "
        "Its purpose is to assess whether partial pooling supports defensible individual z and friend–stranger contrasts. "
        "It runs no age, FEVS, closeness or neural association analysis.","",
        f"Observed-fit stage: **{f['status']}**. Recovery stage: **{r['status']}**.","",
        "**A scientific GREEN decision is not automatic.** Accepted sampler diagnostics must be considered with full-cohort recovery, "
        "uncertainty calibration, RT/choice predictive fit, drift–z confounding and prior/specification sensitivity. "
        "Diagnostic failures are execution/estimation findings, not participant exclusion rules.","",
        "## Cohort and timing","",
        f"- Full behavioral roster: **{c['participants']} participants**, {c['certified_runs']} certified runs.",
        f"- Scientifically retained runs after upstream settled decisions: **{c['scientifically_retained_runs']}**; "
        f"settled run holds applied: {c['settled_run_holds_applied']}.",
        f"- Retained modeled decisions: **{c['valid_model_decisions']}**; observed missed decisions: {c['misses']}.",
        f"- Preregistered variability subset: **{c['preregistered_rule_pass']}**, recorded for comparison; it does not restrict primary hierarchical fitting.",
        f"- Participants with no retained RTs: {c['participants_without_retained_RT']}; their latent estimates, if any, have no personal likelihood information.",
        f"- Responses under 200 ms **from offer availability**: {c['decision_rt_under_200ms']}; retained with the contaminant model.",
        f"- Recorded responses beyond the nominal cue-based response window: {c['beyond_nominal_window']}; withheld from the nominal-window likelihood, without a participant exclusion.","",
        f"The verified task script has a {c['timing']['cue_seconds']:g}-s partner-only cue before offers. "
        f"Its response window ends {c['timing']['response_window_seconds_from_cue']:g} s from cue onset. "
        "The model subtracts the known cue interval and estimates remaining nondecision time. Current task code is verified; "
        "historical per-run executable versions remain a provenance limitation. Previously reported cue-based RT thresholds "
        "must not be interpreted as offer-based response thresholds.","",
        "## Imaging intersection","",
        f"Status: `{c['imaging']['scientific_imaging_eligibility']}`; overlapping imaging participants: "
        f"{c['imaging'].get('imaging_members','unresolved')}.","",
        "The separate Trust imaging intersection uses saved Trust/session FD and tSNR thresholds, coverage ≥95%, "
        "and recorded upstream holds/source exclusions. Good individual runs remain available. It uses no age restriction, "
        "TEDANA-count cutoff or effect-size exclusion. Missing imaging/QC records are not assumed to pass. "
        "Imaging eligibility does not discard valid behavior from the full behavioral fit, and this table does not certify completed activation/PPI contrasts. "
        "An imaging-member sensitivity fit uses all certified, scientifically retained behavior of those members.","",
        "## Model","",
        "Both models use a joint choice/RT Wiener likelihood with unit diffusion noise, conditional on the retained RT window, "
        "and a fixed 1% unconditional uniform contaminant mixture before conditioning. The upper boundary denotes higher investment. "
        "The shared model has one starting point across partners within each person; the partner model has friend/stranger/computer starting points. "
        "Every person is estimated jointly through a correlated noncentered population distribution (LKJ(2) correlation prior). "
        "Boundary separation is positive without the pilot's upper ceiling; z spans its natural open unit interval. "
        "Nondecision time has intrinsic support below the response window, without the pilot's 0.8-s cap. "
        "Individual drift intercepts and an offer-gap slope are partially pooled; common zero-offer drift and motor-side bias terms are included. "
        "The history sensitivity adds lagged observed partner feedback and trial position to drift; the wider-prior sensitivity doubles population prior scales. "
        "These changes mean improvement over the old pilot cannot be attributed to pooling alone.","",
        "A disclosed numerical guard rejects extreme proposals before Stan Math's CDF series can become nonterminating: "
        "a²/(deadline−t0)≤1000, a≥1e−6 and |drift|≤1e6. Prior-predictive populations are conditioned on this domain, "
        "and posterior proximity to the guard is exported. Estimates near it require numerical-domain sensitivity.","",
        "Stan's native Wiener density/CDF are used and checked against the independent reference likelihood, analytic probabilities and finite-difference gradients. "
        "The independent Euler simulator checks recovery without reusing the fitting density. "
        "Missed responses are not imputed; fixed empirical missingness assumes omissions are noninformative. "
        "Generative history reveals programmed feedback only after a retained positive investment.","",
        "## Sampler diagnostics","",
        "| Model | Status | N | Divergences | Max R-hat | Minimum bulk ESS | Minimum tail ESS |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for state in rows:
        lines.append(f"| {state['name']} | {state['status']} | {state.get('participants','—')} | {state.get('divergences','—')} | "
                     f"{state.get('max_rhat','—')} | {state.get('min_bulk_ess','—')} | {state.get('min_tail_ess','—')} |")
    lines += ["","Thresholds: zero divergences/depth-limit hits, R-hat <1.01, bulk/tail ESS ≥400, BFMI >.3. "
        "Raw chains, participant summaries and parameter correlations remain private. Only accepted fits supply scientific summaries.","",
        "## Posterior predictive fit","",
        "Predictive checks integrate parameter uncertainty and cover all participants. They compare participant-weighted partner, "
        "exact-offer and choice-specific RT cells; report the full cohort and preregistered subset separately. "
        "Conditional checks preserve the observed response mask; a separate unconditional simulation reports predicted retention.",""]
    for state in rows:
        if state["name"] in predictive_current:
            lines += [f"![{state['name']} posterior predictive](fits/{state['name']}/posterior_predictive.png)",""]
    lines += ["## Full-cohort recovery","",
        "Each recovery dataset uses the complete roster, actual partner/offer/run order and original missingness pattern. "
        "New participant effects are generated from accepted population posterior draws. Small/moderate/large latent z variability "
        "is controlled explicitly; realized natural-z variability is reported rather than assumed. "
        "Additional datasets set the friend–stranger z contrast to zero while allowing drift differences. "
        "Two further stress datasets change partner boundary separation or nondecision time by 10% with zero true z contrast, "
        "testing whether the simpler fitted structure misattributes those effects to z. "
        "No simulated constant responder is dropped. Recovery is stratified by original eligibility, realized simulated constant responding, "
        "trial count and absence of observations. Cohort mean/SD contrasts, participant ranks, bias, RMSE, 95% interval coverage "
        "and null contrast exclusions are reported. Four population datasets per regime and two null datasets remain a limited "
        "Monte Carlo calibration study; participant cases within a dataset are not independent population replications.",""]
    path=PUBLIC/"recovery_summary.tsv"
    if path.is_file() and r.get("status")=="complete":
        d=pd.read_csv(path,sep="\t")
        g=d[d.group.eq("full")&d.parameter.isin(["z_friend","z_stranger","z_difference"])]
        lines += ["| Regime | Parameter | Datasets | Median r | Median RMSE | Mean coverage |","|---|---|---:|---:|---:|---:|"]
        for (regime,parameter),x in g.groupby(["regime","parameter"]):
            lines.append(f"| {regime} | {parameter} | {len(x)} | {x.pearson.median():.3f} | {x.rmse.median():.3f} | {x.coverage.mean():.3f} |")
        lines += ["","![Hierarchical recovery](hierarchical_recovery.png)",""]
    else:lines += ["Full-cohort hierarchical recovery has not produced accepted summaries yet.",""]
    lines += ["## Review before any subsequent association analysis","",
        "Inspect predictive tails, prior/specification dependence and posterior drift–z dependence; verify recovery for constant responders "
        "and low-variation populations; inspect uncertainty rather than treating shrinkage as identification. "
        "Resolve any upstream scientific-validity reviews still open. A useful cohort effect does not establish precise participant ranking. "
        "No automatic report label can authorize downstream association tests.","",
        "The earlier independent-fit pilot and completed RL analyses are preserved. "
        "Publication includes only aggregate tables/figures/provenance. Participant identifiers, data, fitted values and chains stay under "
        "ignored `work/ddm_hierarchical/`. See [the full-cohort workflow](../../feasibility/ddm/HIERARCHICAL.md).",""]
    (PUBLIC/"ddm_hierarchical_report.md").write_text("\n".join(lines))
