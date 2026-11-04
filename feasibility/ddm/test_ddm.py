"""Scientific contracts for the isolated DDM prototype; no real data required."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad

sys.path.insert(0,str(Path(__file__).parent))
from model import lower_density,lower_cdf,upper_probability,retained_mass,simulate_paths,simulate_learning,fit
from audit import response_flags,variability,contract,load_run

CONFIG=json.loads((Path(__file__).parent/"config.json").read_text())


def test_wiener_normalization_and_cdf_derivative():
    for a,z,v in [(1.4,.5,0.),(.5,.1,4.),(3.,.9,-4.),(2.,.3,-1.),(1.4,.7,1.)]:
        low=quad(lambda t:lower_density(t,a,z,v),0,50,epsabs=1e-8,points=[.001,.01,.1,1.])[0]
        up=quad(lambda t:lower_density(t,a,1-z,-v),0,50,epsabs=1e-8,points=[.001,.01,.1,1.])[0]
        assert abs(low+up-1)<1e-6
        assert abs(up-upper_probability(a,z,v))<1e-6
        for t in [.01,.1,.5,2.]:
            numeric=(lower_cdf(t+1e-6,a,z,v)-lower_cdf(t-1e-6,a,z,v))/2e-6
            assert np.isclose(numeric,lower_density(t,a,z,v),atol=1e-6,rtol=1e-4)
        mass=quad(lambda t:lower_density(t-.15,a,z,v)+lower_density(t-.15,a,1-z,-v),.2,3,epsabs=1e-8)[0]
        assert np.isclose(mass,retained_mass(a,z,v,.15,.2,3),atol=1e-7)


def test_simulator_agrees_with_independent_analytic_probability():
    n=20000; p=np.zeros(n,dtype=np.int64); zero=np.zeros(n)
    theta=np.array([1.4,.2,.5,0.,0.,0.,.6])
    c,rt=simulate_paths(p,zero,zero,zero,theta,True,.00025,19)
    assert abs(c.mean()-upper_probability(1.4,.6,.5))<.015
    # Analytic unconditional mean decision time: (a*P(upper)-a*z)/v.
    expected=.2+(1.4*upper_probability(1.4,.6,.5)-1.4*.6)/.5
    assert abs(rt.mean()-expected)<.02


def test_missed_placeholder_and_fast_rts_never_enter_likelihood():
    d=pd.DataFrame(dict(valid_choice=[False,True,True,True,True,True],
        response_time=[3.,999.,.199,.2,1.,3.01],chosen_amount=[np.nan,8,8,8,0,8],c_right=[8]*6))
    result=response_flags(d,CONFIG)
    assert result.ddm_valid.tolist()==[False,False,False,True,True,False]
    assert np.isnan(result.loc[0,"choice_is_right"])


def test_choice_variability_is_condition_specific_and_misses_do_not_count():
    d=pd.DataFrame(dict(partner=["friend","friend","stranger","stranger","computer"],
        chose_high=[1.,np.nan,0.,1.,0.],valid_choice=[True,False,True,True,True]))
    assert variability(d,"valid_choice")==([], ["friend"])
    d.loc[0,"valid_choice"]=False
    assert variability(d,"valid_choice")==(["friend"], [])


def test_certificate_detects_modified_private_exclusion_table(tmp_path):
    q=tmp_path/"qc/trust_analysis"; q.mkdir(parents=True)
    (q/"provenance.json").write_text(json.dumps(dict(run_eligibility_sha256="bad",source_exclusions_sha256="bad")))
    (q/"run_eligibility.tsv").write_text("participant_id\n")
    import pytest
    with pytest.raises(ValueError,match="Uncertified"):
        contract(tmp_path,"01")


def synthetic_schedule(repeats=2):
    """Nominal engineering fixture only; never represented as RF1 observed data."""
    pairs=[(0,2),(0,4),(0,8),(2,4),(2,8),(4,8),(2,8)]
    rows=[]; rng=np.random.default_rng(32)
    for run in range(1,repeats+1):
        design=[(p,lo,hi,feedback) for p in ['friend','stranger','computer'] for lo,hi in pairs for feedback in [0,1]]
        for i,k in enumerate(rng.permutation(len(design)),1):
            p,lo,hi,feedback=design[k]
            rows.append(dict(participant_id="synthetic",session="01",run=run,trial_in_run=i,partner=p,
                c_low=lo,c_high=hi,high_is_right=bool(rng.integers(0,2)),ddm_valid=True,
                chose_high=0.,response_time=.7,scheduled_reciprocation=feedback,
                previous_partner_feedback=np.nan,chosen_amount=lo))
    return pd.DataFrame(rows)


def test_miss_and_zero_feedback_contract_in_reused_canonical_loader():
    common=dict(cLeft="0",cRight="8",cLow="0",cHigh="8",partner="friend",reciprocate="n/a",scheduled_reciprocation="recip")
    e=pd.DataFrame([
        dict(common,trial_id="1",onset="0",duration="3",trial_type="missed_trial",response_time="3",trust_value="n/a",choice="n/a"),
        dict(common,trial_id="2",onset="5",duration=".5",trial_type="choice_friend",response_time=".5",trust_value="0",choice="low")])
    d=load_run(e,"synthetic","01",1)
    result=response_flags(d,CONFIG)
    assert result.ddm_valid.tolist()==[False,True]
    assert not result.feedback_observed.any()


def test_learning_has_no_feedback_leak_on_missing_choices():
    n=40; p=np.zeros(n,dtype=np.int64); gap=np.zeros(n)
    low=np.zeros(n); high=np.full(n,8.); run=np.ones(n,dtype=np.int64)
    mask=np.zeros(n,dtype=np.bool_); theta=np.array([1.4,.2,.2,0.,0.,0.,.5,.5,.5])
    c1,t1=simulate_learning(p,gap,low,high,np.zeros(n),run,mask,theta,.0005,13,.2,3.)
    c2,t2=simulate_learning(p,gap,low,high,np.ones(n),run,mask,theta,.0005,13,.2,3.)
    assert np.array_equal(c1,c2) and np.array_equal(t1,t2)


def test_fit_and_recovery_pipeline_on_nominal_engineering_fixture():
    from pilot import simulate,record_fit,metrics
    d=synthetic_schedule(); theta=np.array([1.4,.2,.4,0.,-.2,-.5,.6,.4,.5])
    s=simulate(d,theta,False,CONFIG,92)
    short=dict(CONFIG,multistarts=2)
    row,result=record_fit(s,"partner",short,31,"synthetic_test","synthetic",0,theta)
    assert np.isfinite(result['objective'])
    assert set(result['parameters'])=={'a','t0','v_friend','v_stranger','v_computer','beta_gap','z_friend','z_stranger','z_computer'}
    assert len(metrics([row]))>=7
    # This smoke checks execution and finite values, not actual-schedule reliability.
