import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample as fs
from rf1_trust_socialvalue.full_sample_behavior import summarize, run


def events():
    rows=[]
    for i,(partner,amount,low,high,outcome) in enumerate([
        ('friend',0,0,2,'recip'),('stranger',4,2,4,'defect'),
        ('computer',None,0,8,'recip'),('computer',8,4,8,'recip')],1):
        r=dict(onset=str(i*10),duration='1',trial_type='missed_trial' if amount is None else 'choice_'+partner,
            response_time='n/a' if amount is None else '1',trust_value='n/a' if amount is None else str(amount),
            choice='n/a' if amount is None else 'high' if amount==high else 'low',cLow=str(low),cHigh=str(high),
            cLeft=str(high),cRight=str(low),partner=partner,reciprocate='n/a',trial_id=str(i),scheduled_reciprocation=outcome)
        rows.append(r)
        if amount is not None and amount>0:
            rows.append(dict(r,onset=str(i*10+2),reciprocate=outcome,trial_type='outcome_'+partner+'_'+outcome))
    return pd.DataFrame(rows)


def test_zero_and_miss_preserve_schedule_without_feedback():
    t=fs.load_run(events(),'sub-1','01',1)
    assert t.feedback_observed.tolist()==[False,True,False,True]
    assert t.scheduled_reciprocation.tolist()==[1,0,1,1]
    assert np.isnan(t.observed_reciprocation.iloc[0])
    assert np.isnan(t.chose_high.iloc[2])
    assert not t.high_is_right.any()


@pytest.mark.parametrize('change,message',[
    ('schedule','outcome'),('side','offer sides'),('duplicate','duplicate'),('order','order'),('orphan','orphan'),('amount','choice/amount'),('zero_feedback','outcome count')])
def test_fail_closed_inconsistent_canonical_events(change,message):
    e=events()
    if change=='schedule':e.loc[2,'reciprocate']='recip'
    if change=='side':e.loc[0,'cRight']='4'
    if change=='duplicate':e.loc[1,'trial_id']='1'
    if change=='order':e.loc[1,'onset']='1'
    if change=='orphan':e.loc[2,'trial_id']='999'
    if change=='amount':e.loc[0,'trust_value']='7'
    if change=='zero_feedback':
        extra=e.iloc[0].copy();extra['trial_type']='outcome_friend_recip';extra['reciprocate']='recip'
        e=pd.concat([e,pd.DataFrame([extra])],ignore_index=True)
    with pytest.raises(ValueError,match=message):fs.load_run(e,'sub-1','01',1)


def setup_cohort(tmp_path,monkeypatch):
    root=tmp_path/'science';up=tmp_path/'upstream';bids=up/'bids'
    (root/'config').mkdir(parents=True);(up/'code').mkdir(parents=True)
    (up/'code/convert_behavior.py').write_text('# synthetic')
    (up/'code/behavior_curation.tsv').write_text('# synthetic curation')
    (up/'qc/events').mkdir(parents=True)
    (up/'qc/events/policy.json').write_text('{}')
    base=Path(__file__).resolve().parents[1]/'config/full_sample_linux2.json'
    c=json.loads(base.read_text())
    c.update(upstream_root=str(up),bids_root=str(bids),participants=str(bids/'participants.tsv'),
        response_qc=str(up/'qc.tsv'),qc_provenance=str(up/'qc.json'),eligibility=str(up/'eligibility.tsv'),
        source_exclusions=str(up/'exclusions.tsv'),source_provenance=str(up/'source.json'),
        legacy_trials='legacy.csv',legacy_sample='sample.csv',parity_resolutions='config/resolutions.tsv',bootstrap_iterations=10)
    (root/'config/full_sample_linux2.json').write_text(json.dumps(c))
    (root/'config/resolutions.tsv').write_text('participant_id\tsession\trun\ttrial_id\tfield\tlegacy_value\tcanonical_value\tclassification\tevidence\tlegacy_trials_sha256\tevents_sha256\n')
    qc=[];elig=[];old=[]
    for sub,age in [('sub-1',25),('sub-2',75)]:
        rel=f'{sub}/ses-01/func/{sub}_ses-01_task-trust_run-1_events.tsv';p=bids/rel;p.parent.mkdir(parents=True,exist_ok=True)
        events().to_csv(p,sep='\t',index=False)
        q=dict(subject=sub[4:],session='01',task='trust',run='1',events_path='bids/'+rel,events_sha256=fs.sha(p),
            expected_trials='42',response_trials='4',misses='1',miss_fraction='.25',review_status='review',review_reasons='short_run')
        qc.append(q);elig.append(dict(participant_id=sub,session='01',run='1',events_path=rel,events_sha256=fs.sha(p),
            source_excluded='false',structural_status='pass',structural_reasons='curated_short'))
        t=fs.load_run(events(),sub,'01',1);old.append(t)
    pd.DataFrame(qc).to_csv(up/'qc.tsv',sep='\t',index=False)
    digest=hashlib.sha256()
    for r in qc:digest.update(r['events_path'].encode()+b'\0'+r['events_sha256'].encode()+b'\n')
    fs.save_json(up/'qc.json',dict(events_manifest_sha256=digest.hexdigest(),policy_sha256=fs.sha(up/'qc/events/policy.json')))
    pd.DataFrame(elig).to_csv(up/'eligibility.tsv',sep='\t',index=False)
    (up/'exclusions.tsv').write_text('participant_id\n')
    fs.save_json(up/'source.json',dict(cohort_validation={'status':'passed'},converter_sha256=fs.sha(up/'code/convert_behavior.py'),curation_sha256=fs.sha(up/'code/behavior_curation.tsv'),
        run_eligibility_sha256=fs.sha(up/'eligibility.tsv'),source_exclusions_sha256=fs.sha(up/'exclusions.tsv'),qc_provenance_sha256=fs.sha(up/'qc.json')))
    (bids/'participants.tsv').write_text('participant_id\tage\tsex\nsub-1\t25\tF\nsub-2\t75\tM\n')
    (bids/'task-trust_events.json').write_text('{}')
    pd.concat(old).to_csv(root/'legacy.csv',index=False)
    (root/'sample.csv').write_text('participant_id,primary_include\nsub-1,True\nsub-2,True\n')
    monkeypatch.setattr(fs,'N111_HASHES',{})
    monkeypatch.setattr(fs,'git_sha',lambda p:'synthetic')
    monkeypatch.setattr('rf1_trust_socialvalue.full_sample_behavior.git_sha',lambda p:'synthetic')
    c=fs.resolve_config(root/'config/full_sample_linux2.json')
    return c


def test_frozen_cohort_retains_review_and_curated_short_runs(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch)
    hashes,files=fs.inventory(c);t,m,a=fs.build_tables(c,files)
    assert len(a)==2 and m.include_primary.all()
    assert not m.include_sensitivity.any() # miss fraction .25 and low feedback
    assert m.response_qc_status.eq('review').all()
    p,n=fs.parity(t,c)
    assert n==2 and p.classification.eq('identical').all()
    fs.freeze(c);fs.verify(c)
    with pytest.raises(ValueError,match='already frozen'):fs.freeze(c)
    Path(c['participants']).write_text(Path(c['participants']).read_text().replace('25','26'))
    with pytest.raises(ValueError,match='frozen inputs changed'):fs.verify(c)


def test_source_certification_and_qc_drift_rejected(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch);_,files=fs.inventory(c)
    files[0].write_text(files[0].read_text().replace('defect','recip'))
    with pytest.raises(ValueError,match='live Trust inventory'):fs.build_tables(c,files)


def test_unexplained_outcome_and_new_public_run_classified(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch);_,files=fs.inventory(c);t,_,_=fs.build_tables(c,files)
    t.loc[t.feedback_observed,'observed_reciprocation']=1
    p,_=fs.parity(t,c)
    assert p.query("field == 'observed_reciprocation'").classification.eq('new unexplained discrepancy').all()
    t['run']=2;p,_=fs.parity(t,c)
    assert p[p.run.eq(2)].classification.eq('expected public-data omission').all()
    assert p[p.run.eq(1)].classification.eq('new unexplained discrepancy').all()


def test_milestone_integration_synthetic(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch);fs.freeze(c);run(c)
    out=Path(c['output'])
    s=json.loads((out/'milestone_status.json').read_text())
    assert s['status']=='ready_for_review' and not s['hierarchical_launch_authorized']
    assert s['cohort_summary']['primary_n']==2
    assert (out/'figures/01_behavior_by_partner_offer.png').exists()
    assert not (out/'tables/hierarchical_fit_status.tsv').exists()


def test_age_gee_saves_clustered_marginal_contrasts(tmp_path):
    from rf1_trust_socialvalue.full_sample_behavior import age_model
    rng=np.random.default_rng(23);rows=[]
    for i in range(30):
        for run in [1,2]:
            for trial in range(1,43):
                partner=['friend','stranger','computer'][(trial-1)%3]
                rows.append(dict(participant_id=f'sub-{i}',age=20+i*2,run=run,partner=partner,
                    offer_pair=rng.choice(['0-2','0-4','0-8','2-4','2-8','4-8']),
                    trial_scaled=(trial-1)/41,valid_choice=True,chose_high=float(rng.random()<(0.8 if partner=='friend' else .45))))
    c={'age_behavior':{'max_iterations':100,'contrast_ages':[25,75]}}
    status=age_model(pd.DataFrame(rows),c,tmp_path)
    assert status['status']=='complete'
    changes=pd.read_csv(tmp_path/'age_partner_change.tsv',sep='\t')
    assert len(changes)==3 and (changes.ci_low < changes.ci_high).all()
    assert 'C(run)' in status['formula']


def test_source_excluded_cannot_enter_primary(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch)
    e=fs.tsv(c['eligibility']);e.loc[e.participant_id=='sub-1',['source_excluded','structural_status']]=['true','unresolved']
    e.to_csv(c['eligibility'],sep='\t',index=False)
    Path(c['source_exclusions']).write_text('participant_id\nsub-1\n')
    q=fs.tsv(c['response_qc']);q=q[q.subject!='1'];q.to_csv(c['response_qc'],sep='\t',index=False)
    digest=hashlib.sha256()
    for r in q.to_dict('records'):digest.update(r['events_path'].encode()+b'\0'+r['events_sha256'].encode()+b'\n')
    fs.save_json(c['qc_provenance'],dict(events_manifest_sha256=digest.hexdigest(),policy_sha256=fs.sha(Path(c['upstream_root'])/'qc/events/policy.json')))
    source=json.loads(Path(c['source_provenance']).read_text())
    for name,key in [('eligibility','run_eligibility_sha256'),('source_exclusions','source_exclusions_sha256'),('qc_provenance','qc_provenance_sha256')]:source[key]=fs.sha(c[name])
    fs.save_json(c['source_provenance'],source)
    _,files=fs.inventory(c);t,m,a=fs.build_tables(c,files)
    assert set(t.participant_id)=={'sub-2'}
    assert m.loc[m.participant_id=='sub-1','exclusion_reason'].item()=='source_excluded'


def test_missing_partner_fails_sensitivity(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch);c['sensitivity']['maximum_miss_fraction']=1
    _,files=fs.inventory(c);_,m,a=fs.build_tables(c,files)
    assert (a.minimum_partner_feedback==0).all() # friend zero choice had no feedback
    assert not m.include_sensitivity.any()



def test_all_miss_run_retained_for_participant_missingness(tmp_path,monkeypatch):
    c=setup_cohort(tmp_path,monkeypatch)
    original=fs.load_run
    def allmiss(events,sub,session,run):
        t=original(events,sub,session,run)
        if sub=='sub-1':
            t['valid_choice']=False;t['feedback_observed']=False
            for field in ['chosen_amount','chose_high','response_time','observed_reciprocation']:t[field]=np.nan
        return t
    # A no-choice participant is removed only after aggregating otherwise valid runs.
    # Give sub-1 a second usable run through the certified synthetic contract.
    elig=fs.tsv(c['eligibility']);qc=fs.tsv(c['response_qc'])
    added=elig.iloc[0].copy();added['run']='2';added['events_path']=added.events_path.replace('run-1','run-2')
    path=Path(c['bids_root'])/added.events_path;events().to_csv(path,sep='\t',index=False);added['events_sha256']=fs.sha(path)
    elig=pd.concat([elig,pd.DataFrame([added])],ignore_index=True)
    q=qc.iloc[0].copy();q['run']='2';q['events_path']=q.events_path.replace('run-1','run-2');q['events_sha256']=fs.sha(path)
    qc=pd.concat([qc,pd.DataFrame([q])],ignore_index=True)
    qc.loc[qc.subject.eq('1')&qc.run.eq('1'),['misses','miss_fraction']]=['4','1']
    monkeypatch.setattr(fs,'check_contract',lambda c,files:(elig,qc.assign(participant_id='sub-'+qc.subject),set()))
    def mixed(e,sub,session,run):
        return allmiss(e,sub,session,run) if run==1 else original(e,sub,session,run)
    monkeypatch.setattr(fs,'load_run',mixed)
    _,files=fs.inventory(c);t,m,a=fs.build_tables(c,files)
    assert m.loc[(m.participant_id=='sub-1')&(m.run==1),'include_primary'].item()
    assert a.set_index('participant_id').loc['sub-1','miss_fraction']==5/8
