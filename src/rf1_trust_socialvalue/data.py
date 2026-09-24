"""Reconstruct one row per decision; verify every shared field against BIDS."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

PARTNERS = ['friend', 'stranger', 'computer']
PARTNER_CODES = {3: 'friend', 2: 'stranger', 1: 'computer'}
SHA = 'a3213b56b7bd27d7e3ac10577558eb26bb7c2a61'


def check_close(a, b, field, source, atol=1.1e-6):
    if not np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=0, equal_nan=True):
        raise ValueError(f'{source}: unexplained {field} mismatch')


def reconstruct(raw, events, sub, run):
    raw = raw.reset_index(drop=True)
    decision_mask = events.trial_type.str.startswith('choice') | events.trial_type.eq('missed_trial')
    decision = events[decision_mask].reset_index(drop=True)
    if len(raw) != len(decision):
        raise ValueError(f'{sub} run {run}: raw/BIDS decision counts disagree')
    source = f'{sub} run {run}'
    low = raw[['cLeft', 'cRight']].min(axis=1)
    high = raw[['cLeft', 'cRight']].max(axis=1)
    valid = raw.resp.isin([0, 2, 4, 8])
    assert (low < high).all(), source
    assert raw.Reciprocate.isin([0, 1]).all(), source
    assert raw.Partner.isin(PARTNER_CODES).all(), source
    assert ((raw.resp[valid] == low[valid]) | (raw.resp[valid] == high[valid])).all(), source
    partner = raw.Partner.map(PARTNER_CODES)
    # Missed BIDS rows may be coded choice_miss without partner information.
    for i, d in decision.iterrows():
        if d.trial_type != 'missed_trial':
            assert d.trial_type == 'choice_' + partner[i], (source, i, d.trial_type)
    check_close(low, decision.cLow, 'cLow', source)
    check_close(high, decision.cHigh, 'cHigh', source)
    check_close(raw.resp.where(valid), decision.trust_value, 'selected amount', source)
    check_close(raw.rt.where(valid, 3), decision.response_time, 'RT (missed=3 in BIDS)', source)
    check_close(raw.onset, decision.onset, 'onset', source)
    assert (raw.highlow[valid].astype(str) == decision.choice[valid].astype(str)).all(), source
    assert (raw.highlow[valid].eq('high') == raw.resp[valid].eq(high[valid])).all(), source
    feedback = valid & raw.resp.gt(0)
    observed = raw.Reciprocate.where(feedback)
    # Each decision's following event is its outcome; retain miss/zero events in audit.
    outcomes = []
    positions = np.flatnonzero(decision_mask.to_numpy())
    for i, pos in enumerate(positions):
        stop = positions[i+1] if i+1 < len(positions) else len(events)
        block = events.iloc[pos+1:stop]
        if not feedback[i]:
            assert len(block) == 0, (source, i, 'unexpected feedback for zero/miss')
            outcomes.append('not_shown')
            continue
        if len(block) != 1:
            raise ValueError(f'{source} trial {i+1}: expected exactly one outcome event')
        out = block.iloc[0]
        outcomes.append(out.trial_type)
        if feedback[i]:
            expected = 'outcome_' + partner[i] + ('_recip' if observed[i] else '_defect')
            assert out.trial_type == expected, (source, i, out.trial_type, expected)
        else:
            assert not out.trial_type.endswith(('_recip', '_defect')), (source, i, out.trial_type)
        check_close([raw.resp[i]], [out.trust_value], 'outcome amount', source)
        check_close([low[i],high[i],raw.rt[i]], [out.cLow,out.cHigh,out.response_time], 'outcome offers/RT', source)
    table = pd.DataFrame(dict(participant_id=sub, run=run, trial_in_run=np.arange(1,len(raw)+1),
        partner=partner, c_left=raw.cLeft, c_right=raw.cRight, c_low=low, c_high=high,
        high_is_right=raw.cRight.eq(high), chosen_amount=raw.resp.where(valid),
        chose_high=raw.resp.eq(high).where(valid), response_time=raw.rt.where(valid),
        scheduled_reciprocation=raw.Reciprocate, feedback_observed=feedback,
        observed_reciprocation=observed, amount_gap=high-low, zero_option_present=low.eq(0),
        valid_choice=valid, bids_outcome=outcomes))
    return table


def build(root=Path('data/ds005123'), out=Path('results')):
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() == SHA
    dest=out/'tables';dest.mkdir(parents=True,exist_ok=True)
    participants=pd.read_csv(root/'participants.tsv',sep='\t')
    rows=[];ratings=[];samples=[];manifest=[]; anomalies=[]
    def record(p):
        manifest.append({'path':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    record(root/'participants.tsv')
    for person in participants.to_dict('records'):
        sub=person['participant_id']; assert (root/sub).is_dir(), sub
        events=sorted((root/sub/'func').glob('*task-trust*_events.tsv'))
        samples.append(dict(participant_id=sub, n_bids_files=len(events), age=person['age'], sex=person['sex']))
        for event in events:
            run=int(event.name.split('run-')[1].split('_')[0])
            raw=root/'sourcedata/Scan-Investment_Game/logs'/sub[4:]/f'{sub}_task-trust_run-{run-1}_raw.csv'
            record(event)
            e=pd.read_csv(event,sep='\t')
            if e.empty or e.trial_type.isna().all():
                observed_raw=0
                if raw.exists():
                    record(raw)
                    observed_raw=int(pd.to_numeric(pd.read_csv(raw).onset,errors='coerce').notna().sum())
                anomalies.append(dict(participant_id=sub,run=run,issue='empty_BIDS_run',raw_observed=observed_raw,action='exclude run; no shared representation'))
                continue
            if not raw.exists(): raise FileNotFoundError(f'BIDS run without raw schedule: {raw}')
            record(raw);a=pd.read_csv(raw)
            header=pd.to_numeric(a.TrialNumber,errors='coerce').isna()
            if header.any():
                assert a.loc[header,'cLeft'].eq('cLeft').all()
                bad=e.choice.eq('highlow')
                assert bad.sum()==2*header.sum() and e.loc[bad,'onset'].isna().all(), (sub,'unexpected BIDS header artifacts')
                e=e[~bad].reset_index(drop=True)
                a=a[~header].reset_index(drop=True)
                anomalies.append(dict(participant_id=sub,run=run,issue='appended_sessions_repeated_header',raw_observed=len(a),action='retain behavioral audit; exclude participant from primary inference'))
            unpresented=a.onset.astype(str).eq('--')
            if unpresented.any():
                bad=e.choice.eq('--')
                assert e.loc[bad,'onset'].isna().all() and bad.sum()==2*unpresented.sum()
                e=e[~bad].reset_index(drop=True);a=a[~unpresented].reset_index(drop=True)
                anomalies.append(dict(participant_id=sub,run=run,issue='unpresented_design_rows',raw_observed=int(unpresented.sum()),action='exclude unpresented rows, not missed decisions'))
            for col in ['TrialNumber','cLeft','cRight','Partner','Reciprocate','resp','rt','onset']:
                a[col]=pd.to_numeric(a[col],errors='raise')
            t=reconstruct(a,e,sub,run)
            t['ambiguous_session']=bool(header.any())
            t['source_run']=run
            # Separate appended segments for descriptive displays; chronological order is unverified.
            segment=(a.TrialNumber.diff().fillna(1)<0).cumsum()
            t['run']=run+segment*.1
            t['trial_in_run']=a.TrialNumber.astype(int)
            t['age']=person['age'];t['sex']=person['sex'];rows.append(t)
        rating_files=list((root/'sourcedata/Scan-Investment_Game/logs'/sub[4:]).glob('*Trust-Ratings*.csv'))
        if len(rating_files)>1: raise ValueError(f'Ambiguous rating files: {rating_files}')
        if rating_files:
            p=rating_files[0];record(p);a=pd.read_csv(p)
            if a.duplicated(['Partner','Trait']).any():
                anomalies.append(dict(participant_id=sub,run=0,issue='multiple_unlabelled_rating_sessions',raw_observed=len(a),action='exclude ratings; retain rating-free analyses'))
                continue
            assert a.Rating.dropna().between(-5,5).all(), sub
            a=a.rename(columns={'Rating':'rating','Trait':'trait'})
            a['participant_id']=sub;a['partner']=a.Partner.map(PARTNER_CODES)
            ratings.append(a[['participant_id','partner','trait','rating']])
    t=pd.concat(rows,ignore_index=True).sort_values(['participant_id','run','trial_in_run'])
    t['global_trial']=t.groupby('participant_id').cumcount()+1
    t['offer_pair']=t.c_low.astype(str)+'-'+t.c_high.astype(str)
    t['trial_scaled']=(t.global_trial-1)/83
    t['trial_bin']=((t.global_trial-1)//14+1)
    t['last_observed_reciprocation']=t.groupby(['participant_id','partner']).observed_reciprocation.transform(lambda s:s.ffill().shift())
    r=pd.concat(ratings,ignore_index=True)
    r['normalized']=(r.rating+5)/10
    t.to_csv(dest/'trial_table.csv',index=False);r.to_csv(dest/'ratings.csv',index=False)
    s=pd.DataFrame(samples)
    a=t.groupby('participant_id').agg(n_trials=('valid_choice','size'),n_valid=('valid_choice','sum'),n_feedback=('feedback_observed','sum'))
    s=s.merge(a,on='participant_id',how='left');s['n_runs']=s.participant_id.map(t.groupby('participant_id').run.nunique()).fillna(0).astype(int)
    ambiguous=t.groupby('participant_id').ambiguous_session.any()
    s['primary_include']=s.participant_id.map(ambiguous).eq(False)&s.n_valid.gt(0)
    s['missed_fraction']=1-s.n_valid/s.n_trials
    f=t.groupby(['participant_id','partner']).feedback_observed.sum().unstack().min(axis=1)
    s['minimum_partner_feedback']=s.participant_id.map(f)
    complete=r[r.trait.eq(2)].groupby('participant_id').rating.count().eq(3)
    s['complete_ratings']=s.participant_id.map(complete).fillna(False)
    s['sensitivity_include']=s.primary_include&(s.missed_fraction<=.2)&(s.minimum_partner_feedback>=4)
    s.to_csv(dest/'sample_audit.csv',index=False)
    t.groupby(['participant_id','run']).agg(observed_trials=('valid_choice','size'),valid=('valid_choice','sum'),feedback=('feedback_observed','sum')).assign(expected_trials=42).to_csv(dest/'run_audit.csv')
    t.groupby(['partner','offer_pair']).agg(n=('valid_choice','size'),programmed_reciprocation=('scheduled_reciprocation','mean'),experienced_reciprocation=('observed_reciprocation','mean'),n_feedback=('feedback_observed','sum')).to_csv(dest/'schedule_audit.csv')
    pd.DataFrame(anomalies).to_csv(dest/'data_anomalies.csv',index=False)
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    audit=dict(dataset='ds005123',version='1.1.3',commit=SHA,n_canonical=len(participants),n_behavior=t.participant_id.nunique(),
        n_trials=len(t),n_valid=int(t.valid_choice.sum()),n_missed=int((~t.valid_choice).sum()),
        runs_distribution={str(k):int(v) for k,v in s.n_runs.value_counts().items()},n_rating_complete=int(s.complete_ratings.sum()),n_primary=int(s.primary_include.sum()),
        rating_timing='unverified: session entered but neither filename nor CSV retain it',
        raw_bids_crosscheck='all shared fields agree',n_files_hashed=len(manifest))
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2),flush=True)
    return t,r,s
