"""Canonical Linux2 Trust integration. This milestone never launches Stan.

Source QC is an upstream contract. Input hashes freeze the cohort. Unexplained
parity differences block the milestone; they are never waived by a count check.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import numpy as np
import pandas as pd

PARTNERS = ['friend', 'stranger', 'computer']
PARITY_FIELDS = ['partner','c_left','c_right','c_low','c_high','chosen_amount','chose_high',
                 'response_time','feedback_observed','observed_reciprocation',
                 'scheduled_reciprocation','trial_in_run']
FROZEN_N111 = 'beac6f4d48421b546aa0c7011d4f60adec21ec56'
N111_HASHES = {'legacy_trials': 'b819604280bf12d0325424b727cb6a4ee585966e62f4c1cc7cb968061d73f8d8',
               'legacy_sample': '1226af5ad06437764d9238071c9b8c5cae7326ad2a4d0baa9461baf61a10a02a'}
EVENT_RE = re.compile(r'^(sub-\d+)_ses-(\d+)_task-trust_run-(\d+)_events\.tsv$')
IMAGING_TEMPLATE_RE = re.compile(r'^sub-\d+_ses-\d+_task-trust_run-\d+_part-(?:mag|phase)_events\.tsv$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha(root):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def tsv(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def resolve_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    root = path.parent.parent
    for key in ['upstream_root', 'bids_root', 'response_qc', 'qc_provenance', 'eligibility',
                'source_exclusions', 'source_provenance', 'participants', 'legacy_trials',
                'legacy_sample', 'parity_resolutions', 'parity_run_mappings', 'output', 'work']:
        c[key] = str((root / c[key]).resolve())
    c['_root'] = str(root); c['_config_path'] = str(path)
    if Path(c['output']) != root / 'results/full_sample' or Path(c['work']) != root / 'work/full_sample':
        raise ValueError('full-sample outputs/work must remain isolated from the frozen N111 analysis')
    if c['frozen_n111_commit'] != FROZEN_N111:
        raise ValueError('N111 reference must stay frozen')
    return c


def load_run(events, participant_id, session, run):
    """Join outcomes by trial_id; latent schedule is never observed feedback."""
    e = events.copy()
    required = {'trial_id', 'trial_type', 'onset', 'duration', 'partner', 'choice', 'trust_value',
                'cLow', 'cHigh', 'cLeft', 'cRight', 'response_time', 'reciprocate', 'scheduled_reciprocation'}
    if required - set(e):
        raise ValueError('canonical schema missing: '+', '.join(sorted(required-set(e))))
    decision = e.trial_type.str.startswith('choice_') | e.trial_type.eq('missed_trial')
    outcomes = e.trial_type.str.startswith('outcome_')
    if not (decision | outcomes).all() or not decision.any():
        raise ValueError('unknown event type or no decisions')
    d = e[decision]
    if d.trial_id.duplicated().any() or d.trial_id.eq('').any():
        raise ValueError('duplicate/empty trial_id')
    if not set(e[outcomes].trial_id).issubset(d.trial_id):
        raise ValueError('orphan outcome')
    onset = pd.to_numeric(d.onset, errors='raise').to_numpy()
    trial_numbers = pd.to_numeric(d.trial_id, errors='raise').to_numpy()
    if not np.isfinite(onset).all() or np.any(np.diff(onset) <= 0) or np.any(np.diff(trial_numbers) <= 0):
        raise ValueError('decision order/onsets/trial identifiers are not strictly increasing')
    if np.any(trial_numbers != trial_numbers.astype(int)):
        raise ValueError('noninteger trial identifier')
    records = []
    for i, (_, r) in enumerate(d.iterrows(), 1):
        block = e[outcomes & e.trial_id.eq(r.trial_id)]
        left, right, low, high = [float(r[k]) for k in ['cLeft', 'cRight', 'cLow', 'cHigh']]
        if not np.isfinite([left,right,low,high]).all() or not 0 <= low < high <= 8 or sorted([left,right]) != [low,high]:
            raise ValueError('offer sides disagree with sorted offers')
        if r.partner not in PARTNERS or r.scheduled_reciprocation not in {'recip','defect'}:
            raise ValueError('unknown partner or scheduled outcome')
        if r.reciprocate not in {'n/a', ''}:
            raise ValueError('choice event has invented observed feedback')
        valid = r.trial_type != 'missed_trial'
        amount = float(r.trust_value) if valid else np.nan
        rt = float(r.response_time) if valid else np.nan
        if valid and (r.trial_type != 'choice_'+r.partner or amount not in {low,high}
                      or r.choice != ('high' if amount == high else 'low') or not np.isfinite(rt) or rt < 0):
            raise ValueError('choice/amount/partner/RT inconsistency')
        if not valid and (r.trust_value not in {'n/a',''} or r.choice not in {'n/a',''}):
            raise ValueError('missed choice has a recorded decision')
        feedback = bool(valid and amount > 0)
        if len(block) != int(feedback):
            raise ValueError('outcome count inconsistent with miss/zero/positive choice')
        observed = np.nan
        if feedback:
            o = block.iloc[0]
            for field in ['partner', 'choice', 'trust_value', 'cLow', 'cHigh', 'cLeft', 'cRight', 'scheduled_reciprocation']:
                if o[field] != r[field]:
                    raise ValueError('choice/outcome mismatch: '+field)
            if o.reciprocate != r.scheduled_reciprocation or o.trial_type != 'outcome_'+r.partner+'_'+o.reciprocate:
                raise ValueError('observed outcome disagrees with schedule/label')
            if not float(r.onset) <= float(o.onset) or (i < len(d) and float(o.onset) >= onset[i]):
                raise ValueError('outcome outside its decision interval')
            observed = float(o.reciprocate == 'recip')
        records.append(dict(participant_id=participant_id, session=str(session).zfill(2), run=int(run),
            trial_in_run=i, trial_id=str(int(float(r.trial_id))), partner=r.partner,
            c_left=left, c_right=right, c_low=low, c_high=high, high_is_right=right==high,
            zero_option_present=low==0, chosen_amount=amount, chose_high=float(amount==high) if valid else np.nan,
            response_time=rt, valid_choice=valid, feedback_observed=feedback,
            observed_reciprocation=observed, scheduled_reciprocation=int(r.scheduled_reciprocation=='recip')))
    return pd.DataFrame(records)


def preflight(c):
    """Check the upstream handoff before running tests or hashing event files."""
    paths = {key: Path(c[key]) for key in ['participants', 'response_qc', 'qc_provenance',
        'eligibility', 'source_exclusions', 'source_provenance', 'legacy_trials',
        'legacy_sample', 'parity_resolutions', 'parity_run_mappings']}
    paths.update(trust_sidecar=Path(c['bids_root'])/'task-trust_events.json',
        converter=Path(c['upstream_root'])/'code/convert_behavior.py',
        curation=Path(c['upstream_root'])/'code/behavior_curation.tsv',
        qc_policy=Path(c['upstream_root'])/'qc/events/policy.json')
    missing = [f'{key}: {path}' for key, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError('Required canonical inputs are missing:\n  '+'\n  '.join(missing)+
            '\nCreate missing exports upstream before freezing. The demographics export must '
            'contain participant_id, age in years, and sex from the authoritative cohort source; '
            'do not substitute the historical N111 table or fabricate missing ages.')
    people = tsv(c['participants'])
    if {'participant_id', 'age', 'sex'}-set(people):
        raise ValueError('participants.tsv must contain participant_id, age and sex')
    if people.participant_id.eq('').any() or people.participant_id.duplicated().any():
        raise ValueError('participants.tsv must have nonempty, unique participant_id values')
    elig = tsv(c['eligibility'])
    excluded = set(tsv(c['source_exclusions']).participant_id)
    required = set(elig.loc[elig.structural_status.eq('pass') & elig.events_sha256.ne(''),
                            'participant_id'])-excluded
    missing_ids = sorted(required-set(people.participant_id))
    if missing_ids:
        raise ValueError('participants.tsv lacks rows for structurally eligible participants: '+
                         ', '.join(missing_ids))


def inventory(c):
    """Hash canonical products only. Never consult private sources downstream."""
    preflight(c)
    for key, expected in N111_HASHES.items():
        if sha(c[key]) != expected:
            raise ValueError('frozen N111 reference changed: '+key)
    bids = Path(c['bids_root'])
    files, imaging_templates = [], []
    for path in sorted(bids.glob('sub-*/ses-*/func/*task-trust*_events.tsv')):
        if EVENT_RE.fullmatch(path.name):
            files.append(path)
        elif IMAGING_TEMPLATE_RE.fullmatch(path.name) and tsv(path).empty:
            # Not behavioral runs. Hash them so any later change invalidates the freeze.
            imaging_templates.append(path)
        else:
            raise ValueError(f'unreviewed or nonempty noncanonical Trust events file: {path}')
    if not files:
        raise ValueError('no canonical Trust events')
    for p in files:
        match = EVENT_RE.match(p.name)
        if not match or match[2].zfill(2) not in c['sessions']:
            raise ValueError(f'unreviewed session or ambiguous filename: {p}')
        if p.resolve().parent != (bids / match[1] / ('ses-'+match[2]) / 'func').resolve():
            raise ValueError('event filename/directory identity mismatch')
    paths = files + imaging_templates + [Path(c[k]) for k in ['participants','response_qc','qc_provenance','eligibility',
        'source_exclusions','source_provenance','legacy_trials','legacy_sample','parity_resolutions','parity_run_mappings']]
    paths += [bids/'task-trust_events.json', Path(c['_config_path']),
              Path(c['upstream_root'])/'code/convert_behavior.py',
              Path(c['upstream_root'])/'code/behavior_curation.tsv',
              Path(c['upstream_root'])/'qc/events/policy.json']
    return {str(p): sha(p) for p in paths}, files


def check_contract(c, files):
    upstream = json.loads(Path(c['source_provenance']).read_text())
    for key, field in [('eligibility','run_eligibility_sha256'),('source_exclusions','source_exclusions_sha256'),('qc_provenance','qc_provenance_sha256')]:
        if sha(c[key]) != upstream[field]:
            raise ValueError('upstream contract hash mismatch: '+key)
    if upstream['cohort_validation']['status'] != 'passed':
        raise ValueError('upstream cohort validation has not passed')
    if upstream['converter_sha256'] != sha(Path(c['upstream_root'])/'code/convert_behavior.py'):
        raise ValueError('upstream converter changed; refresh canonical certification')
    if upstream['curation_sha256'] != sha(Path(c['upstream_root'])/'code/behavior_curation.tsv'):
        raise ValueError('upstream curation changed; refresh certification')
    qc = tsv(c['response_qc'])
    digest = hashlib.sha256()
    for r in qc.to_dict('records'):
        digest.update(r['events_path'].encode()+b'\0'+r['events_sha256'].encode()+b'\n')
    prov = json.loads(Path(c['qc_provenance']).read_text())
    if prov['policy_sha256'] != sha(Path(c['upstream_root'])/'qc/events/policy.json'):
        raise ValueError('response QC policy changed; refresh canonical QC')
    if digest.hexdigest() != prov['events_manifest_sha256']:
        raise ValueError('response QC inventory hash mismatch')
    elig = tsv(c['eligibility'])
    keys = ['participant_id','session','run']
    if elig.duplicated(keys).any() or qc.duplicated(['subject','session','task','run']).any():
        raise ValueError('duplicate/ambiguous QC runs')
    live = {str(p.relative_to(c['bids_root'])):sha(p) for p in files}
    certified = {r.events_path:r.events_sha256 for r in elig.itertuples() if r.events_sha256}
    if live != certified:
        raise ValueError('live Trust inventory differs from upstream source certification')
    excluded = set(tsv(c['source_exclusions']).participant_id)
    if set(elig.loc[elig.source_excluded.eq('true'),'participant_id']) != set(elig.participant_id)&excluded:
        raise ValueError('source exclusion inventory disagreement')
    qtrust = qc[qc.task.eq('trust')].copy()
    qtrust['participant_id'] = 'sub-'+qtrust.subject
    qtrust['session'] = qtrust.session.str.zfill(2)
    eligible_keys = set(map(tuple, elig.loc[~elig.participant_id.isin(excluded)&elig.events_sha256.ne(''),keys].to_numpy()))
    if eligible_keys != set(map(tuple,qtrust[keys].to_numpy())):
        raise ValueError('response QC does not cover the live nonexcluded Trust cohort exactly')
    return elig, qtrust, excluded


def build_tables(c, files):
    elig, qc, excluded = check_contract(c, files)
    people = tsv(c['participants'])
    if people.participant_id.duplicated().any() or {'age','sex'}-set(people):
        raise ValueError('participants.tsv must have unique IDs, age and sex')
    people = people.set_index('participant_id')
    qmap = {(r.participant_id,r.session,int(r.run)):r for r in qc.itertuples()}
    chunks=[]; manifest=[]
    for r in elig.itertuples():
        sub, session, run = r.participant_id, str(r.session).zfill(2), int(r.run)
        q = qmap.get((sub,session,run))
        include = sub not in excluded and r.structural_status == 'pass' and bool(r.events_sha256)
        reason = 'source_excluded' if sub in excluded else (r.structural_reasons or 'structural_unresolved') if not include else ''
        n = valid = 0
        if include:
            path = Path(c['bids_root']) / r.events_path
            t = load_run(tsv(path),sub,session,run)
            if sub not in people.index:
                raise ValueError('participant missing demographics row: '+sub)
            t['age'] = pd.to_numeric(people.loc[sub,'age'],errors='coerce')
            t['sex'] = people.loc[sub,'sex']
            n=len(t);valid=int(t.valid_choice.sum())
            if q is None or q.events_sha256 != r.events_sha256 or int(q.response_trials)!=n or int(q.misses)!=n-valid:
                raise ValueError('canonical decision counts/hashes disagree with response QC')
            chunks.append(t)
        manifest.append(dict(participant_id=sub,session=session,run=run,events_path=r.events_path,
            events_sha256=r.events_sha256,expected_trials=int(q.expected_trials) if q else c['expected_trials'],
            presented_trials=n if include or n else (int(q.response_trials) if q else 0),
            valid_choices=valid if include or n else (int(q.response_trials)-int(q.misses) if q else 0),
            missed_choices=n-valid if n else (int(q.misses) if q else 0),
            miss_fraction=(n-valid)/n if n else (float(q.miss_fraction) if q else np.nan),
            response_qc_status=q.review_status if q else 'not_audited',response_qc_reasons=q.review_reasons if q else '',
            structural_status=r.structural_status, source_excluded=sub in excluded,include_primary=include,
            include_sensitivity=False,exclusion_reason=reason))
    if not chunks:
        raise ValueError('no usable primary choices')
    t = pd.concat(chunks,ignore_index=True).sort_values(['participant_id','session','run','trial_in_run']).reset_index(drop=True)
    has_choices=t.groupby('participant_id').valid_choice.any()
    no_choices=set(has_choices.index[~has_choices])
    t=t[~t.participant_id.isin(no_choices)].copy()
    if t.empty:raise ValueError('no participants with valid primary choices')
    for row in manifest:
        if row['participant_id'] in no_choices and row['include_primary']:
            row['include_primary']=False;row['exclusion_reason']='participant_has_no_valid_choices'
    t['global_trial']=t.groupby('participant_id').cumcount()+1
    t['offer_pair']=t.c_low.map(lambda v:f'{v:g}')+'-'+t.c_high.map(lambda v:f'{v:g}')
    t['trial_scaled']=(t.trial_in_run-1)/(c['expected_trials']-1)
    t['trial_bin']=np.minimum(3, ((t.trial_in_run-1)*3/c['expected_trials']).astype(int)+1)
    t['last_observed_reciprocation']=t.groupby(['participant_id','partner']).observed_reciprocation.transform(lambda s:s.ffill().shift())
    old_sample=pd.read_csv(c['legacy_sample'])
    t['cohort_tag']=np.where(t.participant_id.isin(old_sample.participant_id),'in_original_openneuro_release','later_participant')
    m=pd.DataFrame(manifest)
    a=t.groupby('participant_id').agg(presented_trials=('valid_choice','size'),valid_choices=('valid_choice','sum'),
        age=('age','first'),sex=('sex','first'),n_runs=('run','nunique'),cohort_tag=('cohort_tag','first'))
    a['miss_fraction']=1-a.valid_choices/a.presented_trials
    f=t.groupby(['participant_id','partner']).feedback_observed.sum().unstack().reindex(columns=PARTNERS).fillna(0)
    a['minimum_partner_feedback']=f.min(axis=1)
    a['include_sensitivity']=(a.miss_fraction<=c['sensitivity']['maximum_miss_fraction'])&(a.minimum_partner_feedback>=c['sensitivity']['minimum_partner_feedback'])
    complete=m[m.include_primary].assign(complete=lambda q:q.presented_trials.eq(q.expected_trials)).groupby('participant_id').complete.all()
    a['complete_run_sensitivity']=a.index.map(complete).astype(bool)
    a['sensitivity_reason']=np.where(a.miss_fraction>c['sensitivity']['maximum_miss_fraction'],'more_than_20_percent_misses;','')+np.where(a.minimum_partner_feedback<c['sensitivity']['minimum_partner_feedback'],'fewer_than_4_feedback_for_at_least_one_partner;','')
    m['sensitivity_exclusion_reason']=m.participant_id.map(a.sensitivity_reason).fillna(m.exclusion_reason)
    m['complete_run_sensitivity']=m.include_primary & m.participant_id.map(a.complete_run_sensitivity).fillna(False).astype(bool)
    m['include_sensitivity']=m.include_primary&m.participant_id.map(a.include_sensitivity).fillna(False).astype(bool)
    t['include_sensitivity']=t.participant_id.map(a.include_sensitivity)
    return t,m,a.reset_index()


def comparable(a,b):
    if pd.isna(a) and pd.isna(b):return True
    if pd.isna(a) or pd.isna(b):return False
    if isinstance(a,(str,bool,np.bool_)) or isinstance(b,(str,bool,np.bool_)):
        return str(a)==str(b)
    return bool(np.isclose(float(a),float(b),atol=1.1e-6,rtol=0))


def mapped_legacy(t,c):
    """Apply explicit, hash-bound run mappings only after full trial equality."""
    old=pd.read_csv(c['legacy_trials'])
    old['trial_id']=old.trial_in_run.astype(int).astype(str)
    old['session']='01'
    rules=tsv(c['parity_run_mappings'])
    corrections=[]
    if rules.empty:return old,corrections
    if rules.duplicated(['participant_id','legacy_run']).any():
        raise ValueError('duplicate legacy run mapping')
    rules=rules.assign(legacy_run=pd.to_numeric(rules.legacy_run,errors='raise'),
                       canonical_run=pd.to_numeric(rules.canonical_run,errors='raise'),
                       expected_legacy_trials=pd.to_numeric(rules.expected_legacy_trials,errors='raise'))
    if rules.duplicated(['participant_id','legacy_run']).any():
        raise ValueError('duplicate numeric legacy run mapping')
    mapped=old.copy(); removed=[]; verified=set()
    for r in rules.itertuples():
        if r.action not in {'map','omit_aborted_segment'} or not r.evidence:
            raise ValueError('invalid run mapping action/evidence')
        if r.canonical_run < 1 or r.canonical_run != int(r.canonical_run):
            raise ValueError('canonical run mapping must be a positive integer')
        if r.legacy_trials_sha256 != sha(c['legacy_trials']):
            raise ValueError('stale run mapping legacy hash')
        event=Path(c['bids_root'])/f'{r.participant_id}/ses-01/func/{r.participant_id}_ses-01_task-trust_run-{int(r.canonical_run)}_events.tsv'
        if not event.is_file() or r.events_sha256 != sha(event):
            raise ValueError('stale run mapping canonical event hash')
        o=old[old.participant_id.eq(r.participant_id)&old.run.eq(r.legacy_run)]
        n=t[t.participant_id.eq(r.participant_id)&t.run.eq(r.canonical_run)]
        if o.empty or n.empty or len(o)!=r.expected_legacy_trials:
            raise ValueError('run mapping missing rows or unexpected trial count')
        if o.trial_id.duplicated().any() or n.trial_id.duplicated().any():
            raise ValueError('duplicate trial ID in run mapping')
        if r.action=='map':
            key=(r.participant_id,int(r.canonical_run))
            if key in verified:raise ValueError('multiple legacy runs map to one canonical run')
            if len(o)!=c['expected_trials'] or set(o.trial_id)!=set(n.trial_id):
                raise ValueError('run mapping requires identical complete trial sets')
            nn=n.set_index('trial_id')
            for a in o.itertuples():
                b=nn.loc[a.trial_id]
                for field in PARITY_FIELDS:
                    av,bv=getattr(a,field),b[field]
                    if field=='chose_high' and pd.notna(av):av=float(av)
                    if not comparable(av,bv):
                        raise ValueError(f'run mapping trial mismatch: {r.participant_id} legacy {r.legacy_run} -> {r.canonical_run}, trial {a.trial_id}, {field}: {av!r} vs {bv!r}')
            verified.add(key)
            mapped.loc[o.index,'run']=r.canonical_run
        else:
            if not 0<len(o)<c['expected_trials'] or 'ambiguous_session' not in o or not o.ambiguous_session.eq(True).all():
                raise ValueError('only a documented incomplete ambiguous segment can be omitted')
            removed.extend(o.index)
        for a in o.itertuples():
            corrections.append(dict(participant_id=r.participant_id,session='01',run=r.canonical_run,
                trial_id=a.trial_id,field='run_identity' if r.action=='map' else 'aborted_segment_presence',
                legacy_value=str(r.legacy_run) if r.action=='map' else 'present',
                canonical_value=str(int(r.canonical_run)) if r.action=='map' else 'absent',
                classification='expected canonical correction',evidence=r.evidence,
                legacy_run=r.legacy_run,events_sha256=r.events_sha256))
    for r in rules[rules.action.eq('omit_aborted_segment')].itertuples():
        if (r.participant_id,int(r.canonical_run)) not in verified:
            raise ValueError('aborted segment omission requires a verified complete replacement mapping')
    mapped=mapped.drop(index=removed)
    if mapped.duplicated(['participant_id','session','run','trial_id']).any():
        raise ValueError('run mapping collides with another historical run')
    return mapped,corrections


def parity(t,c):
    old,corrections=mapped_legacy(t,c)
    sample=pd.read_csv(c['legacy_sample'])
    overlap=set(t.participant_id)&set(sample.loc[sample.primary_include,'participant_id'])
    release=set(sample.participant_id)
    fields=PARITY_FIELDS
    key=['participant_id','session','run','trial_id']
    resolutions=tsv(c['parity_resolutions'])
    if len(resolutions) and resolutions.duplicated(key+['field']).any():
        raise ValueError('duplicate parity resolution')
    rows=[dict(r,in_n111_primary=r['participant_id'] in overlap) for r in corrections]
    oldgroups={k:v for k,v in old.groupby(['participant_id','run'])}
    newgroups={k:v for k,v in t.groupby(['participant_id','run']) if k[0] in release}
    for sub,run in sorted(set(oldgroups)|set(newgroups)):
        if sub not in set(t.participant_id):continue
        o=oldgroups.get((sub,run),old.iloc[:0]); n=newgroups.get((sub,run),t.iloc[:0])
        om={str(r.trial_id):r for r in o.itertuples()};nm={str(r.trial_id):r for r in n.itertuples()}
        if len(om)!=len(o) or len(nm)!=len(n):
            raise ValueError('duplicate trial key in parity inputs')
        for trial in sorted(set(om)|set(nm),key=lambda s:int(s)):
            a=om.get(trial);b=nm.get(trial)
            differences=[]
            if a is None or b is None:
                differences=[('row_presence','present' if a else 'absent','present' if b else 'absent')]
            else:
                for f in fields:
                    av,bv=getattr(a,f),getattr(b,f)
                    if f=='chose_high' and pd.notna(av):av=float(av)
                    if not comparable(av,bv):differences.append((f,av,bv))
            if not differences:differences=[('all','same','same')]
            for f,av,bv in differences:
                category='identical' if f=='all' else ('expected public-data omission' if o.empty and b is not None else 'new unexplained discrepancy')
                evidence='historical public run absent from frozen decision table' if category=='expected public-data omission' else ''
                rr=resolutions[(resolutions.participant_id==sub)&(resolutions.session=='01')&
                    (pd.to_numeric(resolutions.run,errors='raise')==run)&(resolutions.trial_id==trial)&(resolutions.field==f)]
                if len(rr):
                    r=rr.iloc[0]
                    if r.classification not in {'expected canonical correction','expected public-data omission'} or not r.evidence:
                        raise ValueError('invalid parity correction evidence')
                    if r.legacy_trials_sha256!=sha(c['legacy_trials']) or r.legacy_value!=str(av) or r.canonical_value!=str(bv):
                        raise ValueError('stale parity correction values or legacy hash')
                    event=Path(c['bids_root'])/f'{sub}/ses-01/func/{sub}_ses-01_task-trust_run-{int(run)}_events.tsv'
                    if not event.exists() or r.events_sha256!=sha(event):raise ValueError('stale parity correction event hash')
                    category=r.classification;evidence=r.evidence
                rows.append(dict(participant_id=sub,session='01',run=run,trial_id=trial,field=f,
                    legacy_value=av,canonical_value=bv,classification=category,evidence=evidence,
                    in_n111_primary=sub in overlap))
    p=pd.DataFrame(rows)
    if p.empty or not overlap:raise ValueError('no N111 overlap: check identities before proceeding')
    return p, len(overlap)


def freeze(c,refresh=False):
    out=Path(c['output']);work=Path(c['work']);out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    provenance=out/'provenance.json'
    if provenance.exists() and not refresh:
        raise ValueError('cohort already frozen; use verify/run or explicit freeze --refresh')
    hashes,files=inventory(c)
    t,m,a=build_tables(c,files)
    mapped_legacy(t,c)  # Validate proposed mappings before replacing any existing freeze.
    if provenance.exists():
        backup=work/('previous_freeze_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        shutil.copytree(out,backup)
        shutil.rmtree(out)
        out.mkdir()
    dest=out/'tables';dest.mkdir(exist_ok=True)
    m.to_csv(dest/'cohort_manifest.tsv',sep='\t',index=False)
    a.to_csv(dest/'data_audit.tsv',sep='\t',index=False)
    t.to_csv(work/'canonical_trials.tsv',sep='\t',index=False)
    versions={}
    for package in ['numpy','pandas','scipy','statsmodels','matplotlib','numba','arviz','cmdstanpy']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    versions['python']=platform.python_version()
    versions['cmdstan']=None
    try:
        import cmdstanpy
        versions['cmdstan']=cmdstanpy.cmdstan_version()
    except (ImportError,ValueError):pass
    code_files=[Path(__file__),Path(__file__).with_name('full_sample_behavior.py')]
    prov=dict(schema_version=1, frozen_at=datetime.now(timezone.utc).isoformat(),
        upstream_git_sha=git_sha(c['upstream_root']),analysis_git_sha=git_sha(c['_root']),
        frozen_n111_commit=FROZEN_N111,config=c,input_hashes=hashes,versions=versions,
        implementation_hashes={str(p):sha(p) for p in code_files},
        artifact_hashes={str(p):sha(p) for p in [dest/'cohort_manifest.tsv',dest/'data_audit.tsv',work/'canonical_trials.tsv']})
    if inventory(c)[0]!=hashes:raise ValueError('inputs changed while freezing; rerun after resolving concurrent writes')
    save_json(provenance,prov)
    save_json(out/'milestone_status.json',dict(status='cohort_frozen_pending_parity',hierarchical_launch_authorized=False))
    print(f'Frozen {a.participant_id.nunique()} primary participants; run parity/behavior next.',flush=True)


def verify(c):
    prov=json.loads((Path(c['output'])/'provenance.json').read_text())
    current,_=inventory(c)
    if current!=prov['input_hashes']:
        changed=sorted(k for k in set(current)|set(prov['input_hashes']) if current.get(k)!=prov['input_hashes'].get(k))
        raise ValueError('frozen inputs changed; explicit refresh required: '+', '.join(changed))
    for p,h in {**prov['implementation_hashes'],**prov['artifact_hashes']}.items():
        if sha(p)!=h:raise ValueError('frozen artifact changed: '+p)
    print('Frozen canonical inputs and derived tables verified.',flush=True)
    return prov


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','freeze','verify','run'])
    p.add_argument('--config',default='config/full_sample_linux2.json')
    p.add_argument('--refresh',action='store_true')
    a=p.parse_args();c=resolve_config(a.config)
    if a.refresh and a.command!='freeze':p.error('--refresh applies only to freeze')
    if a.command=='preflight':
        preflight(c)
        print('Canonical input paths and demographics coverage passed preflight.',flush=True)
    elif a.command=='freeze':freeze(c,a.refresh)
    elif a.command=='verify':verify(c)
    else:
        from .full_sample_behavior import run
        run(c)

if __name__=='__main__':main()
