"""Closeout-only authorization of one reviewed, unrelated SharedReward QC addition.

Original amount sources and fingerprints remain immutable. No cohort refresh,
new Trust data, changed Trust QC, or arbitrary provenance drift is accepted.
"""
import hashlib
import json
from pathlib import Path

from . import full_sample_amount as a
from . import full_sample_sampling as s

REVIEW_FILE = 'config/full_sample_closeout_qc_review.json'
REVIEW_SHA256 = '0e9e0a1d740a5f2c4befccf83fccd6a0b81e5cbd93fc2ffffdecabea442b89d7'


def review(c):
    path = Path(c['_root'])/REVIEW_FILE
    if s.fs.sha(path) != REVIEW_SHA256:
        raise ValueError('Closeout QC review artifact changed')
    return json.loads(path.read_text())


def validate_live_audit(audit, c):
    """Keep the whole audit; allow only the exact authenticated QC pair."""
    proof = review(c)
    expected = {str(Path(c['upstream_root'])/rel): values
                for rel, values in proof['approved_transitions'].items()}
    if set(expected) != {c['response_qc'], c['qc_provenance']}:
        raise ValueError('Closeout QC review does not match configured paths')
    qc_changes = [row for row in audit['changes'] if row['path'] in expected]
    if qc_changes:
        # The table and its provenance must be the reviewed pair, not a partial update.
        if len(qc_changes) != 2 or {row['path'] for row in qc_changes} != set(expected):
            raise ValueError('Incomplete/unreviewed global QC update; both reviewed files required')
        for row in qc_changes:
            approved = expected[row['path']]
            if (row['change'] != 'modified' or row['frozen_sha256'] != approved['frozen_sha256']
                    or row['current_sha256'] != approved['current_sha256']):
                raise ValueError('Unreviewed global QC contents: '+row['path'])
            if s.fs.sha(row['path']) != approved['current_sha256']:
                raise ValueError('Global QC file changed during audit: '+row['path'])
        content = Path(c['response_qc']).read_bytes()
        added = proof['added_qc_line'].encode()
        lines = content.splitlines(keepends=True)
        if lines.count(added) != 1:
            raise ValueError('Reviewed SharedReward QC addition missing/duplicated')
        restored = b''.join(line for line in lines if line != added)
        if hashlib.sha256(restored).hexdigest() != expected[c['response_qc']]['frozen_sha256']:
            raise ValueError('Removing SharedReward addition does not restore exact frozen QC table')
        provenance = json.loads(Path(c['qc_provenance']).read_text())
        if provenance != proof['approved_qc_provenance']:
            raise ValueError('Global QC provenance differs from review')
    remaining = dict(audit, changes=[row for row in audit['changes'] if row['path'] not in expected])
    # Still verifies read errors, every other input, and the original repair receipt.
    a.validate_live_audit(remaining, c)
    return dict(review_sha256=REVIEW_SHA256, reviewed_upstream_commit=proof['reviewed_upstream_commit'],
                qc_transition_applied=bool(qc_changes), trust_qc_rows_unchanged=proof['trust_rows_unchanged'],
                changes_accepted=qc_changes,
                reason=proof['reason'] if qc_changes else 'Global QC files still match the frozen inputs.')


def snapshot(old, phase, c, audit_path):
    """Authenticate saved inputs, persist the live audit, then validate all changes."""
    t, prov = a.ppc.geometry.saved_trials(phase, c)
    audit = a.ppc.geometry.source_drift_audit(c, prov)
    audit['scope'] = ('New closeout fits use the authenticated frozen canonical table. Live inputs '
                      'must match except for receipt-authenticated removed empty templates and the '
                      'exact reviewed SharedReward QC addition. No refreshed data enter any fit.')
    audit['closeout_validation'] = {'status': 'pending'}
    s.save(audit_path, audit)  # Persist all changes, including unexpected ones, before raising.
    try:
        exception = validate_live_audit(audit, c)
        train = a.batch.entry_trials(t, dict(subset='train'))
        test = t[t.participant_id.isin(a.batch.paired_ids(t)) & t.run.eq(2)]
        if (train.participant_id.nunique() != old['expected_train_n']
                or int(train.valid_choice.sum()) != old['expected_train_choices']
                or int(test.valid_choice.sum()) != old['expected_test_choices']):
            raise ValueError('Paired cohort/count changed')
        reused = [dict(model=model, variant='zero', subset='train', name=name)
                  for model, name in a.BASELINES.items()]
        a.batch.review_reused(dict(reuse=reused, gamma_population_mean_prior_sd=phase['gamma_population_mean_prior_sd'],
                                   cmdstan_version=phase['cmdstan_version']), c, t)
        audit['closeout_validation'] = dict(status='passed', **exception)
    except BaseException as exc:
        audit['closeout_validation'] = dict(status='blocked', error=repr(exc))
        raise
    finally:
        s.save(audit_path, audit)
    return t, audit
