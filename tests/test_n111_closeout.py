import numpy as np
import pandas as pd
import pytest

from rf1_trust_socialvalue import n111_closeout as c
from rf1_trust_socialvalue.n111_diagnostics import verify_scope
from pathlib import Path


@pytest.fixture(scope='module')
def evidence():
    return (pd.read_csv(c.TABLE/'realistic_model_recovery.csv'),
            pd.read_csv(c.TABLE/'realistic_model_recovery_confusion.csv'),
            pd.read_csv('results/tables/trial_table.csv'), verify_scope(Path.cwd()))


def test_independent_reconstruction_agrees_with_all_posted_confusion_rates(evidence):
    frame, confusion, trials, ids = evidence
    rebuilt = c.validate_recovery(frame, confusion, {}, ids, trials)
    assert len(rebuilt) == len(confusion)
    assert rebuilt.groupby(['variant','regime','unit','metric','generating']).selection_probability.sum().between(1-1e-12,1+1e-12).all()


def test_changed_confusion_or_criterion_is_rejected(evidence):
    frame, confusion, trials, ids = evidence
    wrong = confusion.copy(); wrong.loc[0,'selection_probability'] += .01
    with pytest.raises(AssertionError):
        c.validate_recovery(frame, wrong, {}, ids, trials)
    wrong = frame.copy(); wrong.loc[0,'AICc'] += .01
    with pytest.raises(AssertionError):
        c.validate_recovery(wrong, confusion, {}, ids, trials)


def test_missing_case_and_changed_sample_are_rejected(evidence):
    frame, confusion, trials, ids = evidence
    with pytest.raises(ValueError, match='Incomplete'):
        c.validate_recovery(frame.iloc[:-1], confusion, {}, ids, trials)
    wrong = frame.copy(); wrong.loc[0,'participant_id'] = 'outside_sample'
    with pytest.raises(ValueError, match='participants'):
        c.validate_recovery(wrong, confusion, {}, ids, trials)


def test_individual_and_dataset_selection_have_distinct_denominators():
    rows=[]
    for i, scores in enumerate([[0., 4.], [3., 0.]]):
        for model, score in zip(['H7','HPreference'],scores):
            rows.append(dict(variant='zero',regime='empirical',generating='H7',replicate=0,participant_id=str(i),fitted=model,
                             AICc=score,BIC=score,heldout_log_loss=score,heldout_brier=score))
    d=c.reconstruct_confusion(pd.DataFrame(rows))
    individual=d[(d.unit=='participant')&(d.metric=='AICc')]
    dataset=d[(d.unit=='dataset')&(d.metric=='AICc')]
    assert individual.selection_probability.eq(.5).all() and individual.n_units.eq(2).all()
    assert dataset.set_index('fitted').loc['H7','selection_probability']==1
    assert dataset.n_units.eq(1).all()
