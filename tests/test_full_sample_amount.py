import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from rf1_trust_socialvalue import full_sample_amount as a
from rf1_trust_socialvalue import full_sample_sampling as s
from test_full_sample_sampling import trials


def test_scope_is_four_training_fits_and_preserves_originals():
    cfg,phase,c=a.configuration()
    assert len(a.entries(cfg))==4 and all(e['subset']=='train' for e in a.entries(cfg))
    assert cfg['chains']*cfg['parallel_fits']==32
    assert set(a.BASELINES)=={'H7','HPreference'}


@pytest.mark.parametrize('model',['H7','HPreference'])
@pytest.mark.parametrize('extension',['bias','amount'])
def test_nested_zero_extension_and_original_data_priors(model,extension):
    t=trials(2);phase={'gamma_population_mean_prior_sd':1.};entry=dict(model=model,extension=extension)
    data,meta,arrays=a.model_data(t,entry,phase)
    original,_,_=s.model_data(t,model,True)
    assert {k:v for k,v in data.items() if k not in ['E','amount_feature']}==original
    assert len(meta['parameter_names'])==5+data['E']
    assert data['amount_feature']==[4/64,12/64,48/64]*2
    for frame in arrays.values():
        base=np.array([.3,.8,1.2,.4,.7])
        x=a.engine(frame,np.r_[base,np.zeros(data['E'])],s.SPECS[model][1])
        old=s.zero_engine(frame,base[:-1],s.SPECS[model][1],base[-1])
        np.testing.assert_allclose(x[:,0],old[:,1],atol=1e-14)
        np.testing.assert_allclose(x[:,1],old[:,3],atol=1e-14)


def test_bias_and_quadratic_correction_have_exact_logit_effect():
    frames=np.array([[2,0,2,0,0,1,0,1],[2,2,4,1,0,1,0,1],[2,4,8,1,0,1,0,1]],float)
    p=np.array([.3,.8,1.,.5,.7,0.,0.])
    baseline=a.engine(frames,p,7)[:,0];p[-2]=.6;p[-1]=-1.2
    changed=a.engine(frames,p,7)[:,0]
    logit=lambda v:np.log(v/(1-v))
    np.testing.assert_allclose(logit(changed)-logit(baseline),.6-1.2*(frames[:,2]**2-frames[:,1]**2)/64,atol=1e-13)
    assert changed[1]>baseline[1] and changed[2]<baseline[2]


def test_missing_and_zero_do_not_reveal_feedback_and_run2_is_online():
    frame=np.array([[0,0,2,0,0,1,0,1],[0,0,4,-1,0,1,0,1],[0,2,4,1,1,1,0,2],[0,2,4,1,1,0,0,2]],float)
    par=np.array([.5,1.,1.,.5,.3,.1,-.2]);base=a.engine(frame,par,7)
    hidden=frame.copy();hidden[:2,5]=0
    np.testing.assert_array_equal(base,a.engine(hidden,par,7))
    changed=frame.copy();changed[2,5]=0;result=a.engine(changed,par,7)
    np.testing.assert_allclose(base[:3,0],result[:3,0])
    assert base[3,0]>result[3,0] and base[1,1]==0


def test_live_gate_allows_only_authenticated_removed_templates():
    root=Path(__file__).resolve().parents[1]
    audit=json.loads((root/'results/full_sample/hierarchical/residual_review/live_source_audit.json').read_text())
    c=dict(bids_root='/ZPOOL/data/projects/rf1-sra-linux2/bids')
    a.validate_live_audit(audit,c)
    for change in ['other_file','modified','no_archive','wrong_hash','read_error']:
        bad=copy.deepcopy(audit)
        if change=='other_file':bad['changes'][0]['path']=c['bids_root']+'/participants.tsv'
        if change=='modified':bad['changes'][0]['change']='modified'
        if change=='no_archive':bad['repair_10668']=[]
        if change=='wrong_hash':bad['changes'][0]['frozen_sha256']='bad'
        if change=='read_error':bad['read_errors']=[{'error':'permission'}]
        with pytest.raises(ValueError):a.validate_live_audit(bad,c)
    clean=dict(audit,status='matches_snapshot',changes=[],repair_10668=[])
    a.validate_live_audit(clean,c)


def test_extension_hyperparameters_are_in_diagnostic_gate():
    _,_,c=a.configuration();cfg={'max_treedepth':14}
    d=pd.DataFrame({'R_hat':[1.,1.2],'ESS_bulk':[1000,1000],'ESS_tail':[1000,1000]},index=['mu[1]','mu_ext[1]'])
    fit=SimpleNamespace(summary=lambda:d,method_variables=lambda:{'energy__':np.tile([0.,1.],(20,2)).reshape(20,4),'divergent__':np.zeros((20,4)),'treedepth__':np.ones((20,4))})
    # Vary energy by iteration; BFMI must independently be finite and > .3.
    fit.method_variables=lambda:{'energy__':np.tile(np.arange(20)%2,(4,1)).T,'divergent__':np.zeros((20,4)),'treedepth__':np.ones((20,4))}
    diagnostics,info=a.diagnostics(fit,cfg,c)
    assert 'mu_ext[1]' in set(diagnostics.parameter) and not info['passed'] and info['max_rhat']==1.2


def test_amount_comparison_is_paired_and_excludes_failed_fits(tmp_path):
    cfg,_,c=a.configuration();c=dict(c,output=str(tmp_path/'out'),work=str(tmp_path/'work'))
    base,_=s.paths(c);out,_=a.paths(c)
    def write(folder,name,model,value,passed=True):
        folder.mkdir(parents=True,exist_ok=True)
        f=folder/'heldout_participants.tsv'
        pd.DataFrame([dict(name=name,model=model,participant_id=f'sub-{i}',n=10,log_loss=value+.01*i,brier=.2) for i in range(10)]).to_csv(f,sep='\t',index=False)
        s.save(folder/'status.json',dict(status='complete' if passed else 'diagnostic_failed',passed=passed,heldout_sha256=s.fs.sha(f),output_sha256={f.name:s.fs.sha(f)}))
    for model,name in a.BASELINES.items():write(base/'fits'/name,name,model,.6)
    for entry in a.entries(cfg):write(out/'fits'/entry['name'],entry['name'],entry['model'],.5 if entry['extension']=='bias' else .4,entry['model']=='H7')
    a.compare(cfg,c)
    result=pd.read_csv(out/'amount_vs_bias.tsv',sep='\t')
    assert result.model.tolist()==['H7']
    assert result.amount_minus_bias.iloc[0]==pytest.approx(-.1)
    assert len(pd.read_csv(out/'available_model_comparison.tsv',sep='\t'))==4


def test_run2_scoring_exactly_nests_original_and_preserves_subject_pairing():
    from rf1_trust_socialvalue import full_sample_batch as batch
    t1=trials(2);t2=t1.copy();t2['run']=2;t=pd.concat([t1,t2],ignore_index=True)
    # Deliberately shuffled data: both adapters restore subject/run/trial order.
    t=t.sample(frac=1,random_state=4)
    phase={'gamma_population_mean_prior_sd':1.};entry=dict(model='H7',extension='bias',name='test',subset='train')
    _,meta,_=a.model_data(t1,entry,phase)
    base=np.tile([.3,.8,1.2,.4,.7],(20,2,1));extended=np.concatenate([base,np.zeros((20,2,1))],axis=2)
    result,cells=a.heldout(t,entry,extended,meta)
    old=batch.heldout_scores(t,dict(model='H7',variant='zero',name='test'),base,dict(meta,parameter_names=meta['parameter_names'][:5]))
    np.testing.assert_allclose(result[['n','log_loss','brier','accuracy']],old[['n','log_loss','brier','accuracy']],atol=1e-14)
    assert len(result)==2 and set(cells[cells.stratification.eq('partner_offer')].group)=={'friend_0/2','stranger_2/4','computer_4/8'}
    with pytest.raises(ValueError,match='identity'):a.heldout(t,entry,extended,dict(meta,ids=meta['ids'][::-1]))
