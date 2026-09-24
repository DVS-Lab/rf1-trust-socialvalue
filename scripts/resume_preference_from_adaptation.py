"""Resume adapted diagonal chains; preserve original warmup and discard pilot retained draws."""
from pathlib import Path
import os,json,time,hashlib,re,argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from cmdstanpy import from_csv,write_stan_json
from rf1_trust_socialvalue.hierarchical import inputs,stan_model,configuration,diagnostics,summaries,predictive,collect,WORK,load_chains
from rf1_trust_socialvalue.fitting import stable_seed
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True,help='Four interrupted CSVs with completed 2000-iteration adaptation')
args=parser.parse_args()
name='HPreference_full_age';source=args.source;folder=WORK/name
if folder.exists():raise RuntimeError('Destination must be absent; archive prior attempts explicitly before continuing')
cfg=configuration();cfg['parallel_chains']=4;cfg['metric']='diag_e'
assert cfg['warmup']==2000 and cfg['draws']==2000
data,meta,arrays,ratings,frames=inputs('HPreference')
prepared=[]
for i,p in enumerate(sorted(source.glob('*.csv')),1):
 text=p.read_text();lines=text.splitlines();assert re.search(r'#\s+num_warmup = 2000',text)
 d=pd.read_csv(p,comment='#').dropna();assert len(d)>0
 last=d.iloc[-1];K=data['K'];N=data['N'];A=data['A']
 init={v:[float(last[f'{v}.{j}']) for j in range(1,K+1)] for v in ['mu','tau']}
 for v,nr,nc in [('beta',K,A),('L',K,K),('z',K,N)]:
  init[v]=[[float(last[f'{v}.{r}.{c}']) for c in range(1,nc+1)] for r in range(1,nr+1)]
 inv=np.array([float(x) for x in lines[next(j for j,x in enumerate(lines) if 'Diagonal elements of inverse mass matrix:' in x)+1].removeprefix('# ').split(',')]);assert np.all(inv>0)
 step=float(re.search(r'# Step size = ([\d.eE+-]+)',text).group(1));assert step>0
 assert np.isfinite(np.linalg.det(init['L'])) and np.linalg.det(init['L'])>0
 prepared.append((i,init,inv,step,str(p.resolve()),len(d)))
 print('Prepared',i,'discarded pilot draws',len(d),'step',step,flush=True)
assert len(prepared)==4
folder.mkdir(parents=True)
lock=WORK/(name+'.lock')
with lock.open('x') as handle:handle.write(str(os.getpid()))
sm=stan_model();start=time.time()
def run(item):
 i,init,inv,step,sourcefile,pilot=item
 metric=folder/f'metric_chain{i}.json';write_stan_json(str(metric),{'inv_metric':inv})
 out=sm.sample(data=data,chains=1,chain_ids=[i],iter_warmup=0,iter_sampling=2000,adapt_engaged=False,
     seed=stable_seed(cfg['seed'],name,'resume_diagonal'),inits=init,metric='diag_e',inv_metric=str(metric.resolve()),step_size=step,
     max_treedepth=12,output_dir=str((folder/f'chain{i}').resolve()),show_console=False,show_progress=False,refresh=200,sig_figs=10)
 return out.runset.csv_files[0]
with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(run,prepared))
assert len(set(files))==4
fit=load_chains(files)
fingerprint=hashlib.sha256((json.dumps(data,sort_keys=True)+json.dumps(cfg,sort_keys=True)+Path('stan/hierarchical_shared.stan').read_text()).encode()).hexdigest()
saved=dict(fingerprint=fingerprint,csv_files=files,seconds=time.time()-start,settings=cfg,meta=meta,
 implementation='exact_analytic_gradient',implementation_sha256=hashlib.sha256(Path('stan/rl_fast.hpp').read_bytes()+Path('stan/hierarchical_fast.stan').read_bytes()).hexdigest(),
 sampling_segments={'warmup_per_chain':2000,'retained_per_chain':2000,'retained_segment_warmup':0,'retained_segment_adaptation':False,'retained_seed':stable_seed(cfg['seed'],name,'resume_diagonal'),
 'adaptation_source':[dict(chain=i,csv=f,pilot_draws_discarded=n,step_size=step) for i,_,_,step,f,n in prepared],
 'note':'Reuse final adapted diagonal metric, step size and last state from each original chain; start an independent random stream. Original pilot retained draws discarded. Same posterior target.'})
(folder/'manifest.json').write_text(json.dumps(saved,indent=2)+'\n')
diagnostics(fit,name,meta,cfg);summaries(fit,meta,frames,name);predictive(fit,meta,arrays,ratings,frames,name);collect();lock.unlink(missing_ok=True)
print('Resumed preference fit complete',flush=True)
