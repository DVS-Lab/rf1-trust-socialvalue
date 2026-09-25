"""Inspect saved divergent transitions and reference gradients; never sample."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

POP=('mu.','tau.','beta.','Omega.')
PARAM=('mu.','tau.','beta.','L.','z.')
METHOD=('divergent__','energy__','lp__','treedepth__','stepsize__','accept_stat__')


def scan_chain(path,expected_draws,chain,chunk_size=2048):
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith('#'):break
            if re.search(r'save_warmup\s*=\s*1',line):raise RuntimeError('Audit requires retained-only CSVs')
    chunks=[];states=[];offset=0;previous=None
    for chunk in pd.read_csv(path,comment='#',chunksize=chunk_size,
            usecols=lambda c:c.startswith(POP+PARAM) or c in METHOD):
        if not np.isfinite(chunk.to_numpy(float)).all():raise RuntimeError(f'Nonfinite saved state in {path}')
        for i in np.flatnonzero(chunk['divergent__'].to_numpy()==1):
            row=chunk.iloc[i]
            states.append({'chain':chain,'iteration':offset+int(i)+1,'kind':'flagged','row':row.copy()})
            before=chunk.iloc[i-1] if i>0 else previous
            if before is not None:states.append({'chain':chain,'iteration':offset+int(i),'kind':'previous','row':before.copy()})
        population=chunk[[c for c in chunk if c.startswith(POP) or c in METHOD]].copy()
        population['chain']=chain;population['iteration']=np.arange(offset+1,offset+len(chunk)+1)
        chunks.append(population);previous=chunk.iloc[-1].copy();offset+=len(chunk)
    if offset!=expected_draws:raise RuntimeError(f'Expected {expected_draws} retained draws in {path}, found {offset}')
    return pd.concat(chunks,ignore_index=True),states


def parameter_state(row,data):
    k,n,a=data['K'],data['N'],data['A']
    values={v:[float(row[f'{v}.{j}']) for j in range(1,k+1)] for v in ('mu','tau')}
    for v,nr,nc in [('beta',k,a),('L',k,k),('z',k,n)]:
        values[v]=[[float(row[f'{v}.{r}.{c}']) for c in range(1,nc+1)] for r in range(1,nr+1)]
    return values


def summarize_population(frame):
    rows=[]
    for column in frame:
        if not column.startswith(POP):continue
        values=frame[column].to_numpy()
        rows.append({'parameter':column,'mean':values.mean(),'sd':values.std(ddof=1),
                     'ci_low':np.quantile(values,.025),'ci_high':np.quantile(values,.975)})
    return pd.DataFrame(rows)


def compare_attempts(current,baseline):
    out=current.merge(baseline,on='parameter',suffixes=('_current','_baseline'),validate='one_to_one')
    out['mean_shift_baseline_sd']=(out.mean_current-out.mean_baseline)/out.sd_baseline.replace(0,np.nan)
    return out


def plot_geometry(frame,states,k,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,k,figsize=(4*k,7),squeeze=False)
    thin=max(1,len(frame)//4000);points=frame.iloc[::thin]
    colors={'flagged':'#c83b38','previous':'#e39d24'}
    for j in range(1,k+1):
        x=f'mu.{j}';y=f'tau.{j}'
        axes[0,j-1].scatter(points[x],points[y],s=3,alpha=.15,c='#336b87',rasterized=True)
        for state in states:axes[0,j-1].scatter(state['row'][x],state['row'][y],s=65,marker='x',c=colors[state['kind']],zorder=4)
        axes[0,j-1].set(xlabel=x,ylabel=y)
        for chain,part in frame.groupby('chain'):
            axes[1,j-1].plot(part.iteration.iloc[::thin],part[y].iloc[::thin],lw=.45,alpha=.6,label=f'Chain {chain}')
        for state in states:axes[1,j-1].scatter(state['iteration'],state['row'][y],s=65,marker='x',c=colors[state['kind']],zorder=4)
        axes[1,j-1].set(xlabel='Retained iteration',ylabel=y)
    axes[1,0].legend(fontsize=8)
    fig.suptitle(title+'\nRed: flagged recorded state; orange: previous draw. These are not the failing trajectory points.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(Path(str(path)+'.png'),dpi=180);fig.savefig(Path(str(path)+'.pdf'));plt.close(fig)


def main():
    import platform
    if platform.system()!='Linux':raise RuntimeError('Audit must read the current Linux posterior caches')
    root=Path.cwd();output=root/'results/diagnostic_review';output.mkdir(parents=True,exist_ok=True)
    from .checkpoint_handoff import verify_inputs
    from .linux_handoff import coordinator_lock,existing_workers,assert_no_live_fit_locks
    checkpoint=json.loads((root/'results/hierarchical_checkpoint.json').read_text());verify_inputs(root,checkpoint)
    with coordinator_lock(root):
        if existing_workers(root):raise RuntimeError('Wait for the sequential runner to exit before auditing')
        assert_no_live_fit_locks(root)
        state=json.loads((root/'results/linux_run_status.json').read_text())
        targets=[e for e in checkpoint['runs'] if state['runs'][e['run']]['status']=='diagnostic_failed' and e['kind']!='recovery']
        if not targets:print('No failed real-data fits to audit.');return
        from .hierarchical import stan_model,inputs,WORK
        from cmdstanpy import CmdStanModel
        fast=stan_model();source=Path('stan/hierarchical_shared.stan').read_text()
        ref_path=WORK/'full_reference_validation.stan'
        if not ref_path.exists() or ref_path.read_text()!=source:ref_path.write_text(source)
        reference=CmdStanModel(stan_file=str(ref_path.resolve()))
        implementation=hashlib.sha256(Path('stan/rl_fast.hpp').read_bytes()+Path('stan/hierarchical_fast.stan').read_bytes()).hexdigest()
        gradients=[];inventory=[]
        try:
            for entry in targets:
                name=entry['run'];folder=WORK/name;saved=json.loads((folder/'manifest.json').read_text());cfg=saved['settings']
                if saved.get('implementation_sha256')!=implementation:raise RuntimeError(f'Implementation changed since {name}; inspect before auditing')
                data=inputs(entry['model'],entry['training'],entry['age_terms'],entry['bounded'],prior_scale=entry['prior_scale'])[0]
                expected=hashlib.sha256((json.dumps(data,sort_keys=True)+json.dumps(cfg,sort_keys=True)+source).encode()).hexdigest()
                if expected!=saved['fingerprint']:raise RuntimeError(f'Input/settings fingerprint differs for {name}')
                frames=[];states=[]
                if len(saved['csv_files'])!=cfg['chains']:raise RuntimeError(f'Chain count differs for {name}')
                for chain,file in enumerate(saved['csv_files'],1):
                    frame,found=scan_chain(file,cfg['draws'],chain);frames.append(frame);states.extend(found)
                frame=pd.concat(frames,ignore_index=True)
                diagnostic=pd.read_csv(root/'results/tables'/f'diagnostics_{name}.csv')
                divergences=int(frame.divergent__.sum())
                if divergences!=int(diagnostic.divergences.iloc[0]):raise RuntimeError(f'Saved draws differ from diagnostic table: {name}')
                population=summarize_population(frame);population.to_csv(output/f'{name}_population.csv',index=False)
                baseline=root/'results/tables'/f'divergence_reference_linux_divergences_20260925_{name}.csv'
                if baseline.exists():compare_attempts(population,pd.read_csv(baseline)).to_csv(output/f'{name}_attempt_comparison.csv',index=False)
                locations=[]
                for event in states:
                    for parameter in population.parameter:
                        value=float(event['row'][parameter]);values=frame[parameter].to_numpy()
                        locations.append(dict(chain=event['chain'],iteration=event['iteration'],kind=event['kind'],parameter=parameter,
                                              value=value,percentile=float(np.mean(values<=value))))
                pd.DataFrame(locations).to_csv(output/f'{name}_locations.csv',index=False)
                chain_rows=[]
                for chain,part in frame.groupby('chain'):
                    energy=part.energy__.to_numpy()
                    chain_rows.append(dict(chain=int(chain),draws=len(part),divergences=int(part.divergent__.sum()),
                        max_depth_hits=int((part.treedepth__>=cfg['max_treedepth']).sum()),
                        bfmi=float(np.mean(np.diff(energy)**2)/np.var(energy)),step_size=float(part.stepsize__.median())))
                pd.DataFrame(chain_rows).to_csv(output/f'{name}_chains.csv',index=False)
                plot_geometry(frame,states,data['K'],output/name,name+' — diagnostic review; fit remains unaccepted')
                inventory.append(dict(run=name,fingerprint=saved['fingerprint'],settings=cfg,divergences=divergences))
                # Audit both states bracketing each marked transition. This is not the unavailable failing trajectory.
                for event in states:
                    pars=parameter_state(event['row'],data)
                    a=reference.log_prob(params=pars,data=data,jacobian=True,sig_figs=16).iloc[0]
                    b=fast.log_prob(params=pars,data=data,jacobian=True,sig_figs=16).iloc[0]
                    if list(a.index)!=list(b.index):raise RuntimeError('Reference/fast gradient parameter ordering differs')
                    difference=np.abs(a.to_numpy(float)-b.to_numpy(float));scaled=difference/(1+np.abs(a.to_numpy(float)))
                    finite=bool(np.isfinite(difference).all())
                    gradients.append(dict(run=name,chain=event['chain'],iteration=event['iteration'],kind=event['kind'],
                        max_absolute_error=float(difference.max()),max_scaled_error=float(scaled.max()),
                        worst_parameter=str(a.index[int(np.argmax(scaled))]),passed=finite and float(scaled.max())<1e-8))
                print(f'{name}: reviewed {divergences} divergent transitions; no sampling performed',flush=True)
        finally:
            pd.DataFrame(gradients).to_csv(output/'reference_gradient_checks.csv',index=False)
            (output/'audit_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        (output/'README.md').write_text('# Remaining-fit diagnostic audit\n\nNo MCMC was run. Original chains were read without modification. Fits remain unaccepted.\n\nThe marked recorded and preceding states are not the unrecorded points where a numerical trajectory failed. Gradient agreement at these states is a useful implementation check, not proof that a divergence is harmless. Mean shifts between attempts are descriptive sensitivity checks, not an acceptance rule.\n')
        if not all(row['passed'] for row in gradients):raise RuntimeError('Reference-gradient discrepancy found; inspect results/diagnostic_review before any further fitting')
        print('Audit complete. Push results/diagnostic_review and results/run_logs for review; no fits were accepted or restarted.',flush=True)
