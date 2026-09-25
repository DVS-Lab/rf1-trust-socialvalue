"""Bounded independent fit workers, with serial aggregation and strict final gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


def execution_chains(cfg):
    # Execution concurrency does not alter the posterior/cache fingerprint.
    value=int(os.environ.get('RF1_PARALLEL_CHAINS',cfg['parallel_chains']))
    if not 1<=value<=cfg['chains']:raise ValueError('RF1_PARALLEL_CHAINS must be between 1 and chains')
    return value


def now():return datetime.now(timezone.utc).isoformat()


def write_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def worker(entry):
    from .hierarchical import sample,summaries,predictive,heldout,TABLE
    from .hierarchical_validation import recover
    from .sampling_retry import DiagnosticFailure
    import pandas as pd
    name=entry['run']
    if entry['kind']=='recovery':
        _,_,condition,replicate=name.split('_');recover(condition,int(replicate));return
    manifest=Path('work/hierarchical')/name/'manifest.json'
    settings=None if manifest.exists() else entry['settings']
    fit,meta,arrays,ratings,frames=sample(entry['model'],entry['training'],entry['age_terms'],entry['bounded'],prior_scale=entry['prior_scale'],settings=settings)
    diag=pd.read_csv(TABLE/f'diagnostics_{name}.csv')
    if not diag.passed.all():
        raise DiagnosticFailure(f'{name}: max Rhat={diag.R_hat.max():.5f}; min bulk ESS={diag.ESS_bulk.min():.1f}; '
            f'min tail ESS={diag.ESS_tail.min():.1f}; divergences={diag.divergences.max()}; '
            f'max-depth hits={diag.max_depth_hits.max()}; min BFMI={diag.min_bfmi.min():.3f}')
    if entry['training']:heldout(fit,meta,arrays,ratings,name)
    else:summaries(fit,meta,frames,name);predictive(fit,meta,arrays,ratings,frames,name)
    # Shared collect()/plots/report are deliberately deferred to the coordinator.


def launch_worker(entry,folder,chains,root):
    env=dict(os.environ,RF1_PARALLEL_CHAINS=str(chains),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    task_file=folder/(entry['run']+'.task.json')
    write_json(task_file,entry)
    command=[sys.executable,'-u','scripts/17_parallel_checkpoint.py','--worker',entry['run'],'--task-file',str(task_file)]
    with (folder/(entry['run']+'.txt')).open('w') as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    status='complete' if result.returncode==0 else 'diagnostic_failed' if result.returncode==2 else 'error'
    return {'status':status,'exit_code':result.returncode,'log':str((folder/(entry['run']+'.txt')).relative_to(root)),'updated_at':now()}


def dispatch(entries,jobs,launch,update):
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        def run(entry):
            update(entry['run'],{'status':'running','updated_at':now()})
            return launch(entry)
        futures={pool.submit(run,e):e for e in entries}
        for future in as_completed(futures):
            name=futures[future]['run']
            try:result=future.result()
            except Exception as exc:result={'status':'error','error':repr(exc),'updated_at':now()}
            update(name,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--jobs',type=int,default=8,choices=range(1,13))
    parser.add_argument('--parallel-chains',type=int,default=4,choices=range(1,5))
    parser.add_argument('--take-over-after-current',action='store_true')
    parser.add_argument('--retry-plan',type=Path,help='Explicit fingerprint-bound sampler retry plan')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    parser.add_argument('--task-file',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args();root=Path.cwd().resolve()
    from .checkpoint_handoff import verify_inputs,prepare_caches
    checkpoint=json.loads((root/'results/hierarchical_checkpoint.json').read_text());verify_inputs(root,checkpoint)
    if not args.run and not args.worker:
        print(f'Plan: {args.jobs} independent fits x {args.parallel_chains} chains, up to {args.jobs*args.parallel_chains} sampling CPUs. No fits started.')
        return
    if platform.system()!='Linux':raise RuntimeError('Parallel continuation runs on Linux only')
    if args.worker:
        from .sampling_retry import DiagnosticFailure
        entry=next(e for e in checkpoint['runs'] if e['run']==args.worker)
        if args.task_file:
            task=json.loads(args.task_file.read_text())
            if {k:v for k,v in task.items() if k!='settings'}!={k:v for k,v in entry.items() if k!='settings'}:
                raise RuntimeError('Worker task changes the checkpoint target')
            entry=task
        try:worker(entry)
        except DiagnosticFailure as exc:
            print(str(exc),flush=True);raise SystemExit(2)
        return
    from .linux_handoff import coordinator_lock,existing_workers,take_over,assert_no_live_fit_locks
    with coordinator_lock(root):
        old=existing_workers(root)
        if old:
            if not args.take_over_after_current:raise RuntimeError('Existing sequential runner is active; use --take-over-after-current')
            take_over(root,old)
        assert_no_live_fit_locks(root)
        prepare_caches(root,checkpoint)
        entries=checkpoint['runs']
        if args.retry_plan:
            from .targeted_retry import prepare_retries
            entries=prepare_retries(root,checkpoint,json.loads(args.retry_plan.read_text()))
        # Compile once before launching workers, avoiding races on the shared executable.
        from .hierarchical import stan_model,collect
        stan_model()
        if args.retry_plan:
            from .targeted_retry import audit_divergent_draws
            audit_divergent_draws(root,json.loads(args.retry_plan.read_text()))
        folder=root/'results/run_logs'/('parallel-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
        status_path=root/'results/linux_run_status.json'
        if status_path.exists():(folder/'prior_linux_run_status.json').write_bytes(status_path.read_bytes())
        state={'status':'running','started_at':now(),'jobs':args.jobs,'parallel_chains':args.parallel_chains,
               'runs':{e['run']:{'status':'pending'} for e in checkpoint['runs']}}
        import threading
        mutex=threading.Lock()
        def update(name,value):
            with mutex:
                state['runs'][name]=value;write_json(status_path,state)
                print(f'{name}: {value["status"]}',flush=True)
        write_json(status_path,state)
        dispatch(entries,args.jobs,lambda e:launch_worker(e,folder,args.parallel_chains,root),update)
        try:collect()
        except BaseException:
            state['status']='aggregation_failed';write_json(status_path,state);raise
        failed=[name for name,value in state['runs'].items() if value['status']!='complete']
        if failed:
            state.update(status='incomplete',finished_at=now());write_json(status_path,state)
            raise RuntimeError('Finalization blocked; inspect per-fit logs for: '+', '.join(failed))
        state['status']='finalizing';write_json(status_path,state)
        try:
            from .hierarchical_finish import finish
            from .hierarchical_figures import figures
            from .second_pass_report import report
            from .second_pass_gallery import overview,gallery
            finish();figures();report();overview();gallery()
            subprocess.run([sys.executable,'scripts/13_validate_second_pass.py','--hierarchical'],check=True)
        except BaseException:
            state['status']='finalization_failed';write_json(status_path,state);raise
        state.update(status='complete',finished_at=now());write_json(status_path,state)
