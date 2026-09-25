"""Portable cache preparation and sequential Linux continuation of a saved checkpoint."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import platform


def verify_inputs(root, checkpoint):
    for relative, expected in checkpoint['input_hashes'].items():
        path=root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Checkpoint input differs or is missing: {relative}')


def relocated_path(value, root):
    parts=PurePosixPath(value).parts
    indexes=[i for i in range(len(parts)-1) if parts[i:i+2]==('work','hierarchical')]
    if len(indexes)!=1:raise ValueError(f'Unexpected cache path: {value}')
    path=(root/Path(*parts[indexes[0]:])).resolve()
    if not path.is_relative_to((root/'work/hierarchical').resolve()):raise ValueError('Cache path escapes the posterior directory')
    return path


def prepare_caches(root, checkpoint):
    """Rebase copied manifests; leave posterior CSVs and statistical fingerprints unchanged."""
    prepared=[]
    for entry in checkpoint['runs']:
        manifest=root/'work/hierarchical'/entry['run']/'manifest.json'
        if not manifest.exists():continue
        saved=json.loads(manifest.read_text())
        files=[relocated_path(value,root) for value in saved['csv_files']]
        if len(files)!=len(set(files)) or not all(p.is_file() for p in files):
            raise RuntimeError(f'Incomplete copied posterior cache: {entry["run"]}')
        saved['csv_files']=[str(p) for p in files]
        if 'sampling_segments' in saved:
            for source in saved['sampling_segments']['adaptation_source']:
                source['csv']=str(relocated_path(source['csv'],root))
        manifest.write_text(json.dumps(saved,indent=2)+'\n')
        prepared.append(entry['run'])
    return prepared


def execute_jobs(entries, execute, update):
    from .sampling_retry import DiagnosticFailure
    failed=[]
    for entry in entries:
        name=entry['run'];update(name,'running',None)
        try:
            execute(entry)
        except DiagnosticFailure as exc:
            failed.append(name);update(name,'diagnostic_failed',str(exc))
            print(f'{name}: {exc}; continuing independent fits',flush=True)
        except Exception as exc:
            update(name,'error',str(exc))
            raise
        else:
            update(name,'complete',None)
    return failed


def prepare_retry(root, name, parallel_chains):
    """Explicitly restart a failed fit from a geometry-clean attempt's settings."""
    import pandas as pd
    from .sampling_retry import archive_fit,retry_settings
    folder=root/'work/hierarchical'/name
    saved=json.loads((folder/'manifest.json').read_text())
    current=pd.read_csv(root/'results/tables'/f'diagnostics_{name}.csv')
    if current.passed.all():raise RuntimeError(f'Refusing to replace passing posterior: {name}')
    candidates=[]
    for manifest in sorted(folder.parent.glob(name+'_attempt*/manifest.json')):
        diag_path=root/'results/tables'/f'diagnostics_{manifest.parent.name}.csv'
        if not diag_path.exists():continue
        diag=pd.read_csv(diag_path);cfg=json.loads(manifest.read_text())['settings']
        if diag.divergences.max()!=0 or diag.max_depth_hits.max()!=0 or diag.min_bfmi.min()<=.3:continue
        retry=retry_settings(cfg,passed=bool(diag.passed.all()),divergences=0,max_depth_hits=0,min_bfmi=diag.min_bfmi.min())
        if retry is not None and retry[0] in ('short','8000'):
            candidates.append((cfg['adapt_delta'],str(manifest),retry[1]))
    if not candidates:raise RuntimeError(f'No geometry-clean attempt eligible for a longer retry: {name}')
    _,source,settings=min(candidates,key=lambda x:(x[0],x[1]))
    archive=archive_fit(folder,saved,'_before_manual_retry')
    current['run']=archive.name;current.to_csv(root/'results/tables'/f'diagnostics_{archive.name}.csv',index=False)
    settings=dict(settings,parallel_chains=parallel_chains)
    print(f'{name}: preserved failed cache in {archive}; retry settings from {source}: {settings}',flush=True)
    return settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Fit on Linux, then finalize and validate; never starts MCMC by default')
    parser.add_argument('--prepare-cache',action='store_true',help='Rebase copied completed-cache manifests without fitting')
    parser.add_argument('--parallel-chains',type=int,default=2,choices=[1,2,3,4])
    parser.add_argument('--retry-run',help='Explicitly archive one failed real-data fit and retry longer at its geometry-clean attempt settings')
    args=parser.parse_args();root=Path.cwd().resolve()
    if args.retry_run and not args.run:parser.error('--retry-run requires --run')
    checkpoint=json.loads((root/'results/hierarchical_checkpoint.json').read_text())
    verify_inputs(root,checkpoint)
    if args.retry_run and args.retry_run not in {e['run'] for e in checkpoint['runs'] if e['kind']!='recovery'}:
        parser.error('--retry-run must name a real-data run in the checkpoint')
    for entry in checkpoint['runs']:
        cached=(root/'work/hierarchical'/entry['run']/'manifest.json').is_file()
        print(entry['run'],entry['status'],'posterior cache present' if cached else 'posterior cache absent',flush=True)
    if not args.run and not args.prepare_cache:return
    if platform.system()!='Linux':raise RuntimeError('This handoff command runs on Linux; laptop jobs must remain paused')
    prepared=prepare_caches(root,checkpoint);print(f'Prepared {len(prepared)} copied posterior manifests',flush=True)
    if not args.run:return
    from .hierarchical import sample,summaries,predictive,heldout,collect,TABLE
    from .hierarchical_validation import recover
    import pandas as pd
    from .sampling_retry import DiagnosticFailure
    from datetime import datetime,timezone
    state={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running',
           'runs':{e['run']:{'status':'pending'} for e in checkpoint['runs']}}
    status_path=root/'results/linux_run_status.json'
    def save_status():
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status_path)
    def update(name,status,error):
        state['runs'][name]={'status':status,'error':error,'updated_at':datetime.now(timezone.utc).isoformat()}
        save_status()
    save_status()
    def execute(entry):
        name=entry['run'];manifest=root/'work/hierarchical'/name/'manifest.json'
        if entry['kind']=='recovery':
            _,_,condition,replicate=name.split('_');recover(condition,int(replicate))
            return
        # Retry only when this run is reached, leaving other valid caches in place.
        if name==args.retry_run:
            settings=prepare_retry(root,name,args.parallel_chains)
        else:
            settings=None if manifest.exists() else dict(entry['settings'],parallel_chains=args.parallel_chains)
        fit,meta,arrays,ratings,frames=sample(entry['model'],entry['training'],entry['age_terms'],entry['bounded'],prior_scale=entry['prior_scale'],settings=settings)
        diag=pd.read_csv(TABLE/f'diagnostics_{name}.csv')
        if not diag.passed.all():
            raise DiagnosticFailure(f'Diagnostic thresholds fail: max Rhat={diag.R_hat.max():.5f}, '
                f'min bulk ESS={diag.ESS_bulk.min():.1f}, min tail ESS={diag.ESS_tail.min():.1f}, '
                f'divergences={diag.divergences.max()}, max-depth hits={diag.max_depth_hits.max()}, '
                f'min BFMI={diag.min_bfmi.min():.3f}')
        if entry['training']:heldout(fit,meta,arrays,ratings,name)
        else:summaries(fit,meta,frames,name);predictive(fit,meta,arrays,ratings,frames,name)
        collect()
    try:
        failed=execute_jobs(checkpoint['runs'],execute,update)
    except BaseException:
        state['status']='error';save_status();raise
    if failed:
        state['status']='incomplete';save_status()
        raise DiagnosticFailure(f'Independent fits finished, but finalization is blocked by: {", ".join(failed)}')
    state['status']='finalizing';save_status()
    from .hierarchical_finish import finish
    from .hierarchical_figures import figures
    from .second_pass_report import report
    from .second_pass_gallery import overview,gallery
    import subprocess,sys
    try:
        finish();figures();report();overview();gallery()
        subprocess.run([sys.executable,'scripts/13_validate_second_pass.py','--hierarchical'],check=True)
    except BaseException:
        state['status']='finalization_failed';save_status();raise
    state['status']='complete';save_status()
    print('Hierarchical continuation complete. Inspect figures/report and commit final results.',flush=True)
