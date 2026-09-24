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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Fit on Linux, then finalize and validate; never starts MCMC by default')
    parser.add_argument('--prepare-cache',action='store_true',help='Rebase copied completed-cache manifests without fitting')
    parser.add_argument('--parallel-chains',type=int,default=2,choices=[1,2,3,4])
    args=parser.parse_args();root=Path.cwd().resolve()
    checkpoint=json.loads((root/'results/hierarchical_checkpoint.json').read_text())
    verify_inputs(root,checkpoint)
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
    # One fit at a time avoids the concurrent queue pressure seen on the laptop.
    for entry in checkpoint['runs']:
        name=entry['run'];manifest=root/'work/hierarchical'/name/'manifest.json'
        if entry['kind']=='recovery':
            if not manifest.exists():
                _,_,condition,replicate=name.split('_');recover(condition,int(replicate))
            continue
        settings=None if manifest.exists() else dict(entry['settings'],parallel_chains=args.parallel_chains)
        fit,meta,arrays,ratings,frames=sample(entry['model'],entry['training'],entry['age_terms'],entry['bounded'],prior_scale=entry['prior_scale'],settings=settings)
        if not pd.read_csv(TABLE/f'diagnostics_{name}.csv').passed.all():
            raise RuntimeError(f'Final diagnostic thresholds still fail: {name}; inspect before interpreting')
        if entry['training']:heldout(fit,meta,arrays,ratings,name)
        else:summaries(fit,meta,frames,name);predictive(fit,meta,arrays,ratings,frames,name)
        collect()
    from .hierarchical_finish import finish
    from .hierarchical_figures import figures
    from .second_pass_report import report
    from .second_pass_gallery import overview,gallery
    import subprocess,sys
    finish();figures();report();overview();gallery()
    subprocess.run([sys.executable,'scripts/13_validate_second_pass.py','--hierarchical'],check=True)
    print('Hierarchical continuation complete. Inspect figures/report and commit final results.',flush=True)
