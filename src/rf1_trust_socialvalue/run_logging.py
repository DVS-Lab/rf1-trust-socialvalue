"""Git-visible console logs and bounded sampler evidence; never copy posterior draws."""
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid


def write_tail(source,destination,limit=250_000):
    destination.parent.mkdir(parents=True,exist_ok=True)
    size=source.stat().st_size
    with source.open('rb') as stream:
        stream.seek(max(0,size-limit));content=stream.read()
    prefix=f'[Last {limit} of {size} bytes; full file remains under work/]\n' if size>limit else ''
    destination.write_text(prefix+content.decode('utf-8',errors='replace'))


def collect_artifacts(root,destination):
    root=Path(root).resolve();destination=Path(destination)
    for folder in sorted((root/'work/hierarchical').glob('*')):
        if not folder.is_dir():continue
        manifest=folder/'manifest.json'
        if manifest.exists():
            saved=json.loads(manifest.read_text())
            # A portable record, not a resumable cache or a copy of the raw draws.
            record={k:saved[k] for k in ('fingerprint','settings','seconds','implementation','implementation_sha256') if k in saved}
            record['posterior_files']=[Path(f).name for f in saved.get('csv_files',[])]
            target=destination/'sampler'/folder.name/'manifest_summary.json'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(record,indent=2)+'\n')
        for source in sorted(folder.rglob('*.txt')):
            write_tail(source,destination/'sampler'/folder.name/source.relative_to(folder))
    for source in sorted((root/'work/logs').glob('linux-resume-*.log')):
        write_tail(source,destination/'legacy'/source.with_suffix('.txt').name,limit=1_000_000)
    status=root/'results/linux_run_status.json'
    if status.exists():
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'linux_run_status.json').write_bytes(status.read_bytes())


def run_logged(command,root):
    root=Path(root).resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    folder=root/'results/run_logs'/stamp;folder.mkdir(parents=True)
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True)
    record={'started_at':datetime.now(timezone.utc).isoformat(),'command':command,
            'git_commit':revision.stdout.strip() if revision.returncode==0 else None,
            'platform':platform.platform(),'status':'running'}
    state=folder/'run.json'
    def save():state.write_text(json.dumps(record,indent=2)+'\n')
    save();print(f'Git-visible run log: {folder}/console.txt',flush=True)
    try:
        with (folder/'console.txt').open('w',buffering=1) as log:
            child=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
            try:
                for line in child.stdout:
                    log.write(line);sys.stdout.write(line);sys.stdout.flush()
                code=child.wait()
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        record.update(status='complete' if code==0 else 'failed',exit_code=code)
    except BaseException as exc:
        record.update(status='interrupted_or_error',error=repr(exc))
        raise
    finally:
        record['finished_at']=datetime.now(timezone.utc).isoformat()
        try:collect_artifacts(root,folder)
        except Exception as exc:record['artifact_collection_error']=repr(exc)
        save()
        print(f'Run record saved in {folder}; include results/run_logs when committing.',flush=True)
    return code
