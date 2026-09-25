"""Linux-only, identity-checked handoff after the active fit's recorded completion."""
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time


def process(pid):
    folder=Path('/proc')/str(pid)
    try:
        if folder.stat().st_uid!=os.getuid():return None
        raw=(folder/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        command=(folder/'cmdline').read_bytes().split(b'\0')
        return {'pid':int(pid),'ppid':int(fields[1]),'started':fields[19],
                'state':fields[0],'cwd':str((folder/'cwd').resolve(strict=True)),
                'command':[x.decode(errors='replace') for x in command if x]}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def is_sequential_worker(info,root):
    return bool(info and info['cwd']==str(root) and '-c' in info['command'] and
        'from rf1_trust_socialvalue.checkpoint_handoff import main; main()' in info['command'] and
        '--run' in info['command'])


def same_process(saved,current):
    return bool(current and all(saved[k]==current[k] for k in ('pid','started','cwd','command')))


def existing_workers(root):
    return [p for entry in Path('/proc').iterdir() if entry.name.isdigit()
            if is_sequential_worker(p:=process(int(entry.name)),root)]


@contextmanager
def coordinator_lock(root):
    import fcntl
    path=root/'work/parallel_coordinator.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another parallel coordinator already owns this checkout')
        handle.seek(0);handle.truncate();handle.write(str(os.getpid()));handle.flush()
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def descendants(pid):
    rows={int(p.name):row for p in Path('/proc').iterdir() if p.name.isdigit() if (row:=process(int(p.name)))}
    selected={pid};changed=True
    while changed:
        new={n for n,r in rows.items() if r['ppid'] in selected}-selected
        selected.update(new);changed=bool(new)
    return [rows[n] for n in selected if n in rows]


def stop_verified_tree(saved):
    current=process(saved['pid'])
    if not same_process(saved,current):raise RuntimeError('Sequential runner identity changed; refusing signals')
    os.kill(saved['pid'],signal.SIGSTOP)
    # Freeze the coordinator first so it cannot launch another fit during the snapshot.
    targets=descendants(saved['pid'])
    for info in targets:
        if info['pid']!=saved['pid'] and same_process(info,process(info['pid'])):
            try:os.kill(info['pid'],signal.SIGSTOP)
            except ProcessLookupError:pass
    for info in reversed(targets):
        if same_process(info,process(info['pid'])):
            try:
                os.kill(info['pid'],signal.SIGTERM);os.kill(info['pid'],signal.SIGCONT)
            except ProcessLookupError:pass
    deadline=time.monotonic()+10
    def alive():return [p for p in targets if same_process(p,c:=process(p['pid'])) and c['state']!='Z']
    while alive() and time.monotonic()<deadline:time.sleep(.2)
    for info in alive():
        try:os.kill(info['pid'],signal.SIGKILL)
        except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while alive() and time.monotonic()<deadline:time.sleep(.2)
    if alive():raise RuntimeError('Old analysis processes did not exit; refusing parallel start')
    return [p['pid'] for p in targets]


def take_over(root,workers):
    if len(workers)!=1:raise RuntimeError('Expected exactly one sequential runner; inspect active processes')
    saved=workers[0];parent=process(saved['ppid'])
    path=root/'results/linux_run_status.json';state=json.loads(path.read_text())
    active=[name for name,info in state['runs'].items() if info['status']=='running']
    if len(active)>1:raise RuntimeError('Status has multiple running fits; refusing sequential handoff')
    target=active[0] if active else None
    print(f'Waiting for {target or "the existing runner"} to finish; current chains stay running.',flush=True)
    stopped=[]
    while same_process(saved,current:=process(saved['pid'])) and current['state']!='Z':
        state=json.loads(path.read_text())
        if target and state['runs'][target]['status'] in ('complete','diagnostic_failed'):
            stopped=stop_verified_tree(saved)
            print(f'{target} reached its saved fit boundary; old coordinator stopped.',flush=True)
            break
        time.sleep(2)
    # The old logging wrapper exports its evidence when its worker exits.
    if parent and any(x.endswith('scripts/15_resume_checkpoint.py') for x in parent['command']):
        deadline=time.monotonic()+60
        while same_process(parent,current:=process(parent['pid'])) and current['state']!='Z':
            if time.monotonic()>deadline:raise RuntimeError('Old logging wrapper is still exporting; rerun after it exits')
            time.sleep(.5)
    note={'at':datetime.now(timezone.utc).isoformat(),'waited_for':target,'stopped_process_ids':stopped,
          'note':'Stopped after saved completion of the active fit; a following fit may have started briefly before polling detected the boundary.'}
    dest=root/'results/run_logs'/('handoff-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(note,indent=2)+'\n')


def assert_no_live_fit_locks(root):
    for lock in (root/'work/hierarchical').glob('*.lock'):
        try:pid=int(lock.read_text())
        except ValueError:raise RuntimeError(f'Unrecognized fit lock: {lock}')
        if pid<=0:raise RuntimeError(f'Invalid fit owner in {lock}')
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise RuntimeError(f'A process still owns {lock.name}; refusing concurrent coordinators')
