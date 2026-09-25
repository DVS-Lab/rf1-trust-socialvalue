import json
import sys
from rf1_trust_socialvalue.run_logging import run_logged,collect_artifacts


def test_nonzero_exit_captures_stdout_stderr_and_status(tmp_path):
    code=run_logged([sys.executable,'-u','-c',"import sys; print('progress'); print('failure',file=sys.stderr); sys.exit(7)"],tmp_path)
    assert code==7
    folder=next((tmp_path/'results/run_logs').iterdir())
    assert 'progress' in (folder/'console.txt').read_text()
    assert 'failure' in (folder/'console.txt').read_text()
    record=json.loads((folder/'run.json').read_text())
    assert record['exit_code']==7 and record['status']=='failed' and 'finished_at' in record


def test_artifacts_exclude_raw_draws_and_bound_console_size(tmp_path):
    folder=tmp_path/'work/hierarchical/fit';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps({'settings':{'draws':2000},'csv_files':[str(folder/'draws.csv')]}))
    (folder/'draws.csv').write_text('raw posterior must not be exported')
    (folder/'chain.txt').write_text('x'*300000+'last error')
    (folder/'diagnose.txt').write_text('depth warning')
    dest=tmp_path/'results/run_logs/test';collect_artifacts(tmp_path,dest)
    assert not list(dest.rglob('*.csv'))
    text=(dest/'sampler/fit/chain.txt').read_text()
    assert text.startswith('[Last ') and text.endswith('last error') and len(text)<251000
    assert (dest/'sampler/fit/diagnose.txt').read_text()=='depth warning'
