"""One reviewed HPreference full-data retry; no automatic escalation or other fits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import pandas as pd

from . import n111_zero as z
from .accepted_review import diagnostic_pass

ORIGINAL = 'N111_HPreference_zero_full'
RETRY = ORIGINAL+'_depth14'
PLAN = Path('config/n111_zero_retry.json')
STATUS = z.OUT/'zero_option_retry_status.json'


def validate_plan():
    """Require the exact reviewed batch and eleven still-accepted source fits."""
    plan = json.loads(PLAN.read_text())
    if plan['original'] != ORIGINAL or plan['retry'] != RETRY or plan['max_attempts'] != 1:
        raise ValueError('This is only the single reviewed HPreference retry')
    for path, digest in plan['evidence_sha256'].items():
        if z.sha(Path(path)) != digest:
            raise ValueError(f'Reviewed evidence changed: {path}')
    cfg = z.settings()
    retry_cfg = dict(cfg, max_treedepth=14)
    if plan['settings'] != retry_cfg or cfg['max_treedepth'] != 12:
        raise ValueError('Only maximum tree depth may change, from 12 to 14')
    states = json.loads((z.OUT/'zero_option_status.json').read_text())['runs']
    for entry in z.entries():
        name = entry['name']
        d = pd.read_csv(z.TABLE/f'zero_diagnostics_{name}.csv')
        if not d.run.eq(name).all():
            raise ValueError(f'Diagnostic source mismatch: {name}')
        if name == ORIGINAL:
            other = d.copy(); other['max_depth_hits'] = 0; other['passed'] = True
            if states[name]['status'] != 'diagnostic_failed' or not d.max_depth_hits.eq(8).all() or not diagnostic_pass(other):
                raise ValueError('Failure is no longer the reviewed eight depth hits alone')
        elif states[name]['status'] != 'complete' or not diagnostic_pass(d):
            raise ValueError(f'Accepted comparison source failed: {name}')
    return plan, states


def run():
    if platform.system() != 'Linux':
        raise RuntimeError('Run the reviewed retry on linux1, not the laptop')
    from .linux_handoff import coordinator_lock, existing_workers, assert_no_live_fit_locks
    from .parallel_checkpoint import write_json
    root = Path.cwd().resolve()
    with coordinator_lock(root):
        if existing_workers(root):
            raise RuntimeError('Another fitting coordinator is active')
        assert_no_live_fit_locks(root)
        z.gate_stage_a()
        plan, states = validate_plan()
        entry = dict(name=RETRY, model='HPreference', zero=True, training=False, seed_name=ORIGINAL)
        state = dict(status='running', original=ORIGINAL, retry=RETRY, settings=plan['settings'],
                     started_at=datetime.now(timezone.utc).isoformat(),
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     plan_sha256=z.sha(PLAN), reused_runs=[e['name'] for e in z.entries() if e['name'] != ORIGINAL])
        write_json(STATUS, state)
        try:
            # Same fixed seed, target, priors, warmup and draws. Separate cache/name.
            # A completed retry cache is reused; incomplete files or a changed fingerprint stop.
            code = z.fit_entry(entry, parallel_chains=4, cfg=plan['settings'])
            state['status'] = 'complete' if code == 0 else 'diagnostic_failed_no_further_retry'
            state['exit_code'] = code
            if code == 0:
                states[ORIGINAL] = dict(status='complete', source_run=RETRY)
                if not z.collect_results(states, {ORIGINAL: RETRY}):
                    raise RuntimeError('Retry passed but accepted comparison coverage is incomplete')
                state['comparison_status'] = 'complete_retention_review_pending'
            else:
                state['comparison_status'] = 'original_partial_results_preserved'
            return code
        except BaseException as exc:
            state.update(status='error_or_interrupted', error=repr(exc))
            raise
        finally:
            state['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(STATUS, state)
            print(f'Reviewed retry status: {STATUS}; no further fit is launched.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        return run()
    validate_plan()
    print('Reviewed plan valid: one HPreference full-data fit, depth 14, four chains.\nEleven accepted fits reused; original failed cache preserved. --run is Linux-only.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
