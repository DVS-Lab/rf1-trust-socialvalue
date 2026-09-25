"""Export diagnostics and reference checks for existing failed fits; never sample."""
from pathlib import Path
import sys
from rf1_trust_socialvalue.run_logging import run_logged

if __name__=='__main__':
    raise SystemExit(run_logged([sys.executable,'-u','-c',
        'from rf1_trust_socialvalue.geometry_audit import main; main()'],Path.cwd()))
