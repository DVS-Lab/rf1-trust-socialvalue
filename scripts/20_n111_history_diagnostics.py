#!/usr/bin/env python3
"""Read accepted no-age posterior caches with tracked logs; never sample."""
from pathlib import Path
import sys
from rf1_trust_socialvalue.run_logging import run_logged

if __name__ == '__main__':
    command = [sys.executable, '-u', '-m', 'rf1_trust_socialvalue.n111_diagnostics', *sys.argv[1:]]
    if '--run' in sys.argv:
        raise SystemExit(run_logged(command, Path.cwd()))
    from rf1_trust_socialvalue.n111_diagnostics import main
    main()
