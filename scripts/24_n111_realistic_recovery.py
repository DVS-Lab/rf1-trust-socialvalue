#!/usr/bin/env python3
"""Tracked, finite Linux recovery screen; no posterior sampling."""
from pathlib import Path
import sys
from rf1_trust_socialvalue.run_logging import run_logged

if __name__ == '__main__':
    if '--run' in sys.argv:
        raise SystemExit(run_logged([sys.executable, '-u', '-m', 'rf1_trust_socialvalue.n111_recovery', *sys.argv[1:]], Path.cwd()))
    from rf1_trust_socialvalue.n111_recovery import main
    raise SystemExit(main())
