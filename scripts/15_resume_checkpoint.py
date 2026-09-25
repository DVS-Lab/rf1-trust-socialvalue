"""Run the Linux checkpoint with tracked console/error logs by default."""
from pathlib import Path
import sys
from rf1_trust_socialvalue.checkpoint_handoff import main
from rf1_trust_socialvalue.run_logging import run_logged

if __name__=='__main__':
    if '--run' in sys.argv[1:]:
        command=[sys.executable,'-u','-c',
                 'from rf1_trust_socialvalue.checkpoint_handoff import main; main()',*sys.argv[1:]]
        raise SystemExit(run_logged(command,Path.cwd()))
    main()
