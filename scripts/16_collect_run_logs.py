"""Export existing Linux console/diagnostic evidence without starting any fits."""
from datetime import datetime,timezone
from pathlib import Path
from rf1_trust_socialvalue.run_logging import collect_artifacts

root=Path.cwd()
destination=root/'results/run_logs'/('collected-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
collect_artifacts(root,destination)
print(f'Collected existing sampler evidence in {destination}; no fits started.')
