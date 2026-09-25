"""Bounded sampling retries that distinguish precision from geometry failures."""
import json
from pathlib import Path


def retry_settings(cfg, *, passed, divergences, max_depth_hits, min_bfmi):
    if passed:
        return None
    clean = divergences == 0 and max_depth_hits == 0 and min_bfmi > .3
    # More precision is needed; tightening the integrator is not the first remedy.
    if clean and cfg['draws'] < 8000:
        return 'short', dict(cfg, warmup=max(3000, cfg['warmup']), draws=8000)
    if clean and cfg['draws'] == 8000:
        return '8000', dict(cfg, warmup=max(3000, cfg['warmup']), draws=16000)
    if cfg['adapt_delta'] < .99:
        return 'delta', dict(cfg, adapt_delta=.99)
    return None


def archive_fit(folder, saved, suffix):
    """Preserve an attempt and rebase its chain paths without overwriting history."""
    folder = Path(folder).resolve()
    relative = [Path(f).resolve().relative_to(folder) for f in saved['csv_files']]
    archive = folder.with_name(folder.name + suffix)
    index = 2
    while archive.exists():
        archive = folder.with_name(folder.name + suffix + f'_{index}')
        index += 1
    folder.rename(archive)
    saved = dict(saved, csv_files=[str(archive / p) for p in relative])
    (archive / 'manifest.json').write_text(json.dumps(saved, indent=2) + '\n')
    return archive


class DiagnosticFailure(RuntimeError):
    """A completed posterior failed scientific acceptance thresholds."""
