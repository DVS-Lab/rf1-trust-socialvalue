# Resume Trust computational modeling from the frozen cohort

This stage continues the computational analysis while the upstream imaging
repair proceeds independently. It reads the authenticated saved canonical trial
table and existing posterior CSVs. It does not rewrite BIDS, regenerate the
cohort, compile Stan or start new MCMC. New-fitting entry points retain their
strict live-source checks.

## Scientific work

Posterior predictive checks run for all **nine accepted full-data models** at
N=343. The unaccepted Full HPreference base fit is excluded. Each model uses 200
posterior replicates (the frozen analysis configuration's count), with posterior
draws selected evenly across the stored chain-concatenated samples. One selected
joint draw is used across all participants in a replicate, preserving posterior
dependence; choice randomness is independent across participants. Base/zero
counterparts use matched random numbers to reduce simulation noise.

Two predictions are kept distinct:

- **Conditional history:** expected choices using actual prior observed feedback.
  Neither missed decisions nor actual zero investments reveal latent outcomes.
- **Generative history:** simulated choices determine which scheduled outcomes
  are revealed and how beliefs update. A simulated positive investment can expose
  the scheduled outcome even when the participant's actual choice was zero.

Both hold offers, scheduled outcomes and observed missingness fixed. Beliefs
carry across available runs, matching the fitted likelihood. These full-data
checks assess model adequacy; they are not heldout prediction or model recovery.
The already computed run-1 to run-2 scores remain the predictive comparison.

Outputs cover:

- high-choice rates and investment amounts by partner, partner × zero-option
  availability, offer, run and early/middle/late thirds of each presented run;
- paired friend–computer, friend–stranger and stranger–computer contrasts;
- generative between-participant SDs and all-low/all-high choice fractions;
- participant-level partner summaries and a nine-panel observed-versus-replicated
  partner/zero-option plot, with 95% predictive intervals.

Cell summaries weight each contributing participant equally. Partner contrasts
use participants represented in both cells. Conditional intervals describe
posterior uncertainty in expected behavior; generative intervals additionally
include replicated-choice variability. With 200 replicates, tail summaries are
coarse checks, not finely calibrated rejection rules. No automatic model
acceptance/exclusion follows a predictive residual.

Four independent workers load and process the nine large posterior caches.
This postprocessing stage uses fewer cores than MCMC and leaves capacity for the
ongoing imaging repair. It should not be confused with another 72-chain run.

The launcher then exports the two unresolved fits' divergent-iteration context
from existing draws. Each stage retains its own status/logs, and the geometry
export still runs if the predictive-check stage fails. See
`linux2_full_sample_geometry.md` for that export's interpretation.

## Linux2 commands

Start or attach a separate tmux session:

```bash
tmux new-session -A -s rf1-trust-postfit 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_postfit.sh
```

Detach with Ctrl-b then d. The upstream repair session can continue. Live-source
changes are recorded separately; any changed saved trial table, historical
integration output, model target or raw posterior still stops processing. The
script's shell options do not change the interactive tmux shell.

After completion or failure:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record Trust predictive checks and remaining model diagnostics"
fi
git push origin main
```

Predictive outputs live under `results/full_sample/hierarchical/ppc/`, including
per-fit `console.txt`/`status.json`, aggregate and participant tables, and the
combined PNG/PDF figure. Divergence exports remain under `geometry_review/`.
The wrapper records the combined exit status under `results/run_logs/`.
Rerunning recomputes derived predictive checks from the same verified posterior;
it never resamples or replaces fitted model draws.

## Subsequent model decisions

Use these checks to identify systematic residual behavior before age hierarchies,
sensitivity runs or targeted recovery. The accepted predictive ranking currently
favors H7 + zero over HPreference + zero by a small amount, but it does not prove
mechanism identification. The two one-divergence controls remain unaccepted until
reviewed; their absence need not block checking the nine accepted full models.
