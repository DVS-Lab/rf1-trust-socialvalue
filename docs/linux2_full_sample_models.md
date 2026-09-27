# Full-cohort model pilot on Linux2

The integration run recorded in commit `a785eb6` passed: 343 primary participants,
338 sensitivity participants, 304 with two runs, 26,560 valid choices, and zero
unexplained parity fields. The audit retains 128 explicit canonical corrections.
The behavioral age analysis completed without warnings. Ages are the documented
scanner-recorded session-01 ages recovered upstream, not independently verified
research demographics. Scientific stimulus-validity questions remain distinct
from successful structural certification and parity.

The user requested proceeding to full-cohort modeling after this integration
review. `config/full_sample_sampling.json` records that next-phase authorization
and pins the accepted integration-status hash. The old integration configuration
and status still correctly describe the earlier stopping point; they are not
rewritten to authorize a new phase. Sampling controls (seed, four chains,
2,000 warmup, 2,000 retained draws per chain, adapt_delta .99, depth 14) are read
from the frozen cohort configuration.

## What this command runs

Four production-length no-age fits, all using the complete primary cohort:

- H2 base: the learning baseline;
- H5 zero: friend value plus the generic zero-option term;
- H7 zero: separate friend/stranger values plus the generic zero-option term;
- HPreference zero: partner preferences plus the generic zero-option term.

These fits have the same non-centered hierarchical prior/likelihood family as
the accepted N111 analysis. The existing Stan sources are reused unchanged;
the N111 data loader, fitting entry points, caches, and output functions are
not invoked. Misses contribute no likelihood/update, and actual zero choices
produce no feedback update. Empirical Stan data contain only observed outcomes
(with ignored zeros in unused outcome slots), not latent feedback after zero
choices. The generic gamma0 is partially pooled by participant and shared across
partners. Beliefs carry across available runs, consistent with the prior model.

Four fits × four chains use up to 16 cores. Numerical-library thread counts are
limited to one per process to avoid nested oversubscription. The pilot records
wall time and peak Python/child RSS (the latter is a per-child peak, not aggregate
memory) so the next full/train batch can be sized from evidence. The 96-CPU host
can support much broader model-level parallelism; an 80-core budget is reserved
in the phase configuration for subsequent reviewed work. This command does not
automatically launch that larger batch.

Before sampling, the launcher installs CmdStan 2.40.0 if necessary (eight build
cores), compiles the optimized and reference models, and runs 20 target/gradient
and Python-likelihood checks using canonical schedules across all five planned
model families and base/zero variants. No laptop compilation or sampling is
permitted. It checks that all reviewed integration outputs and live canonical
inputs retain their hashes before fitting and after fitting.

Acceptance remains Rhat <1.01, bulk/tail ESS >=400, no divergences, no maximum-
treedepth hits, and BFMI >.3. Failed fits retain diagnostics and console logs;
there is no automatic retry or loosening of thresholds. Accepted parameter
summaries are saved, but broader PPCs, full versus training comparisons, age
hierarchies, sensitivity fits, and recovery are subsequent stages.

## Launch

On Linux2:

```bash
tmux new-session -A -s rf1-trust-full 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_pilot.sh
```

Detach with Ctrl-b then d. Do not rerun the cohort freeze or upstream conversion.
All script failures leave the interactive tmux shell available.

The script captures setup/tests/compilation and final status in
`results/run_logs/*-full-cohort-pilot/`. Each fit has its own live console log,
status, diagnostics, and portable posterior manifest under
`results/full_sample/hierarchical/fits/`. Raw posterior draws remain ignored under
`work/full_sample/hierarchical/fits/`. A process lock prevents two pilot batches
from running simultaneously.

In another Linux2 terminal, inspect progress without interrupting:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
NUMBA_CACHE_DIR="$PWD/work/numba" .venv-linux2/bin/python -m rf1_trust_socialvalue.full_sample_sampling status
```

For a live chain log, for example:

```bash
tail -n 30 /ZPOOL/data/projects/rf1-trust-socialvalue/results/full_sample/hierarchical/fits/Full_H7_zero_noage/console.txt
```

At completion or failure:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record full-cohort model pilot diagnostics and logs"
fi
git push origin main
```

Reissuing the same pilot command reuses complete posterior files after checking
model/data/settings fingerprints and posterior hashes; it does not silently
resample failed diagnostic fits. Partial CSVs without a completed manifest stop
for inspection. Preserve them rather than deleting evidence to force a retry.
No N111 posterior is reused for the new cohort.

After reviewing these four fits, retain the accepted pilot fits and expand to
matched base/zero models across H2/H5/H7/HPreference/H8 and actual run-1→run-2
prediction among eligible two-run participants. The historical 65/35 single-run
fallback must not be used for that full-cohort comparison.
