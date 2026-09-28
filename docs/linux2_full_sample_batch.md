# Full-cohort matched-model batch on Linux2

The four representative full-data fits are accepted. The HPreference zero-option
retry pushed in `182d8c7` passed at 4,000 retained draws per chain: maximum Rhat
1.00498, minimum bulk ESS 702.858, minimum tail ESS 1377.5, no divergences or
maximum-depth hits, and minimum BFMI .627. Sampling took 4.07 hours. The retry
replaces the original failed HPreference fit; the two posteriors are not pooled.
Original pilot/retry records remain intact.

## Work and resources

This batch reuses the accepted H2 base, H5 zero, H7 zero, and HPreference zero
fits and runs six remaining full-cohort fits plus ten training fits. The matched
family is H2, H5, H7, HPreference and H8, each with and without the generic
zero-option term. All are hierarchical and have no age covariates in this phase.

There are up to **16 concurrent fits × four chains = 64 active sampling cores**,
with an 80-core ceiling and CPU-affinity checks. Single-threaded numerical
libraries avoid nested oversubscription. Utilization falls as fits finish; the
slowest remaining fit can again use only four cores. Increasing chain counts
mid-run will not help. Expect several hours; the measured 4.1-hour retry is an
anchor, not a guarantee for other targets under concurrent load.

HPreference-family new fits use 4,000 retained draws per chain based on the pilot
ESS evidence. Other fits retain 2,000. All use 2,000 warmup, four chains, diag_e,
adapt_delta .99, max_treedepth 14, and unchanged prior scales. The strict
acceptance thresholds and prohibition on automatic retries remain in place.
A failed fit does not interrupt the other independent fits.

Full-data fits use 343 participants and 26,560 valid choices. Training requires
valid choices in actual runs 1 and 2. The exported cohort manifest predicts 304
eligible participants, 12,498 training choices and 12,494 heldout choices; the
launcher verifies live frozen inputs and writes the actual paired cohort. There
is no within-run 65/35 fallback. Run-2 decisions never enter the fitting target.

Heldout scores use the unchanged run-1 posterior. Beliefs carry from run 1 and
update sequentially from observed feedback during run 2. This is online
conditional prediction, not a joint marginal likelihood for the entire second
run. Each trial probability is integrated over all retained posterior draws;
parameters are not updated using run-2 decisions. Misses do not score or update,
and zero choices do not reveal scheduled outcomes. Summary comparisons give
each participant equal weight and use paired participant-bootstrap intervals.

Two sets of 20 optimized/reference Stan target/gradient and Python-likelihood
checks run before sampling, one on full data and one on the training subset.
Tests and checks run before the 64-core sampling stage, so initial usage is low.
Accepted pilot summaries are hash checked and their target fingerprints are
reconstructed from the current frozen data; accepted fits are never resampled.

## Launch

The prior retry is complete. On Linux2:

```bash
tmux new-session -A -s rf1-trust-batch 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_batch.sh
```

Detach with Ctrl-b then d. The wrapper's shell options are confined to the child
script; failure leaves the interactive shell available. Do not rerun upstream
conversion or cohort freezing. Do not change analysis code/config while running.
The parent holds the same lock as pilot/retry launchers to prevent overlap.

From a separate terminal:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
NUMBA_CACHE_DIR="$PWD/work/numba" .venv-linux2/bin/python -m rf1_trust_socialvalue.full_sample_batch status
```

Per-fit progress is in `results/full_sample/hierarchical/fits/<name>/console.txt`.
Batch status and the paired cohort are under `results/full_sample/hierarchical/batch/`.
The run wrapper records its console and exit status under `results/run_logs/`.
Raw posterior CSVs and compilation products stay in ignored `work/`.

After completion or failure:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record full-cohort matched fits and run-2 predictions"
fi
git push origin main
```

Re-running the batch verifies and reuses completed posterior caches; it does not
resample diagnostic failures. Partial draws without a completed manifest stop
for inspection. The batch checks all diagnostic thresholds before producing
accepted parameter summaries or predictive scores. Broad comparison tables and
plots are generated only when all scheduled fits have passed.

## Results and next interpretation

Successful completion writes `heldout_comparison.tsv`, `zero_option_comparison.tsv`,
and `zero_option_heldout.png`/`.pdf` in the batch directory, with score/output
hashes. Negative zero-minus-base log-loss differences favor the generic
zero-option term. Confidence intervals describe participant sampling variation
and are not a multiple-comparison-corrected mechanism selection rule.

Review these results before choosing targeted posterior predictive checks,
age hierarchies, sensitivity fits and recovery. Predictive improvement alone
cannot establish that value and preference mechanisms are distinguishable.
The preserved N111 endpoint and its limitations are unchanged.
