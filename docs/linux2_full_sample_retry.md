# Linux2 targeted batch retry

Review `results/full_sample/hierarchical/batch_diagnostic_review.md` for the
original diagnostics. The `f08e77f` results contain seven accepted new fits and
nine diagnostic failures; with four accepted pilots, 11/20 targets are complete.
All raw samplers completed, so this is not a Python crash or missing-data issue.

Run exactly the reviewed nine retries. Each has eight parallel chains, 4,000
warmup, and 4,000 retained draws per chain except Full/Train H2 zero, which have
8,000 retained draws per chain. The three fits with one divergence each use
adapt_delta .995; the other six retain .99. Depth14, diag_e, seeds, priors,
likelihood, data and diagnostic thresholds remain unchanged. There are at most
nine concurrent fits, hence **72 sampling cores** on a host with 96 available.
Startup checks initially use fewer cores; utilization falls as fits finish.

The longest original fit took about six hours. Budget an overnight window for
its retry; more warmup and stricter adaptation may make it slower despite the
additional chains. This is one bounded attempt, not an automatic retry loop.
Persistent failures need chain/geometry inspection. Increasing adaptation target
for divergences and using longer chains for slow mixing follows [Stan guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html);
neither guarantees convergence.

The config pins the original batch, every original status, diagnostic table and
posterior manifest. Each retry reconstructs its original target fingerprint
before sampling so only reviewed sampler changes can differ. All eleven accepted
fits are verified and reused, including four accepted training fits. Original
attempts and logs remain untouched. No original and retry posterior draws are
combined. No cohort refresh or upstream conversion is needed.

## Launch on Linux2

```bash
tmux new-session -A -s rf1-trust-retry 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/run_full_sample_batch_retry.sh
```

Detach with Ctrl-b then d. The interactive shell is not given fail-fast options.
Do not change analysis code/config or rerun the original batch while this runs.

Check from another terminal:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
NUMBA_CACHE_DIR="$PWD/work/numba" .venv-linux2/bin/python \
  -m rf1_trust_socialvalue.full_sample_batch status \
  --config config/full_sample_batch_retry.json
```

## Outputs and handoff

Per-fit outputs and logs use new names ending `_retry1` under
`results/full_sample/hierarchical/fits/`. Batch status and implementation checks
are under `results/full_sample/hierarchical/batch_retry/`. If all retries pass,
that directory also contains heldout comparison tables and plots using the six
accepted replacement training fits plus four original accepted training fits.
Scores retain actual run-1 fitting/run-2 evaluation and equal-participant paired
bootstrap comparisons. Full-data fits still use all 343 participants; training
fits use the 304 eligible paired participants.

If a retry fails diagnostics, its full posterior remains in ignored `work/` and
its tracked output includes chain means/SDs over all draws and a trace excerpt
of up to 200 evenly spaced retained draws per chain plus all divergent draws.
These exports help review failures without transferring raw posterior CSVs.

After completion **or failure**, publish outputs and logs:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Record targeted full-cohort retry diagnostics and predictions"
fi
git push origin main
```

Reissuing the retry command validates completed caches, without generating
additional attempts. A partial run without a completed manifest stops for
inspection. The original batch status is historical and will continue to report
a failure even if its replacements pass; use the separate retry status.
