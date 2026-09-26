# One reviewed HPreference retry on linux1

The original batch `e6eb3b3` finished all twelve fits in about 3 h 18 min; eleven passed. The full-data HPreference extension alone failed the unchanged zero-depth-hit gate (8/16,000 retained transitions at depth 12). It had no divergences and passed Rhat, ESS and BFMI. Its accepted training fit already provides held-out scores. See the [scientific review](../results/n111_wrapup/zero_option_initial_review.md).

This is one explicit, evidence-bound retry to obtain the missing matched PPC and preference-parameter comparison, and an accepted empirical recovery generator. It is not an automatic escalation to reach a passing count. Only maximum depth increases from 12 to 14. The seed, target, priors, four chains, 3,000 warmup, 4,000 retained draws per chain, adapt_delta .99 and metric are unchanged. The depth-14 gate still requires no depth hits, zero divergences, Rhat <1.01, bulk/tail ESS >=400, BFMI >.3.

- New cache: `work/n111_wrapup/fits/N111_HPreference_zero_full_depth14/`.
- Old failed cache, diagnostics and batch status are preserved.
- All eleven accepted fits are reused from their exported results; no other fit runs.
- The plan binds the original batch status, all twelve diagnostics, reused numerical outputs, baseline settings and Stan sources by SHA-256. Unexpected changes stop execution.
- Repeating this command reuses a completed retry cache, including one that failed diagnostics. An incomplete cache stops for review. It never automatically samples a second attempt.
- On success, aggregate comparisons and Figure 3 are rebuilt using the named retry source. On failure, the original partial inference is preserved and the command exits 2.
- No realistic recovery runs automatically. Retention remains a scientific review decision, especially because positive-positive PPC changes are mixed.

## Start on linux1

Start or rejoin tmux:

```bash
tmux new-session -A -s rf1-n111
```

If another analysis is visibly running, let it finish first. Otherwise run this **one-fit command**, not the original twelve-fit launcher:

```bash
bash <<'BASH'
set -euo pipefail
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git pull --ff-only
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/matplotlib" MPLBACKEND=Agg
python -m pytest -q
python -u scripts/22_n111_zero_retry.py --run
BASH
```

This uses four sampling cores, one per chain. More cores cannot run additional independent fits because only this one fit is selected and the current likelihood is not threaded. The old HPreference full fit took about 3 h 15 min while sharing the server with other fits; the new runtime is uncertain because deeper trees can require more work. Detach with Ctrl-b then d. There is no sampling on the laptop.

When the command exits, whether it passes or fails:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue
git add results/n111_wrapup results/run_logs
git commit -m "Record reviewed HPreference depth-14 retry and logs"
git push origin main
```

The retry state is `results/n111_wrapup/zero_option_retry_status.json`; the original `zero_option_status.json` intentionally remains an unmodified historical record. Combined console and exit status are tracked in `results/run_logs/`; bounded retry sampler evidence is under `results/n111_wrapup/sampler_evidence/N111_HPreference_zero_full_depth14/`. Raw posterior CSV files stay ignored under `work/`.

## Local review only

`python scripts/22_n111_zero_retry.py` checks the plan without sampling. `python scripts/23_review_n111_zero_batch.py` regenerates the first-batch review and explicitly partial figure from the eleven accepted fits. The latter remains a historical review after the retry; it cannot replace the later full comparison figure.
