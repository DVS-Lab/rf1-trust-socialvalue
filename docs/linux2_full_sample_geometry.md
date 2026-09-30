# Review after the first targeted retry

The Linux2 results in `aaababe` resolve seven of nine retried fits. Eighteen of
20 target fits are now accepted, including nine of ten run-1 training fits.
Published result/diagnostic/score hashes have been checked. The report under
`results/full_sample/hierarchical/accepted_review/` uses accepted fits only and
labels the remaining gaps explicitly; it does not replace the failed historical
batch statuses with a success.

The four available matched model families all improve heldout prediction when
the generic zero-option term is added. H7 zero has the lowest available mean
log loss (.5055), followed by HPreference zero (.5118). Their paired difference
(H7 minus preference) is -.00628, 95% participant-bootstrap interval
[-.00887, -.00364]. This is a small predictive advantage, not proof of a unique
psychological mechanism. The missing H5 base training fit prevents the fifth
matched zero-option comparison. Evaluation uses actual run 2 for 304 paired
participants (12,494 decisions), fixed run-1 parameters, and observed feedback
history online. Bootstrap intervals are not multiplicity adjusted.

## Two unresolved diagnostic failures

| Fit | Max Rhat | Min bulk ESS | Min tail ESS | Divergences | Depth hits | Min BFMI |
|---|---:|---:|---:|---:|---:|---:|
| Train H5 base retry1 | 1.00409 | 2088 | 3485 | 1/32000 | 0 | .708 |
| Full HPreference base retry1 | 1.00762 | 1285 | 2826 | 1/32000 | 0 | .576 |

Both also had one divergence in the original attempt. The retry's global
chain means are similar, and neither divergence is enough to identify a specific
geometry issue from the thinned global traces. Participant-level latent and
natural parameter states were not part of that global excerpt. Keep the
predeclared zero-divergence rule and inspect the existing posterior context
before deciding on another sampler or parameterization change. Do not discard
a chain or truncate the posterior to make the warning disappear.

## Export existing posterior context on Linux2

This reads the two existing raw posterior caches. It does **not** compile models,
launch MCMC, change diagnostic thresholds, or write over previous fit evidence.
It uses the logged wrapper and leaves the interactive shell available on error.
It is primarily a file-reading/numerical-summary job; it does not need 72 cores.

```bash
tmux new-session -A -s rf1-trust-inspect 'bash --noprofile --norc -i'
```

Inside tmux:

```bash
export PATH="/ZPOOL/data/projects/rf1-trust-socialvalue/.venv-linux2/bin:$PATH"
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git pull --ff-only &&
bash scripts/export_full_sample_geometry.sh
```

The exporter checks the frozen canonical inputs, reconstructs both fit targets,
checks raw posterior files against their published hashes, and exports:

- every variable's mean, SD, 5th/50th/95th percentiles and divergent-state rank
  for mu, tau, Cholesky/correlation, latent z and natural participant parameters;
- the saved iteration's sampler diagnostics and five neighboring draws on each
  side in its chain;
- chain-level retained draw counts, divergence/depth counts, BFMI and acceptance;
- provenance and output hashes under `results/full_sample/hierarchical/geometry_review/`.

The saved state is not the internal leapfrog point where integration failed.
Extreme ranks among thousands of variables are expected and cannot alone locate
a causal problem. These exports support a targeted review; no automatic refit
follows them. The full-data preference retry already took 13.1 hours, so another
long run should have a better-supported justification.

After the export, including if it fails:

```bash
cd /ZPOOL/data/projects/rf1-trust-socialvalue &&
git add results/full_sample/hierarchical results/run_logs &&
if ! git diff --cached --quiet; then
    git commit -m "Export remaining divergent-fit diagnostic context"
fi
git push origin main
```
