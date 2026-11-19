# Testing the Linux2 partner fit

The November run's primary partner model exited during sampling. CmdStanPy 1.3
collects Stan messages from all chains when a process returns a nonzero code.
The launcher log also records the individual process codes. For the original
run, inspect the lines containing `error: code`, `retcodes=`, and the last
reported iteration for each chain. A later error about partial CSV files means
the runner found the existing attempt when it tried to obtain the primary
posterior for recovery.

In the archived model, a synthetic response 10 microseconds above nondecision
time produced a finite mixture log likelihood and 23 nonfinite gradient
components. The retry uses log-scale image and spectral tails for the Wiener
density so differentiation stays finite there. It also skips the empty history
row operation in the primary model. Proposals outside the existing CDF
numerical domain now receive zero target density before the CDF is called.
This preserves the domain restriction and
avoids throwing that exception from a worker thread. Priors, response window,
cohort and four-chain acceptance criteria carry forward. The
correction needs a test on the original Linux2 input before a long run.

## Local synthetic results, November 19

On identical inputs and parameters, the original and corrected executables
returned the same log likelihood to 11 decimal places (−156.55594152648).
The count of nonfinite gradient components fell from 23 to zero. The corrected
executable also passed reference likelihood and finite-difference gradient
checks at eight boundary and crossover settings. A four-thread evaluation of
an out-of-domain proposal returned a log target of negative infinity.

On 12 synthetic participants and 423 trials, all four corrected chains
completed 50 warmup iterations and 25 saved draws per chain in 87 seconds
of sampling wall time, using two threads per chain. There were zero divergences
and zero maximum-depth hits among the 100 saved draws. These short chains test
execution. Convergence and friend–stranger parameter recovery require the
longer primary fit and recovery simulations, respectively.

On a separate synthetic input with 343 participants and 24,678 trials, median
time per gradient command was 0.199, 0.137 and 0.142 seconds with one, two and
four threads. The gradients agreed at every thread count. These local timings
include process startup and file I/O. Measure thread scaling again on Linux2.

## First run on Linux2

Use the existing environment and the hash-verified partner input saved by the
November job. Run from a new checkout so its tests and attempts have a separate
output directory. Compilation runs once. The default exercise uses all four
chains, 50 warmup iterations and 25 retained draws with a 30-minute per-chain
sampling timeout. Each invocation writes a unique private directory beneath
`work/ddm_hierarchical/partner_preflight/`.

```bash
cd /ZPOOL/data/projects
git clone --single-branch --branch codex/ddm-partner-retry \
  https://github.com/DVS-Lab/rf1-trust-socialvalue.git rf1-trust-ddm-retry
cd rf1-trust-ddm-retry
original=/ZPOOL/data/projects/rf1-trust-socialvalue
bash scripts/run_ddm_partner_preflight.sh "$original" --threads-per-chain 4
```

The launcher prints the private log path and a compact JSON status. Share the
status and per-chain exit codes. Exit 2 means the test completed with
divergences or depth-limit hits. A timeout preserves partial outputs. Compare
progress across chains before deciding to extend the time limit. Clean short
chains support a longer primary-model diagnostic run. Individual recovery needs
the later simulations. If the old environment used a custom CmdStan installation
or Python interpreter, set `CMDSTAN` or `DDM_PYTHON` as in the original job.

## Choosing processor counts

Four chains at four threads per chain use up to 16 CPUs. Increasing threads
parallelizes the likelihood across participants within each chain. Every chain
still advances through its own warmup and sampling iterations in sequence.
Within a larger CPU allocation, use this quick comparison on the archived input.
For example, a 32-CPU allocation supports testing up to eight threads per chain.

```bash
bash scripts/run_ddm_partner_preflight.sh "$original" --threads-per-chain 8 --benchmark-only
```

The benchmark checks equal log densities and gradients at the same initial
parameters across thread counts. Its median time includes CmdStan startup and
JSON I/O. Measure elapsed time and sampling diagnostics in the actual preflight
before extrapolating its speedup. The test checks visible CPU affinity. Size
jobs to the allocation on the shared server, accounting for memory, disk and
other users.

## Longer runs after inspecting the preflight

For a longer primary fit, use the new checkout, its prepared canonical data,
and the existing Linux2 environment. The scientific sampler settings are defined
in `hierarchical.json`. The production `primary` stage fits just the partner
model and runs predictive checks if its diagnostics pass. For a different
thread count, save a private copy of the config and pass the same `--config`
path to preparation and every subsequent stage.

```bash
export DDM_PYTHON="$original/.venv-linux2/bin/python"
export CMDSTAN="$original/work/cmdstan/cmdstan-2.40.0"
bash scripts/run_ddm_hierarchical.sh prepare
bash scripts/run_ddm_hierarchical.sh primary
```

Once the scientific primary fit is accepted, its independent simulated datasets
can run concurrently. For a 32-CPU allocation with enough RAM for two fits and
their summaries, the production command is:

```bash
bash scripts/run_ddm_hierarchical.sh recovery --recovery-workers 2
```

It schedules two distinct datasets at a time, each with four chains and four
threads per chain. The default is one dataset at a time. A single coordinator
owns the aggregate status and tables. Seeds and generated population effects are
assigned in the same order for each worker count. Each completed fit has its own
cache. The launcher continues to allow manual publication of aggregate results.
