"""Time-limited four-chain partner-model preflight. All outputs are private."""
import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from hierarchical import cmdstan, compile_model, config, execution_failure, initial_values, model_data, synthetic_fixture
from hier_data import WORK, digest, save, sha


def partner_input(root):
    original = root / "work/ddm_hierarchical"
    paths = sorted((original / "fits/partner").glob("*/data.json"))
    if len(paths) != 1:
        raise ValueError("Expected one archived partner input. Use --data with its full path.")
    return archived_data(paths[0])


def archived_data(path):
    target = json.loads(path.with_name("target.json").read_text())
    data = json.loads(path.read_text())
    if path.parent.parent.name != "partner" or digest(data) != target["data_sha256"]:
        raise ValueError("Expected a hash-matched input from the archived primary partner fit.")
    if (data["Z"], data["H"], data["prior_only"], data["likelihood_only"]) != (3, 0, 0, 0):
        raise ValueError("Unexpected primary partner specification.")
    return data


def check_gradients(model, c):
    # Probe both response boundaries and both numerical crossovers, including
    # observations just above/below t0 that use the uniform contaminant.
    from hierarchical import reference_loglik
    trials = synthetic_fixture(2).groupby("participant_id").head(1)
    data, _ = model_data(trials, c, "partner")
    data["likelihood_only"] = 1
    p = initial_values(data, 412)
    p["raw"] = np.zeros((9, 2)).tolist()
    t0 = data["hi"] * expit(p["mu"][1])
    checked = 0
    for a, dt, z in [(1.25, -1e-5, .5), (1.25, 1e-5, .5),
                     (1.25, 1e-5, .998), (1., .00999, .27), (1., .01001, .27),
                     (.2, .39996, .27), (.2, .40004, .27), (.02, 1., .5)]:
        p["mu"][0] = np.log(a)
        p["mu"][6:] = [np.log(z/(1-z))] * 3
        data["rt"] = [t0 + dt] * 2
        data["y"] = [0, 1]
        result = model.log_prob(params=p, data=data, jacobian=False, sig_figs=14).iloc[0].to_numpy()
        reference = reference_loglik(data, p)
        if not np.isfinite(result).all() or abs(result[0] - reference) > 1e-5:
            raise ArithmeticError(f"Invalid value or gradient at a={a}, dt={dt}, z={z}")
        for index in (0, 1, 2, 6):
            h = 1e-9 if abs(dt) < 1e-4 else 1e-6
            plus, minus = dict(p, mu=p["mu"].copy()), dict(p, mu=p["mu"].copy())
            plus["mu"][index] += h
            minus["mu"][index] -= h
            slope = (reference_loglik(data, plus) - reference_loglik(data, minus)) / (2*h)
            if not np.isclose(result[1+index], slope, atol=2e-4, rtol=2e-4):
                raise ArithmeticError(f"Gradient mismatch at a={a}, dt={dt}, index={index}")
        checked += 1
    return checked


def time_gradients(model, data, seed, cap, folder):
    """Compare the exact same log density and gradients across thread counts."""
    save(folder/"data.json", data)
    save(folder/"initial.json", initial_values(data, seed))
    timings = {}
    reference = None
    for threads in sorted({1, cap, *(n for n in (2, 4, 8, 16) if n < cap)}):
        samples = []
        for repeat in range(3):
            csv_path = folder/f"gradient-{threads}-{repeat}.csv"
            cmd = [model.exe_file, "log_prob", "constrained_params="+str(folder/"initial.json"),
                   "jacobian=true", "data", "file="+str(folder/"data.json"), "output",
                   "file="+str(csv_path), "sig_figs=14", f"num_threads={threads}"]
            start = time.monotonic()
            with csv_path.with_suffix(".log").open("w") as log:
                subprocess.run(cmd, env=dict(os.environ, STAN_NUM_THREADS=str(threads)),
                               stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
            samples.append(time.monotonic()-start)
            with csv_path.open() as f:
                rows = (line for line in f if line.strip() and not line.startswith("#"))
                next(rows)
                values = np.fromstring(next(rows), sep=",")
            if values.size < 2 or not np.isfinite(values).all():
                raise ArithmeticError("Nonfinite initial log density or gradient")
            if reference is None:
                reference = values
            elif not np.allclose(values, reference, rtol=1e-8, atol=1e-8):
                raise ArithmeticError("Log density or gradients changed across thread counts")
        timings[str(threads)] = round(float(np.median(samples)), 4)
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--original-root", type=Path)
    select.add_argument("--data", type=Path, help="Exact archived primary model data.json")
    select.add_argument("--synthetic", action="store_true")
    parser.add_argument("--threads-per-chain", type=int, default=4)
    parser.add_argument("--benchmark-only", action="store_true", help="Time identical gradient evaluations, including process and I/O overhead")
    parser.add_argument("--minutes", type=int, default=30, help="Sampling timeout per chain, all four start together")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--draws", type=int, default=25)
    args = parser.parse_args()
    if min(args.threads_per_chain, args.minutes, args.warmup, args.draws) < 1:
        parser.error("Sampling options must be positive")
    budget = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if 4*args.threads_per_chain > budget:
        parser.error(f"Requested more than the {budget} visible CPUs")
    c = config()
    if args.synthetic:
        data, _ = model_data(synthetic_fixture(12), c, "partner")
    else:
        if platform.system() != "Linux":
            parser.error("Use Linux2 for archived participant data")
        if args.original_root:
            os.environ["DDM_ORIGINAL_ROOT"] = str(args.original_root.resolve())
        data = archived_data(args.data) if args.data else partner_input(args.original_root)
    parent = WORK / "partner_preflight"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=parent))
    status = dict(purpose="computational preflight", synthetic=args.synthetic, N=data["N"], T=data["T"],
                  input_hash=digest(data), model_hash=sha(Path(__file__).with_name("hierarchical.stan")),
                  chains=4, threads_per_chain=args.threads_per_chain, minutes=args.minutes,
                  warmup=args.warmup, draws=args.draws, seed=c["seed"]+1)
    print(f"Private attempt: {folder}", flush=True)
    save(folder/"status.json", dict(status, status="started"))
    phase = "locate_cmdstan"
    try:
        backend = cmdstan(c)
        status["cmdstan_path"] = backend.cmdstan_path()
        print(f"Using CmdStan {c['cmdstan_version']} at {status['cmdstan_path']}", flush=True)
        phase = "compile_model"
        model = compile_model(c, backend=backend)
        phase = "check_gradients"
        status["gradient_states_checked"] = check_gradients(model, c)
        if args.benchmark_only:
            phase = "benchmark"
            status["gradient_seconds_including_startup"] = time_gradients(
                model, data, status["seed"], args.threads_per_chain, folder)
            status["status"] = "benchmark_complete"
            return
        print("Gradient checks passed. Launching four primary-model chains.", flush=True)
        phase = "sampling"
        start = time.monotonic()
        fit = model.sample(data=data, seed=status["seed"], chains=4, parallel_chains=4,
                           force_one_process_per_chain=True, threads_per_chain=args.threads_per_chain,
                           iter_warmup=args.warmup, iter_sampling=args.draws, timeout=args.minutes*60,
                           adapt_delta=c["adapt_delta"], max_treedepth=c["max_treedepth"],
                           inits=[initial_values(data, status["seed"]+i) for i in range(4)],
                           output_dir=str(folder), show_progress=False, show_console=True, refresh=5)
        status["sampling_seconds"] = time.monotonic()-start
        phase = "summarize_draws"
        method = fit.method_variables()
        status["divergences_by_chain"] = method["divergent__"].sum(axis=0).astype(int).tolist()
        status["depth_limit_by_chain"] = (method["treedepth__"] >= c["max_treedepth"]).sum(axis=0).astype(int).tolist()
        status["completed_draws_per_chain"] = fit.num_draws_sampling
        warnings = any(status["divergences_by_chain"]) or any(status["depth_limit_by_chain"])
        status["status"] = "completed_with_sampler_warnings" if warnings else "completed_short_test"
    except Exception as error:
        status.update(execution_failure("partner", error))
        status["failed_phase"] = phase
        (folder/"exception.txt").write_text(str(error))
        raise
    finally:
        save(folder/"status.json", status)
        print(json.dumps(status, indent=2), flush=True)
    if warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
