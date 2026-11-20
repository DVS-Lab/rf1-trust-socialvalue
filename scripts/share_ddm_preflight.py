"""Export one Linux2 partner preflight's diagnostics, optionally pushing them to Git."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/ddm_hierarchical"
REPORTS = Path("results/ddm_hierarchical/partner_preflight")
BRANCH = "codex/ddm-partner-retry"
# The private status writer controls these fields. Paths, exception text, raw
# console messages and participant-level files are outside the export.
FIELDS = """status failed_phase error_type synthetic N T input_hash model_hash chains
threads_per_chain minutes warmup draws seed gradient_states_checked sampling_seconds
completed_draws_per_chain divergences_by_chain depth_limit_by_chain chain_exit_codes
gradient_seconds_including_startup""".split()


def diagnostic_report(log, work):
    text = log.read_text(errors="replace")
    match = re.search(r"^Private attempt: .*/(\d{8}-\d{6}-[a-z0-9_]+)\s*$", text, re.M)
    if match:
        attempt = match[1]
        status_file = work / "partner_preflight" / attempt / "status.json"
        try:
            status = json.loads(status_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            # An interrupted writer can leave an empty or partial status file.
            status = {}
    else:
        # Python can exit before creating an attempt (e.g. import or CLI error).
        attempt = log.name
        if not re.fullmatch(r"partner-preflight\.[A-Za-z0-9]{8}", attempt):
            raise ValueError("Expected a partner preflight launcher log or a Private attempt line")
        status = {}
    report = {key: status[key] for key in FIELDS if key in status}
    report.setdefault("status", "no_recorded_status")
    report["attempt"] = attempt
    for label, pattern in (
        ("runner_exit_code", r"^Preflight runner exit (-?\d+)$"),
        ("code_revision", r"^Preflight code revision ([0-9a-f]{40,64})$"),
        ("cmdstan_version", r"^Using CmdStan (\d+\.\d+\.\d+) at "),
    ):
        found = re.search(pattern, text, re.M)
        if found:
            report[label] = int(found[1]) if label == "runner_exit_code" else found[1]
    if report.get("runner_exit_code") and report["status"] in {"started", "no_recorded_status"}:
        report["status"] = "exited_without_final_status"

    chains = {}
    for index in range(1, status.get("chains", 4) + 1):
        prefix = rf"Chain \[{index}\] "
        iterations = [int(n) for n in re.findall(prefix + r"Iteration:\s*(\d+)\s*/", text)]
        rejected = bool(re.search(prefix + r"(?:User-specified initialization failed|Initialization failed)", text))
        chain = {"initialization": "unobserved"}
        if iterations:
            chain.update(initialization="passed", last_reported_iteration=max(iterations))
        elif rejected:
            chain["initialization"] = "failed"
        if re.search(prefix + r"Gradient evaluated at the initial value is not finite", text):
            chain["nonfinite_initial_gradient"] = True
        error = re.search(prefix + r"error: code '(-?\d+)'", text)
        if error:
            chain["exit_code"] = int(error[1])
        elif index <= len(status.get("chain_exit_codes", [])):
            chain["exit_code"] = status["chain_exit_codes"][index - 1]
        chains[str(index)] = chain
    report["chain_progress"] = chains

    # Keep numerical details from the synthetic reference check. Other tracebacks
    # contribute a type and source location, never arbitrary exception strings.
    number = r"[-+.\deE]+"
    probe = re.search(rf"^ArithmeticError: ((?:Gradient mismatch|Invalid value or gradient) at "
                      rf"a={number}, dt={number}, (?:index|z)={number})$", text, re.M)
    if probe:
        report["reference_check_failure"] = probe[1]
    frames = re.findall(r'File "[^"\n]*/(partner_preflight|hierarchical)\.py", line (\d+)', text)
    if frames:
        module, line = frames[-1]
        report["last_project_frame"] = f"{module}.py:{line}"
    errors = re.findall(r"^([A-Za-z]+(?:Error|Exception)):", text, re.M)
    if errors:
        report.setdefault("error_type", errors[-1])
    for phrase, reason in (
        ("processes timed out", "chain_timeout"), ("process timed out", "chain_timeout"),
        ("Could not locate CmdStan", "cmdstan_not_found"), ("Set CMDSTAN to its installation", "cmdstan_not_found"),
        ("Permission denied", "permission_denied"), ("No space left on device", "out_of_disk_space"),
        ("No module named", "python_dependency_missing"), ("Killed signal terminated program", "compiler_killed"),
    ):
        if phrase in text:
            report["reported_issue"] = reason
            break
    return report


def push_report(path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("branch", "--show-current") != BRANCH:
        raise SystemExit(f"Report saved. To push with this helper, use the {BRANCH} checkout.")
    git("fetch", "origin", BRANCH)
    # A push also publishes the tip's ancestors. Avoid sending unrelated local
    # commits when sharing one report. Unrelated staged files can stay staged.
    pending = git("log", "--format=%h", "FETCH_HEAD..HEAD", "--", ".", f":(exclude){REPORTS}/*.json")
    if pending:
        raise SystemExit("Report saved. This checkout has unpublished commits outside the preflight reports. "
                         "Review those commits before pushing.")
    git("add", "--", str(path))
    if git("diff", "--cached", "--name-only", "--", str(path)):
        git("commit", "--only", "-m", f"Record DDM preflight {path.stem}", "--", str(path))
    git("push", "origin", f"HEAD:refs/heads/{BRANCH}")
    print(f"Shared {path} on {BRANCH} at {git('rev-parse', '--short', 'HEAD')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="Private launcher log (defaults to most recently written)")
    parser.add_argument("--push", action="store_true", help="Commit and push the generated report on the retry branch")
    args = parser.parse_args()
    log = args.log or max((WORK / "logs").glob("partner-preflight.*"),
                          key=lambda p: p.stat().st_mtime_ns, default=None)
    if log is None:
        parser.error("No partner preflight launcher log found in this checkout")
    report = diagnostic_report(log, WORK)
    path = REPORTS / (report["attempt"] + ".json")
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=dest.parent, delete=False) as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    os.replace(out.name, dest)
    print(f"Saved {path}. Chain 1 initialization: {report['chain_progress']['1']['initialization']}", flush=True)
    if args.push:
        push_report(path)


if __name__ == "__main__":
    main()
