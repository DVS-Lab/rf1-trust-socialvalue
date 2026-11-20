"""Public diagnostics for existing attempts and failed or interrupted runs."""
import importlib.util
import json
from pathlib import Path
import subprocess

script = Path(__file__).resolve().parents[2] / "scripts/share_ddm_preflight.py"
spec = importlib.util.spec_from_file_location("share_ddm_preflight", script)
share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share)


def test_report_links_to_its_attempt_and_keeps_private_content_out(tmp_path):
    attempt = "20261119-120000-fake1234"
    folder = tmp_path / "partner_preflight" / attempt
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps(dict(
        status="execution_failed", failed_phase="sampling", N=343, chains=4,
        cmdstan_path="/private_token/toolchain", private="private_token",
        chain_exit_codes=[0, 1, 0, 0])))
    log = tmp_path / "partner-preflight.abcdefgh"
    log.write_text(f"Private attempt: {folder}\nChain [1] Iteration: 1 / 75 (Warmup)\n"
                   "Chain [1] Iteration: 10 / 75 (Warmup)\nChain [2] Initialization failed.\n"
                   "Chain [2] Gradient evaluated at the initial value is not finite.\n"
                   "RuntimeError: private_token\nPreflight runner exit 1\n")
    report = share.diagnostic_report(log, tmp_path)
    assert report["chain_progress"]["1"] == dict(initialization="passed", last_reported_iteration=10, exit_code=0)
    assert report["chain_progress"]["2"] == dict(initialization="failed", nonfinite_initial_gradient=True, exit_code=1)
    assert report["failed_phase"] == "sampling" and report["runner_exit_code"] == 1
    assert "private_token" not in json.dumps(report)
    log.write_text("ModuleNotFoundError: No module named 'private_token'\nPreflight runner exit 1\n")
    early = share.diagnostic_report(log, tmp_path)
    assert early["chain_progress"]["1"]["initialization"] == "unobserved"
    assert early["reported_issue"] == "python_dependency_missing"
    assert "private_token" not in json.dumps(early)


def test_git_push_is_scoped_and_repeatable(tmp_path, monkeypatch):
    checkout = tmp_path / "retry"
    checkout.mkdir()
    monkeypatch.setattr(share, "ROOT", checkout)
    remote = tmp_path / "remote.git"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()

    git("init", "--bare", str(remote))
    git("init", "-b", share.BRANCH)
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.org")
    git("config", "commit.gpgsign", "false")
    git("commit", "--allow-empty", "-m", "base")
    git("remote", "add", "origin", str(remote))
    git("push", "origin", share.BRANCH)
    (checkout / "unrelated").write_text("stay staged")
    git("add", "unrelated")
    path = share.REPORTS / "attempt.json"
    (checkout / path).parent.mkdir(parents=True)
    (checkout / path).write_text("{}\n")
    share.push_report(path)
    head = git("rev-parse", "HEAD")
    share.push_report(path)
    assert git("rev-parse", "HEAD") == head == git("--git-dir", str(remote), "rev-parse", share.BRANCH)
    assert git("ls-tree", "-r", "--name-only", "HEAD") == str(path)
    assert git("diff", "--cached", "--name-only") == "unrelated"
