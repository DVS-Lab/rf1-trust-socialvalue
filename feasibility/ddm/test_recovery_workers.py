"""Exercise the spawn-based recovery scheduler with tiny aggregate fixtures."""
import json
import pandas as pd
import hierarchical as h


def fake_fit(job):
    name, value = job
    if value == 2:
        raise RuntimeError("synthetic chain failure")
    return (dict(name=name, passed=True),
            pd.DataFrame([dict(fit=name, estimate=value)]), [dict(fit=name, truth=value)])


def test_serial_and_parallel_keep_failures_and_write_identical_summaries(tmp_path, monkeypatch):
    jobs = [("one", 1), ("two", 2), ("three", 3)]
    outputs = []
    for workers in (1, 2):
        out = tmp_path / str(workers)
        monkeypatch.setattr(h, "PUBLIC", out)
        assert not h.run_recovery_jobs(jobs, workers, fake_fit, ())
        status = json.loads((out / "recovery_status.json").read_text())
        assert status["status"] == "incomplete" and status["workers"] == workers
        assert [s["name"] for s in status["fits"]] == ["one", "two", "three"]
        assert status["fits"][1]["error_type"] == "RuntimeError"
        outputs.append([(out / name).read_text() for name in ("recovery_summary.tsv", "recovery_population.tsv")])
    assert outputs[0] == outputs[1]
