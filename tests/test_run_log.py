import json

import pytest

from rhmoo.run_log import run_log

pytestmark = pytest.mark.smoke


def test_run_log_writes_expected_fields(tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    with run_log(log_path, config_hash="deadbeef", run_name="unit-test") as state:
        state["molecules_evaluated"] = 42

    record = json.loads(log_path.read_text().strip())
    assert record["config_hash"] == "deadbeef"
    assert record["run_name"] == "unit-test"
    assert record["molecules_evaluated"] == 42
    assert "wall_time_seconds" in record
    assert "started_at" in record and "finished_at" in record


def test_run_log_appends(tmp_path):
    log_path = tmp_path / "run_log.jsonl"
    with run_log(log_path, config_hash="a", run_name="r1"):
        pass
    with run_log(log_path, config_hash="b", run_name="r2"):
        pass

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_run_log_records_arbitrary_provenance_keys(tmp_path):
    """Run provenance is write-once: a finished run cannot be re-interrogated
    for something it never wrote down. brief SS3.3 in particular requires
    recording whether the ensemble-variance check was available, and that key
    used to be silently dropped.
    """
    import json

    from rhmoo.run_log import run_log

    path = tmp_path / "run_log.jsonl"
    with run_log(path, "deadbeef", "demo") as state:
        state["molecules_evaluated"] = 7
        state["ensemble_variance"] = "available"
        state["something_future"] = {"nested": 1}

    record = json.loads(path.read_text().strip())
    assert record["molecules_evaluated"] == 7
    assert record["ensemble_variance"] == "available"
    assert record["something_future"] == {"nested": 1}
    # The standard fields survive alongside the extras.
    for key in ("run_name", "config_hash", "git_sha", "started_at", "finished_at", "wall_time_seconds"):
        assert key in record
