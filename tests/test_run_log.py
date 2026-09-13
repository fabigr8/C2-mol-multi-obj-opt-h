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
