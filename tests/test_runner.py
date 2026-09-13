from pathlib import Path

import pytest

from rhmoo.runner import run_experiment

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

pytestmark = pytest.mark.smoke


def test_run_experiment_writes_artifacts(tmp_path):
    out_dir = run_experiment(CONFIG_DIR / "smoke.yaml", results_root=tmp_path)
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "run_log.jsonl").exists()
