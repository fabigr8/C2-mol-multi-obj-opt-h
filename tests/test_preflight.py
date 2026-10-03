"""brief SS4.5 requires calibration before any arm A molecule exists, and SS9
requires pre-declared thresholds. Both are process claims unless something
enforces them -- these tests check that the enforcement actually bites.
"""

import dataclasses
import json

import pytest

from rhmoo.config import load_config
from rhmoo.preflight import PreflightError, check_experiment

pytestmark = pytest.mark.smoke


def test_committed_configs_pass_preflight():
    for path in ("configs/main.yaml", "configs/smoke.yaml", "configs/dress_rehearsal.yaml"):
        check_experiment(load_config(path))


def test_an_edited_battery_threshold_is_refused():
    config = load_config("configs/main.yaml")
    tampered = dict(config.implausibility_battery)
    tampered["mw_max"] = 700  # the superseded Phase 8 value
    config = dataclasses.replace(config, implausibility_battery=tampered)

    with pytest.raises(PreflightError, match="does not match the frozen calibration"):
        check_experiment(config)


def test_a_missing_calibration_is_refused_with_the_remedy(tmp_path):
    config = load_config("configs/main.yaml")
    with pytest.raises(PreflightError, match="make calibrate"):
        check_experiment(config, frozen_path=tmp_path / "absent.json")


def test_a_tampered_arm_r_artifact_is_refused(tmp_path, monkeypatch):
    """The calibration set and the reported arm R must be the same bytes."""
    import shutil
    from pathlib import Path

    import rhmoo.preflight as preflight

    staging = tmp_path / "calibration"
    staging.mkdir()
    shutil.copy("calibration/arm_r_scored.parquet", staging / "arm_r_scored.parquet")
    manifest = json.loads(Path("calibration/arm_r_manifest.json").read_text())
    manifest["artifact_sha256"] = "0" * 64
    (staging / "arm_r_manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    config = load_config("configs/main.yaml")
    with pytest.raises(PreflightError, match="does not match its manifest hash"):
        preflight.check_arm_r_artifact(config)


def test_a_missing_scorer_variant_is_refused():
    config = load_config("configs/main.yaml")
    scorers = dataclasses.replace(
        config.independent_scorers,
        variants=(*config.independent_scorers.variants, "data_control_imaginary"),
    )
    config = dataclasses.replace(config, independent_scorers=scorers)

    with pytest.raises(PreflightError, match="train-independent-scorer"):
        check_experiment(config)
