"""Pre-run invariant checks.

Brief SS4.5 requires the implausibility battery to be calibrated against arm R
and frozen before any arm A molecule is generated, and brief SS9 requires
pre-declared thresholds. Those are process claims unless something enforces
them, so the runner calls `check_experiment` first and refuses to start when an
invariant is violated.

This is what turns "we calibrated before running" from a statement in a
document into a property of the program.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rhmoo.config import ExperimentConfig
from rhmoo.independent_scorer import model_dir

REPO_ROOT = Path(__file__).resolve().parents[2]


class PreflightError(RuntimeError):
    """Raised when a pre-run invariant fails. Carries the remedy, not just the fault."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def check_battery_is_frozen(config: ExperimentConfig, frozen_path: str | Path) -> None:
    frozen_path = _resolve(frozen_path)
    if not frozen_path.exists():
        raise PreflightError(
            f"the implausibility battery has not been calibrated: {frozen_path} is missing.\n"
            "Run `make arm-r` then `make calibrate` (brief SS4.5 -- blocking before any arm A run)."
        )

    frozen = json.loads(frozen_path.read_text())["implausibility_battery"]
    active = config.implausibility_battery

    if set(frozen) != set(active):
        raise PreflightError(
            f"config battery keys {sorted(active)} do not match the frozen "
            f"calibration keys {sorted(frozen)}."
        )
    differing = {k: (active[k], frozen[k]) for k in frozen if float(active[k]) != float(frozen[k])}
    if differing:
        raise PreflightError(
            "the config's implausibility_battery does not match the frozen calibration "
            f"({frozen_path}):\n"
            + "\n".join(f"  {k}: config={c!r} frozen={f!r}" for k, (c, f) in sorted(differing.items()))
            + "\nChanging a pre-declared threshold is a new experiment (brief SS9). Either restore the "
            "frozen values or re-run `make calibrate` and commit the new freeze."
        )


def check_arm_r_artifact(config: ExperimentConfig) -> None:
    artifact = _resolve(config.reference_arm.artifact_path)
    manifest_path = artifact.parent / "arm_r_manifest.json"

    if not artifact.exists():
        raise PreflightError(
            f"arm R artifact missing at {artifact}. Run `make arm-r` (brief SS3.3)."
        )
    if not manifest_path.exists():
        raise PreflightError(f"arm R manifest missing at {manifest_path}. Re-run `make arm-r`.")

    manifest = json.loads(manifest_path.read_text())
    actual = sha256_file(artifact)
    if manifest.get("artifact_sha256") != actual:
        raise PreflightError(
            f"arm R artifact at {artifact} does not match its manifest hash "
            f"(expected {manifest.get('artifact_sha256')}, got {actual}).\n"
            "The calibration set and the reported arm R must be the same bytes; re-run `make arm-r` "
            "and `make calibrate` together."
        )

    source = Path(manifest["source_file"])
    if source.exists() and sha256_file(source) != manifest["source_sha256"]:
        raise PreflightError(
            f"the approved-drug source {source} has changed since arm R was scored. "
            "Re-run `make arm-r` and `make calibrate`."
        )

    declared = set(manifest.get("independent_scorer_variants", []))
    if declared != set(config.independent_scorers.variants):
        raise PreflightError(
            f"arm R was scored with independent scorer variants {sorted(declared)} but the config "
            f"declares {sorted(config.independent_scorers.variants)}. Re-run `make arm-r`."
        )


def check_scorer_artifacts(config: ExperimentConfig) -> None:
    for variant in config.independent_scorers.variants:
        directory = model_dir(variant)
        if not directory.exists():
            raise PreflightError(
                f"independent scorer variant {variant!r} is missing at {directory}. "
                "Run `make train-independent-scorer`."
            )
        for filename in ("metadata.json", "herg_lightgbm.txt", "solubility_aqsoldb_lightgbm.txt"):
            if not (directory / filename).exists():
                raise PreflightError(f"scorer variant {variant!r} is missing {filename} in {directory}.")


def check_experiment(config: ExperimentConfig, frozen_path: str | Path = "calibration/battery_frozen.json") -> None:
    """All pre-run invariants. Raises `PreflightError` with the remedy."""
    check_scorer_artifacts(config)
    check_arm_r_artifact(config)
    check_battery_is_frozen(config, frozen_path)
